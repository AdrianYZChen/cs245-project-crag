import os
from collections import defaultdict
from typing import Any, Dict, List

import numpy as np
import ray
import torch
import vllm
from rank_bm25 import BM25Okapi
from sentence_transformers import SentenceTransformer

from openai import OpenAI

from tqdm import tqdm

from rag_baseline import ChunkExtractor

#### CONFIG PARAMETERS ---

# Define the number of context sentences to consider for generating an answer.
NUM_CONTEXT_SENTENCES = 20
# Set the maximum length for each context sentence (in characters).
MAX_CONTEXT_SENTENCE_LENGTH = 1000
# Set the maximum context references length (in characters).
MAX_CONTEXT_REFERENCES_LENGTH = 4000

# Batch size you wish the evaluators will use to call the `batch_generate_answer` function
AICROWD_SUBMISSION_BATCH_SIZE = 16 # TUNE THIS VARIABLE depending on the number of GPUs you are requesting and the size of your model.

# VLLM Parameters 
VLLM_TENSOR_PARALLEL_SIZE = 1 # TUNE THIS VARIABLE depending on the number of GPUs you are requesting and the size of your model.
VLLM_GPU_MEMORY_UTILIZATION = 0.85 # TUNE THIS VARIABLE depending on the number of GPUs you are requesting and the size of your model.

# Sentence Transformer Parameters
SENTENTENCE_TRANSFORMER_BATCH_SIZE = 32 # TUNE THIS VARIABLE depending on the size of your embedding model and GPU mem available

#### CONFIG PARAMETERS END---

class RAGModelOurs:
    """
    An example RAGModel for the KDDCup 2024 Meta CRAG Challenge
    which includes all the key components of a RAG lifecycle.
    """
    def __init__(
        self, 
        llm_name="meta-llama/Llama-3.2-3B-Instruct", 
        is_server=False, 
        vllm_server=None, 
        use_rephrase=False, 
        bm25_score_ratio=0.0, 
        use_scores_for_prompt=False,
    ):
        self.initialize_models(llm_name, is_server, vllm_server)
        self.chunk_extractor = ChunkExtractor()
        
        self.use_rephrase = use_rephrase
        self.bm25_score_ratio = bm25_score_ratio
        self.use_scores_for_prompt = use_scores_for_prompt
        
    def initialize_models(self, llm_name, is_server, vllm_server):
        self.llm_name = llm_name
        self.master_cache_dir = "/local2/shared_cache_huggingface"
        self.llm_full_name = os.path.join(self.master_cache_dir, llm_name)
        self.is_server = is_server
        self.vllm_server = vllm_server

        if self.is_server:
            # initialize the model with vllm server
            openai_api_key = "EMPTY"
            openai_api_base = self.vllm_server
            self.llm_client = OpenAI(
                api_key=openai_api_key,
                base_url=openai_api_base,
            )
        else:
            # initialize the model with vllm offline inference
            self.llm = vllm.LLM(
                model=self.llm_full_name,
                worker_use_ray=True,
                tensor_parallel_size=VLLM_TENSOR_PARALLEL_SIZE,
                gpu_memory_utilization=VLLM_GPU_MEMORY_UTILIZATION,
                trust_remote_code=True,
                dtype="half",  # note: bfloat16 is not supported on nvidia-T4 GPUs
                enforce_eager=True
            )
            self.tokenizer = self.llm.get_tokenizer()

        # Load a sentence transformer model optimized for sentence embeddings, using CUDA if available.
        self.sentence_model = SentenceTransformer(
            "all-MiniLM-L6-v2",
            device=torch.device(
                "cuda" if torch.cuda.is_available() else "cpu"
            ),
        )

    def calculate_embeddings(self, sentences):
        """
        Compute normalized embeddings for a list of sentences using a sentence encoding model.

        This function leverages multiprocessing to encode the sentences, which can enhance the
        processing speed on multi-core machines.

        Args:
            sentences (List[str]): A list of sentences for which embeddings are to be computed.

        Returns:
            np.ndarray: An array of normalized embeddings for the given sentences.

        """
        embeddings = self.sentence_model.encode(
            sentences=sentences,
            normalize_embeddings=True,
            batch_size=SENTENTENCE_TRANSFORMER_BATCH_SIZE,
        )
        # Note: There is an opportunity to parallelize the embedding generation across 4 GPUs
        #       but sentence_model.encode_multi_process seems to interefere with Ray
        #       on the evaluation servers. 
        #       todo: this can also be done in a Ray native approach.
        #       
        return embeddings

    def get_batch_size(self) -> int:
        """
        Determines the batch size that is used by the evaluator when calling the `batch_generate_answer` function.
        
        The evaluation timeouts linearly scale with the batch size. 
            i.e.: time out for the `batch_generate_answer` call = batch_size * per_sample_timeout 
        

        Returns:
            int: The batch size, an integer between 1 and 16. It can be dynamic
                 across different batch_generate_answer calls, or stay a static value.
        """
        self.batch_size = AICROWD_SUBMISSION_BATCH_SIZE  
        return self.batch_size

    def rephrase_queries(self, queries: List[str]) -> List[str]:
        messages = [
            {"role": "system", 
             "content": """You are a search query optimization expert. Rephrase questions to:
             1. Include key entities and concepts
             2. Remove unnecessary words
             3. Make implicit context explicit
             4. Break complex queries into key aspects
             Keep responses concise and focused."""},
            {"role": "user", 
             "content": "Rephrase these questions:\n" + "\n".join([f"{i+1}. {query}" for i, query in enumerate(queries)])}
        ]
        
        try:
            openai_client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))  # Get API key from environment
            response = openai_client.chat.completions.create(
                model="gpt-4o-mini", 
                messages=messages,
                temperature=0.2,  # Lower temperature for more focused responses
                max_tokens=100 * len(queries)
            )
            rephrased_queries = response.choices[0].message.content.strip().split("\n")
            # Remove numbering from rephrased queries
            rephrased_queries = [q.split(". ", 1)[1] if ". " in q else q for q in rephrased_queries]
            
            return rephrased_queries
        except:
            # Fallback to original queries if API call fails
            print(f"Failed to rephrase queries: {queries}")
            return queries

    def batch_generate_answer(self, batch: Dict[str, Any]) -> List[str]:
        """
        Generates answers for a batch of queries using associated (pre-cached) search results and query times.

        Parameters:
            batch (Dict[str, Any]): A dictionary containing a batch of input queries with the following keys:
                - 'interaction_id;  (List[str]): List of interaction_ids for the associated queries
                - 'query' (List[str]): List of user queries.
                - 'search_results' (List[List[Dict]]): List of search result lists, each corresponding
                                                      to a query. Please refer to the following link for
                                                      more details about the individual search objects:
                                                      https://gitlab.aicrowd.com/aicrowd/challenges/meta-comprehensive-rag-benchmark-kdd-cup-2024/meta-comphrehensive-rag-benchmark-starter-kit/-/blob/master/docs/dataset.md#search-results-detail
                - 'query_time' (List[str]): List of timestamps (represented as a string), each corresponding to when a query was made.

        Returns:
            List[str]: A list of plain text responses for each query in the batch. Each response is limited to 75 tokens.
            If the generated response exceeds 75 tokens, it will be truncated to fit within this limit.

        Notes:
        - If the correct answer is uncertain, it's preferable to respond with "I don't know" to avoid
          the penalty for hallucination.
        - Response Time: Ensure that your model processes and responds to each query within 30 seconds.
          Failing to adhere to this time constraint **will** result in a timeout during evaluation.
        """
        batch_interaction_ids = batch["interaction_id"]
        queries = batch["query"]
        batch_search_results = batch["search_results"]
        query_times = batch["query_time"]

        # Chunk all search results using ChunkExtractor
        chunks, chunk_interaction_ids = self.chunk_extractor.extract_chunks(
            batch_interaction_ids, batch_search_results
        )

        # Calculate all chunk embeddings
        chunk_embeddings = self.calculate_embeddings(chunks)

        # Calculate embeddings for queries
        query_embeddings = self.calculate_embeddings(queries)
        
        # Calculate rephrased queries if needed
        if self.use_rephrase:
            rephrased_queries = self.rephrase_queries(queries)
            query_embeddings = self.calculate_embeddings(rephrased_queries)

        # Retrieve top matches for the whole batch
        batch_retrieval_results = []
        cosine_scores = []
        bm25_scores = []
        for _idx, interaction_id in enumerate(batch_interaction_ids):
            query = queries[_idx]
            if self.use_rephrase:
                rephrased_query = rephrased_queries[_idx]
            query_time = query_times[_idx]
            query_embedding = query_embeddings[_idx]

            # Identify chunks that belong to this interaction_id
            relevant_chunks_mask = chunk_interaction_ids == interaction_id

            # Filter out the said chunks and corresponding embeddings
            relevant_chunks = chunks[relevant_chunks_mask]
            relevant_chunks_embeddings = chunk_embeddings[relevant_chunks_mask]

            # Calculate cosine similarity between query and chunk embeddings,
            cosine_scores.append((relevant_chunks_embeddings * query_embedding).sum(1))
            
            # Calculate BM25 scores
            corpus = [chunk.split() for chunk in relevant_chunks]
            bm25 = BM25Okapi(corpus)
            if self.use_rephrase:
                bm25_scores.append(np.array(bm25.get_scores(rephrased_query.split())))
            else:
                bm25_scores.append(np.array(bm25.get_scores(query.split())))

            # Combine cosine and BM25 scores
            weighted_scores = (1 - self.bm25_score_ratio) * cosine_scores[_idx] + self.bm25_score_ratio * bm25_scores[_idx]

            # and retrieve top-N results.
            retrieval_results = relevant_chunks[
                (-weighted_scores).argsort()[:NUM_CONTEXT_SENTENCES]
            ]
            
            # You might also choose to skip the steps above and 
            # use a vectorDB directly.
            batch_retrieval_results.append(retrieval_results)
            
        # Prepare formatted prompts from the LLM
        formatted_prompts = self.format_prompts(
            queries, query_times, batch_retrieval_results, 
            use_scores=self.use_scores_for_prompt, cosine_scores=cosine_scores, bm25_scores=bm25_scores
        )

        # Generate responses via vllm
        # note that here self.batch_size = 1
        if self.is_server:
            answers = []
            for prompt in formatted_prompts:
                response = self.llm_client.chat.completions.create(
                    model=self.llm_name,
                    messages=prompt,
                    n=1,  # Number of output sequences to return for each prompt.
                    top_p=0.9,  # Float that controls the cumulative probability of the top tokens to consider.
                    temperature=0.1,  # randomness of the sampling
                    # skip_special_tokens=True,  # Whether to skip special tokens in the output.
                    max_tokens=50,  # Maximum number of tokens to generate per output sequence.
                )
                answers.append(response.choices[0].message.content)
        else:
            responses = self.llm.generate(
                formatted_prompts,
                vllm.SamplingParams(
                    n=1,  # Number of output sequences to return for each prompt.
                    top_p=0.9,  # Float that controls the cumulative probability of the top tokens to consider.
                    temperature=0.1,  # randomness of the sampling
                    skip_special_tokens=True,  # Whether to skip special tokens in the output.
                    max_tokens=50,  # Maximum number of tokens to generate per output sequence.
                ),
                use_tqdm=False
            )
            answers = []
            for response in responses:
                answers.append(response.outputs[0].text)

        return answers

    def format_prompts(self, queries, query_times, batch_retrieval_results=[], 
                       use_scores=False, cosine_scores=None, bm25_scores=None):
        """
        Formats queries, corresponding query_times and retrieval results using the chat_template of the model.
            
        Parameters:
        - queries (List[str]): A list of queries to be formatted into prompts.
        - query_times (List[str]): A list of query_time strings corresponding to each query.
        - batch_retrieval_results (List[str])
        - use_scores (bool): Whether to use cosine and BM25 scores to determine the relevance of the references to the question.
        - cosine_scores (List[float]): A list of cosine scores corresponding to each query.
        - bm25_scores (List[float]): A list of BM25 scores corresponding to each query.
        """        
        
        if use_scores:
            assert cosine_scores is not None and bm25_scores is not None
            assert len(cosine_scores) == len(bm25_scores) == len(batch_retrieval_results)

        system_prompt = """
            You are provided with a question and various references. 
            Your task is to answer the question succinctly, using the fewest words possible. 
            If the question is not well-defined or does not make sense, respond with 'invalid question'.
            If the references do not contain the necessary information to answer the question, respond with 'I don't know'. 
            There is no need to explain the reasoning behind your answers.
        """
        if use_scores:
            system_prompt += """
                You are also provided with the cosine and BM25 scores between the query and the references. 
                Use these scores to determine the relevance of the references to the question.
            """
        formatted_prompts = []
        
        for _idx, query in enumerate(queries):
            query_time = query_times[_idx]
            retrieval_results = batch_retrieval_results[_idx]

            user_message = ""
            references = ""
            
            if len(retrieval_results) > 0:
                references += "# References \n"
                # Format the top sentences as references in the model's prompt template.
                if use_scores:
                    for _snippet_idx, (snippet, cosine_score, bm25_score) in enumerate(zip(retrieval_results, cosine_scores[_idx], bm25_scores[_idx])):
                        references += f"- {snippet.strip()}\n- Cosine Score: {cosine_score:.4f}, BM25 Score: {bm25_score:.4f}\n"
                else:
                    for _snippet_idx, snippet in enumerate(retrieval_results):
                        references += f"- {snippet.strip()}\n"
            
            references = references[:MAX_CONTEXT_REFERENCES_LENGTH]
            # Limit the length of references to fit the model's input size.

            user_message += f"{references}\n------\n\n"
            user_message += f"Using only the references listed above, answer the following question: \n"
            user_message += f"Current Time: {query_time}\n"
            user_message += f"Question: {query}\n"

            if self.is_server:
                # there is no need to wrap the messages into chat when using the server
                # because we use the chat API: chat.completions.create
                formatted_prompts.append(
                    [
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": user_message},
                    ]
                )
            else:
                formatted_prompts.append(
                    self.tokenizer.apply_chat_template(
                        [
                            {"role": "system", "content": system_prompt},
                            {"role": "user", "content": user_message},
                        ],
                        tokenize=False,
                        add_generation_prompt=True,
                    )
                )

        return formatted_prompts
