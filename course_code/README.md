# CRAG Project Runner

## Quick Start
To run the full ablation experiments:
```bash
python run_experiments.py
```

## Running Individual Experiments
You can run single experiments using `generate.py` and `evaluate.py` separately.

### Example Commands
```bash
# Generate predictions
python generate.py \
    --dataset_path data/crag_task_1_dev_v4_release.jsonl.bz2 \
    --model_name rag_ours \
    --llm_name meta-llama/Llama-3.2-3B-Instruct \
    --split 1 \
    --is_server \
    --gpu 0

# Evaluate results
python evaluate.py \
    --dataset_path data/crag_task_1_dev_v4_release.jsonl.bz2 \
    --model_name rag_ours \
    --llm_name meta-llama/Llama-3.2-3B-Instruct \
    --split 1 \
    --is_server \
    --gpu 0
```

## Configuration
To modify experiment parameters, edit the `rag_ours_config.json` file.

## Important Note
When using `generate.py` or `evaluate.py` with `is_server=True`, you must set your `OPENAI_API_KEY` in the respective files.