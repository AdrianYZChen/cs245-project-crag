import subprocess
import json
import time
from datetime import datetime
import os
from typing import List, Dict

def run_command(command: List[str], description: str = None) -> None:
    """Run a command and print its output in real-time."""
    if description:
        print(f"\n=== {description} ===")
    print(f"Running command: {' '.join(command)}")
    
    process = subprocess.Popen(
        command,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        universal_newlines=True
    )
    
    # Print output in real-time
    for line in process.stdout:
        print(line, end='')
    
    process.wait()
    if process.returncode != 0:
        raise subprocess.CalledProcessError(process.returncode, command)

def run_experiment(config: Dict) -> None:
    """Run a single experiment (generate + evaluate) with the given configuration."""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Save experiment config
    if config.get("rag_config"):
        with open("rag_ours_config.json", "w") as f:
            json.dump(config["rag_config"], f, indent=4)
    
    # Build generate command
    generate_cmd = [
        "python", "generate.py",
        "--dataset_path", config["dataset_path"],
        "--model_name", config["model_name"],
        "--llm_name", config["llm_name"],
    ]
    
    if config.get("split") is not None:
        generate_cmd.extend(["--split", str(config["split"])])
    if config.get("is_server"):
        generate_cmd.append("--is_server")
    if config.get("vllm_server"):
        generate_cmd.extend(["--vllm_server", config["vllm_server"]])
    if config.get("gpu"):
        generate_cmd.extend(["--gpu", config["gpu"]])
    
    # Run generate
    run_command(generate_cmd, f"Generating predictions for {config['experiment_name']}")
    
    # Build evaluate command
    evaluate_cmd = [
        "python", "evaluate.py",
        "--dataset_path", config["dataset_path"],
        "--model_name", config["model_name"],
        "--llm_name", config["llm_name"],
        "--timestamp", timestamp,
    ]
    
    if config.get("is_server"):
        evaluate_cmd.append("--is_server")
    if config.get("vllm_server"):
        evaluate_cmd.extend(["--vllm_server", config["vllm_server"]])
    if config.get("gpu"):
        evaluate_cmd.extend(["--gpu", config["gpu"]])
    
    # Run evaluate
    run_command(evaluate_cmd, f"Evaluating predictions for {config['experiment_name']}")

def main():
    # Define your experiment configurations
    experiments = [
        {
            "experiment_name": f"ours_rephrase_{use_rephrase}_bm25_{bm25_score_ratio}_promptScores_{use_scores_for_prompt}",
            "dataset_path": "data/crag_task_1_dev_v4_release.jsonl.bz2",
            "model_name": "rag_ours",
            "llm_name": "meta-llama/Llama-3.2-3B-Instruct",
            "split": 1,
            "is_server": True,
            "gpu": "0",
            "rag_config": {
                "use_rephrase": use_rephrase,
                "bm25_score_ratio": bm25_score_ratio,
                "use_scores_for_prompt": use_scores_for_prompt
            }
        }
        for use_rephrase in [True, False]
        for bm25_score_ratio in [0.0, 0.3]
        for use_scores_for_prompt in [True, False]
    ]
    
    # Run each experiment
    for config in experiments:
        try:
            print(f"\n\n=== Starting experiment: {config['experiment_name']} ===")
            run_experiment(config)
            print(f"=== Completed experiment: {config['experiment_name']} ===\n")
        except Exception as e:
            print(f"Error in experiment {config['experiment_name']}: {str(e)}")
            continue

if __name__ == "__main__":
    main() 