nohup python generate.py --dataset_path data/crag_task_1_dev_v4_release.jsonl.bz2 --model_name rag_baseline --is_server --split 1 --gpu "0" > outputs/generate_rag_baseline.log 2>&1 &
nohup python generate.py --dataset_path data/crag_task_1_dev_v4_release.jsonl.bz2 --model_name vanilla_baseline --is_server --split 1 --gpu "0" > outputs/generate_vanilla_baseline.log 2>&1 &
nohup python generate.py --dataset_path data/crag_task_1_dev_v4_release.jsonl.bz2 --model_name rag_ours --is_server --split 1 --gpu "0" > outputs/generate_rag_ours.log 2>&1 &

nohup python evaluate.py --dataset_path data/crag_task_1_dev_v4_release.jsonl.bz2 --model_name rag_baseline --is_server --gpu "0" > outputs/evaluate_rag_baseline.log 2>&1 &
nohup python evaluate.py --dataset_path data/crag_task_1_dev_v4_release.jsonl.bz2 --model_name vanilla_baseline --is_server --gpu "0" > outputs/evaluate_vanilla_baseline.log 2>&1 &
nohup python evaluate.py --dataset_path data/crag_task_1_dev_v4_release.jsonl.bz2 --model_name rag_ours --is_server --gpu "0" > outputs/evaluate_rag_ours.log 2>&1 &