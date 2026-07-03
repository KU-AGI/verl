#!/bin/bash

# for i in 6 7; do
#     port=$((8000 + i))
#     CUDA_VISIBLE_DEVICES=$i python -m sglang.launch_server \
#         --model-path /home/work/AGILAB/mllm_reasoning/data/checkpoints/Qwen3.5-35B-A3B \
#         --served-model-name Qwen/Qwen3.5-35B-A3B \
#         --trust-remote-code \
#         --host 0.0.0.0 \
#         --port $port \
#         --mem-fraction-static 0.95 \
#         --max-running-requests 96 \
#         --chunked-prefill-size 8192 \
#         --max-prefill-tokens 32768 \
#         --schedule-conservativeness 0.1 \
#         --cuda-graph-max-bs 96 \
#         --mamba-ssm-dtype bfloat16 \
#         --attention-backend flashinfer \
#         --sampling-backend flashinfer &
# done
# wait

#!/bin/bash

# 종료 시 백그라운드 프로세스들도 함께 종료하기 위한 설정
# trap "kill 0" EXIT

# for i in 6 7; do
#     port=$((8000 + i))
    
#     # 각 GPU별로 무한 루프를 백그라운드에서 실행
#     (
#         while true; do
#             echo "[GPU $i] Starting sglang_server on port $port..."
            
#             CUDA_VISIBLE_DEVICES=$i python -m sglang.launch_server \
#                 --model-path /home/work/AGILAB/mllm_reasoning/data/checkpoints/Qwen3.5-35B-A3B \
#                 --served-model-name Qwen/Qwen3.5-35B-A3B \
#                 --trust-remote-code \
#                 --host 0.0.0.0 \
#                 --port $port \
#                 --reasoning-parser qwen3 \
#                 --mem-fraction-static 0.87 \
#                 --max-running-requests 48 \
#                 --chunked-prefill-size 4096 \
#                 --max-prefill-tokens 16384 \
#                 --schedule-conservativeness 0.1 \
#                 --mamba-ssm-dtype bfloat16 \
#                 --attention-backend flashinfer \
#                 --sampling-backend flashinfer
            
#             echo "[GPU $i] Server crashed with exit code $?. Restarting in 5 seconds..."
#             sleep 5
#         done
#     ) &
# done

# echo "All 8 servers are launching with auto-restart enabled."
# wait


# 종료 시 백그라운드 프로세스들도 함께 종료하기 위한 설정
trap "kill 0" EXIT

for i in 6; do
    port=$((8000 + i))
    
    # 각 GPU별로 무한 루프를 백그라운드에서 실행
    (
        while true; do
            echo "[GPU $i] Starting sglang_server on port $port..."
            
            CUDA_VISIBLE_DEVICES=$i python -m sglang.launch_server \
                --model-path /home/work/AGILAB/mllm_reasoning/data/checkpoints/Qwen3.5-35B-A3B \
                --served-model-name Qwen/Qwen3.5-35B-A3B \
                --trust-remote-code \
                --host 0.0.0.0 \
                --port $port \
                --mem-fraction-static 0.90 \
                --max-running-requests 32 \
                --chunked-prefill-size 2048 \
                --max-prefill-tokens 16384 \
                --schedule-conservativeness 0.1 \
                --cuda-graph-max-bs 16 \
                --mamba-ssm-dtype bfloat16 \
                --attention-backend flashinfer \
                --sampling-backend flashinfer
            
            echo "[GPU $i] Server crashed with exit code $?. Restarting in 5 seconds..."
            sleep 5
        done
    ) &
done

echo "All 8 servers are launching with auto-restart enabled."
wait