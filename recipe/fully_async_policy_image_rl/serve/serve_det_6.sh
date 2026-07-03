#!/bin/bash

# 메인 스크립트 종료 시 모든 자식 프로세스(detector들)를 함께 종료
trap "kill 0" EXIT

# 환경 변수 설정
GDINO_MODEL_PATH=/home/work/AGILAB/mllm_reasoning/data/checkpoints/mm_grounding_dino_large_all

# 대상 GPU 번호들
GPUS=(6)

for i in "${GPUS[@]}"; do
    port=$((8080 + i))
    
    # 각 GPU별로 무한 루프를 백그라운드에서 실행
    (
        while true; do
            echo "[GPU $i] Starting detector on port $port..."
            
            CUDA_VISIBLE_DEVICES=$i python recipe/image_rl/detector.py \
                --gdino_ckpt_path "$GDINO_MODEL_PATH" \
                --host 0.0.0.0 \
                --port $port
            
            EXIT_CODE=$?
            echo "[GPU $i] Detector crashed (Exit Code: $EXIT_CODE). Restarting in 5 seconds..."
            sleep 5
        done
    ) &
done

echo "Detectors on GPUs ${GPUS[*]} are running with auto-restart."
echo "Press Ctrl+C to stop all processes."

# 모든 백그라운드 프로세스가 종료될 때까지 대기
wait