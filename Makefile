CONTAINER_NAME=verl-$(USER)
IMAGE_NAME_TAG=verlai/verl:app-verl0.5-transformers4.55.4-vllm0.10.0-mcore0.13.0-te2.2

init-container:
	docker run -d \
	--gpus all \
	--network host \
	-v ${PWD}:/verl \
	-v /data:/data \
	-v /home:/home \
	-v /data/.cache:/root/.cache \
	--shm-size=10g \
	--ulimit memlock=1 \
	--name $(CONTAINER_NAME) \
	$(IMAGE_NAME_TAG) \
	tail -f /dev/null

init-container-with-infiniband:
	docker run -d \
		--gpus all \
		--network host \
		--ipc=host \
		--shm-size=16g \
		--ulimit memlock=-1:-1 --ulimit stack=67108864 \
		--cap-add IPC_LOCK \
		--device /dev/infiniband \
		-e NCCL_IB_DISABLE=0 \
		-e NCCL_IB_HCA="mlx5_0,mlx5_4,mlx5_5,mlx5_8" \
		-e NCCL_CROSS_NIC=1 \
		-e NCCL_SOCKET_IFNAME="bond-srv.1518" \
		-e NCCL_P2P_LEVEL=NVL \
		-e NCCL_NET_GDR_LEVEL=0 \
		-e NCCL_CUDA_DEVICE_MAX_CONNECTIONS=1 \
		-e NCCL_DEBUG=INFO \
		-e NCCL_DEBUG_SUBSYS=INIT,NET,IB \
		-e NCCL_ASYNC_ERROR_HANDLING=1 \
		-e OMP_NUM_THREADS=4 \
		-v ${PWD}:/verl \
		-v /data:/data \
		-v /home:/home \
		-v /data/.cache:/root/.cache \
		-v /mnt/nvme1:/mnt/nvme1 \
		-v /mnt/nvme2:/mnt/nvme2 \
		-v /mnt/nvme3:/mnt/nvme3 \
		-v /mnt/nvme4:/mnt/nvme4 \
		--name $(CONTAINER_NAME) \
		$(IMAGE_NAME_TAG) \
		tail -f /dev/null

CACHE_PATH=/data/.cache
MODEL_PATH=Qwen/Qwen3-VL-30B-A3B-Instruct # OpenGVLab/InternVL3_5-38B
VLLM_CONTAINER_NAME_PREFIX=vllm-g

# https://github.com/vllm-project/vllm/pull/22386
start-vllm-servers:
	for GPU in 4 ; do \
		PORT=$$((8000 + $$GPU)) ; \
		docker run --rm -d --name ${VLLM_CONTAINER_NAME_PREFIX}$$GPU \
			--gpus all \
			-v /data:/data \
			-v /home:/home \
			-e HF_HOME=${CACHE_PATH} \
			-e CUDA_VISIBLE_DEVICES=$$GPU,$$(($$GPU + 1)) \
			-e VLLM_WORKER_MULTIPROC_METHOD=spawn \
			-p $${PORT}:8000 \
			--ipc=host \
			vllm/vllm-openai:v0.11.0 \
			--model ${MODEL_PATH} \
			--served-model-name ${MODEL_PATH} \
			--trust-remote-code \
			--host 0.0.0.0 \
			--port 8000 \
			--tensor-parallel-size 2 \
			--limit-mm-per-prompt.video 0 \
			--async-scheduling ; \
	done

stop-vllm-servers:
	for GPU in 0 1 2 3 4 5 6 7 ; do \
		docker stop ${VLLM_CONTAINER_NAME_PREFIX}$$GPU || true ; \
	done

DYNAMO_CONTAINER_NAME_PREFIX=dynamo-g

# https://github.com/ai-dynamo/dynamo/blob/main/examples/backends/vllm/launch/agg_multimodal.sh
# -e DYN_REQUEST_PLANE: NATS default 1MB max payload limit (multimodal base64 images can exceed this)
start-dynamo-vllm-servers:
	for GPU in 4 ; do \
		PORT=$$((8000 + $$GPU)) ; \
		docker run --rm -d --name ${DYNAMO_CONTAINER_NAME_PREFIX}$$GPU \
			--gpus all \
			--network host \
			--shm-size=10G \
			--ulimit memlock=-1 \
			--ulimit stack=67108864 \
			--ulimit nofile=65536:65536 \
			-w /workspace \
			--cap-add CAP_SYS_PTRACE \
			-v /data:/data \
			-v /home:/home \
			-e HF_HOME=${CACHE_PATH} \
			-e CUDA_VISIBLE_DEVICES=$$GPU,$$(($$GPU + 1)) \
			-e VLLM_WORKER_MULTIPROC_METHOD=spawn \
			-e DYN_REQUEST_PLANE=tcp \
			-p $${PORT}:8000 \
			--ipc=host \
			dynamo:latest-vllm \
			bash -lc " \
				python -m dynamo.frontend --http-port $${PORT} & \
				python -m dynamo.vllm \
					--enable-multimodal \
					--model ${MODEL_PATH} \
					--enforce-eager \
					--connector none \
					--trust-remote-code \
					--tensor-parallel-size 2 \
			" ; \
	done

# https://github.com/ai-dynamo/dynamo/blob/main/examples/backends/vllm/launch/agg_multimodal.sh
# -e DYN_REQUEST_PLANE: NATS default 1MB max payload limit (multimodal base64 images can exceed this)
start-dynamo-trt-servers:
	for GPU in 4 ; do \
		PORT=$$((8000 + $$GPU)) ; \
		docker run --rm -d --name ${DYNAMO_CONTAINER_NAME_PREFIX}$$GPU \
			--gpus all \
			--network host \
			--shm-size=10G \
			--ulimit memlock=-1 \
			--ulimit stack=67108864 \
			--ulimit nofile=65536:65536 \
			-w /workspace \
			--cap-add CAP_SYS_PTRACE \
			-v /data:/data \
			-v /home:/home \
			-e HF_HOME=${CACHE_PATH} \
			-e CUDA_VISIBLE_DEVICES=$$GPU,$$(($$GPU + 1)) \
			-e VLLM_WORKER_MULTIPROC_METHOD=spawn \
			-e DYN_REQUEST_PLANE=tcp \
			-p $${PORT}:8000 \
			--ipc=host \
			dynamo:latest-trtllm \
			bash -lc " \
				python -c 'import transformers; print(transformers.__version__)' && \
				python -m dynamo.frontend --http-port $${PORT} & \
				python -m dynamo.trtllm \
					--model-path ${MODEL_PATH} \
					--served-model-name ${MODEL_PATH} \
					--modality "multimodal" \
					--tensor-parallel-size 2 \
			" ; \
	done