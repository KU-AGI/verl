CONTAINER_NAME=verl-$(USER)
IMAGE_NAME_TAG=verlai/verl:base-v4-cu126-cudnn9.8-torch2.7.1-fa2.8.0-te2.3-fi0.2.6

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

CACHE_PATH=/data/.cache
MODEL_PATH=$(CACHE_PATH)/huggingface/hub/models--OpenGVLab/InternVL3_5-38B
VLLM_CONTAINER_NAME_PREFIX=vllm-g

# https://github.com/vllm-project/vllm/pull/22386
start-vllm-servers:
	for GPU in 0 ; do \
		PORT=$$((8000 + $$GPU)) ; \
		docker run --rm -d --name ${VLLM_CONTAINER_NAME_PREFIX}$$GPU \
			--runtime nvidia \
			--gpus all \
			-v /data:/data \
			-v /home:/home \
			-v /data/.cache:/root/.cache \
			-e CUDA_VISIBLE_DEVICES=$$GPU,$$(($$GPU + 1)) \
			-e VLLM_WORKER_MULTIPROC_METHOD=spawn \
			-p $${PORT}:8000 \
			--ipc=host \
			vllm/vllm-openai:v0.10.1 \
			--model ${MODEL_PATH} \
			--served-model-name OpenGVLab/InternVL3_5-38B \
			--trust-remote-code \
			--host 0.0.0.0 \
			--port 8000 \
			--tensor-parallel-size 2 ; \
	done

stop-vllm-servers:
	for GPU in 0 1 2 3 4 5 6 7 ; do \
		docker stop ${VLLM_CONTAINER_NAME_PREFIX}$$GPU || true ; \
	done
