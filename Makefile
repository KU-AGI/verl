CONTAINER_NAME=verl-vllm-$(USER)
IMAGE_NAME_TAG=verlai/verl:vllm015.dev
# CONTAINER_NAME=verl-sglang-$(USER)
# IMAGE_NAME_TAG=verlai/verl:sgl059.latest
HUGGING_FACE_HUB_TOKEN=hf_xx
CACHE_PATH=/data/.cache

init-container:
	docker run -d \
	--init \
	--gpus all \
	--network host \
	-v ${PWD}:/verl \
	-v /data:/data \
	-v /home:/home \
	-v /data/.cache:/root/.cache \
	--shm-size=10g \
	--ulimit memlock=-1 \
	--name $(CONTAINER_NAME) \
	$(IMAGE_NAME_TAG) \
	tail -f /dev/null
