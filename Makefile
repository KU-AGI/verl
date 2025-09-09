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