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
		-e NCCL_SOCKET_IFNAME="ibp26s0" \
		-e NCCL_P2P_LEVEL=NVL \
		-e NCCL_NET_GDR_LEVEL=0 \
		-e NCCL_CUDA_DEVICE_MAX_CONNECTIONS=8 \
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
