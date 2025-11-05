# export MASTER_ADDR=$(hostname -I | awk '{print $1}')
export MASTER_ADDR=10.10.0.1
export MASTER_PORT=6379
export GLOO_SOCKET_IFNAME="ibp26s0"

export NUM_GPUS=8

ray start \
  --node-ip-address=10.10.0.2 \
  --address=${MASTER_ADDR}:${MASTER_PORT} \
  --num-gpus=${NUM_GPUS}