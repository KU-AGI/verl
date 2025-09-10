export MASTER_ADDR=$(hostname -I | awk '{print $1}')
export MASTER_PORT=6379

export NUM_GPUS=8

ray start --address=${MASTER_ADD}:${MASTER_PORT} --num-gpus=${NUM_GPUS}