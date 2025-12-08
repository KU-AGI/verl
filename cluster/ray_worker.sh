export MASTER_ADDR= # $(hostname -I | awk '{print $1}')
export MASTER_PORT=6379
export NUM_GPUS=8
export GLOO_SOCKET_IFNAME="bond-srv.1518"

# ray stop --force
# rm -rf /tmp/ray

ray start \
  --node-ip-address=$MASTER_ADDR \
  --address=${MASTER_ADDR}:${MASTER_PORT} \
  --num-gpus=${NUM_GPUS}