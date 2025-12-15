export MASTER_ADDR= # $(hostname -I | awk '{print $1}')
export MASTER_PORT=6379
export NUM_GPUS=8
export GLOO_SOCKET_IFNAME="bond-srv.1518"

# ray stop --force
# rm -rf /tmp/ray

RAY_memory_monitor_refresh_ms=0 ray start \
  --node-ip-address=$WORKER_ADDR \
  --address=${MASTER_ADDR}:${MASTER_PORT} \
  --num-gpus=${NUM_GPUS}