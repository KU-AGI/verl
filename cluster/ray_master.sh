export MASTER_ADDR=$(hostname -I | awk '{print $1}')
export MASTER_PORT=6379
export DASHBOARD_PORT=8265
export NUM_GPUS=8
export GLOO_SOCKET_IFNAME="bond-srv.1518"

# ray stop --force
# rm -rf /tmp/ray

ray start --head \
  --node-ip-address=$MASTER_ADDR \
  --port=$MASTER_PORT \
  --dashboard-host=0.0.0.0 \
  --dashboard-port=$DASHBOARD_PORT \
  --include-dashboard=True \
  --num-gpus=${NUM_GPUS} \
  --disable-usage-stats