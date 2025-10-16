export MASTER_ADDR=$(hostname -I | awk '{print $1}')
export MASTER_PORT=6380
export DASHBOARD_PORT=8266
export NUM_GPUS=4

# ray stop --force
# rm -rf /tmp/ray

RAY_memory_monitor_refresh_ms=0 ray start --head \
  --port=$MASTER_PORT \
  --dashboard-host=0.0.0.0 \
  --dashboard-port=$DASHBOARD_PORT \
  --include-dashboard=True \
  --num-gpus=${NUM_GPUS} \
  --disable-usage-stats \
  --temp-dir="/data/ray"