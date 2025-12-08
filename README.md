
# Main Docs

- https://verl.readthedocs.io/en/latest/

# Setup

## Using Docker

- docker image pull
    - https://hub.docker.com/r/verlai/verl/tags
    ```bash
    docker pull verlai/verl:app-verl0.5-transformers4.55.4-vllm0.10.0-mcore0.13.0-te2.2
    ```

- docker run
    
    ```bash
    make init-container # single node
    make init-container-with-infiniband # InfiniBand server
    ```

## Ray clustering

```python
cluster  
  ├── ray_master.sh
  └── ray_worker.sh
```

- For head node, run `ray_master.sh`
- For worker node, run `ray_worker.sh`
