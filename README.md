# Main Docs

- https://verl.readthedocs.io/en/latest/

# Setup

## Using Docker

- docker image pull
    - https://hub.docker.com/r/verlai/verl/tags
    
    ```bash
    docker pull verlai/verl:<image tag>
    ```
    
- docker run
    
    ```bash
    # Makefile example
    
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
    ```
    
- Install

    ```
    bash scripts/install_vllm_sglang_mcore.sh
    ```
    
## Ray clustering

```python
cluster  
  ├── ray_master.sh
  └── ray_worker.sh
```

- For head node, run `ray_master.sh`
- For worker node, run `ray_worker.sh`

## Using conda

- https://verl.readthedocs.io/en/latest/start/install.html#install-dependencies

# Core Code

### Data Preprocess

- https://verl.readthedocs.io/en/latest/preparation/prepare_data.html
    - `data_source` : reward function class를 구분짓는 중요한 요소
    - `prompt` : List of “role”, “content” key messages ← generation을 위한 user prompt
    - `reward_model`
        - style “rule” : exact match (math) ← X model base
        - reward model base면 컬럼 없애기 → [참조](https://www.notion.so/verl-Recipe-26b7e0e795c9807b8de0e225f12bf94c?pvs=21)
- example/data_preprocess
    - chem_dataset.py
        
        ```python
        """
        Preprocess the syntheticreact dataset to parquet format
        """
        
        import argparse
        import os
        import re
        
        import datasets
        
        if __name__ == "__main__":
            parser = argparse.ArgumentParser()
            parser.add_argument("--local_dir", default="/data/verl/data/chem_dapo")
            parser.add_argument("--train", action="store_true")
            parser.add_argument("--test", action="store_true")
        
            args = parser.parse_args()
        
            if not os.path.exists(args.local_dir):
                os.makedirs(args.local_dir)
        
            data_source = "chem_dapo"
        
            if args.train:
                dataset = datasets.load_from_disk(f"{args.local_dir}/train", "main")
            elif args.test:
                dataset = datasets.load_from_disk(f"{args.local_dir}/test", "main")
            else:
                raise ValueError("Either --train or --test must be specified")
        
            # add a row to each data item that represents a unique id
            def make_map_fn(split):
                def process_fn(example, idx):
                    messages = example.pop("messages")
                    solution = example.pop("answers")
                    
                    # Choose extra columns
                    extra_columns = ["rxn_str", "reactants", "reagents", "products", "solvent", "yields", "class_name"]
                    
                    data = {
                        "data_source": data_source,
                        "prompt": messages,
                        "ability": "chemistry",
                        "reward_model": {"style": "rule", "ground_truth": solution},
                        "extra_info": {
                            k: v for k, v in example.items() if k in extra_columns
                        },
                    }
                    return data
        
                return process_fn
        
            if args.train:
                dataset = dataset.map(function=make_map_fn("train"), with_indices=True, load_from_cache_file=False)
                dataset.to_parquet(os.path.join(args.local_dir, "syntheticreact_9k_train.parquet"))
            elif args.test:
                dataset = dataset.map(function=make_map_fn("test"), with_indices=True, load_from_cache_file=False)
                dataset.to_parquet(os.path.join(args.local_dir, "syntheticreact_3k_test.parquet"))
        
        ```
        

### Rule-Based Reward Function

- verl/utils/reward_score
    - \__init__.py
        
        ```python
        from verl.utils.import_utils import deprecated
        
        def default_compute_score(
            data_source,
            solution_str,
            ground_truth,
            extra_info=None,
            sandbox_fusion_url=None,
            concurrent_semaphore=None,
            memory_limit_mb=None,
        ):
        		...
            elif **data_source** == "chem_dapo": <- add point
                from . import chem_dapo <- verl/utils/reward_score에 func 추가
        
                res = chem_dapo.**compute_score**(solution_str, ground_truth)
        		...
        ```
        
- verl/utils/reward_score
    - chem_dapo.py (**compute_score**)
        
        ```python
        import re
        from typing import Optional
        from rdkit import Chem, RDLogger
        RDLogger.DisableLog('rdApp.*')
        
        ANSWER_TAG = r"<ANSWER>(.*?)</ANSWER>"
        
        def exact_match(ot_smi, gt_smi):
            """SMILES exact match"""
            try:
                m_out = Chem.MolFromSmiles(ot_smi)
                m_gt = Chem.MolFromSmiles(gt_smi)
                
                if m_out is None or m_gt is None:
                    return 0
                    
                if Chem.MolToInchi(m_out) == Chem.MolToInchi(m_gt):
                    return 1
            except:
                pass
            return 0
        
        def remove_tag(s: str) -> str:
            """Remove LaTeX tags from a string.
        
            Args:
                tags: String with LaTeX tags
        
            Returns:
                String without LaTeX tags
            """
            smiles_match = re.search(ANSWER_TAG, s, re.DOTALL)
            if smiles_match:
                tags = smiles_match.group(1)
            else:
                tags = s
            return tags.strip()
        
        def is_correct_strict_tag(
            pred: str, gt: str
        ) -> tuple[int, Optional[str]]:
            """Check if the prediction is correct using strict boxed answer criteria.
        
            Args:
                pred: The prediction string
                gt: The ground truth answer
                pause_tokens_index: Indices of pause tokens
        
            Returns:
                Tuple of (score, extracted_prediction)
            """
            # Extract and check the boxed answer
            extracted_pred = remove_tag(pred) if pred is not None else None
        
            return 1 if exact_match(extracted_pred, gt) else -1, extracted_pred
        
        def verify(
            solution_str: str, answer: str
        ) -> bool:
            """Verify if the solution is correct.
        
            Args:
                solution_str: The solution string to verify
                answer: The ground truth answer
        
            Returns:
                True if the solution is correct, False otherwise
            """
            correct, pred = is_correct_strict_tag(solution_str, answer)
            return correct == 1, pred
        
        def compute_score(
            solution_str: str,
            ground_truth: str,
        ) -> float:
            """Compute the reward score for a solution.
        
            Args:
                solution_str: The solution string
                ground_truth: The ground truth answer
                strict_tag_verify: Whether to use strict tag verification
                pause_tokens_index: Indices of pause tokens
        
            Returns:
                Reward score (1.0 for correct, -1.0 for incorrect)
            """
            # Verify the solution
            correct, pred = verify(solution_str, ground_truth)
        
            reward = 1.0 if correct else -1.0
            acc = correct
        
            return {
                "score": reward,
                "acc": acc,
                "pred": pred,
            }
        ```
        

### Generative Reward Model-Based

- Model serving 후 reward 계산 시 request
    - https://github.com/volcengine/verl/tree/main/recipe/genrm_remote
    - reward_function.py
        
        ```python
        from concurrent.futures import ThreadPoolExecutor
        from time import sleep
        
        import requests
        
        from verl.utils.reward_score.math_reward import last_boxed_only_string, remove_boxed
        
        BASE_URL = "http://localhost:30000"
        API_KEY = "EMPTY"
        MAX_RETRIES = 3
        BASE_DELAY = 2
        MAX_WORKERS = 32
        MODEL_NAME = "genrm-demo"
        GENRM_PROMPT_TEMPLATE = """
        The following is a math problem and an AI solution:
        
        [Math Problem]
        
        {problem}
        
        [AI Solution]
        
        {solution}
        
        Your task is to review and critique the solution step by step, and output whether the AI solution is correct.
        
        Please put your final answer (i.e., 'True' or 'False') in \\boxed{{}}.
        """.strip()
        
        def get_response(problem, solution_str, ground_truth):
            prompt = GENRM_PROMPT_TEMPLATE.format(problem=problem, solution=solution_str)
            messages = [{"role": "user", "content": prompt}]
            for attempt in range(MAX_RETRIES):
                try:
                    headers = {"Content-Type": "application/json"}
                    chat_url = f"{BASE_URL}/v1/chat/completions"
                    data = {"model": MODEL_NAME, "messages": messages}
                    output = requests.post(chat_url, headers=headers, json=data, timeout=30)
                    response = output.json()["choices"][0]["message"]["content"]
                    return response
                except Exception as e:
                    if attempt < MAX_RETRIES - 1:
                        print("Exception: ", repr(e))
                        delay = BASE_DELAY * (2**attempt)
                        print(f"Retrying in {delay} seconds...")
                        sleep(delay)
                    else:
                        print(f"Failed after {MAX_RETRIES} attempts. Error: {e}")
        
            raise ConnectionRefusedError(f"Failed to run the model for {prompt}!")
        
        def compute_reward(response):
            reward_score = 0.0
            try:
                boxed_result = last_boxed_only_string(response)
                if boxed_result is not None:
                    result = remove_boxed(boxed_result)
                    reward_score = float(result == "True")
            except Exception as e:
                print(e)
            return reward_score
        
        def compute_score(data_source, solution_str, ground_truth, extra_info):
            split = extra_info["split"]
            from verl.utils.reward_score import default_compute_score
        
            func_rm_score = default_compute_score(data_source, solution_str, ground_truth, extra_info)
        
            if split == "test":
                return func_rm_score
            else:
                problem = extra_info["question"]
                response = get_response(problem, solution_str, ground_truth)
                if response is not None:
                    reward_score = compute_reward(response)
                else:
                    reward_score = 0.0
        
                return reward_score
        
        def compute_score_batch(data_sources, solution_strs, ground_truths, extra_infos):
            with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
                futures = []
                for data_source, solution_str, ground_truth, extra_info in zip(
                    data_sources, solution_strs, ground_truths, extra_infos, strict=True
                ):
                    future = executor.submit(compute_score, data_source, solution_str, ground_truth, extra_info)
                    futures.append(future)
        
                results = [future.result() for future in futures]
        
            return result
        ```