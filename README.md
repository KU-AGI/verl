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
    # https://github.com/ranaroussi/quantstats/issues/365
    pip install numpy==1.24.4
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

<<<<<<< HEAD
- https://verl.readthedocs.io/en/latest/start/install.html#install-dependencies
=======
- [TinyZero](https://github.com/Jiayi-Pan/TinyZero): a reproduction of **DeepSeek R1 Zero** recipe for reasoning tasks ![GitHub Repo stars](https://img.shields.io/github/stars/Jiayi-Pan/TinyZero)
- [SkyThought](https://github.com/NovaSky-AI/SkyThought): RL training for Sky-T1-7B by NovaSky AI team. ![GitHub Repo stars](https://img.shields.io/github/stars/NovaSky-AI/SkyThought)
- [simpleRL-reason](https://github.com/hkust-nlp/simpleRL-reason): SimpleRL-Zoo: Investigating and Taming Zero Reinforcement Learning for Open Base Models in the Wild ![GitHub Repo stars](https://img.shields.io/github/stars/hkust-nlp/simpleRL-reason)
- [Easy-R1](https://github.com/hiyouga/EasyR1): **Multi-modal** RL training framework ![GitHub Repo stars](https://img.shields.io/github/stars/hiyouga/EasyR1)
- [OpenManus-RL](https://github.com/OpenManus/OpenManus-RL): LLM Agents RL tunning framework for multiple agent environments. ![GitHub Repo stars](https://img.shields.io/github/stars/OpenManus/OpenManus-RL)
- [rllm](https://github.com/agentica-project/rllm): async RL training with [verl-pipeline](https://github.com/agentica-project/verl-pipeline) ![GitHub Repo stars](https://img.shields.io/github/stars/agentica-project/rllm)
- [RAGEN](https://github.com/ZihanWang314/ragen): a general-purpose reasoning **agent** training framework ![GitHub Repo stars](https://img.shields.io/github/stars/ZihanWang314/ragen)
- [Search-R1](https://github.com/PeterGriffinJin/Search-R1): RL with reasoning and **searching (tool-call)** interleaved LLMs ![GitHub Repo stars](https://img.shields.io/github/stars/PeterGriffinJin/Search-R1)
- [ReSearch](https://github.com/Agent-RL/ReSearch): Learning to **Re**ason with **Search** for LLMs via Reinforcement Learning ![GitHub Repo stars](https://img.shields.io/github/stars/Agent-RL/ReSearch)
- [Skywork-OR1](https://github.com/SkyworkAI/Skywork-OR1): Skywork open reaonser series ![GitHub Repo stars](https://img.shields.io/github/stars/SkyworkAI/Skywork-OR1)
- [ToRL](https://github.com/GAIR-NLP/ToRL): Scaling tool-integrated RL ![GitHub Repo stars](https://img.shields.io/github/stars/GAIR-NLP/ToRL)
- [Absolute Zero Reasoner](https://github.com/LeapLabTHU/Absolute-Zero-Reasoner): [A no human curated data self-play framework for reasoning](https://arxiv.org/abs/2505.03335) ![GitHub Repo stars](https://img.shields.io/github/stars/LeapLabTHU/Absolute-Zero-Reasoner)
- [verl-agent](https://github.com/langfengQ/verl-agent): A scalable training framework for **long-horizon LLM/VLM agents**, along with a new algorithm **GiGPO** ![GitHub Repo stars](https://img.shields.io/github/stars/langfengQ/verl-agent)
- [RL-Factory](https://github.com/Simple-Efficient/RL-Factory): An easy and efficient RL post-training framework for Agentic Learning ![GitHub Repo stars](https://img.shields.io/github/stars/Simple-Efficient/RL-Factory)
- [ReTool](https://retool-rl.github.io/): ReTool: reinforcement learning for strategic tool use in LLMs. Code release is in progress...
- [verl-tool](https://github.com/TIGER-AI-Lab/verl-tool): An unified and easy-to-extend tool-agent training framework based on verl![GitHub Repo stars](https://img.shields.io/github/stars/TIGER-AI-Lab/verl-tool)
- [PRIME](https://github.com/PRIME-RL/PRIME): Process reinforcement through implicit rewards ![GitHub Repo stars](https://img.shields.io/github/stars/PRIME-RL/PRIME)
- [MemAgent](https://github.com/BytedTsinghua-SIA/MemAgent): MemAgent: Reshaping Long-Context LLM with Multi-Conv RL based Memory Agent ![GitHub Repo stars](https://img.shields.io/github/stars/BytedTsinghua-SIA/MemAgent)
- [POLARIS](https://github.com/ChenxinAn-fdu/POLARIS): A Post-training recipe for scaling RL on Advanced Reasoning models ![GitHub Repo stars](https://img.shields.io/github/stars/ChenxinAn-fdu/POLARIS)
- [GUI-R1](https://github.com/ritzz-ai/GUI-R1): **GUI-R1**: A Generalist R1-style Vision-Language Action Model For **GUI Agents** ![GitHub Repo stars](https://img.shields.io/github/stars/ritzz-ai/GUI-R1)
- [DeepRetrieval](https://github.com/pat-jj/DeepRetrieval): RL Training of **Search Agent** with **Search/Retrieval Outcome** ![GitHub Repo stars](https://img.shields.io/github/stars/pat-jj/DeepRetrieval)
- [Code-R1](https://github.com/ganler/code-r1): Reproducing R1 for **Code** with Reliable Rewards ![GitHub Repo stars](https://img.shields.io/github/stars/ganler/code-r1)
- [DeepResearcher](https://github.com/GAIR-NLP/DeepResearcher): Scaling deep research via reinforcement learning in real-world environments ![GitHub Repo stars](https://img.shields.io/github/stars/GAIR-NLP/DeepResearcher)
- [VAGEN](https://github.com/RAGEN-AI/VAGEN): Training VLM agents with multi-turn reinforcement learning ![GitHub Repo stars](https://img.shields.io/github/stars/RAGEN-AI/VAGEN)
- [RM-R1](https://arxiv.org/abs/2505.02387): RL training of reasoning reward models ![GitHub Repo stars](https://img.shields.io/github/stars/RM-R1-UIUC/RM-R1)
- [LUFFY](https://arxiv.org/pdf/2504.14945): Learning to Reason under Off-Policy Guidance![GitHub Repo stars](https://img.shields.io/github/stars/ElliottYan/LUFFY)
- [DeepMath](https://github.com/zwhe99/DeepMath): DeepMath-103K data and series models for math reasoning![GitHub Repo stars](https://img.shields.io/github/stars/zwhe99/DeepMath)
- [PACS](https://github.com/ritzz-ai/PACS): Implicit Actor Critic Coupling via a Supervised Learning Framework for RLVR ![GitHub Repo stars](https://img.shields.io/github/stars/ritzz-ai/PACS)
- [Entropy Mechanism of RL](https://github.com/PRIME-RL/Entropy-Mechanism-of-RL): The Entropy Mechanism of Reinforcement Learning for Large Language Model Reasoning![GitHub Repo stars](https://img.shields.io/github/stars/PRIME-RL/Entropy-Mechanism-of-RL)
- [LLaSA-TTS-GRPO](https://github.com/channel-io/ch-tts-llasa-rl-grpo): TTS fine-tuning with GRPO optimization based on LLASA models ![GitHub Repo stars](https://img.shields.io/github/stars/channel-io/ch-tts-llasa-rl-grpo)
- [PF-PPO](https://arxiv.org/abs/2409.06957): Policy Filtration for PPO based on the reliability of reward signals for more efficient and robust RLHF.
- [RACRO](https://github.com/gyhdog99/RACRO2): Build multi-modal reasoning models via decoupling it into query-conditioned captioning and text-only reasoning ![GitHub Repo stars](https://img.shields.io/github/stars/gyhdog99/RACRO2)
- [Agent Lightning](https://github.com/microsoft/agent-lightning): A flexible and extensible framework that enables seamless agent optimization for any existing agent framework. ![GitHub Repo stars](https://img.shields.io/github/stars/microsoft/agent-lightning)
- [VTool-R1](https://github.com/VTOOL-R1/vtool-r1): VLMs Learn to Think with Images via Reinforcement Learning on Multimodal Tool Use. ![GitHub Repo stars](https://img.shields.io/github/stars/VTOOL-R1/vtool-r1)
- [Kimina-Prover-RL](https://github.com/project-numina/kimina-prover-rl/tree/main/recipe/kimina_prover_rl): Training pipeline for formal theorem proving, based on a paradigm inspired by DeepSeek-R1.
- [RL-PLUS](https://github.com/YihongDong/RL-PLUS): Countering Capability Boundary Collapse of LLMs in Reinforcement Learning with Hybrid-policy Optimization.
- [rStar2-Agent](https://github.com/microsoft/rStar): Using reinforcement learning with multi-step tool-calling for math tasks, rStar2-Agent-14B reaches frontier-level math reasoning in just 510 RL training steps ![GitHub Repo stars](https://img.shields.io/github/stars/microsoft/rStar)
- [Vision-SR1](https://github.com/zli12321/Vision-SR1): Self-Rewarding Vision-Language Model via Reasoning Decomposition ![GitHub Repo stars](https://img.shields.io/github/stars/zli12321/Vision-SR1)
- [SimpleVLA-RL](https://github.com/PRIME-RL/SimpleVLA-RL): SimpleVLA-RL: A Simple yet Effective Vision-Language Action Model for Reinforcement Learning ![GitHub Repo stars](https://img.shields.io/github/stars/PRIME-RL/SimpleVLA-RL)
- [Table-R1](https://github.com/Table-R1/Table-R1): Table-R1: Inference-Time Scaling for Table Reasoning ![GitHub Repo stars](https://img.shields.io/github/stars/Table-R1/Table-R1)
- [Revisual-R1](https://github.com/CSfufu/Revisual-R1): Revisual-R1: Advancing Multimodal Reasoning From Optimized Cold Start to Staged Reinforcement Learning ![GitHub Repo stars](https://img.shields.io/github/stars/CSfufu/Revisual-R1)
- [ARES](https://github.com/shawn0728/ARES): ARES: Multimodal Adaptive Reasoning via Difficulty-Aware Token-Level Entropy Shaping ![GitHub Repo stars](https://img.shields.io/github/stars/shawn0728/ARES)
- [Meta-Bandit-LLM](https://github.com/sanxing-chen/meta-bandit-llm): Meta-Bandit-LLM: Long-horizon multiturn interactive training for meta-bandit agents ![GitHub Repo stars](https://img.shields.io/github/stars/sanxing-chen/meta-bandit-llm)
- [PokeeResearch](https://github.com/Pokee-AI/PokeeResearchOSS): PokeeResearch: State-of-the-art 7B DeepResearch Agent that leverages web search and content reading capabilities to answer complex questions using the most up-to-date information available online. ![Github Repo Stars](https://img.shields.io/github/stars/Pokee-AI/PokeeResearchOSS)
>>>>>>> origin/main

# Core Code

### Load pretrained model 

- With LoRA trained

    ```python
    python -m utils.model_merge_and_save.py --{args}
    ```

- Full-fine tuned model

    ```python
    python -m utils.model_loader.py --{args}
    ```

### Data Preprocess

- https://verl.readthedocs.io/en/latest/preparation/prepare_data.html
    - `data_source` : reward function class를 구분짓는 중요한 요소
    - `prompt` : List of “role”, “content” key messages ← generation을 위한 user prompt
    - `reward_model`
        - style “rule” : exact match (math) ← X model base
        - reward model base면 컬럼 없애기 → [참조](https://www.notion.so/verl-Recipe-26b7e0e795c9807b8de0e225f12bf94c?pvs=21)
- example/data_preprocess
    - chem_dataset.py (after warm up from chem_select.py)
        
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

### Save pretrained model 

- FSDP & Megatron

    ```bash
    bash utils/model_merge.sh
    ```
