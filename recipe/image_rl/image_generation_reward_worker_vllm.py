"""
vLLM-based Image Generation Reward Model Worker for TP > 1 (Async Engine).

Only rank 0 initializes AsyncLLMEngine, other ranks receive results
via torch.distributed.broadcast.
"""

import datetime
import logging
import os
import re
import subprocess
from collections import defaultdict
from typing import Optional, List

import numpy as np
import torch
import torch.distributed
from omegaconf import DictConfig

from verl import DataProto
from verl.single_controller.base import Worker
from verl.single_controller.base.decorator import Dispatch, register
from verl.utils.config import omega_conf_to_dataclass
from verl.utils.device import get_nccl_backend, get_device_id
from verl.utils.profiler import DistProfiler, DistProfilerExtension, ProfilerConfig
from verl.workers.fsdp_workers import create_device_mesh


logger = logging.getLogger(__file__)
logger.setLevel(os.getenv("VERL_PPO_LOGGING_LEVEL", "WARN"))


# ============== Reward Scoring Logic ==============

TASK1_TASK3_IMAGE_GENERATOR_SYSTEM_PROMPT_TEMPLATE = """
You are an AI assistant that answers a batch of yes/no questions.

Protocol:
1) For each question, output EXACTLY two lines, in order, both starting with "<index> |".
2) Line 1 (justification): one or two concise sentences; no lists, no newlines, do NOT include "Answer:".
3) Line 2 (final): exactly "<index> | Answer: Yes" or "<index> | Answer: No".
4) Preserve the original question order and indices. Do not skip or renumber.
5) If evidence is insufficient or the question is malformed/ambiguous, include the phrase "Insufficient evidence." in the justification and set the final line to "Answer: No".
6) Do NOT reveal chain-of-thought; provide only brief conclusions based on observable evidence.
7) Do not add any extra text before, between, or after answers.

Output template per question (two lines per question):
<index> | <one or two concise sentences for justification>
<index> | Answer: Yes or Answer: No""".strip()

TASK1_TASK3_IMAGE_GENERATOR_USER_PROMPT_TEMPLATE = """
You will receive multiple questions, one per line, in the format "<index> | <question>".
For each question, follow the protocol and produce EXACTLY two lines using the template:

<index> | <one or two concise sentences for justification>
<index> | Answer: Yes or Answer: No

Questions:
{questions}""".strip()


def image_evaluator_parser(text):
    """Parse VLM response to extract yes/no answers."""
    ans_line_re = re.compile(r'^\s*(\d+)\s*\|\s*Answer:\s*(Yes|No)\s*$', re.IGNORECASE | re.MULTILINE)
    idx_to_ans = {}
    for idx_str, yn in ans_line_re.findall(text):
        idx = int(idx_str)
        idx_to_ans[idx] = (yn.strip().lower() == "yes")
    return idx_to_ans


def convert_image_to_base64(gen_img) -> Optional[str]:
    """Convert image to base64 data URL."""
    import base64
    from io import BytesIO
    import PIL.Image
    
    if gen_img is None:
        return None
    
    if isinstance(gen_img, str):
        gen_img = PIL.Image.open(gen_img)
    elif isinstance(gen_img, torch.Tensor):
        gen_img = gen_img.detach().cpu().numpy()
        if gen_img.ndim == 3 and gen_img.shape[0] in (1, 3, 4):
            gen_img = np.transpose(gen_img, (1, 2, 0))
        if gen_img.dtype in (np.float32, np.float64):
            gen_img = np.clip((gen_img + 1) / 2 * 255, 0, 255)
        gen_img = gen_img.astype(np.uint8)
        if gen_img.shape[-1] == 1:
            gen_img = gen_img.squeeze(-1)
        gen_img = PIL.Image.fromarray(gen_img)
    elif isinstance(gen_img, np.ndarray):
        if gen_img.ndim == 3 and gen_img.shape[0] in (1, 3, 4):
            gen_img = np.transpose(gen_img, (1, 2, 0))
        if gen_img.dtype in (np.float32, np.float64):
            gen_img = np.clip((gen_img + 1) / 2 * 255, 0, 255)
        gen_img = gen_img.astype(np.uint8)
        if gen_img.ndim == 3 and gen_img.shape[-1] == 1:
            gen_img = gen_img.squeeze(-1)
        gen_img = PIL.Image.fromarray(gen_img)

    if not isinstance(gen_img, PIL.Image.Image):
        raise TypeError(f"Unsupported image type: {type(gen_img)}")

    buffer = BytesIO()
    gen_img.save(buffer, format="PNG")
    img_base64 = base64.b64encode(buffer.getvalue()).decode("utf-8")
    return f"data:image/png;base64,{img_base64}"


def decode_base64_to_pil(img_base64: str):
    """Decode base64 image to PIL Image."""
    import base64
    from io import BytesIO
    import PIL.Image
    
    if img_base64.startswith("data:image"):
        img_data = img_base64.split(",")[1]
    else:
        img_data = img_base64
    
    img_bytes = base64.b64decode(img_data)
    return PIL.Image.open(BytesIO(img_bytes))


def get_all_gpu_ids():
    """Get all GPU IDs available on this node using nvidia-smi."""
    try:
        result = subprocess.run(
            ["nvidia-smi", "--query-gpu=index", "--format=csv,noheader,nounits"],
            capture_output=True, 
            text=True,
            timeout=10,
        )
        if result.returncode == 0:
            gpu_ids = [x.strip() for x in result.stdout.strip().split("\n") if x.strip()]
            return gpu_ids
    except Exception as e:
        logger.warning(f"Failed to get GPU IDs via nvidia-smi: {e}")
    return []


# ============== vLLM Reward Model Worker (Async) ==============

class ImageGenerationRewardModelWorker(Worker, DistProfilerExtension):
    """
    vLLM-based Image Generation Reward Model Worker for TP > 1 (AsyncLLMEngine).
    
    Key design:
    - Only rank 0 initializes AsyncLLMEngine
    - Rank 0 overrides CUDA_VISIBLE_DEVICES to see all TP GPUs
    - Other ranks receive results via torch.distributed.broadcast
    """

    def __init__(self, config: DictConfig):
        Worker.__init__(self)
        self.config = config
        
        # Initialize distributed
        if not torch.distributed.is_initialized():
            torch.distributed.init_process_group(
                backend=get_nccl_backend(),
                timeout=datetime.timedelta(seconds=self.config.get("nccl_timeout", 600)),
                init_method=os.environ.get("DIST_INIT_METHOD", None),
            )
        
        # Profiler setup
        omega_profiler_config = config.get("profiler", {})
        profiler_config = omega_conf_to_dataclass(omega_profiler_config, dataclass_type=ProfilerConfig)
        tool_config = None
        if omega_profiler_config.get("tool", None) in ["npu", "nsys", "torch", "torch_memory"]:
            tool_config = omega_conf_to_dataclass(
                omega_profiler_config.get("tool_config", {}).get(omega_profiler_config.get("tool"))
            )
        DistProfilerExtension.__init__(
            self, DistProfiler(rank=self.rank, config=profiler_config, tool_config=tool_config)
        )
        
        # Device mesh
        fsdp_size = self.config.model.get("fsdp_config", {}).get("fsdp_size", self.world_size)
        self.device_mesh = create_device_mesh(world_size=self.world_size, fsdp_size=fsdp_size)
        
        # vLLM async engine (only on rank 0)
        self.engine = None
        self.sampling_params = None
        
        # num_examine for logging
        self.num_examine = config.get("num_examine", 2)
        
        # Register dispatch
        self._register_dispatch_collect_info("reward", dp_rank=self.rank, is_collect=True)

    @register(dispatch_mode=Dispatch.ONE_TO_ALL)
    def init_model(self):
        """Initialize Async vLLM engine for reward model."""
        
        original_cuda_visible = os.environ.get('CUDA_VISIBLE_DEVICES', 'not set')
        
        print(f"[RewardModel Rank {self.rank}] Initializing...")
        print(f"[RewardModel Rank {self.rank}] Original CUDA_VISIBLE_DEVICES: {original_cuda_visible}")
        print(f"[RewardModel Rank {self.rank}] world_size: {self.world_size}")
        
        # Engine parameters
        model_path = self.config.model.path
        tensor_parallel_size = self.config.model.get("tensor_parallel_size", 4)
        gpu_memory_utilization = self.config.model.get("gpu_memory_utilization", 0.6)
        max_model_len = self.config.get("max_length", 4096)
        dtype = self.config.model.get("dtype", "bfloat16")
        
        # Only rank 0 initializes vLLM
        if self.rank == 0:
            print(f"[RewardModel] Rank 0: Initializing AsyncLLMEngine with TP={tensor_parallel_size}")
            
            # CRITICAL: Override CUDA_VISIBLE_DEVICES to see all TP GPUs
            all_gpu_ids = get_all_gpu_ids()
            print(f"[RewardModel] All GPUs on node: {all_gpu_ids}")
            
            if len(all_gpu_ids) < tensor_parallel_size:
                raise RuntimeError(
                    f"[RewardModel] ERROR: Need {tensor_parallel_size} GPUs for TP but only "
                    f"{len(all_gpu_ids)} GPUs available on this node."
                )
            
            # Use specified GPU IDs or first TP GPUs
            gpu_ids = self.config.model.get("gpu_ids", None)
            if gpu_ids is None:
                gpu_ids = all_gpu_ids[:tensor_parallel_size]
            
            new_cuda_visible = ",".join(map(str, gpu_ids))
            os.environ["CUDA_VISIBLE_DEVICES"] = new_cuda_visible
            print(f"[RewardModel] Overriding CUDA_VISIBLE_DEVICES: {original_cuda_visible} -> {new_cuda_visible}")
            
            # Set environment to use multiprocessing instead of Ray
            os.environ["VLLM_WORKER_MULTIPROC_METHOD"] = "spawn"
            
            # Import async vLLM AFTER setting CUDA_VISIBLE_DEVICES
            from vllm import AsyncEngineArgs, AsyncLLMEngine, SamplingParams
            
            engine_args = AsyncEngineArgs(
                model=model_path,
                tensor_parallel_size=tensor_parallel_size,
                gpu_memory_utilization=gpu_memory_utilization,
                max_model_len=max_model_len,
                dtype=dtype,
                trust_remote_code=self.config.model.get("trust_remote_code", True),
                disable_log_stats=self.config.model.get("disable_log_stats", True),
                enforce_eager=self.config.model.get("enforce_eager", False),
                max_num_seqs=self.config.get("max_num_seqs", 256),
                enable_chunked_prefill=self.config.model.get("enable_chunked_prefill", True),
                # Use multiprocessing instead of Ray
                distributed_executor_backend="mp",
                # Disable custom all reduce to avoid NCCL conflicts
                disable_custom_all_reduce=True,
            )
            
            self.engine = AsyncLLMEngine.from_engine_args(engine_args)
            
            self.sampling_params = SamplingParams(
                max_tokens=self.config.get("max_new_tokens", 2048),
                temperature=0.0,
                top_p=1.0,
            )
            
            print(f"[RewardModel Rank 0] AsyncLLMEngine initialized successfully with TP={tensor_parallel_size}")
        else:
            print(f"[RewardModel Rank {self.rank}] Skipping vLLM init (only rank 0 runs inference)")
        
        # Synchronize all ranks
        torch.distributed.barrier()
        print(f"[RewardModel Rank {self.rank}] Ready")

    def _generate_batch(self, prompts: List[dict]) -> List[str]:
        """Generate responses for a batch of prompts (rank 0 only, async engine)."""
        if self.rank != 0 or not prompts:
            return [""] * len(prompts) if prompts else []
        
        # Prepare inputs for vLLM (same as 기존 코드)
        vllm_inputs = []
        for p in prompts:
            text = p.get("text", "")
            images = p.get("images", [])
            
            if images:
                # Multi-modal: convert base64 to PIL
                pil_images = []
                for img in images:
                    if isinstance(img, str):
                        pil_images.append(decode_base64_to_pil(img))
                    else:
                        pil_images.append(img)
                
                vllm_inputs.append({
                    "prompt": text,
                    "multi_modal_data": {"image": pil_images}
                })
            else:
                vllm_inputs.append({"prompt": text})

        import asyncio

        async def _async_generate():
            # AsyncLLMEngine.generate is coroutine
            outputs = await self.engine.generate(
                vllm_inputs,
                self.sampling_params,
                use_tqdm=False,
            )
            results = []
            for output in outputs:
                if output.outputs:
                    results.append(output.outputs[0].text)
                else:
                    results.append("")
            return results

        # 블로킹 환경에서 한 번 돌려서 결과만 가져옴
        return asyncio.run(_async_generate())

    def _broadcast_responses(self, responses: List[str]) -> List[str]:
        """Broadcast responses from rank 0 to all ranks."""
        import pickle
        
        if self.world_size == 1:
            return responses
        
        device = f"cuda:{get_device_id()}"
        
        if self.rank == 0:
            data = pickle.dumps(responses)
            size = torch.tensor([len(data)], dtype=torch.long, device=device)
        else:
            size = torch.tensor([0], dtype=torch.long, device=device)
        
        torch.distributed.broadcast(size, src=0)
        
        if self.rank == 0:
            data_tensor = torch.ByteTensor(list(data)).to(device)
        else:
            data_tensor = torch.empty(size[0].item(), dtype=torch.uint8, device=device)
        
        torch.distributed.broadcast(data_tensor, src=0)
        
        if self.rank != 0:
            responses = pickle.loads(bytes(data_tensor.cpu().tolist()))
        
        return responses

    def _build_prompt_task1(self, gen_img, vqa_question: str) -> Optional[dict]:
        """Build prompt for Task 1 (image evaluation)."""
        system_prompt = TASK1_TASK3_IMAGE_GENERATOR_SYSTEM_PROMPT_TEMPLATE
        user_prompt = TASK1_TASK3_IMAGE_GENERATOR_USER_PROMPT_TEMPLATE.format(questions=vqa_question)
        
        img_base64 = convert_image_to_base64(gen_img)
        if img_base64 is None:
            return None
        
        full_prompt = f"System: {system_prompt}\n\nUser: {user_prompt}\n\nAssistant:"
        
        return {
            "text": full_prompt,
            "images": [img_base64]
        }

    def _build_prompt_task3(self, regen_img, vqa_question: str) -> Optional[dict]:
        """Build prompt for Task 3 (regenerated image evaluation)."""
        system_prompt = TASK1_TASK3_IMAGE_GENERATOR_SYSTEM_PROMPT_TEMPLATE
        user_prompt = TASK1_TASK3_IMAGE_GENERATOR_USER_PROMPT_TEMPLATE.format(questions=vqa_question)
        
        img_base64 = convert_image_to_base64(regen_img)
        if img_base64 is None:
            return None
        
        full_prompt = f"System: {system_prompt}\n\nUser: {user_prompt}\n\nAssistant:"
        
        return {
            "text": full_prompt,
            "images": [img_base64]
        }

    def _compute_score_single(
        self,
        response: str,
        task_id: int,
        feedback_text: str = "",
        feedback_tuple: str = None,
    ) -> dict:
        """Compute score for a single sample based on VLM response."""
        reward_score = 0.0
        reward_extra_info = {}

        if task_id in [1, 3]:
            if response:
                idx_to_ans = image_evaluator_parser(response)
                reward_score_sum = sum(idx_to_ans.values())
                ans_count = len(idx_to_ans)
                score_mean = 1.0 if ans_count > 0 and reward_score_sum == ans_count else 0.0
                reward_score = score_mean
                reward_extra_info[f"task{task_id}_reward_response"] = response
            else:
                reward_extra_info[f"task{task_id}_reward_response"] = "No response"

        elif task_id == 2:
            from recipe.image_rl.utils import FormattingEvaluator
            
            task2_reward_score = 0.0
            task2_ans_count = 0
            
            formatting_evaluator = FormattingEvaluator()
            part1, part2, part3 = formatting_evaluator._split_text_into_parts(feedback_text.strip())
            
            task2_reward_score += 1.0 if all(part is not None for part in [part1, part2, part3]) else 0.0
            task2_ans_count += 1
            
            if feedback_tuple and part1:
                feedback_parsed_tuple = formatting_evaluator._parse_part1(feedback_tuple)
                predict_parsed_tuple = formatting_evaluator._parse_part1(part1)
                predict_decomposed_ans = formatting_evaluator._extract_answer_paragraphs(part2) if part2 else []
                
                part1_reward_dict = formatting_evaluator._calculate_metrics_for_reward(
                    feedback_parsed_tuple, predict_parsed_tuple, predict_decomposed_ans
                )
                task2_reward_score += sum(part1_reward_dict.values())
                task2_ans_count += len(part1_reward_dict)
            
            reward_score = (task2_reward_score / task2_ans_count) if task2_ans_count > 0 else 0.0
            reward_extra_info[f"task{task_id}_reward_response"] = "Rule-based scoring"

        return {"score": reward_score, "reward_extra_info": reward_extra_info}

    def _verify(self, data: DataProto, task_id: int) -> List[dict]:
        """Verify and compute scores for batch."""
        prompt = data.non_tensor_batch.get('prompt', [])
        gen_imgs_pil_list = data.non_tensor_batch.get('task1_gen_imgs_pil_list', [])
        feedback_texts = data.non_tensor_batch.get('task2_feedback_texts', [])
        regen_imgs_pil_list = data.non_tensor_batch.get('task3_regen_imgs_pil_list', [])
        ground_truth = data.non_tensor_batch.get('reward_model', {})
        
        batch_size = len(prompt)
        if self.rank == 0:
            print(f"[VERIFY] Processing batch of {batch_size} samples for task {task_id}")
        
        feedback_texts_padded = [feedback_texts[i] if i < len(feedback_texts) else "" for i in range(batch_size)]
        feedback_tuples = [ground_truth[i]["tuple"] if i < len(ground_truth) and ground_truth[i] else None for i in range(batch_size)]
        vqa_questions = [ground_truth[i]["vqa_question"] if i < len(ground_truth) and ground_truth[i] else None for i in range(batch_size)]
        
        scores = []
        
        if task_id == 1:
            # Build all prompts
            prompts_to_generate = []
            valid_indices = []
            
            for i in range(batch_size):
                gen_img = gen_imgs_pil_list[i] if i < len(gen_imgs_pil_list) else None
                vqa_q = vqa_questions[i]
                
                if gen_img is not None and vqa_q:
                    prompt_dict = self._build_prompt_task1(gen_img, vqa_q)
                    if prompt_dict:
                        prompts_to_generate.append(prompt_dict)
                        valid_indices.append(i)
            
            # Generate on rank 0 (async engine)
            responses = [""] * batch_size
            if prompts_to_generate:
                generated = self._generate_batch(prompts_to_generate)
                if self.rank == 0:
                    for idx, gen_idx in enumerate(valid_indices):
                        responses[gen_idx] = generated[idx] if idx < len(generated) else ""
            
            # Broadcast to all ranks
            responses = self._broadcast_responses(responses)
            
            # Compute scores
            for i in range(batch_size):
                score_dict = self._compute_score_single(responses[i], task_id)
                scores.append(score_dict)
                
        elif task_id == 2:
            # Task 2: rule-based, no VLM needed
            for i in range(batch_size):
                score_dict = self._compute_score_single(
                    "", task_id, 
                    feedback_text=feedback_texts_padded[i],
                    feedback_tuple=feedback_tuples[i]
                )
                scores.append(score_dict)
                
        elif task_id == 3:
            from recipe.image_rl.utils import FormattingEvaluator
            formatting_evaluator = FormattingEvaluator()
            
            prompts_to_generate = []
            valid_indices = []
            skip_indices = set()
            
            for i in range(batch_size):
                feedback_text = feedback_texts_padded[i]
                parts = formatting_evaluator._split_text_into_parts(feedback_text)
                last = parts[-1] if parts else None
                
                if last is not None and "No need to generate feedback.".lower() in last.lower():
                    skip_indices.add(i)
                    continue
                
                regen_img = regen_imgs_pil_list[i] if i < len(regen_imgs_pil_list) else None
                vqa_q = vqa_questions[i]
                
                if regen_img is not None and vqa_q:
                    prompt_dict = self._build_prompt_task3(regen_img, vqa_q)
                    if prompt_dict:
                        prompts_to_generate.append(prompt_dict)
                        valid_indices.append(i)
            
            # Generate on rank 0 (async engine)
            responses = [""] * batch_size
            if prompts_to_generate:
                generated = self._generate_batch(prompts_to_generate)
                if self.rank == 0:
                    for idx, gen_idx in enumerate(valid_indices):
                        responses[gen_idx] = generated[idx] if idx < len(generated) else ""
            
            # Broadcast to all ranks
            responses = self._broadcast_responses(responses)
            
            # Compute scores
            for i in range(batch_size):
                if i in skip_indices:
                    scores.append({"score": -100, "reward_extra_info": {f"task{task_id}_reward_response": "None"}})
                else:
                    score_dict = self._compute_score_single(responses[i], task_id)
                    scores.append(score_dict)
        
        return scores

    @register(dispatch_mode=Dispatch.DP_COMPUTE_PROTO)
    def compute_rm_score(self, data: DataProto) -> DataProto:
        """Compute reward model scores."""
        task_id = data.meta_info.get("task_id", 1)
        
        if self.rank == 0:
            print(f"[REWARD] Computing rewards for batch_size={len(data)}, task_id={task_id}")
        
        response_mask = data.batch.get(f"task{task_id}_response_mask", 
                                       torch.ones(len(data), 1))
        
        reward_tensor = torch.zeros_like(response_mask, dtype=torch.float32)
        reward_extra_info = defaultdict(list)
        
        prompt = data.non_tensor_batch.get('prompt', [])
        data_sources = data.non_tensor_batch.get("data_source", ["unknown"] * len(data))
        
        # Compute scores
        scores = self._verify(data, task_id)
        rewards = []
        already_printed = {}

        for i in range(len(data)):
            valid_response_length = response_mask[i].sum().int()
            score_dict = scores[i] if scores[i] is not None else {"score": 0.0}
            reward = score_dict.get("score", 0.0)
            
            if "reward_extra_info" in score_dict:
                for key, value in score_dict["reward_extra_info"].items():
                    reward_extra_info[key].append(value)

            rewards.append(reward)
            reward_tensor[i, valid_response_length - 1] = reward

            if reward == -100:
                data.batch[f"task{task_id}_response_mask"][i] = torch.zeros_like(
                    response_mask[i], dtype=torch.float32
                )

            # Log samples (rank 0 only)
            if self.rank == 0:
                data_source = data_sources[i] if i < len(data_sources) else "unknown"
                if already_printed.get(data_source, 0) < self.num_examine:
                    prompt_text = prompt[i] if i < len(prompt) else "N/A"
                    ground_truth_data = data.non_tensor_batch.get("reward_model", {})
                    ground_truth_val = ground_truth_data[i].get("ground_truth", "N/A") if i < len(ground_truth_data) and ground_truth_data[i] else "N/A"
                    
                    print(f"\n[EXAMINE {i}]")
                    print(f"Data Source: {data_source}")
                    print(f"Prompt: {prompt_text[:200]}..." if len(str(prompt_text)) > 200 else f"Prompt: {prompt_text}")
                    print(f"Ground Truth: {ground_truth_val}")
                    print(f"Score: {score_dict}")
                    print("-" * 80)
                    
                    already_printed[data_source] = already_printed.get(data_source, 0) + 1

        valid_rewards = [r for r in rewards if r != -100]
        mean_reward = sum(valid_rewards) / len(valid_rewards) if valid_rewards else 0.0
        
        if self.rank == 0:
            print(f"[REWARD] Computed {len(rewards)} rewards, valid={len(valid_rewards)}, mean={mean_reward:.4f}")

        return DataProto.from_dict(
            tensors={
                f"task{task_id}_reward_tensor": reward_tensor,
                f"task{task_id}_acc": torch.tensor(rewards, dtype=torch.float32),
            },
            non_tensors={
                f"task{task_id}_reward_extra_info": dict(reward_extra_info),
            }
        )

    @register(dispatch_mode=Dispatch.ONE_TO_ALL)
    def shutdown(self):
        """Shutdown Async vLLM engine."""
        if self.rank == 0 and self.engine is not None:
            print("[RewardModel] Shutting down AsyncLLMEngine...")
            import asyncio

            async def _stop():
                await self.engine.stop()

            try:
                asyncio.run(_stop())
            except RuntimeError:
                # 이미 루프가 돌아가는 환경이면 그냥 best-effort로 넘어감
                pass

            self.engine = None
            torch.cuda.empty_cache()
            print("[RewardModel] vLLM engine shutdown complete")