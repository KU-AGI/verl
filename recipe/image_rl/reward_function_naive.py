import os
import base64
import PIL
import re
import json
from io import BytesIO
from typing import Optional, List, Dict, Any
import PIL.Image
from openai import OpenAI, AsyncOpenAI
import numpy as np
from transformers import AutoTokenizer
from concurrent.futures import ThreadPoolExecutor
import time
from collections import defaultdict
from verl.utils.reward_score.math_reward import last_boxed_only_string, remove_boxed
from recipe.image_rl.utils import FormattingEvaluator, FormattingEvaluatorV2
from recipe.image_rl.prompts import *
import asyncio
import threading
import random
from enum import Enum
import torch
import aiohttp
from recipe.image_rl.gdino_regex import _CONNECTORS, SKIP_KEYWORDS, _COMPILED_RELATIONS

# Configuration
VLM_BASE_URLS = [
    # "http://10.100.44.4:8006/v1", # main1
    # "http://10.100.44.8:8006/v1", # sub1
    # "http://10.100.44.8:8007/v1",
    "http://10.100.44.2:8000/v1", # sub2
    "http://10.100.44.2:8001/v1",
    "http://10.100.44.2:8002/v1",
    "http://10.100.44.2:8003/v1",
    "http://10.100.44.2:8004/v1", # sub2
    "http://10.100.44.2:8005/v1",
    "http://10.100.44.2:8006/v1",
    "http://10.100.44.2:8007/v1",
    
]
LLM_BASE_URLS = [
    # "http://10.100.44.2:8004/v1", # sub2
    # "http://10.100.44.2:8005/v1",
    # "http://10.100.44.2:8006/v1",
    # "http://10.100.44.2:8007/v1",
]
API_KEY = "EMPTY"
MAX_RETRIES = 3
BASE_DELAY = 2
RM_VLM_MODEL_PATH = os.environ.get("RM_VLM_MODEL_PATH", "Qwen/Qwen3-VL-30B-A3B-Instruct")
RM_LLM_MODEL_PATH = os.environ.get("RM_LLM_MODEL_PATH", "Qwen/Qwen3-30B-A3B-Instruct-2507")

# Health checking configuration
HEALTH_CHECK_INTERVAL = 30  # seconds
FAILURE_THRESHOLD = 3  # consecutive failures before marking as unhealthy
RECOVERY_CHECK_INTERVAL = 60  # seconds to wait before checking if unhealthy server recovered

# Detector configuration
DETECTOR_URLS = [
    "http://10.100.44.2:8084",
    "http://10.100.44.2:8085",
    "http://10.100.44.2:8086",
    "http://10.100.44.2:8087",
]
DETECTOR_TIMEOUT = 300000.0

DETECTOR_MAX_RETRIES = 2

RM_PER_SERVER_INFLIGHT = 16
_rm_slot_lock = threading.Lock()
_rm_slot_queues = {}  # {(loop_id, is_vlm): queue}

DET_PER_SERVER_INFLIGHT = 4  
_det_slot_lock = threading.Lock()
_det_slot_queues = {}  # {loop_id: queue}

async def _ensure_rm_slots(is_vlm=True) -> asyncio.Queue:
    """Ensure slot queue exists for current event loop and model type"""
    loop = asyncio.get_running_loop()
    loop_id = id(loop)
    key = (loop_id, is_vlm)
    
    # Quick check without lock
    if key in _rm_slot_queues:
        queue = _rm_slot_queues[key]
        # Verify queue is still valid for this loop
        try:
            # Try a quick operation to verify queue is accessible
            queue.qsize()
            return queue
        except RuntimeError:
            # Queue is bound to different loop, need to recreate
            pass
    
    # Create new queue with lock
    with _rm_slot_lock:
        # Double-check after acquiring lock
        if key in _rm_slot_queues:
            queue = _rm_slot_queues[key]
            try:
                queue.qsize()
                return queue
            except RuntimeError:
                # Remove invalid queue
                del _rm_slot_queues[key]
        
        # Create new queue
        q = asyncio.Queue()
        base_urls = VLM_BASE_URLS if is_vlm else LLM_BASE_URLS
        start = random.randrange(len(base_urls))
        for i in range(RM_PER_SERVER_INFLIGHT * len(base_urls)):
            sid = (start + i) % len(base_urls)
            q.put_nowait(sid)
        
        _rm_slot_queues[key] = q
        return q


async def borrow_rm_client(is_vlm=True):
    q = await _ensure_rm_slots(is_vlm)
    manager = vlm_client_manager if is_vlm else llm_client_manager
    while True:
        sid = await q.get()
        s = manager.servers[sid]
        if s.status != ServerStatus.UNHEALTHY or s.should_retry_unhealthy():
            return s.client, sid, is_vlm
        q.put_nowait(sid)
        await asyncio.sleep(0.05)


async def release_rm_client(server_id: int, is_vlm: bool):
    q = await _ensure_rm_slots(is_vlm)
    q.put_nowait(server_id)


class ServerStatus(Enum):
    HEALTHY = "healthy"
    DEGRADED = "degraded"
    UNHEALTHY = "unhealthy"

class ServerInfo:
    def __init__(self, url: str, client: AsyncOpenAI):
        self.url = url
        self.client = client
        self.status = ServerStatus.HEALTHY
        self.consecutive_failures = 0
        self.last_success_time = time.time()
        self.last_failure_time = None
        self.total_requests = 0
        self.successful_requests = 0
        
    def record_success(self):
        self.consecutive_failures = 0
        self.last_success_time = time.time()
        self.total_requests += 1
        self.successful_requests += 1
        if self.status == ServerStatus.UNHEALTHY:
            print(f"Server {self.url} recovered!")
        self.status = ServerStatus.HEALTHY
        
    def record_failure(self):
        self.consecutive_failures += 1
        self.last_failure_time = time.time()
        self.total_requests += 1
        
        if self.consecutive_failures >= FAILURE_THRESHOLD:
            if self.status != ServerStatus.UNHEALTHY:
                print(f"Server {self.url} marked as UNHEALTHY after {self.consecutive_failures} consecutive failures")
            self.status = ServerStatus.UNHEALTHY
        elif self.consecutive_failures >= 1:
            self.status = ServerStatus.DEGRADED
            
    @property
    def success_rate(self):
        if self.total_requests == 0:
            return 1.0
        return self.successful_requests / self.total_requests
        
    def should_retry_unhealthy(self):
        """Check if we should retry an unhealthy server"""
        if self.status != ServerStatus.UNHEALTHY:
            return True
        if self.last_failure_time is None:
            return True
        return time.time() - self.last_failure_time > RECOVERY_CHECK_INTERVAL

# client manager with failover support and round robin load balancing
class ClientManager:
    def __init__(self, base_urls: List[str], name: str = ""):
        self.servers = []
        self.lock = threading.Lock()
        self.current_index = 0  # Round robin counter
        self.name = name
        
        # Initialize servers
        for url in base_urls:
            client = AsyncOpenAI(api_key=API_KEY, base_url=url)
            server_info = ServerInfo(url, client)
            self.servers.append(server_info)
            
        # Start health monitoring thread
        self.health_monitor_thread = threading.Thread(target=self._health_monitor, daemon=True)
        self.health_monitor_thread.start()
        
    def get_healthy_servers(self) -> List[tuple]:
        """Get list of (index, server) tuples that are healthy or degraded (not unhealthy)"""
        with self.lock:
            healthy_servers = []
            for i, server in enumerate(self.servers):
                if server.status in [ServerStatus.HEALTHY, ServerStatus.DEGRADED]:
                    healthy_servers.append((i, server))
                elif server.should_retry_unhealthy():
                    # Give unhealthy servers a chance to recover
                    healthy_servers.append((i, server))
            return healthy_servers
    
    def get_next_server_round_robin(self) -> tuple:
        """Get next server using round robin among healthy servers"""
        healthy_servers = self.get_healthy_servers()
        
        if not healthy_servers:
            print(f"WARNING: No healthy {self.name} servers available! Using any available server...")
            if self.servers:
                return 0, self.servers[0]
            return None, None
        
        with self.lock:
            # Build a set of healthy server indices for quick lookup
            healthy_indices = {idx for idx, _ in healthy_servers}
            num_servers = len(self.servers)
            
            # Find the next healthy server starting from current_index
            for _ in range(num_servers):
                idx = self.current_index % num_servers
                self.current_index = (self.current_index + 1) % num_servers
                
                # Check if this server is healthy
                if idx in healthy_indices:
                    return idx, self.servers[idx]
            
            # Fallback: return first healthy server
            return healthy_servers[0]
    
    def get_best_server(self) -> Optional[ServerInfo]:
        """Get the best available server using weighted selection (legacy method)"""
        healthy_servers = self.get_healthy_servers()
        
        if not healthy_servers:
            print(f"WARNING: No healthy {self.name} servers available! Using any available server...")
            return self.servers[0] if self.servers else None
            
        # Sort by status (healthy first) then by success rate
        servers_only = [s for _, s in healthy_servers]
        servers_only.sort(key=lambda s: (
            s.status.value,  # healthy < degraded < unhealthy
            -s.success_rate,  # higher success rate first
            s.consecutive_failures  # fewer failures first
        ))
        
        return servers_only[0]
    
    def get_next_client_with_fallback(self) -> tuple:
        """Get next client using round robin with automatic fallback to healthy servers"""
        server_id, server = self.get_next_server_round_robin()
        if server is None:
            return None, None
            
        return server.client, server_id
        
    def record_request_result(self, server_id: int, success: bool, error: Exception = None):
        if 0 <= server_id < len(self.servers):
            server = self.servers[server_id]
            if success:
                server.record_success()
            else:
                prev_status = server.status
                server.record_failure()
                # Log error: status transition
                if error and (server.status != prev_status) and (server.status == ServerStatus.UNHEALTHY):
                    print(f"Server {server.url} marked UNHEALTHY: {repr(error)}")
    
    def get_server_status(self) -> Dict[str, Any]:
        """Get current status of all servers"""
        with self.lock:
            status = {}
            for i, server in enumerate(self.servers):
                status[f"{self.name}_server_{i}"] = {
                    "url": server.url,
                    "status": server.status.value,
                    "consecutive_failures": server.consecutive_failures,
                    "success_rate": f"{server.success_rate:.2%}",
                    "total_requests": server.total_requests,
                    "last_success": server.last_success_time,
                    "last_failure": server.last_failure_time
                }
            return status
    
    def _health_monitor(self):
        """Background thread to monitor server health"""
        while True:
            try:
                # Print server status periodically
                if time.time() % 120 < 1:  # Every 2 minutes
                    status = self.get_server_status()
                    print(f"{self.name} Server Health Status: {json.dumps(status, indent=2, default=str)}")
                
                time.sleep(HEALTH_CHECK_INTERVAL)
            except Exception as e:
                print(f"{self.name} Health monitor error: {e}")
                time.sleep(HEALTH_CHECK_INTERVAL)

# Initialize the client managers
vlm_client_manager = ClientManager(VLM_BASE_URLS, name="VLM")
llm_client_manager = ClientManager(LLM_BASE_URLS, name="LLM")


def convert_gen_img_to_base64(gen_img) -> Optional[str]:
    """Convert image to base64 data URL.
    
    Supports: PIL.Image, str (file path), np.ndarray, torch.Tensor
    """
    if isinstance(gen_img, str):
        gen_img = PIL.Image.open(gen_img)
    elif isinstance(gen_img, torch.Tensor):
        gen_img = gen_img.detach().cpu().numpy()
        # Convert [C, H, W] -> [H, W, C] if channel-first
        if gen_img.ndim == 3 and gen_img.shape[0] in (1, 3, 4):
            gen_img = np.transpose(gen_img, (1, 2, 0))
        # Normalize if float
        if gen_img.dtype in (np.float32, np.float64):
            gen_img = np.clip((gen_img + 1) / 2 * 255, 0, 255)
        gen_img = gen_img.astype(np.uint8)
        if gen_img.shape[-1] == 1:
            gen_img = gen_img.squeeze(-1)
        gen_img = PIL.Image.fromarray(gen_img)
    elif isinstance(gen_img, np.ndarray):
        # Convert [C, H, W] -> [H, W, C] if channel-first
        if gen_img.ndim == 3 and gen_img.shape[0] in (1, 3, 4):
            gen_img = np.transpose(gen_img, (1, 2, 0))
        # Normalize if float
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
    

def image_evaluator_parser(text):
    ans_line_re = re.compile(r'^\s*(\d+)\s*\|\s*Answer:\s*(Yes|No)\s*$', re.IGNORECASE | re.MULTILINE)
    
    idx_to_ans = {} # 1 | ... Answer: Yes or No -> {1: True or False}
    for idx_str, yn in ans_line_re.findall(text):
        idx = int(idx_str)
        idx_to_ans[idx] = (yn.strip().lower() == "yes")
    
    return idx_to_ans


# Main message construction function
def get_messages(*args):
    prompt, gen_img, feedback_text, regen_img, ground_truth_img, summarize, feedback_tuple, predicted_summarize, predicted_tuple, predicted_answer, predicted_feedback, vqa_question, extra_info, task_id = args

    if task_id == 1:
        system_prompt = TASK1_TASK3_IMAGE_GENERATOR_SYSTEM_PROMPT_TEMPLATE
        user_prompt = TASK1_TASK3_IMAGE_GENERATOR_USER_PROMPT_TEMPLATE.format(questions=vqa_question)
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": [
                {"type": "image_url", "image_url": {"url": convert_gen_img_to_base64(gen_img)}},
                {"type": "text", "text": user_prompt}
            ]}
        ]
        model = RM_VLM_MODEL_PATH
    elif task_id == 2:
        system_prompt = TASK2_FEEDBACK_GENERATOR_SYSTEM_PROMPT_TEMPLATE_NAIVE
        user_prompt = TASK2_FEEDBACK_GENERATOR_USER_PROMPT_TEMPLATE_NAIVE.format(prompt=prompt, predicted_feedback=predicted_feedback)
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": [
                {"type": "image_url", "image_url": {"url": convert_gen_img_to_base64(gen_img)}},
                {"type": "text", "text": user_prompt},
            ]}
        ]
        model = RM_VLM_MODEL_PATH
    elif task_id == 3:
        system_prompt = TASK1_TASK3_IMAGE_GENERATOR_SYSTEM_PROMPT_TEMPLATE
        user_prompt = TASK1_TASK3_IMAGE_GENERATOR_USER_PROMPT_TEMPLATE.format(questions=vqa_question)
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": [
                {"type": "image_url", "image_url": {"url": convert_gen_img_to_base64(regen_img)}},
                {"type": "text", "text": user_prompt}
            ]}
        ]
        model = RM_VLM_MODEL_PATH
    else:
        raise ValueError(f"Invalid task: {task_id} is must be one of task1, task2, or task3.")

    return messages, model


# Additional message constructors for task 2 subtasks
def get_messsages_task2_vqa_pass_or_fail(*args): # 5: part 1
    prompt, gen_img, feedback_text, regen_img, ground_truth_img, summarize, feedback_tuple, predicted_summarize, predicted_tuple, predicted_answer, predicted_feedback, vqa_question, extra_info, task_id = args

    system_prompt = TASK2_VQA_PASS_OR_FAIL_SYSTEM_PROMPT_TEMPLATE
    task1_vlm_reward_response = extra_info.get("task1_reward_response", "")
    user_prompt = TASK2_VQA_PASS_OR_FAIL_USER_PROMPT_TEMPLATE.format(prompt=prompt, task1_vlm_reward_response=task1_vlm_reward_response, predicted_answer=predicted_answer)
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": [
            {"type": "text", "text": user_prompt},
        ]}
    ]
    model = RM_VLM_MODEL_PATH

    return messages, model


def get_messsages_task2_edit_instruction_following(*args): # 5: part 1
    prompt, gen_img, feedback_text, regen_img, ground_truth_img, summarize, feedback_tuple, predicted_summarize, predicted_tuple, predicted_answer, predicted_feedback, vqa_question, extra_info, task_id = args

    system_prompt = TASK3_EDIT_INSTRUCTION_FOLLOWING_SYSTEM_PROMPT
    user_prompt = TASK3_EDIT_INSTRUCTION_FOLLOWING_USER_PROMPT.format(predicted_feedback=predicted_feedback)
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": [
            {"type": "image_url", "image_url": {"url": convert_gen_img_to_base64(gen_img)}},
            {"type": "text", "text": user_prompt},
        ]}
    ]
    model = RM_VLM_MODEL_PATH

    return messages, model


async def get_response_with_client(client, messages, model):
    """Get response from a specific client with improved error handling"""
    response = await client.chat.completions.create(
        model=model,
        messages=messages,
        max_tokens=4096,
        extra_body={"repetition_penalty": 1.2},
        timeout=300000.0,
    )
    return response.choices[0].message.content


async def get_response(message_builder_fn, *args):
    """Generic response fetcher with automatic server fallback

    Args:
        message_builder_fn: Function that takes *args and returns (messages list, model_name)
        *args: Arguments to pass to message_builder_fn
    """
    messages, model = message_builder_fn(*args)
    
    # Determine if we need VLM or LLM based on model name
    is_vlm = (model == RM_VLM_MODEL_PATH)
    base_urls = VLM_BASE_URLS if is_vlm else LLM_BASE_URLS

    # Try different servers until one succeeds
    max_attempts = len(base_urls) * MAX_RETRIES

    for attempt in range(max_attempts):
        client, sid, _ = await borrow_rm_client(is_vlm)
        manager = vlm_client_manager if is_vlm else llm_client_manager
        
        try:
            response = await get_response_with_client(client, messages, model)
            if not is_meaningful_response(response):
                # Back off 
                manager.record_request_result(sid, success=False,
                                                     error=ValueError("Non-meaningful response"))
                print(response) # DEBUG
                continue
            else:
                manager.record_request_result(sid, success=True)
                return response

        except Exception as e:
            manager.record_request_result(sid, success=False, error=e)

        finally:
            await release_rm_client(sid, is_vlm)

    return None


async def compute_score_single_async(prompt, gen_img, feedback_text, regen_img, ground_truth_img, summarize, feedback_tuple, predicted_summarize, predicted_tuple, predicted_answer, predicted_feedback, vqa_question, extra_info, task_id):
    """Async version of compute_score"""
    reward_score = 0.0
    reward_extra_info = {}

    if task_id == 1: # Total score: 3.0
        # Create tasks
        vqa_task = asyncio.create_task(
            get_response(get_messages, prompt, gen_img, feedback_text, regen_img, ground_truth_img, summarize, feedback_tuple, predicted_summarize, predicted_tuple, predicted_answer, predicted_feedback, vqa_question, extra_info, task_id)
        )

        # Gather results
        vqa_response = await vqa_task

        # Process VLM response
        task1_vqa_reward_score = 0.0
        if not isinstance(vqa_response, Exception) and vqa_response is not None:
            try:
                task1_idx_to_ans: dict = image_evaluator_parser(vqa_response)
                task1_vqa_reward_score_sum = sum(task1_idx_to_ans.values())
                task1_ans_count = len(task1_idx_to_ans)
                task1_vqa_reward_score = 1.0 if task1_vqa_reward_score_sum == task1_ans_count else 0.0
                reward_score += task1_vqa_reward_score
                reward_extra_info[f"task{task_id}_vqa_reward"] = task1_vqa_reward_score
            except Exception as e:
                task1_vqa_reward_score = 0.0
                reward_score += 0.0
                reward_extra_info[f"task{task_id}_vqa_reward"] = 0.0
        else:
            reward_score += 0.0
            reward_extra_info[f"task{task_id}_vqa_reward"] = 0.0

        reward_extra_info[f"task{task_id}_vqa_reward_response"] = vqa_response if not isinstance(vqa_response, Exception) else str(vqa_response)

    elif task_id == 2:
        task2_reward_score = 0.0

        # Call FormattingEvaluator
        formatting_evaluator = FormattingEvaluatorV2()

        # Rule-based: formatting reward
        task2_format_reward = 1.0 if all(part is not None for part in [predicted_summarize, predicted_tuple, predicted_answer, predicted_feedback]) else 0.0
        task2_reward_score += task2_format_reward
        reward_extra_info[f"task{task_id}_format_reward"] = task2_format_reward

        # Rule-based: part 1 scoring
        feedback_parsed_tuple = formatting_evaluator._parse_part1(feedback_tuple) # GT Tuple
        predict_parsed_tuple = formatting_evaluator._parse_part1(predicted_tuple) # Pred Tuple
        predict_decomposed_ans = formatting_evaluator._extract_answer_paragraphs(predicted_answer)

        part1_reward_dict = formatting_evaluator._calculate_metrics_for_reward(feedback_parsed_tuple, predict_parsed_tuple, predict_decomposed_ans)
        task2_part1_reward = sum(part1_reward_dict.values())
        task2_reward_score += task2_part1_reward
        reward_extra_info[f"task{task_id}_part1_reward"] = task2_part1_reward # +2

        # Launch all API requests in parallel
        args = (prompt, gen_img, feedback_text, regen_img, ground_truth_img, summarize, feedback_tuple, predicted_summarize, predicted_tuple, predicted_answer, predicted_feedback, vqa_question, extra_info, task_id)

        # Gather results
        vqa_pass_or_fail_task = asyncio.create_task(get_response(get_messsages_task2_vqa_pass_or_fail, *args)) # pass or fail
        feedback_task = asyncio.create_task(get_response(get_messages, *args)) # +1

        # Await VQA response
        vqa_pass_or_fail_response = await vqa_pass_or_fail_task

        task2_vqa_reward = 0.0
        vqa_judge = False
        if not isinstance(vqa_pass_or_fail_response, Exception) and vqa_pass_or_fail_response is not None:
            try:
                vqa_pass_or_fail = safe_json_loads(vqa_pass_or_fail_response)
                if vqa_pass_or_fail and vqa_pass_or_fail.get("judge", "").lower() == "pass":
                    task2_vqa_reward = 1.0
                    vqa_judge = True
                else:
                    task2_vqa_reward = 0.0
            except:
                task2_vqa_reward = 0.0

        # Always add VQA reward info
        reward_extra_info["task2_vqa_pass_or_fail_reward"] = task2_vqa_reward
        reward_extra_info["task2_vqa_pass_or_fail_reward_response"] = vqa_pass_or_fail

        if not vqa_judge:
            reward_extra_info[f"task{task_id}_vqa_pass_or_fail_reward"] = 0.0
            reward_extra_info[f"task{task_id}_vqa_pass_or_fail_reward_response"] = "N/A due to VQA alignment fail"
            return {
                "score": task2_reward_score,
                "reward_extra_info": reward_extra_info,
            }

        # Await feedback response
        feedback_response = await feedback_task

        # Process feedback response
        task2_feedback_reward_score = 0.0
        if not isinstance(feedback_response, Exception) and feedback_response is not None:
            try:
                feedback_success = safe_json_loads(feedback_response)
                if feedback_success and feedback_success.get("answer", "").lower() == "yes":
                    task2_feedback_reward_score = 1.0
                else:
                    task2_feedback_reward_score = 0.0

                task2_reward_score += task2_feedback_reward_score
                # task2_ans_count += 1
            except:
                task2_feedback_reward_score = 0.0
                task2_reward_score += 0.0
        else:
            task2_feedback_reward_score = 0.0
            task2_reward_score += 0.0

        # reward_score += (task2_reward_score / task2_ans_count) if task2_ans_count > 0 else 0.0
        reward_extra_info[f"task{task_id}_feedback_reward"] = task2_feedback_reward_score
        reward_extra_info[f"task{task_id}_feedback_reward_response"] = feedback_response if not isinstance(feedback_response, Exception) else str(feedback_response)

        # reward_score += (task2_reward_score / task2_ans_count) if task2_ans_count > 0 else 0.0
        reward_score += task2_reward_score # not normalizing

    elif task_id == 3: # Total score: 4.0
        last = predicted_feedback
        if last is not None and "No need to generate feedback.".lower() in last.lower():
            reward_score = -100
            reward_extra_info[f"task{task_id}_vqa_reward"] = reward_score
            reward_extra_info[f"task{task_id}_vqa_reward_response"] = "None"
            return {
                "score": reward_score,
                "reward_extra_info": reward_extra_info,
            }

        args = (prompt, gen_img, feedback_text, regen_img, ground_truth_img, summarize, feedback_tuple, predicted_summarize, predicted_tuple, predicted_answer, predicted_feedback, vqa_question, extra_info, task_id)

        # Gather results
        vqa_task = asyncio.create_task(get_response(get_messages, *args))
        edit_task = asyncio.create_task(get_response(get_messsages_task2_edit_instruction_following, *args))

        vqa_response = await vqa_task

        # Process VLM response
        task3_vqa_reward_score = 0.0
        if not isinstance(vqa_response, Exception) and vqa_response is not None:
            task3_vqa_idx_to_ans: dict = image_evaluator_parser(vqa_response)
            task3_vqa_reward_score_sum = sum(task3_vqa_idx_to_ans.values())
            task3_vqa_ans_count = len(task3_vqa_idx_to_ans)
            task3_vqa_reward_score = 1.0 if task3_vqa_reward_score_sum == task3_vqa_ans_count else 0.0
            reward_score += task3_vqa_reward_score
            reward_extra_info[f"task{task_id}_vqa_reward"] = task3_vqa_reward_score
        else:
            reward_score += 0.0
            reward_extra_info[f"task{task_id}_vqa_reward"] = 0.0
        
        reward_extra_info[f"task{task_id}_vqa_reward_response"] = vqa_response if not isinstance(vqa_response, Exception) else str(vqa_response)

        edit_response = await edit_task

        # Process VLM response
        task3_edit_reward_score = 0.0
        if not isinstance(edit_response, Exception) and edit_response is not None:
            task3_edit_idx_to_ans: dict = image_evaluator_parser(edit_response)
            task3_edit_reward_score_sum = sum(task3_edit_idx_to_ans.values())
            task3_edit_ans_count = len(task3_edit_idx_to_ans)
            task3_edit_reward_score = 1.0 if task3_edit_reward_score_sum == task3_edit_ans_count else 0.0
            reward_score += task3_edit_reward_score
            reward_extra_info[f"task{task_id}_edit_reward"] = task3_edit_reward_score
        else:
            reward_score += 0.0
            reward_extra_info[f"task{task_id}_edit_reward"] = 0.0
        
        reward_extra_info[f"task{task_id}_edit_reward_response"] = edit_response if not isinstance(edit_response, Exception) else str(edit_response)

    return {
        "score": reward_score,
        "reward_extra_info": reward_extra_info,
    }


async def compute_score_batch_async(prompts, gen_imgs, feedback_texts, regen_imgs, ground_truth_imgs, summarizes, feedback_tuples, vqa_questions, extra_infos, task_ids):
    """Async batch processing with better load balancing"""
    n = len(prompts)
    if n == 0:
        return []

    async def process_single_request(idx, args):
        (prompt, gen_img, feedback_text, regen_img, ground_truth_img, summarize, feedback_tuple, vqa_question, extra_info, task_id) = args
        
        # Load ground truth image
        ground_truth_img = await asyncio.to_thread(lambda p=ground_truth_img: PIL.Image.open(p).convert("RGB")) if ground_truth_img is not None else ground_truth_img

        # Extract predicted feedback tuple for task 2
        formatting_evaluator = FormattingEvaluatorV2() # task 2 only
        par1, part2, part3, part4 = formatting_evaluator._split_text_into_parts(feedback_text.strip())
        predicted_summarize, predicted_tuple, predicted_answer, predicted_feedback = par1, part2, part3, part4
        
        # Process the request with improved error handling
        result = await compute_score_single_async(
            prompt, gen_img, feedback_text, regen_img, ground_truth_img, summarize, feedback_tuple, predicted_summarize, predicted_tuple, predicted_answer, predicted_feedback, vqa_question, extra_info, task_id
        )
        
        return idx, result

    # Create tasks for all requests
    tasks = []
    for idx, args in enumerate(zip(
        prompts, gen_imgs, feedback_texts, regen_imgs, ground_truth_imgs, summarizes, feedback_tuples, vqa_questions, extra_infos, task_ids
    )):
        task = asyncio.create_task(process_single_request(idx, args))
        tasks.append(task)

    # Wait for all tasks to complete
    results = [None] * n
    completed_tasks = await asyncio.gather(*tasks, return_exceptions=True)
    
    for result in completed_tasks:
        if isinstance(result, Exception):
            print(f"Task failed with exception: {result}")
        else:
            idx, res = result
            results[idx] = res

    return results


# Make this async to work with the async reward loop
async def compute_score_batch(prompts, gen_imgs, feedback_texts, regen_imgs, ground_truth_imgs, summarizes, feedback_tuples, vqa_questions, extra_infos, task_ids):
    """Async batch processing - directly calls the async implementation"""
    return await compute_score_batch_async(prompts, gen_imgs, feedback_texts, regen_imgs, ground_truth_imgs, summarizes, feedback_tuples, vqa_questions, extra_infos, task_ids)


def compute_score_batch_sync(prompts, gen_imgs, feedback_texts, regen_imgs, ground_truth_imgs, summarizes, feedback_tuples, vqa_questions, extra_infos, task_ids):
    """Synchronous wrapper for non-async contexts"""
    return asyncio.run(
        compute_score_batch_async(prompts, gen_imgs, feedback_texts, regen_imgs, ground_truth_imgs, summarizes, feedback_tuples, vqa_questions, extra_infos, task_ids)
    )


def get_server_health_status():
    """Get current server health status - useful for monitoring"""
    return {
        "vlm": vlm_client_manager.get_server_status(),
        "llm": llm_client_manager.get_server_status()
    }


def is_meaningful_response(text: str) -> bool:
    if not text or not text.strip():
        return False

    start_idx = text.find('{')
    end_idx = text.rfind('}')
    
    if (start_idx == -1) != (end_idx == -1):
        return False

    if start_idx != -1:
        if end_idx <= start_idx: return False
        try:
            json.loads(text[start_idx : end_idx + 1])
            return True
        except:
            return False
            
    return True

def safe_json_loads(text):
    try:
        # Markdown 블록 제거 및 순수 JSON 추출
        match = re.search(r'\{.*\}', text, re.DOTALL)
        if match:
            return json.loads(match.group())
        return json.loads(text)
    except:
        return None