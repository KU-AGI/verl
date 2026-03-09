import os
import base64
import PIL
import re
import json
from io import BytesIO
from typing import Optional, List, Dict, Any
import PIL.Image
from openai import AsyncOpenAI
import numpy as np
import time
from recipe.image_rl.utils import FormattingEvaluatorV2
from recipe.image_rl.prompts import *
from recipe.image_rl.prompts_test import INTEGRATED_PIPELINE_REWARD_SYSTEM_PROMPT
import asyncio
import threading
import random
from enum import Enum
import torch
from mathruler.grader import extract_boxed_content

# Configuration
VLM_BASE_URLS = [
    # "http://10.100.44.4:8006/v1", # main1
    # "http://10.100.44.4:8006/v1", # sub2
    # "http://10.100.44.4:8007/v1",
    # "http://10.100.44.6:8006/v1",
    # "http://192.169.0.2:8006/v1",
    "http://192.169.0.2:8007/v1",
]
LLM_BASE_URLS = [
    # "http://10.100.44.2:8004/v1", # sub2
    # "http://10.100.44.2:8005/v1",
    # "http://10.100.44.2:8006/v1",
    # "http://10.100.44.2:8007/v1",
]
API_KEY = "EMPTY"
MAX_RETRIES = 3
# Default model paths (can be overridden via reward_kwargs)
DEFAULT_RM_VLM_MODEL_PATH = "Qwen/Qwen3.5-27B"
DEFAULT_RM_LLM_MODEL_PATH = "Qwen/Qwen3-30B-A3B-Instruct-2507"
RM_VLM_MODEL_PATH = os.environ.get("RM_VLM_MODEL_PATH", DEFAULT_RM_VLM_MODEL_PATH)
RM_LLM_MODEL_PATH = os.environ.get("RM_LLM_MODEL_PATH", DEFAULT_RM_LLM_MODEL_PATH)

# Health checking configuration
HEALTH_CHECK_INTERVAL = 30  # seconds
FAILURE_THRESHOLD = 3  # consecutive failures before marking as unhealthy
RECOVERY_CHECK_INTERVAL = 60  # seconds to wait before checking if unhealthy server recovered

RM_PER_SERVER_INFLIGHT = 16
_rm_slot_lock = threading.Lock()
_rm_slot_queues = {}  # {(loop_id, is_vlm): queue}

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

def content_from_prompt_with_images(prompt: str, image_urls: list[str]):
    parts = prompt.split("<image>")
    if len(parts) - 1 != len(image_urls):
        raise ValueError(f"placeholder <image> cnt({len(parts)-1})and image_urls cnt({len(image_urls)}) different")

    content = []
    for i, part in enumerate(parts):
        if part:
            content.append({"type": "text", "text": part})
        if i < len(image_urls):
            content.append({"type": "image_url", "image_url": {"url": image_urls[i]}})
    return content

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
    ans_line_re = re.compile(r'(?:step\s+)?(\d+)\s*\|\s*Answer:\s*(Yes|No)', re.IGNORECASE) # whether step <index> or <index>
    
    idx_to_ans = {} # 1 | ... Answer: Yes or No -> {1: True or False}
    for idx_str, yn in ans_line_re.findall(text):
        idx = int(idx_str)
        idx_to_ans[idx] = (yn.strip().lower() == "yes")
    
    return idx_to_ans


# Main message construction function
def get_messages(*args):
    prompt, gen_img, feedback_text, regen_img, ground_truth_img, summarize, feedback_tuple, predicted_summarize, predicted_tuple, predicted_answer, predicted_feedback, vqa_question, extra_info, task_id = args

    if task_id == 1:
        user_prompt = REASONGEN_R1_TEMPLATE.format(prompt=prompt.replace("A photo of ", ""))
        messages = [
            {"role": "user", "content": content_from_prompt_with_images(user_prompt, [convert_gen_img_to_base64(gen_img)])}
        ]
    elif task_id == 2:
        if predicted_answer:
            numbered_answer = "\n".join(
                f"{i+1} | {line}" for i, line in enumerate(
                    l for l in predicted_answer.splitlines() if l.strip()
                )
            )
        else:
            numbered_answer = ''
        user_content = (
            f"IMAGE:\n<image>\n\n"
            f"PROMPT:\n{prompt}\n\n"
            f"SUMMARY:\n{predicted_summarize or ''}\n\n"
            f"PRED_TUPLES:\n{predicted_tuple or ''}\n\n"
            f"VQA_RESULTS:\n{numbered_answer}\n\n"
            f"FEEDBACK:\n{predicted_feedback or 'No need to generate feedback.'}"
        )
        messages = [
            {"role": "system", "content": INTEGRATED_PIPELINE_REWARD_SYSTEM_PROMPT},
            {"role": "user", "content": content_from_prompt_with_images(user_content, [convert_gen_img_to_base64(gen_img)])}
        ]
    elif task_id == 3:
        user_prompt = REASONGEN_R1_TEMPLATE.format(prompt=prompt.replace("A photo of ", ""))
        messages = [
            {"role": "user", "content": content_from_prompt_with_images(user_prompt, [convert_gen_img_to_base64(regen_img)])}
        ]
    else:
        raise ValueError(f"Invalid task: {task_id} is must be one of task1, task2, or task3.")

    return messages, RM_VLM_MODEL_PATH


async def get_response_with_client(client, messages, model):
    """Get response from a specific client with improved error handling"""
    extra_body = {
        "top_k": -1,
        "min_p": 0.0,
        "best_of": 1,
        "repetition_penalty": 1.05,
    }

    if "qwen3.5" in model.lower():
        extra_body.update(chat_template_kwargs={"enable_thinking": False})

    response = await client.chat.completions.create(
        model=model,
        messages=messages,

        max_tokens=2048,

        temperature=0.0,
        top_p=1.0,

        extra_body=extra_body,

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
                manager.record_request_result(sid, success=False, error=ValueError("Non-meaningful response"))
                continue
            else:
                manager.record_request_result(sid, success=True)
                return response

        except Exception as e:
            manager.record_request_result(sid, success=False, error=e)
            print(f"[REWARD] API call failed (attempt {attempt+1}/{max_attempts}): {type(e).__name__}: {e}")

        finally:
            await release_rm_client(sid, is_vlm)

    print(f"[REWARD] All {max_attempts} attempts failed for {message_builder_fn.__name__}, returning None")
    return None


async def compute_score_single_async(prompt, gen_img, feedback_text, regen_img, ground_truth_img, summarize, feedback_tuple, predicted_summarize, predicted_tuple, predicted_answer, predicted_feedback, vqa_question, extra_info, task_id):
    """Async version of compute_score"""
    reward_score = 0.0
    reward_extra_info = {}

    if task_id == 1: # Total score: 1.0
        vqa_response = await get_response(get_messages, prompt, gen_img, feedback_text, regen_img, ground_truth_img, summarize, feedback_tuple, predicted_summarize, predicted_tuple, predicted_answer, predicted_feedback, vqa_question, extra_info, task_id)

        vqa_score = 0.0
        if vqa_response is None:
            print(f"[REWARD] Task {task_id}: vqa_response is None")
        elif not isinstance(vqa_response, Exception):
            try:
                vqa_score = _parse_vqa_reward_score(vqa_response)
            except Exception:
                pass

        reward_score += vqa_score
        reward_extra_info[f"task{task_id}_vqa_reward"] = vqa_score
        reward_extra_info[f"task{task_id}_vqa_reward_response"] = vqa_response if not isinstance(vqa_response, Exception) else str(vqa_response)

    elif task_id == 2: # Total score: 4.0
        formatting_evaluator = FormattingEvaluatorV2()

        # Rule-based: formatting reward
        task2_format_reward = 1.0 if all(part is not None for part in [predicted_summarize, predicted_tuple, predicted_answer, predicted_feedback]) else 0.0
        reward_extra_info[f"task{task_id}_format_reward"] = task2_format_reward

        # Rule-based: part 1 scoring
        feedback_parsed_tuple = formatting_evaluator._parse_part2(feedback_tuple)  # GT Tuple
        predict_parsed_tuple = formatting_evaluator._parse_part2(predicted_tuple)  # Pred Tuple
        predict_decomposed_ans = formatting_evaluator._extract_answer_paragraphs(predicted_answer)
        part1_reward_dict = formatting_evaluator._calculate_metrics_for_reward(feedback_parsed_tuple, predict_parsed_tuple, predict_decomposed_ans)
        reward_extra_info[f"task{task_id}_part1_reward"] = sum(part1_reward_dict.values())  # +2

        # Get reward from API
        args = (prompt, gen_img, feedback_text, regen_img, ground_truth_img, summarize, feedback_tuple, predicted_summarize, predicted_tuple, predicted_answer, predicted_feedback, vqa_question, extra_info, task_id)
        feedback_response = await get_response(get_messages, *args)  # +1

        # Process feedback response
        task2_feedback_reward_score = 0.0
        if feedback_response is None:
            print(f"[REWARD] Task {task_id}: feedback_response is None")
        elif not isinstance(feedback_response, Exception):
            try:
                task2_feedback_reward_score = _parse_pipeline_reward_score(feedback_response)
            except Exception:
                pass

        reward_extra_info[f"task{task_id}_feedback_reward"] = task2_feedback_reward_score
        reward_extra_info[f"task{task_id}_feedback_reward_response"] = feedback_response if not isinstance(feedback_response, Exception) else None
        reward_score += task2_feedback_reward_score

    elif task_id == 3: # Total score: 2.0
        if predicted_feedback is not None and "no need to generate feedback." in predicted_feedback.lower():
            reward_score = -100
            reward_extra_info[f"task{task_id}_vqa_reward"] = reward_score
            reward_extra_info[f"task{task_id}_vqa_reward_response"] = "No need to get VQA reward."
            return {"score": reward_score, "reward_extra_info": reward_extra_info}

        args = (prompt, gen_img, feedback_text, regen_img, ground_truth_img, summarize, feedback_tuple, predicted_summarize, predicted_tuple, predicted_answer, predicted_feedback, vqa_question, extra_info, task_id)
        vqa_response = await get_response(get_messages, *args)

        vqa_score = 0.0
        if vqa_response is None:
            print(f"[REWARD] Task {task_id}: vqa_response is None")
        elif not isinstance(vqa_response, Exception):
            try:
                vqa_score = _parse_vqa_reward_score(vqa_response)
            except Exception:
                pass

        reward_score += vqa_score
        reward_extra_info[f"task{task_id}_vqa_reward"] = vqa_score
        reward_extra_info[f"task{task_id}_vqa_reward_response"] = vqa_response if not isinstance(vqa_response, Exception) else str(vqa_response)

    return {
        "score": reward_score,
        "reward_extra_info": reward_extra_info,
    }


def postprocess_task2_rewards(results: List[Dict], extra_infos: List[Dict], task_ids: List[int]) -> List[Dict]:
    for idx, (result, extra_info, task_id) in enumerate(zip(results, extra_infos, task_ids)):
        if result is None or task_id != 2:
            continue
        result["reward_extra_info"] = result.get("reward_extra_info", {})
        results[idx] = result
    return results


async def compute_score_batch_async(prompts, gen_imgs, feedback_texts, regen_imgs, ground_truth_imgs, summarizes, feedback_tuples, vqa_questions, extra_infos, task_ids):
    """Async batch processing with better load balancing"""
    n = len(prompts)
    if n == 0:
        return []

    async def process_single_request(idx, args):
        (prompt, gen_img, feedback_text, regen_img, ground_truth_img, summarize, feedback_tuple, vqa_question, extra_info, task_id) = args

        if ground_truth_img is not None:
            ground_truth_img = await asyncio.to_thread(lambda p=ground_truth_img: PIL.Image.open(p).convert("RGB"))

        formatting_evaluator = FormattingEvaluatorV2()
        predicted_summarize, predicted_tuple, predicted_answer, predicted_feedback = formatting_evaluator._split_text_into_parts(feedback_text.strip())

        result = await compute_score_single_async(
            prompt, gen_img, feedback_text, regen_img, ground_truth_img, summarize, feedback_tuple, predicted_summarize, predicted_tuple, predicted_answer, predicted_feedback, vqa_question, extra_info, task_id
        )
        return idx, result

    tasks = [asyncio.create_task(process_single_request(idx, args)) for idx, args in enumerate(zip(
        prompts, gen_imgs, feedback_texts, regen_imgs, ground_truth_imgs, summarizes, feedback_tuples, vqa_questions, extra_infos, task_ids
    ))]

    results = [None] * n
    none_indices = []
    for result in await asyncio.gather(*tasks, return_exceptions=True):
        if isinstance(result, Exception):
            print(f"[REWARD] Task failed with exception: {result}")
        else:
            idx, res = result
            results[idx] = res
            if res is None:
                none_indices.append(idx)

    if none_indices:
        print(f"[REWARD] Warning: {len(none_indices)}/{n} results are None at indices: {none_indices}")

    return results


# Make this async to work with the async reward loop
async def compute_score_batch(prompts, gen_imgs, feedback_texts, regen_imgs, ground_truth_imgs, summarizes, feedback_tuples, vqa_questions, extra_infos, task_ids, **kwargs):
    """Async batch processing - directly calls the async implementation"""
    # Update model paths from kwargs if provided
    global RM_VLM_MODEL_PATH, RM_LLM_MODEL_PATH
    if 'rm_vlm_model_path' in kwargs:
        RM_VLM_MODEL_PATH = kwargs['rm_vlm_model_path']
    if 'rm_llm_model_path' in kwargs:
        RM_LLM_MODEL_PATH = kwargs['rm_llm_model_path']

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
    return bool(text and text.strip())

def safe_json_loads(text):
    try:
        match = re.search(r'\{.*\}', text, re.DOTALL)
        if match:
            return json.loads(match.group())
        return json.loads(text)
    except:
        return None


def _parse_vqa_reward_score(vqa_response: str) -> int:
    """Extract boxed integer reward score from VQA response."""
    raw = extract_boxed_content(vqa_response).strip("{}")
    return int(raw)


def _parse_pipeline_reward_score(response: str) -> float:
    """Extract final_score from INTEGRATED_PIPELINE_REWARD_SYSTEM_PROMPT JSON response."""
    parsed = safe_json_loads(response)
    if parsed and "final_score" in parsed:
        return float(parsed["final_score"])
    raise ValueError(f"final_score not found in response: {response}")


# Make this async to work with the async reward loop
async def compute_score_batch_with_postprocess(prompts, gen_imgs, feedback_texts, regen_imgs, ground_truth_imgs, summarizes, feedback_tuples, vqa_questions, extra_infos, task_ids, **kwargs):
    """Async batch processing - directly calls the async implementation"""
    # Update model paths from kwargs if provided
    global RM_VLM_MODEL_PATH, RM_LLM_MODEL_PATH
    if 'rm_vlm_model_path' in kwargs:
        RM_VLM_MODEL_PATH = kwargs['rm_vlm_model_path']
    if 'rm_llm_model_path' in kwargs:
        RM_LLM_MODEL_PATH = kwargs['rm_llm_model_path']

    # First, compute all scores
    results = await compute_score_batch_async(
        prompts, gen_imgs, feedback_texts, regen_imgs, ground_truth_imgs, summarizes, feedback_tuples, vqa_questions, extra_infos, task_ids
    )
    
    # Build a mapping from prompt to task1 vqa_reward for cross-task reference
    prompt_to_task1_vqa = {}
    for idx, (prompt, task_id, result) in enumerate(zip(prompts, task_ids, results)):
        if result is None:
            continue
        if task_id == 1:
            vqa_reward = result.get("reward_extra_info", {}).get("task1_vqa_reward", 0.0)
            prompt_to_task1_vqa[prompt] = vqa_reward
    
    # Update extra_infos with task1 vqa rewards and feedback_text for task2
    updated_extra_infos = []
    for idx, (prompt, feedback_text, extra_info, task_id) in enumerate(zip(prompts, feedback_texts, extra_infos, task_ids)):
        updated_info = dict(extra_info) if extra_info else {}
        
        if task_id == 2:
            # Add task1 vqa reward if available
            if prompt in prompt_to_task1_vqa:
                updated_info["task1_vqa_reward"] = prompt_to_task1_vqa[prompt]
            # Store feedback_text for post-processing
            updated_info["feedback_text"] = feedback_text
            
        updated_extra_infos.append(updated_info)
    
    # Run post-processing for task2
    results = postprocess_task2_rewards(results, updated_extra_infos, task_ids)
    
    return results


def compute_score_batch_with_postprocess_sync(
    prompts, gen_imgs, feedback_texts, regen_imgs, ground_truth_imgs,
    summarizes, feedback_tuples, vqa_questions, extra_infos, task_ids
):
    """Synchronous wrapper for compute_score_batch_with_postprocess"""
    return asyncio.run(
        compute_score_batch_with_postprocess(
            prompts, gen_imgs, feedback_texts, regen_imgs, ground_truth_imgs, summarizes, feedback_tuples, vqa_questions, extra_infos, task_ids
        )
    )