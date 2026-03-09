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
from recipe.image_rl.prompts import REASONGEN_R1_TEMPLATE
from recipe.image_rl.prompts_finegrained import (
    TASK1_TASK3_IMAGE_GENERATOR_SYSTEM_PROMPT_TEMPLATE,
    PROMPT_TO_SUMMARY_REWARD_SYSTEM_PROMPT,
    SUMMARY_TO_TUPLE_DECOMPOSITION_REWARD_SYSTEM_PROMPT,
    TUPLE_DECOMPOSITION_TO_VQA_REWARD_SYSTEM_PROMPT,
    VQA_TO_FEEDBACK_REWARD_SYSTEM_PROMPT,
)
import asyncio
import threading
import random
from enum import Enum
import torch
from mathruler.grader import extract_boxed_content

# Configuration
VLM_BASE_URLS = [
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


# --- Normalize helpers for task-2 stage judges ---

def _normalize_tuple_lines(text: str) -> str:
    """Strip 'N | ' prefix from tuple lines before sending to stage judges.

    Model generates: '1 | entity - whole (X)'
    Judge expects:   'entity - whole (X)'
    """
    if not text:
        return ''
    lines = []
    for line in text.splitlines():
        stripped = line.strip()
        if not stripped:
            continue
        m = re.match(r'^\d+\s*\|\s*(.*)', stripped)
        lines.append(m.group(1).strip() if m else stripped)
    return '\n'.join(lines)


# --- Task-2 stage message builders ---

def get_messages_task2_stage1(prompt: str, predicted_summarize: str):
    """Stage 1: PROMPT -> SUMMARY reward judge (text-only)."""
    user_content = (
        f"PROMPT:\n{prompt or ''}\n\n"
        f"SUMMARY:\n{predicted_summarize or ''}"
    )
    messages = [
        {"role": "system", "content": PROMPT_TO_SUMMARY_REWARD_SYSTEM_PROMPT},
        {"role": "user", "content": user_content},
    ]
    return messages, RM_VLM_MODEL_PATH


def get_messages_task2_stage2(predicted_summarize: str, tuple_raw: str):
    """Stage 2: SUMMARY -> TUPLE_DECOMPOSITION reward judge (text-only)."""
    user_content = (
        f"SUMMARY:\n{predicted_summarize or ''}\n\n"
        f"PRED_TUPLES:\n{tuple_raw or ''}"
    )
    messages = [
        {"role": "system", "content": SUMMARY_TO_TUPLE_DECOMPOSITION_REWARD_SYSTEM_PROMPT},
        {"role": "user", "content": user_content},
    ]
    return messages, RM_VLM_MODEL_PATH


def get_messages_task2_stage3(gen_img, tuple_raw: str, vqa_raw: str):
    """Stage 3: TUPLE_DECOMPOSITION -> VQA reward judge (requires image)."""
    user_content = (
        f"IMAGE:\n<image>\n\n"
        f"PRED_TUPLES:\n{tuple_raw or ''}\n\n"
        f"VQA_RESULTS:\n{vqa_raw or ''}"
    )
    messages = [
        {"role": "system", "content": TUPLE_DECOMPOSITION_TO_VQA_REWARD_SYSTEM_PROMPT},
        {"role": "user", "content": content_from_prompt_with_images(user_content, [convert_gen_img_to_base64(gen_img)])},
    ]
    return messages, RM_VLM_MODEL_PATH


def get_messages_task2_stage4(prompt: str, predicted_summarize: str, tuple_raw: str, vqa_raw: str, predicted_feedback: str):
    """Stage 4: VQA -> FEEDBACK reward judge (text-only)."""
    user_content = (
        f"PROMPT:\n{prompt or ''}\n\n"
        f"SUMMARY:\n{predicted_summarize or ''}\n\n"
        f"PRED_TUPLES:\n{tuple_raw or ''}\n\n"
        f"VQA_RESULTS:\n{vqa_raw or ''}\n\n"
        f"FEEDBACK:\n{predicted_feedback or 'No need to generate feedback.'}"
    )
    messages = [
        {"role": "system", "content": VQA_TO_FEEDBACK_REWARD_SYSTEM_PROMPT},
        {"role": "user", "content": user_content},
    ]
    return messages, RM_VLM_MODEL_PATH


# Main message construction function
def get_messages(*args):
    prompt, gen_img, feedback_text, regen_img, ground_truth_img, summarize, feedback_tuple, predicted_summarize, predicted_tuple, predicted_answer, predicted_feedback, vqa_question, extra_info, task_id = args

    if task_id == 1:
        
        user_content =(
            f"[IMAGE]:\n<image>\n\n"
            f"[QUESTIONS]:\n{vqa_question}"
        )
        messages = [
            {"role": "system", "content": TASK1_TASK3_IMAGE_GENERATOR_SYSTEM_PROMPT_TEMPLATE},
            {"role": "user", "content": content_from_prompt_with_images(user_content, [convert_gen_img_to_base64(gen_img)])}
        ]
        
    elif task_id == 3:
        user_content =(
            f"[IMAGE]:\n<image>\n\n"
            f"[QUESTIONS]:\n{vqa_question}"
        )
        messages = [
            {"role": "system", "content": TASK1_TASK3_IMAGE_GENERATOR_SYSTEM_PROMPT_TEMPLATE},
            {"role": "user", "content": content_from_prompt_with_images(user_content, [convert_gen_img_to_base64(gen_img)])}
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
    try:
        messages, model = message_builder_fn(*args)
    except Exception as e:
        print(f"[REWARD] Message builder {message_builder_fn.__name__} failed: {type(e).__name__}: {e}")
        return None

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
                vqa_score = _parse_vqa_reward_score(vqa_response, is_fine_grained=True)
            except Exception:
                pass

        reward_score += vqa_score
        reward_extra_info[f"task{task_id}_vqa_reward"] = vqa_score
        reward_extra_info[f"task{task_id}_vqa_reward_response"] = vqa_response if not isinstance(vqa_response, Exception) else str(vqa_response)

    elif task_id == 2: # Total score: format 1.0 + decompose 0..1 + stage judges geometric mean 0..1
        formatting_evaluator = FormattingEvaluatorV2()

        all_parts_present = all(part is not None for part in [predicted_summarize, predicted_tuple, predicted_answer, predicted_feedback])
        task2_rule_based_format_reward = 1.0 if all_parts_present else 0.0
        reward_extra_info["task2_rule_based_format_reward"] = task2_rule_based_format_reward

        # If any part is missing, skip all stage judges and return 0
        if not all_parts_present:
            reward_extra_info["task2_rule_based_decompose_reward"] = 0.0
            reward_extra_info["task2_rule_based_feedback_format_ok"] = 0
            for key in ["task2_prompt_to_summary_reward", "task2_summary_to_tuple_reward", "task2_tuple_to_vqa_reward", "task2_vqa_to_feedback_reward"]:
                reward_extra_info[key] = 0.0
            for key in ["task2_prompt_to_summary_response", "task2_summary_to_tuple_response", "task2_tuple_to_vqa_response", "task2_vqa_to_feedback_response"]:
                reward_extra_info[key] = "Skipped: missing parts"
            return {"score": 0.0, "reward_extra_info": reward_extra_info}

        reward_score += task2_rule_based_format_reward  # 1.0

        # Rule-based: decompose — internal_consistency_ok gates the F1 score (0..1)
        feedback_parsed_tuple = formatting_evaluator._parse_part2(feedback_tuple)
        predict_parsed_tuple = formatting_evaluator._parse_part2(predicted_tuple)
        predict_decomposed_ans = formatting_evaluator._extract_answer_paragraphs(predicted_answer)
        part2_reward_dict = formatting_evaluator._calculate_metrics_for_reward(feedback_parsed_tuple, predict_parsed_tuple, predict_decomposed_ans)

        consistency_ok = part2_reward_dict.get("part2_internal_consistency_ok", 0)
        reward_extra_info["part2_internal_consistency_ok"] = int(consistency_ok)
        f1_score = part2_reward_dict.get("part2_accuracy", 0.0)
        task2_rule_based_decompose_reward = float(f1_score * consistency_ok)  # 0..1
        reward_score += task2_rule_based_decompose_reward
        reward_extra_info["task2_rule_based_decompose_reward"] = task2_rule_based_decompose_reward

        feedback_step_format_ok = formatting_evaluator.check_feedback_step_format(predicted_feedback)
        no_feedback_needed = predicted_feedback is not None and "no need to generate feedback" in predicted_feedback.lower()
        reward_extra_info["task2_rule_based_feedback_format_ok"] = int(feedback_step_format_ok)
        reward_extra_info["task2_no_feedback_needed"] = int(no_feedback_needed)

        # Prepare normalized inputs for stage judges
        tuple_raw = _normalize_tuple_lines(predicted_tuple or '')
        vqa_raw = predicted_answer or ''

        # Format gates: wrong format → skip judge (saves API call), _safe_stage_score maps None → 0.0
        tuple_format_ok = len(predict_parsed_tuple) > 0         # valid "N | content" lines
        vqa_format_ok = len(predict_decomposed_ans) > 0         # has "Answer: Yes/No"
        # s4 runs when feedback has proper "Step N:" format OR when model explicitly says no feedback needed
        s4_format_ok = feedback_step_format_ok or no_feedback_needed
        reward_extra_info["task2_tuple_format_ok"] = int(tuple_format_ok)
        reward_extra_info["task2_vqa_format_ok"] = int(vqa_format_ok)

        async def _none():
            return None

        # Run stage judges in parallel with format gating
        s1_resp, s2_resp, s3_resp, s4_resp = await asyncio.gather(
            get_response(get_messages_task2_stage1, prompt, predicted_summarize),
            get_response(get_messages_task2_stage2, predicted_summarize, tuple_raw) if tuple_format_ok else _none(),
            get_response(get_messages_task2_stage3, gen_img, tuple_raw, vqa_raw) if (gen_img is not None and vqa_format_ok) else _none(),
            get_response(get_messages_task2_stage4, prompt, predicted_summarize, tuple_raw, vqa_raw, predicted_feedback) if s4_format_ok else _none(),
            return_exceptions=True,
        )

        # Parse each stage score; None/Exception → 0.0
        def _safe_stage_score(resp) -> float:
            if resp is None or isinstance(resp, Exception):
                return 0.0
            try:
                return _parse_json_score(resp)
            except Exception:
                return 0.0

        s1 = _safe_stage_score(s1_resp)
        s2 = _safe_stage_score(s2_resp)
        s3 = _safe_stage_score(s3_resp)
        s4 = _safe_stage_score(s4_resp)
        
        vlm_reward = (s1 * s2 * s3 * s4)**0.25
        reward_extra_info["task2_vlm_reward"] = vlm_reward
        reward_score += vlm_reward

        reward_extra_info["task2_prompt_to_summary_reward"] = s1
        reward_extra_info["task2_prompt_to_summary_response"] = s1_resp if not isinstance(s1_resp, Exception) else str(s1_resp)
        reward_extra_info["task2_summary_to_tuple_reward"] = s2
        reward_extra_info["task2_summary_to_tuple_response"] = s2_resp if not isinstance(s2_resp, Exception) else str(s2_resp)
        reward_extra_info["task2_tuple_to_vqa_reward"] = s3
        reward_extra_info["task2_tuple_to_vqa_response"] = s3_resp if not isinstance(s3_resp, Exception) else str(s3_resp)
        reward_extra_info["task2_vqa_to_feedback_reward"] = s4
        reward_extra_info["task2_vqa_to_feedback_response"] = s4_resp if not isinstance(s4_resp, Exception) else str(s4_resp)

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


def _parse_vqa_reward_score(vqa_response: str, is_fine_grained=True) -> int:
    """Extract boxed integer reward score from VQA response."""

    if is_fine_grained:
        task1_idx_to_ans: dict = image_evaluator_parser(vqa_response)
        task1_vqa_reward_score_sum = sum(task1_idx_to_ans.values())
        task1_ans_count = len(task1_idx_to_ans)
        task1_vqa_reward_score = (task1_vqa_reward_score_sum / task1_ans_count) if task1_ans_count != 0 else 0.0
        return float(task1_vqa_reward_score)
    else:
        raw = extract_boxed_content(vqa_response).strip("{}")
        return int(raw)


def _parse_json_score(response: str, key: str = "score") -> float:
    """Extract float score from a JSON response produced by stage judges."""
    parsed = safe_json_loads(response)
    if parsed and key in parsed:
        return float(parsed[key])
    raise ValueError(f"'{key}' not found in response: {response}")


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