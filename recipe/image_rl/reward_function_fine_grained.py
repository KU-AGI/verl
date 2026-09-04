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
from recipe.image_rl.utils import (
    FormattingEvaluatorV3,
    classify_task2_feedback,
    filter_entity_questions,
    should_route_task3,
)
from recipe.image_rl.prompts import REASONGEN_R1_TEMPLATE
from recipe.image_rl.prompts_finegrained_simple import (
    TASK1_TASK3_IMAGE_GENERATOR_SYSTEM_PROMPT_TEMPLATE,
    TASK3_REGENERATION_FOLLOWED_BY_EDITING_SYSTEM_PROMPT,
    PROMPT_TO_TUPLE_DECOMPOSITION_REWARD_SYSTEM_PROMPT,
    TUPLE_DECOMPOSITION_TO_VQA_REWARD_SYSTEM_PROMPT,
    VQA_TO_FEEDBACK_REWARD_SYSTEM_PROMPT,
)
import asyncio
import threading
import random
from enum import Enum
import torch
import math
import aiohttp
from mathruler.grader import extract_boxed_content
from recipe.image_rl.gdino_regex import _CONNECTORS, SKIP_KEYWORDS, _COMPILED_RELATIONS

# Configuration
VLM_BASE_URLS = [
    "http://10.100.87.2:8005/v1",
    "http://10.100.87.2:8006/v1",
    "http://10.100.87.2:8007/v1",
    "http://10.100.87.6:8005/v1",
    "http://10.100.87.6:8006/v1",
    "http://10.100.87.6:8007/v1",
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
DEFAULT_RM_VLM_MODEL_PATH = "Qwen/Qwen3.5-35B-A3B"
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

# Detector configuration
DETECTOR_URLS = [
    "http://10.100.87.2:8086",
    "http://10.100.87.2:8087",
    "http://10.100.87.6:8086",
    "http://10.100.87.6:8087",
]
DETECTOR_TIMEOUT = 300000.0
DETECTOR_MAX_RETRIES = 2
DET_PER_SERVER_INFLIGHT = 4
DETECTOR_ALIGN_BONUS_WEIGHT = 0.2
MDP_REASONING_REWARD_WEIGHT = 0.03
MDP_REASONING_REWARD_COST = 0.02
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
    idx_to_ans = {}  # 1 | ... Answer: Yes or No -> {1: True or False}

    for line in text.splitlines():
        line = line.strip()
        if not line:
            continue
        # Match lines starting with "N | " (or "step N | ")
        m = re.match(r'(?:step\s+)?(\d+)\s*\|(.+)', line, re.IGNORECASE)
        if not m:
            continue
        idx = int(m.group(1))
        rest = m.group(2)
        # Find Answer: Yes/No anywhere in the rest of the line (handles both
        # split-line format "1 | Answer: No" and inline format "1 | Reason: ... Answer: No")
        ans_m = re.search(r'\bAnswer:\s*(Yes|No)\b', rest, re.IGNORECASE)
        if ans_m:
            idx_to_ans[idx] = (ans_m.group(1).lower() == "yes")

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


def _add_index_to_vqa_lines(text: str) -> str:
    """Add 'N | ' index prefix to each VQA result line before sending to stage judges.

    Model generates: 'The image shows a dog... Answer: Yes'
    Judge expects:   '1 | The image shows a dog... Answer: Yes'
    """
    if not text:
        return ''
    lines = [line.strip() for line in text.splitlines() if line.strip()]
    return '\n'.join(f'{i + 1} | {line}' for i, line in enumerate(lines))


def _add_index_to_vqa_entries(text: str) -> str:
    """Add 'N | ' index prefix to each VQA result entry.

    Unlike `_add_index_to_vqa_lines`, this respects the actual task2 answer
    structure where one tuple result may span multiple lines but terminates with
    `Answer: Yes/No`. This keeps Stage3/4 inputs aligned 1:1 with PRED_TUPLES.
    """
    if not text:
        return ''
    formatting_evaluator = FormattingEvaluatorV3()
    entries = formatting_evaluator._extract_verify_paragraphs(text)
    if not entries:
        return ''
    return '\n\n'.join(f'{i + 1} | {entry.strip()}' for i, entry in enumerate(entries))


# --- Task-2 stage message builders ---

def get_messages_task2_stage1(prompt: str, tuple_raw: str):
    """Stage 1: PROMPT -> TUPLE_DECOMPOSITION reward judge (text-only)."""
    user_content = (
        f"PROMPT:\n{prompt or ''}\n\n"
        f"PRED_TUPLES:\n{tuple_raw or ''}"
    )
    messages = [
        {"role": "system", "content": PROMPT_TO_TUPLE_DECOMPOSITION_REWARD_SYSTEM_PROMPT},
        {"role": "user", "content": user_content},
    ]
    return messages, RM_VLM_MODEL_PATH


def get_messages_task2_stage2(gen_img, tuple_raw: str, vqa_raw: str):
    """Stage 2: TUPLE_DECOMPOSITION -> VQA reward judge (requires image)."""
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


def get_messages_task2_stage3(tuple_raw: str, vqa_raw: str, predicted_feedback: str):
    """Stage 3: VQA -> FEEDBACK reward judge (text-only)."""
    user_content = (
        f"PRED_TUPLES:\n{tuple_raw or ''}\n\n"
        f"VQA_RESULTS:\n{vqa_raw or ''}\n\n"
        f"FEEDBACK:\n{predicted_feedback or 'No need to generate feedback.'}"
    )
    messages = [
        {"role": "system", "content": VQA_TO_FEEDBACK_REWARD_SYSTEM_PROMPT},
        {"role": "user", "content": user_content},
    ]
    return messages, RM_VLM_MODEL_PATH


def get_messages_task2_stage4(prompt: str, predicted_summarize: str, tuple_raw: str, vqa_raw: str, predicted_feedback: str):
    """Compatibility wrapper for old callers; summary is ignored in V3."""
    return get_messages_task2_stage3(tuple_raw, vqa_raw, predicted_feedback)


def get_messages_task3_edit(gen_img, predicted_feedback: str, regen_img):
    """Task 3 edit reward: SOURCE_IMAGE + FEEDBACK -> EDITED_IMAGE judge.

    Uses TASK3_REGENERATION_FOLLOWED_BY_EDITING_SYSTEM_PROMPT.
    Returns JSON {"REWARD": 0.0..2.0}.
    """
    user_content = (
        "SOURCE_IMAGE:\n<image>\n\n"
        f"FEEDBACK:\n{predicted_feedback or ''}\n\n"
        "EDITED_IMAGE:\n<image>"
    )
    messages = [
        {"role": "system", "content": TASK3_REGENERATION_FOLLOWED_BY_EDITING_SYSTEM_PROMPT},
        {
            "role": "user",
            "content": content_from_prompt_with_images(
                user_content,
                [convert_gen_img_to_base64(gen_img), convert_gen_img_to_base64(regen_img)],
            ),
        },
    ]
    return messages, RM_VLM_MODEL_PATH


# Main message construction function
def get_messages(*args):
    prompt, gen_img, feedback_text, regen_img, ground_truth_img, summarize, feedback_tuple, predicted_summarize, predicted_tuple, predicted_answer, predicted_feedback, vqa_question, extra_info, task_id = args

    if task_id == 1:
        filtered_vqa = filter_entity_questions(feedback_tuple, vqa_question)
        user_content =(
            f"[IMAGE]:\n<image>\n\n"
            f"[QUESTIONS]:\n{filtered_vqa}"
        )
        messages = [
            {"role": "system", "content": TASK1_TASK3_IMAGE_GENERATOR_SYSTEM_PROMPT_TEMPLATE},
            {"role": "user", "content": content_from_prompt_with_images(user_content, [convert_gen_img_to_base64(gen_img)])}
        ]

    elif task_id == 3:
        filtered_vqa = filter_entity_questions(feedback_tuple, vqa_question)
        user_content =(
            f"[IMAGE]:\n<image>\n\n"
            f"[QUESTIONS]:\n{filtered_vqa}"
        )
        messages = [
            {"role": "system", "content": TASK1_TASK3_IMAGE_GENERATOR_SYSTEM_PROMPT_TEMPLATE},
            {"role": "user", "content": content_from_prompt_with_images(user_content, [convert_gen_img_to_base64(regen_img)])}
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


# =============================================================================
# Detector helpers
# =============================================================================
def _parse_tuple_lines(text):
    if not isinstance(text, str):
        return []
    return [
        (int(m.group(1)), m.group(2).strip())
        for line in text.strip().split("\n")
        if (m := re.match(r"(\d+)\s*\|\s*(.*)", line))
    ]


def verify_detection_single(feedback_tuple) -> List[Dict[str, Any]]:
    parsed_tup = _parse_tuple_lines(feedback_tuple)

    results = []
    for num, content in parsed_tup:
        info = None
        if 'spatial' in content:
            if m := re.search(r'\((.*?)\)', content):
                parts = [p.strip() for p in m.group(1).split(',')]
                if len(parts) >= 3:
                    s, o = parts[0], parts[1]
                    r_text = ", ".join(parts[2:])

                    cs = re.sub(r"_\d+$", "", _CONNECTORS.sub('', s).strip())
                    co = re.sub(r"_\d+$", "", _CONNECTORS.sub('', o).strip())

                    if not (cs.lower() in SKIP_KEYWORDS or co.lower() in SKIP_KEYWORDS or cs.startswith('[')):
                        canonical_rel = next((c for c, p, _ in _COMPILED_RELATIONS if p.search(r_text)), None)
                        if canonical_rel:
                            info = {
                                "subject": cs,
                                "object": co,
                                "relation": canonical_rel,
                                "tuple_idx": num,
                                "type": "spatial",
                            }

        elif 'count' in content:
            if m := re.search(r'\((.*?)\)', content):
                parts = [p.strip() for p in m.group(1).split(',')]
                if len(parts) >= 2:
                    s, expr = parts[0], parts[1]
                    cs = re.sub(r"_\d+$", "", _CONNECTORS.sub('', s).strip())
                    if not (cs.lower() in SKIP_KEYWORDS or cs.startswith('[')) and re.search(r'\d', expr):
                        info = {
                            "subject": cs,
                            "object": cs,
                            "num": expr,
                            "tuple_idx": num,
                            "type": "counting",
                        }

        if info is not None:
            results.append(info)

    return results


async def request_detector_single(detection_list: List[Dict[str, Any]], img) -> Dict[str, Any]:
    """Send detection request to detector server with slot-based load balancing and retry."""

    if not detection_list or not DETECTOR_URLS:
        return {"results": {}, "details": [], "errors": []}

    # ---- loop-local slot queue ----
    async def _ensure_det_slots() -> asyncio.Queue:
        loop = asyncio.get_running_loop()
        loop_id = id(loop)

        if loop_id in _det_slot_queues:
            queue = _det_slot_queues[loop_id]
            try:
                queue.qsize()
                return queue
            except RuntimeError:
                pass

        with _det_slot_lock:
            if loop_id in _det_slot_queues:
                queue = _det_slot_queues[loop_id]
                try:
                    queue.qsize()
                    return queue
                except RuntimeError:
                    del _det_slot_queues[loop_id]

            q = asyncio.Queue()
            start = random.randrange(len(DETECTOR_URLS))
            for i in range(DET_PER_SERVER_INFLIGHT * len(DETECTOR_URLS)):
                sid = (start + i) % len(DETECTOR_URLS)
                q.put_nowait(sid)

            _det_slot_queues[loop_id] = q
            return q

    slot_q = await _ensure_det_slots()

    # ---- image -> raw base64 ----
    img_b64 = convert_gen_img_to_base64(img)
    if img_b64.startswith("data:"):
        img_b64 = img_b64.split(",", 1)[1]

    # ---- build payload ----
    info_list = []
    idx_mapping = {}

    for det_info in detection_list:
        det_type = det_info.get("type", "")
        api_info = None

        if det_type == "spatial":
            api_info = {
                "type": "spatial",
                "subject": det_info.get("subject"),
                "object": det_info.get("object"),
                "relation": det_info.get("relation"),
            }
        elif det_type in ["counting", "numeracy"]:
            api_info = {
                "type": "numeracy",
                "object": det_info.get("object"),
                "num": str(det_info.get("num", "")),
            }

        if api_info is None:
            continue

        idx_mapping[len(info_list)] = det_info.get("tuple_idx", len(info_list))
        info_list.append(api_info)

    if not info_list:
        return {"results": {}, "details": [], "errors": ["No valid detection items"]}

    payload = {"info_list": info_list, "img_url": img_b64}

    # ---- retry bookkeeping ----
    per_server_attempts = {sid: 0 for sid in range(len(DETECTOR_URLS))}
    max_total_attempts = len(DETECTOR_URLS) * DETECTOR_MAX_RETRIES

    results: Dict[int, bool] = {}
    details: List[Dict[str, Any]] = []
    errors: List[str] = []

    timeout = aiohttp.ClientTimeout(total=DETECTOR_TIMEOUT)

    async with aiohttp.ClientSession(timeout=timeout) as session:
        for attempt in range(max_total_attempts):
            sid = await slot_q.get()

            if per_server_attempts[sid] >= DETECTOR_MAX_RETRIES:
                slot_q.put_nowait(sid)
                continue
            per_server_attempts[sid] += 1

            detect_url = f"{DETECTOR_URLS[sid]}/detect"

            try:
                async with session.post(detect_url, json=payload) as resp:
                    if resp.status != 200:
                        txt = await resp.text()
                        errors.append(f"{detect_url} -> {resp.status}: {txt[:200]}")
                        continue

                    data = await resp.json()
                    api_results = data.get("results", [])

                    for api_idx, result_list in enumerate(api_results):
                        if api_idx not in idx_mapping or not result_list:
                            continue
                        tuple_idx = idx_mapping[api_idx]
                        r0 = result_list[0]

                        det_judge = bool(r0.get("det_judge", False))
                        results[tuple_idx] = det_judge
                        details.append({
                            "tuple_idx": tuple_idx,
                            "det_judge": det_judge,
                            "det_reason": r0.get("det_reason", ""),
                            "det_info": r0.get("det_info", {}),
                            "vis_data": r0.get("vis_data"),
                            "server": DETECTOR_URLS[sid],
                        })

                    return {"results": results, "details": details, "errors": errors}

            except Exception as e:
                errors.append(f"{detect_url} exception: {repr(e)}")

            finally:
                slot_q.put_nowait(sid)

            await asyncio.sleep(0.1 * (attempt + 1))

    return {
        "results": results,
        "details": details,
        "errors": errors if errors else ["All detector servers failed"],
    }


def _compute_detector_bonus(detector_response, detection_results) -> float:
    """Compute detector bonus reward: ratio of passed detections (0..1)."""
    if not detection_results or not detector_response:
        return 0.0
    det_results_dict = detector_response.get("results", {})
    if not det_results_dict:
        return 0.0
    return sum(det_results_dict.values()) / len(det_results_dict)


def _shape_mdp_reasoning_reward(
    raw_reward: float,
    reward_weight: Optional[float] = None,
    reward_cost: Optional[float] = None,
) -> float:
    """Shape a [0, 2] reasoning RM score into penalty-only MDP reward.

    New formulation:
        r = ((raw_reward / 2) - 1) / 3
    so a perfect reasoning step receives 0 and failures receive a penalty in
    [-1/3, 0]. The weight/cost arguments are kept for config compatibility.
    """
    try:
        normalized = max(0.0, min(float(raw_reward) / 2.0, 1.0))
    except (TypeError, ValueError):
        normalized = 0.0
    return (normalized - 1.0) / 3.0


async def compute_score_single_async(
    prompt,
    gen_img,
    feedback_text,
    regen_img,
    ground_truth_img,
    summarize,
    feedback_tuple,
    predicted_summarize,
    predicted_tuple,
    predicted_answer,
    predicted_feedback,
    vqa_question,
    extra_info,
    task_id,
    mdp_reasoning_reward_weight: Optional[float] = None,
    mdp_reasoning_reward_cost: Optional[float] = None,
):
    """Async version of compute_score"""
    reward_score = 0.0
    reward_extra_info = {}

    if task_id == 1: # Total score: vqa (0..1) + detector bonus (0..1)
        # Parse detection items from feedback_tuple
        detection_results = verify_detection_single(feedback_tuple)

        # Launch VQA and detector in parallel
        vqa_task = asyncio.create_task(
            get_response(get_messages, prompt, gen_img, feedback_text, regen_img, ground_truth_img, summarize, feedback_tuple, predicted_summarize, predicted_tuple, predicted_answer, predicted_feedback, vqa_question, extra_info, task_id)
        )
        detector_task = None
        if detection_results and DETECTOR_URLS:
            detector_task = asyncio.create_task(
                request_detector_single(detection_results, gen_img)
            )

        vqa_response = await vqa_task
        detector_response = (await detector_task) if detector_task else {"results": {}, "details": [], "errors": []}

        # VQA score
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

        # Detector bonus
        detector_bonus = _compute_detector_bonus(detector_response, detection_results)
        reward_score += detector_bonus
        reward_extra_info[f"task{task_id}_detector_reward"] = detector_bonus
        reward_extra_info[f"task{task_id}_detector_details"] = detector_response.get("details", [])
        reward_extra_info[f"task{task_id}_detector_active"] = int(bool(detection_results))
        reward_extra_info[f"task{task_id}_detector_count"] = len(detection_results)
        reward_extra_info[f"task{task_id}_detector_active_score"] = int(bool(detection_results))
        reward_extra_info[f"task{task_id}_detector_active_only_reward"] = detector_bonus if detection_results else None

        # MDP image score: V_t + m_x D_t, with max score 1 + m_x.
        detector_active = float(bool(detection_results))
        task1_image_score = vqa_score + detector_active * detector_bonus
        task1_image_score_max = 1.0 + detector_active
        reward_extra_info["task1_align"] = task1_image_score
        reward_extra_info["task1_image_score"] = task1_image_score
        reward_extra_info["task1_image_score_max"] = task1_image_score_max
        reward_extra_info["task1_mdp_reward"] = task1_image_score

    elif task_id == 2: # Total score: post-hoc finalized from stage scores
        formatting_evaluator = FormattingEvaluatorV3()
        raw_stage_only = bool((extra_info or {}).get("task2_raw_stage_only", False))

        all_parts_present = all(part is not None for part in [predicted_tuple, predicted_answer, predicted_feedback])
        task2_rule_based_format_reward = 1.0 if all_parts_present else 0.0
        reward_extra_info["task2_rule_based_format_reward"] = task2_rule_based_format_reward

        # If any part is missing, skip all stage judges and return 0
        if not all_parts_present:
            reward_extra_info["task2_rule_based_decompose_reward"] = 0.0
            reward_extra_info["task2_rule_based_feedback_format_ok"] = 0
            reward_extra_info["task2_no_feedback_needed"] = 0
            reward_extra_info["task2_no_feedback_needed_score"] = 0
            reward_extra_info["task2_noncanonical_no_edit"] = 0
            reward_extra_info["task2_tuple_format_ok"] = 0
            reward_extra_info["task2_vqa_format_ok"] = 0
            reward_extra_info["task2_stage_raw_only"] = int(raw_stage_only)
            for key in ["task2_prompt_to_tuple_reward", "task2_tuple_to_vqa_reward", "task2_vqa_to_feedback_reward"]:
                reward_extra_info[key] = 0.0
            reward_extra_info["task2_vqa_to_feedback_content_reward"] = 0.0
            reward_extra_info["task2_step2_reward"] = _shape_mdp_reasoning_reward(
                0.0,
                mdp_reasoning_reward_weight,
                mdp_reasoning_reward_cost,
            )
            reward_extra_info["task2_step3_reward"] = _shape_mdp_reasoning_reward(
                0.0,
                mdp_reasoning_reward_weight,
                mdp_reasoning_reward_cost,
            )
            reward_extra_info["task2_step4_reward"] = _shape_mdp_reasoning_reward(
                0.0,
                mdp_reasoning_reward_weight,
                mdp_reasoning_reward_cost,
            )
            for key in ["task2_prompt_to_tuple_response", "task2_tuple_to_vqa_response", "task2_vqa_to_feedback_response"]:
                reward_extra_info[key] = "Skipped: missing parts"
            return {"score": 0.0, "reward_extra_info": reward_extra_info}

        reward_score += task2_rule_based_format_reward  # 1.0

        # Rule-based: decompose — internal_consistency_ok gates the F1 score (0..1)
        feedback_parsed_tuple = formatting_evaluator._parse_tuples(feedback_tuple)
        predict_parsed_tuple = formatting_evaluator._parse_tuples(predicted_tuple)
        predict_decomposed_ans = formatting_evaluator._extract_verify_paragraphs(predicted_answer)
        part2_reward_dict = formatting_evaluator._calculate_metrics_for_reward(feedback_parsed_tuple, predict_parsed_tuple, predict_decomposed_ans)

        consistency_ok = part2_reward_dict.get("task2_internal_consistency_ok", 0)
        reward_extra_info["task2_internal_consistency_ok"] = int(consistency_ok)
        f1_score = part2_reward_dict.get("task2_part2_accuracy", 0.0)
        task2_rule_based_decompose_reward = float(f1_score * consistency_ok)  # 0..1
        #reward_score += task2_rule_based_decompose_reward
        reward_extra_info["task2_rule_based_decompose_reward"] = task2_rule_based_decompose_reward

        feedback_step_format_ok = formatting_evaluator.check_feedback_step_format(predicted_feedback)
        feedback_class = classify_task2_feedback(predicted_feedback)
        canonical_no_edit = feedback_class == "canonical_no_edit"
        noncanonical_no_edit = feedback_class == "noncanonical_no_edit"
        reward_extra_info["task2_rule_based_feedback_format_ok"] = int(feedback_step_format_ok)
        reward_extra_info["task2_no_feedback_needed"] = int(canonical_no_edit)
        reward_extra_info["task2_no_feedback_needed_score"] = int(canonical_no_edit)
        reward_extra_info["task2_noncanonical_no_edit"] = int(noncanonical_no_edit)
        # Prepare normalized inputs for stage judges
        # tuple_raw = _normalize_tuple_lines(predicted_tuple or '')
        tuple_raw = predicted_tuple or ''
        vqa_raw = _add_index_to_vqa_entries(predicted_answer or '')

        # Format gates: wrong format → skip judge (saves API call), _safe_stage_score maps None → 0.0
        tuple_format_ok = formatting_evaluator.check_tuple_schema_ok(predict_parsed_tuple)
        vqa_format_ok = len(predict_decomposed_ans) > 0         # has "Answer: Yes/No"
        # Feedback judge runs only when feedback is needed AND format is correct.
        feedback_should_run = feedback_class == "other" and feedback_step_format_ok
        reward_extra_info["task2_tuple_format_ok"] = int(tuple_format_ok)
        reward_extra_info["task2_vqa_format_ok"] = int(vqa_format_ok)

        async def _none():
            return None

        # Run stage judges in parallel with format gating
        prompt_to_tuple_resp, tuple_to_vqa_resp, feedback_resp = await asyncio.gather(
            get_response(get_messages_task2_stage1, prompt, tuple_raw) if tuple_format_ok else _none(),
            get_response(get_messages_task2_stage2, gen_img, tuple_raw, vqa_raw) if (gen_img is not None and tuple_format_ok and vqa_format_ok) else _none(),
            get_response(get_messages_task2_stage3, tuple_raw, vqa_raw, predicted_feedback) if (vqa_format_ok and feedback_should_run) else _none(),
            return_exceptions=True,
        )

        # Parse each stage score; None/Exception → 0.0
        def _safe_stage_score(resp) -> float:
            if resp is None or isinstance(resp, Exception):
                return 0.0
            try:
                return max(0.0, min(2.0, _parse_json_score(resp)))
            except Exception:
                return 0.0

        prompt_to_tuple = _safe_stage_score(prompt_to_tuple_resp)
        tuple_to_vqa = _safe_stage_score(tuple_to_vqa_resp)
        feedback_content = _safe_stage_score(feedback_resp)

        reward_extra_info["task2_prompt_to_tuple_reward"] = prompt_to_tuple
        reward_extra_info["task2_prompt_to_tuple_response"] = prompt_to_tuple_resp if not isinstance(prompt_to_tuple_resp, Exception) else str(prompt_to_tuple_resp)
        reward_extra_info["task2_tuple_to_vqa_reward"] = tuple_to_vqa
        reward_extra_info["task2_tuple_to_vqa_response"] = tuple_to_vqa_resp if not isinstance(tuple_to_vqa_resp, Exception) else str(tuple_to_vqa_resp)
        reward_extra_info["task2_vqa_to_feedback_content_reward"] = feedback_content
        reward_extra_info["task2_vqa_to_feedback_response"] = feedback_resp if not isinstance(feedback_resp, Exception) else str(feedback_resp)
        reward_extra_info["task2_step2_reward"] = _shape_mdp_reasoning_reward(
            prompt_to_tuple,
            mdp_reasoning_reward_weight,
            mdp_reasoning_reward_cost,
        )
        reward_extra_info["task2_step3_reward"] = _shape_mdp_reasoning_reward(
            tuple_to_vqa,
            mdp_reasoning_reward_weight,
            mdp_reasoning_reward_cost,
        )
        reward_extra_info["task2_step4_content_reward"] = _shape_mdp_reasoning_reward(
            feedback_content,
            mdp_reasoning_reward_weight,
            mdp_reasoning_reward_cost,
        )
        reward_extra_info["task2_step4_reward"] = reward_extra_info["task2_step4_content_reward"]
        reward_extra_info["task2_stage_raw_only"] = int(raw_stage_only)

        if not raw_stage_only:
            decision_vqa = float((extra_info or {}).get("task2_decision_vqa_reward", 0.0))
            decision_source = (extra_info or {}).get("task2_decision_vqa_source", "missing")
            vlm_reward, reward_extra_info = finalize_task2_reward_extra_info(
                reward_extra_info,
                decision_vqa=decision_vqa,
                decision_source=decision_source,
                mdp_reasoning_reward_weight=mdp_reasoning_reward_weight,
                mdp_reasoning_reward_cost=mdp_reasoning_reward_cost,
            )
            reward_score += vlm_reward

    elif task_id == 3: # Total score: sqrt(vqa*2 * edit) + detector bonus (0..1)
        if not should_route_task3(predicted_feedback):
            reward_extra_info.update({
                "task3_vqa_reward": 0.0,
                "task3_vqa_reward_response": "Skipped: no-edit feedback",
                "task3_edit_reward": 0.0,
                "task3_edit_reward_response": "Skipped: no-edit feedback",
                "task3_detector_reward": 0.0,
                "task3_detector_details": [],
                "task3_detector_active": 0,
                "task3_detector_count": 0,
                "task3_detector_active_score": 0,
                "task3_detector_active_only_reward": None,
                "task3_align": 0.0,
                "task3_image_score": 0.0,
                "task3_image_score_max": 1.0,
                "task3_if": 0.0,
                "task3_edit_if_reward": 0.0,
            })
            return {"score": 0.0, "reward_extra_info": reward_extra_info}

        # Parse detection items from feedback_tuple
        detection_results = verify_detection_single(feedback_tuple)

        call_args = (prompt, gen_img, feedback_text, regen_img, ground_truth_img, summarize, feedback_tuple, predicted_summarize, predicted_tuple, predicted_answer, predicted_feedback, vqa_question, extra_info, task_id)
        async def _none():
            return None

        # Launch VQA, edit, and detector in parallel
        vqa_response, edit_response, detector_response = await asyncio.gather(
            get_response(get_messages, *call_args) if regen_img is not None else _none(),
            get_response(get_messages_task3_edit, gen_img, predicted_feedback, regen_img) if regen_img is not None else _none(),
            request_detector_single(detection_results, regen_img) if (detection_results and DETECTOR_URLS and regen_img is not None) else _none(),
            return_exceptions=True,
        )

        # Handle detector exception from gather
        if isinstance(detector_response, Exception):
            print(f"[REWARD] Task {task_id}: detector exception: {detector_response}")
            detector_response = None

        vqa_score = 0.0
        if vqa_response is None:
            print(f"[REWARD] Task {task_id}: vqa_response is None")
        elif not isinstance(vqa_response, Exception):
            try:
                vqa_score = _parse_vqa_reward_score(vqa_response)
            except Exception:
                pass

        edit_score = 0.0
        if edit_response is None:
            print(f"[REWARD] Task {task_id}: edit_response is None")
        elif not isinstance(edit_response, Exception):
            try:
                edit_score = _parse_json_score(edit_response)
            except Exception:
                pass

        reward_score = np.sqrt((vqa_score * 2) * edit_score)
        reward_extra_info[f"task{task_id}_vqa_reward"] = vqa_score
        reward_extra_info[f"task{task_id}_vqa_reward_response"] = vqa_response if not isinstance(vqa_response, Exception) else str(vqa_response)
        reward_extra_info[f"task{task_id}_edit_reward"] = edit_score
        reward_extra_info[f"task{task_id}_edit_reward_response"] = edit_response if not isinstance(edit_response, Exception) else str(edit_response)

        # Detector bonus
        detector_bonus = _compute_detector_bonus(detector_response, detection_results)
        reward_score += detector_bonus
        reward_extra_info[f"task{task_id}_detector_reward"] = detector_bonus
        reward_extra_info[f"task{task_id}_detector_details"] = detector_response.get("details", []) if detector_response else []
        reward_extra_info[f"task{task_id}_detector_active"] = int(bool(detection_results))
        reward_extra_info[f"task{task_id}_detector_count"] = len(detection_results)
        reward_extra_info[f"task{task_id}_detector_active_score"] = int(bool(detection_results))
        reward_extra_info[f"task{task_id}_detector_active_only_reward"] = detector_bonus if detection_results else None

        # MDP image score: V_t + m_x D_t, with max score 1 + m_x.
        detector_active = float(bool(detection_results))
        task3_image_score = vqa_score + detector_active * detector_bonus
        task3_image_score_max = 1.0 + detector_active
        task3_edit_if_reward = edit_score / 2.0
        reward_extra_info["task3_align"] = task3_image_score
        reward_extra_info["task3_image_score"] = task3_image_score
        reward_extra_info["task3_image_score_max"] = task3_image_score_max
        reward_extra_info["task3_if"] = task3_edit_if_reward
        reward_extra_info["task3_edit_if_reward"] = task3_edit_if_reward

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


async def compute_score_batch_async(
    prompts,
    gen_imgs,
    feedback_texts,
    regen_imgs,
    ground_truth_imgs,
    summarizes,
    feedback_tuples,
    vqa_questions,
    extra_infos,
    task_ids,
    mdp_reasoning_reward_weight: Optional[float] = None,
    mdp_reasoning_reward_cost: Optional[float] = None,
):
    """Async batch processing with better load balancing"""
    n = len(prompts)
    if n == 0:
        return []

    async def process_single_request(idx, args):
        (prompt, gen_img, feedback_text, regen_img, ground_truth_img, summarize, feedback_tuple, vqa_question, extra_info, task_id) = args

        if ground_truth_img is not None:
            ground_truth_img = await asyncio.to_thread(lambda p=ground_truth_img: PIL.Image.open(p).convert("RGB"))

        formatting_evaluator = FormattingEvaluatorV3()
        predicted_tuple, predicted_answer, predicted_feedback = formatting_evaluator._split_text_into_parts(feedback_text.strip())
        predicted_summarize = None

        result = await compute_score_single_async(
            prompt,
            gen_img,
            feedback_text,
            regen_img,
            ground_truth_img,
            summarize,
            feedback_tuple,
            predicted_summarize,
            predicted_tuple,
            predicted_answer,
            predicted_feedback,
            vqa_question,
            extra_info,
            task_id,
            mdp_reasoning_reward_weight=mdp_reasoning_reward_weight,
            mdp_reasoning_reward_cost=mdp_reasoning_reward_cost,
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
    global RM_VLM_MODEL_PATH, RM_LLM_MODEL_PATH, MDP_REASONING_REWARD_WEIGHT, MDP_REASONING_REWARD_COST
    if 'rm_vlm_model_path' in kwargs:
        RM_VLM_MODEL_PATH = kwargs['rm_vlm_model_path']
    if 'rm_llm_model_path' in kwargs:
        RM_LLM_MODEL_PATH = kwargs['rm_llm_model_path']
    if 'mdp_reasoning_reward_weight' in kwargs:
        MDP_REASONING_REWARD_WEIGHT = float(kwargs['mdp_reasoning_reward_weight'])
    if 'mdp_reasoning_reward_cost' in kwargs:
        MDP_REASONING_REWARD_COST = float(kwargs['mdp_reasoning_reward_cost'])

    return await compute_score_batch_async(
        prompts,
        gen_imgs,
        feedback_texts,
        regen_imgs,
        ground_truth_imgs,
        summarizes,
        feedback_tuples,
        vqa_questions,
        extra_infos,
        task_ids,
        mdp_reasoning_reward_weight=MDP_REASONING_REWARD_WEIGHT,
        mdp_reasoning_reward_cost=MDP_REASONING_REWARD_COST,
    )


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


def _parse_json_score(response: str, key: str = "REWARD") -> float:
    """Extract float score from a JSON response or text format produced by stage judges.

    Supports two formats:
    1. JSON: {"REWARD": 1.0, ...}
    2. Text: "REWARD: 1.00" or "REWARD: -0.5"
    """
    # Try JSON format first
    parsed = safe_json_loads(response)
    if parsed and key in parsed:
        return float(parsed[key])

    # Try text format: "REWARD: <value>"
    pattern = rf'{key}\s*:\s*(-?\d+(?:\.\d+)?)'
    match = re.search(pattern, response, re.IGNORECASE)
    if match:
        return float(match.group(1))

    raise ValueError(f"'{key}' not found in response: {response}")


def finalize_task2_reward_extra_info(
    reward_extra_info: Dict,
    decision_vqa: float,
    decision_source: str = "missing",
    mdp_reasoning_reward_weight: Optional[float] = None,
    mdp_reasoning_reward_cost: Optional[float] = None,
) -> tuple[float, Dict]:
    """Finalize task2 reward fields from raw stage outputs plus decision target.

    This is shared by the live final-reward path and the Phase1 post-hoc
    finalization path.
    """
    finalized = dict(reward_extra_info or {})
    target_no_edit = bool(decision_vqa >= 1.0 - 1e-6)
    model_no_edit = bool(finalized.get("task2_no_feedback_needed", 0))
    noncanonical_no_edit = bool(finalized.get("task2_noncanonical_no_edit", 0))

    prompt_to_tuple = float(finalized.get("task2_prompt_to_tuple_reward", 0.0) or 0.0)
    tuple_to_vqa = float(finalized.get("task2_tuple_to_vqa_reward", 0.0) or 0.0)
    feedback_content = float(finalized.get("task2_vqa_to_feedback_content_reward", 0.0) or 0.0)

    if noncanonical_no_edit:
        feedback_total = 0.0
    elif target_no_edit and model_no_edit:
        feedback_total = 2.0
    elif target_no_edit and not model_no_edit:
        feedback_total = 0.0
    elif (not target_no_edit) and model_no_edit:
        feedback_total = 0.0
    else:
        feedback_total = feedback_content

    vlm_reward = (prompt_to_tuple + tuple_to_vqa + feedback_total) / 3.0
    finalized["task2_decision_vqa_reward"] = float(decision_vqa)
    finalized["task2_decision_vqa_source"] = decision_source
    finalized["task2_target_no_edit"] = int(target_no_edit)
    finalized["task2_target_no_edit_score"] = int(target_no_edit)
    finalized["task2_no_feedback_needed_score"] = int(model_no_edit)
    finalized["task2_vqa_to_feedback_reward"] = feedback_total
    finalized["task2_step2_reward"] = _shape_mdp_reasoning_reward(
        prompt_to_tuple,
        mdp_reasoning_reward_weight,
        mdp_reasoning_reward_cost,
    )
    finalized["task2_step3_reward"] = _shape_mdp_reasoning_reward(
        tuple_to_vqa,
        mdp_reasoning_reward_weight,
        mdp_reasoning_reward_cost,
    )
    finalized["task2_step4_reward"] = _shape_mdp_reasoning_reward(
        feedback_total,
        mdp_reasoning_reward_weight,
        mdp_reasoning_reward_cost,
    )
    finalized["task2_vlm_reward"] = vlm_reward
    finalized["task2_total_reward"] = float(finalized.get("task2_rule_based_format_reward", 0.0) or 0.0) + vlm_reward
    finalized["task2_process"] = vlm_reward / 2.0
    return vlm_reward, finalized


# Make this async to work with the async reward loop
async def compute_score_batch_with_postprocess(prompts, gen_imgs, feedback_texts, regen_imgs, ground_truth_imgs, summarizes, feedback_tuples, vqa_questions, extra_infos, task_ids, **kwargs):
    """Async batch processing - directly calls the async implementation"""
    # Update model paths from kwargs if provided
    global RM_VLM_MODEL_PATH, RM_LLM_MODEL_PATH, MDP_REASONING_REWARD_WEIGHT, MDP_REASONING_REWARD_COST
    if 'rm_vlm_model_path' in kwargs:
        RM_VLM_MODEL_PATH = kwargs['rm_vlm_model_path']
    if 'rm_llm_model_path' in kwargs:
        RM_LLM_MODEL_PATH = kwargs['rm_llm_model_path']
    if 'mdp_reasoning_reward_weight' in kwargs:
        MDP_REASONING_REWARD_WEIGHT = float(kwargs['mdp_reasoning_reward_weight'])
    if 'mdp_reasoning_reward_cost' in kwargs:
        MDP_REASONING_REWARD_COST = float(kwargs['mdp_reasoning_reward_cost'])

    # First, compute all scores
    results = await compute_score_batch_async(
        prompts,
        gen_imgs,
        feedback_texts,
        regen_imgs,
        ground_truth_imgs,
        summarizes,
        feedback_tuples,
        vqa_questions,
        extra_infos,
        task_ids,
        mdp_reasoning_reward_weight=MDP_REASONING_REWARD_WEIGHT,
        mdp_reasoning_reward_cost=MDP_REASONING_REWARD_COST,
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
