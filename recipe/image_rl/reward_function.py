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
BASE_URLS = [
    # "http://10.100.44.4:8006/v1", # main1
    "http://10.100.44.8:8006/v1", # sub1
    "http://10.100.44.8:8007/v1",
    "http://10.100.44.2:8006/v1", # sub2
    "http://10.100.44.2:8007/v1",
]
API_KEY = "EMPTY"
MAX_RETRIES = 3
BASE_DELAY = 2
MODEL_NAME = os.environ.get("RM_MODEL_PATH", "Qwen/Qwen3-VL-30B-A3B-Instruct")

# Health checking configuration
HEALTH_CHECK_INTERVAL = 30  # seconds
FAILURE_THRESHOLD = 3  # consecutive failures before marking as unhealthy
RECOVERY_CHECK_INTERVAL = 60  # seconds to wait before checking if unhealthy server recovered

# Detector configuration
DETECTOR_URLS = [
    # "http://10.100.44.4:8086", # main1
    # "http://10.100.44.4:8087",
    # "http://10.100.44.8:8086", # sub1
    # "http://10.100.44.8:8087",
    "http://10.100.44.2:8086", # sub2
    "http://10.100.44.2:8087",
]
DETECTOR_TIMEOUT = 300000.0

DETECTOR_MAX_RETRIES = 2

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

RM_PER_SERVER_INFLIGHT = 16
_rm_slot_state = {}

DET_PER_SERVER_INFLIGHT = 4  
_det_slot_state = {}       

async def _ensure_rm_slots() -> asyncio.Queue:
    loop = asyncio.get_running_loop()
    state = _rm_slot_state.get(loop)
    if state is None:
        state = {"lock": asyncio.Lock(), "q": None}
        _rm_slot_state[loop] = state

    if state["q"] is not None:
        return state["q"]

    async with state["lock"]:
        if state["q"] is None:
            q = asyncio.Queue()

            start = random.randrange(len(BASE_URLS))
            for i in range(RM_PER_SERVER_INFLIGHT * len(BASE_URLS)):
                sid = (start + i) % len(BASE_URLS)
                q.put_nowait(sid)

            state["q"] = q

    return state["q"]


async def borrow_rm_client():
    q = await _ensure_rm_slots()
    while True:
        sid = await q.get()
        s = client_manager.servers[sid]
        if s.status != ServerStatus.UNHEALTHY or s.should_retry_unhealthy():
            return s.client, sid
        q.put_nowait(sid)
        await asyncio.sleep(0.05)


async def release_rm_client(server_id: int):
    q = await _ensure_rm_slots()
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
    def __init__(self, base_urls: List[str]):
        self.servers = []
        self.lock = threading.Lock()
        self.current_index = 0  # Round robin counter
        
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
            print("WARNING: No healthy servers available! Using any available server...")
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
            print("WARNING: No healthy servers available! Using any available server...")
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
                status[f"server_{i}"] = {
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
                    print(f"Server Health Status: {json.dumps(status, indent=2, default=str)}")
                
                time.sleep(HEALTH_CHECK_INTERVAL)
            except Exception as e:
                print(f"Health monitor error: {e}")
                time.sleep(HEALTH_CHECK_INTERVAL)

# Initialize the client manager
client_manager = ClientManager(BASE_URLS)


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
    elif task_id == 2:
        system_prompt = TASK2_FEEDBACK_GENERATOR_SYSTEM_PROMPT_TEMPLATE
        user_prompt = TASK2_FEEDBACK_GENERATOR_USER_PROMPT_TEMPLATE.format(prompt=prompt, part2_tuples=predicted_tuple, part3_answers=predicted_answer, part4_feedback=predicted_feedback)
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": [
                {"type": "text", "text": user_prompt},
            ]}
        ]
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
    else:
        raise ValueError(f"Invalid task: {task_id} is must be one of task1, task2, or task3.")

    return messages


# Additional message constructors for task 2 subtasks
def get_messsages_task2_comparison_summarize(*args): # 5: part 1
    prompt, gen_img, feedback_text, regen_img, ground_truth_img, summarize, feedback_tuple, predicted_summarize, predicted_tuple, predicted_answer, predicted_feedback, vqa_question, extra_info, task_id = args

    system_prompt = TASK2_COMPARISON_SUMMARIZE_SYSTEM_PROMPT
    user_prompt = json.dumps({"prompt": prompt, "summarize": predicted_summarize})
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": [
            {"type": "text", "text": user_prompt},
        ]}
    ]

    return messages

def get_messages_task2_comparison_tuple(*args): # 1: part 2
    prompt, gen_img, feedback_text, regen_img, ground_truth_img, summarize, feedback_tuple, predicted_summarize, predicted_tuple, predicted_answer, predicted_feedback, vqa_question, extra_info, task_id = args

    system_prompt = TASK2_COMPARISON_TUPLE_SYSTEM_PROMPT
    user_prompt = json.dumps({"GT": feedback_tuple, "PRED": predicted_tuple})
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": [
            {"type": "text", "text": user_prompt},
        ]}
    ]

    return messages


def get_messages_task2_hallucination_check(*args): # 2: part 3
    prompt, gen_img, feedback_text, regen_img, ground_truth_img, summarize, feedback_tuple, predicted_summarize, predicted_tuple, predicted_answer, predicted_feedback, vqa_question, extra_info, task_id = args

    system_prompt = TASK2_HALLUCINATION_CHECK_SYSTEM_PROMPT
    user_prompt = json.dumps({"tuple": predicted_tuple, "answer": predicted_answer})
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": [
            {"type": "image_url", "image_url": {"url": convert_gen_img_to_base64(gen_img)}},
            {"type": "text", "text": user_prompt},
        ]}
    ]

    return messages


def get_messages_task2_edit_instruction(*args): # 3: part 4
    prompt, gen_img, feedback_text, regen_img, ground_truth_img, summarize, feedback_tuple, predicted_summarize, predicted_tuple, predicted_answer, predicted_feedback, vqa_question, extra_info, task_id = args

    system_prompt = TASK2_EDIT_INSTRUCTION_SYSTEM_PROMPT
    user_prompt = json.dumps({"prompt": prompt, "answer": predicted_answer, "edit_instruction": predicted_feedback})
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": [
            {"type": "text", "text": user_prompt},
        ]}
    ]

    return messages


# Additional message constructors for task 1 alignment
def get_messages_task3_regeneration_followed_by_editing(*args):
    prompt, gen_img, feedback_text, regen_img, ground_truth_img, summarize, feedback_tuple, predicted_summarize, predicted_tuple, predicted_answer, predicted_feedback, vqa_question, extra_info, task_id = args

    system_prompt = TASK3_REGENERATION_FOLLOWED_BY_EDITING_SYSTEM_PROMPT
    user_prompt = json.dumps({"prompt": prompt, "edit_instruction": predicted_feedback})
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": [
            {"type": "image_url", "image_url": {"url": convert_gen_img_to_base64(gen_img)}},
            {"type": "image_url", "image_url": {"url": convert_gen_img_to_base64(regen_img)}},
            {"type": "text", "text": user_prompt},
        ]}
    ]

    return messages


async def get_response_with_client(client, messages):
    """Get response from a specific client with improved error handling"""
    response = await client.chat.completions.create(
        model=MODEL_NAME,
        messages=messages,
        max_tokens=4096,
        extra_body={"repetition_penalty": 1.2},
        timeout=300000.0,
    )
    return response.choices[0].message.content


async def get_response(message_builder_fn, *args):
    """Generic response fetcher with automatic server fallback

    Args:
        message_builder_fn: Function that takes *args and returns messages list
        *args: Arguments to pass to message_builder_fn
    """
    messages = message_builder_fn(*args)

    # Try different servers until one succeeds
    max_attempts = len(BASE_URLS) * MAX_RETRIES

    for attempt in range(max_attempts):
        client, sid = await borrow_rm_client()
        try:
            response = await get_response_with_client(client, messages)
            if not is_meaningful_response(response):
                # Back off 
                client_manager.record_request_result(sid, success=False,
                                                     error=ValueError("Non-meaningful response"))
                print(response) # DEBUG
                continue
            else:
                client_manager.record_request_result(sid, success=True)
                return response

        except Exception as e:
            client_manager.record_request_result(sid, success=False, error=e)

        finally:
            await release_rm_client(sid)

    return None


# Detection parsing and request functions
def parse_lines(text):
    if not isinstance(text, str): return []
    return [(int(m.group(1)), m.group(2).strip()) for line in text.strip().split("\n") if (m := re.match(r"(\d+)\s*\|\s*(.*)", line))]


def verify_detection_single(feedback_tuple: list) -> List[Dict[str, Any]]: # 
    parsed_tup = parse_lines(feedback_tuple)
    
    results = []
    for num, content in parsed_tup: # [(1, 'entity - whole (banana)'), ...]
        info = None
        if 'spatial' in content:
            if m := re.search(r'\((.*?)\)', content): # (obj1, obj2, relation) 파싱
                parts = [p.strip() for p in m.group(1).split(',')]
                if len(parts) >= 3: # Spatial
                    s, o = parts[0], parts[1]
                    r_text = ", ".join(parts[2:]) # 관계 텍스트 (예: "is on the left of")
                    
                    cs = re.sub(r"_\d+$", "", _CONNECTORS.sub('', s).strip())
                    co = re.sub(r"_\d+$", "", _CONNECTORS.sub('', o).strip())
                    
                    # Subject/Object가 방향 지시어면 스킵
                    if not (cs.lower() in SKIP_KEYWORDS or co.lower() in SKIP_KEYWORDS or cs.startswith('[')):
                        
                        # [핵심] 정규식 리스트를 순회하며 매칭 확인
                        # 매칭되면 'c'(Canonical Name, 예: "left of")를 반환
                        canonical_rel = next((c for c, p, _ in _COMPILED_RELATIONS if p.search(r_text)), None)
                        
                        if canonical_rel:
                            # Prompt는 GDino가 이해하기 쉬운 단순 형태로 구성 (필요시 canonical_rel 사용 가능)
                            info = {
                                "subject": cs,
                                "object": co,
                                "relation": canonical_rel,  # <--- 여기에 통일된 단어("left of")가 들어감
                                "tuple_idx": num,
                                "type": "spatial"
                            }

        elif 'count' in content:
            if m := re.search(r'\((.*?)\)', content):
                parts = [p.strip() for p in m.group(1).split(',')]
                if len(parts) >= 2: # Counting
                    s, expr = parts[0], parts[1]
                    cs = re.sub(r"_\d+$", "", _CONNECTORS.sub('', s).strip())
                    if not (cs.lower() in SKIP_KEYWORDS or cs.startswith('[')) and re.search(r'\d', expr):
                        info = {
                            "subject": cs, 
                            "object": cs, 
                            "num": expr, 
                            "tuple_idx": num,
                            "type": "counting"
                        }
        
        if info is not None:    
            results.append(info)

    return results # [], [info1, info2, ...]

    
async def request_detector_single(detection_list: List[Dict[str, Any]], img) -> Dict[str, Any]:
    """
    - detector 서버 4개를 고루 사용 (slot token 기반)
    - 서버당 inflight cap (DET_PER_SERVER_INFLIGHT)
    - failover/retry (DETECTOR_MAX_RETRIES) 동안 슬롯 점유하지 않음
    """

    if not detection_list:
        return {"results": {}, "details": [], "errors": []}

    # -------- loop-local slot queue init --------
    async def _ensure_det_slots() -> asyncio.Queue:
        loop = asyncio.get_running_loop()
        state = _det_slot_state.get(loop)
        if state is None:
            state = {"lock": asyncio.Lock(), "q": None}
            _det_slot_state[loop] = state

        if state["q"] is not None:
            return state["q"]

        async with state["lock"]:
            if state["q"] is None:
                q = asyncio.Queue()

                start = random.randrange(len(DETECTOR_URLS))
                for i in range(DET_PER_SERVER_INFLIGHT * len(DETECTOR_URLS)):
                    sid = (start + i) % len(DETECTOR_URLS)
                    q.put_nowait(sid)

                state["q"] = q
        return state["q"]

    slot_q = await _ensure_det_slots()

    # -------- image -> raw base64 --------
    img_b64 = convert_gen_img_to_base64(img)
    if img_b64.startswith("data:"):
        img_b64 = img_b64.split(",", 1)[1]

    # -------- build payload --------
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

    # -------- retry bookkeeping --------
    per_server_attempts = {sid: 0 for sid in range(len(DETECTOR_URLS))}
    max_total_attempts = len(DETECTOR_URLS) * DETECTOR_MAX_RETRIES

    results: Dict[int, bool] = {}
    details: List[Dict[str, Any]] = []
    errors: List[str] = []

    timeout = aiohttp.ClientTimeout(total=DETECTOR_TIMEOUT)

    async with aiohttp.ClientSession(timeout=timeout) as session:
        for attempt in range(max_total_attempts):
            sid = await slot_q.get()

            # 서버당 시도 횟수 제한
            if per_server_attempts[sid] >= DETECTOR_MAX_RETRIES:
                slot_q.put_nowait(sid)
                continue
            per_server_attempts[sid] += 1

            detect_url = f"{DETECTOR_URLS[sid]}/detect"

            err = None
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
                        details.append(
                            {
                                "tuple_idx": tuple_idx,
                                "det_judge": det_judge,
                                "det_reason": r0.get("det_reason", ""),
                                "det_info": r0.get("det_info", {}),
                                "vis_data": r0.get("vis_data"),
                                "server": DETECTOR_URLS[sid],
                            }
                        )

                    return {"results": results, "details": details, "errors": errors}

            except Exception as e:
                err = e
                errors.append(f"{detect_url} exception: {repr(e)}")

            finally:
                # slot return
                slot_q.put_nowait(sid)

            # backoff before next attempt
            await asyncio.sleep(0.1 * (attempt + 1))

    # All servers failed
    return {
        "results": results,
        "details": details,
        "errors": errors if errors else ["All detector servers failed"]
    }


async def request_detector_batch(detection_requests: List[tuple]) -> List[Dict[str, Any]]:
    """
    Batch process multiple detection requests concurrently.
    
    Args:
        detection_requests: List of (detection_list, img) tuples
    
    Returns:
        List of detection results corresponding to each request
    """
    if not detection_requests:
        return []
    
    # Create semaphore to limit concurrent detector requests
    semaphore = asyncio.Semaphore(4)  # Limit concurrent detector calls
    
    async def process_single(idx, det_list, img):
        async with semaphore:
            result = await request_detector_single(det_list, img)
            return idx, result
    
    tasks = [
        asyncio.create_task(process_single(i, det_list, img))
        for i, (det_list, img) in enumerate(detection_requests)
    ]
    
    results = [None] * len(detection_requests)
    completed = await asyncio.gather(*tasks, return_exceptions=True)
    
    for item in completed:
        if isinstance(item, Exception):
            print(f"Detector batch request failed: {item}")
        else:
            idx, result = item
            results[idx] = result
    
    return results


async def compute_score_single_async(prompt, gen_img, feedback_text, regen_img, ground_truth_img, summarize, feedback_tuple, predicted_summarize, predicted_tuple, predicted_answer, predicted_feedback, vqa_question, extra_info, task_id):
    """Async version of compute_score"""
    reward_score = 0.0
    reward_extra_info = {}

    if task_id == 1: # Total score: 3.0
        # Launch all API requests in parallel
        detection_results = verify_detection_single(feedback_tuple)

        # Create tasks
        vlm_task = asyncio.create_task(
            get_response(get_messages, prompt, gen_img, feedback_text, regen_img, ground_truth_img, summarize, feedback_tuple, predicted_summarize, predicted_tuple, predicted_answer, predicted_feedback, vqa_question, extra_info, task_id)
        )

        detector_task = None
        if detection_results:
            detector_task = asyncio.create_task(
                request_detector_single(detection_results, gen_img)
            )

        # Gather results
        response = await vlm_task
        detector_response = await detector_task if detector_task else {"results": {}, "details": [], "errors": ["No valid detection items"]}

        # Process VLM response
        task1_vlm_reward_score = 0.0
        if not isinstance(response, Exception) and response is not None:
            try:
                task1_idx_to_ans: dict = image_evaluator_parser(response)
                task1_vlm_reward_score_sum = sum(task1_idx_to_ans.values())
                task1_ans_count = len(task1_idx_to_ans)
                task1_vlm_reward_score = 1.0 if task1_vlm_reward_score_sum == task1_ans_count else 0.0
                reward_score += task1_vlm_reward_score
                reward_extra_info[f"task{task_id}_vlm_reward"] = task1_vlm_reward_score
            except Exception as e:
                task1_vlm_reward_score = 0.0
                reward_score += 0.0
                reward_extra_info[f"task{task_id}_vlm_reward"] = 0.0
        else:
            reward_score += 0.0
            reward_extra_info[f"task{task_id}_vlm_reward"] = 0.0

        reward_extra_info[f"task{task_id}_reward_response"] = response if not isinstance(response, Exception) else str(response)

        # Process detector response
        detector_reward = 0.0
        if detection_results and detector_response:
            det_results_dict = detector_response.get("results", {})
            if det_results_dict:
                all_true = all(det_results_dict.values())
                detector_reward = 1.0 if all_true else 0.0
                reward_score += detector_reward
                reward_extra_info[f"task{task_id}_detector_reward"] = detector_reward
            else:
                reward_score += 0.0
                reward_extra_info[f"task{task_id}_detector_reward"] = 0.0
        else:
            reward_score += 0.0
            reward_extra_info[f"task{task_id}_detector_reward"] = 0.0

        reward_extra_info[f"task{task_id}_detector_response"] = detector_response

        # Bonus if both perfect
        if task1_vlm_reward_score == 1 and detector_reward == 1:
            reward_score += 1.0
            reward_extra_info[f"task{task_id}_vlm_detector_bonus"] = 1.0
        else:
            reward_score += 0.0
            reward_extra_info[f"task{task_id}_vlm_detector_bonus"] = 0.0

    elif task_id == 2:
        task2_reward_score = 0.0

        # Call FormattingEvaluator
        formatting_evaluator = FormattingEvaluatorV2()

        # Rule-based: formatting reward
        task2_reward_score += 1.0 if all(part is not None for part in [predicted_summarize, predicted_tuple, predicted_answer, predicted_feedback]) else 0.0 # +1

        # Rule-based: part 1 scoring
        feedback_parsed_tuple = formatting_evaluator._parse_part1(feedback_tuple) # GT Tuple
        predict_parsed_tuple = formatting_evaluator._parse_part1(predicted_tuple) # Pred  Tuple
        predict_decomposed_ans = formatting_evaluator._extract_answer_paragraphs(predicted_answer)

        part1_reward_dict = formatting_evaluator._calculate_metrics_for_reward(feedback_parsed_tuple, predict_parsed_tuple, predict_decomposed_ans)
        task2_part1_reward = sum(part1_reward_dict.values())
        task2_reward_score += task2_part1_reward
        reward_extra_info[f"task{task_id}_part1_reward"] = task2_part1_reward # +2

        # Launch all API requests in parallel
        args = (prompt, gen_img, feedback_text, regen_img, ground_truth_img, summarize, feedback_tuple, predicted_summarize, predicted_tuple, predicted_answer, predicted_feedback, vqa_question, extra_info, task_id)

        feedback_task = asyncio.create_task(get_response(get_messages, *args)) # +1
        comparison_summarize_task = asyncio.create_task(get_response(get_messsages_task2_comparison_summarize, *args)) # +1
        comparison_tuple_task = asyncio.create_task(get_response(get_messages_task2_comparison_tuple, *args)) # +1
        hallucination_check_task = asyncio.create_task(get_response(get_messages_task2_hallucination_check, *args)) # +1
        edit_instruction_task = asyncio.create_task(get_response(get_messages_task2_edit_instruction, *args)) # +1

        # Gather all results at once
        (
            response,
            comparison_summarize_response,
            comparison_tuple_response,
            hallucination_check_response,
            edit_instruction_response
        ) = await asyncio.gather(
            feedback_task,
            comparison_summarize_task,
            comparison_tuple_task,
            hallucination_check_task,
            edit_instruction_task,
            return_exceptions=True
        )

        # Process feedback response
        task2_vlm_reward_score = 0.0
        if not isinstance(response, Exception) and response is not None:
            try:
                raw_json = safe_json_loads(response)
                label_response = raw_json.get("label", "").lower()
                if label_response in ["targeted_only", "no_feedback_needed"]:
                    task2_vlm_reward_score = 1.0
                elif label_response in ["non_target_touched", "global_or_irrelevant"]:
                    task2_vlm_reward_score = 0.0
                else:
                    task2_vlm_reward_score = 0.0

                task2_reward_score += task2_vlm_reward_score
                # task2_ans_count += 1
            except:
                task2_vlm_reward_score = 0.0
                task2_reward_score += 0.0
        else:
            task2_vlm_reward_score = 0.0
            task2_reward_score += 0.0

        # reward_score += (task2_reward_score / task2_ans_count) if task2_ans_count > 0 else 0.0
        reward_extra_info[f"task{task_id}_vlm_reward"] = task2_vlm_reward_score
        reward_extra_info[f"task{task_id}_reward_response"] = response

        # Process comparison_summarize response
        task2_comparison_summarize_score = 0.0
        if not isinstance(comparison_summarize_response, Exception) and comparison_summarize_response is not None:
            try:
                reward_data = safe_json_loads(comparison_summarize_response)
                task2_comparison_summarize_score = float(reward_data.get("score", 0.0))
                task2_reward_score += task2_comparison_summarize_score
                # task2_ans_count += 1
            except:
                task2_comparison_summarize_score = 0.0
                task2_reward_score += 0.0
        else:
            task2_comparison_summarize_score = 0.0
            task2_reward_score += 0.0

        reward_extra_info[f"task{task_id}_comparison_summarize_score"] = task2_comparison_summarize_score
        reward_extra_info[f"task{task_id}_comparison_summarize_response"] = comparison_summarize_response if not isinstance(comparison_summarize_response, Exception) else str(comparison_summarize_response)

        # Process comparison_tuple response
        task2_comparison_tuple_score = 0.0
        if not isinstance(comparison_tuple_response, Exception) and comparison_tuple_response is not None:
            try:
                reward_data = safe_json_loads(comparison_tuple_response)
                task2_comparison_tuple_score = float(reward_data.get("accuracy", 0.0))
                task2_reward_score += task2_comparison_tuple_score
                # task2_ans_count += 1
            except:
                task2_comparison_tuple_score = 0.0
                task2_reward_score += 0.0
        else:
            task2_comparison_tuple_score = 0.0
            task2_reward_score += 0.0

        reward_extra_info[f"task{task_id}_comparison_tuple_score"] = task2_comparison_tuple_score
        reward_extra_info[f"task{task_id}_comparison_tuple_response"] = comparison_tuple_response if not isinstance(comparison_tuple_response, Exception) else str(comparison_tuple_response)

        # Process hallucination_check response
        task2_hallucination_check_score = 0.0
        if not isinstance(hallucination_check_response, Exception) and hallucination_check_response is not None:
            try:
                reward_data = safe_json_loads(hallucination_check_response)
                task2_hallucination_check_score = float(reward_data.get("accuracy", 0.0))
                task2_reward_score += task2_hallucination_check_score
                # task2_ans_count += 1
            except:
                task2_hallucination_check_score = 0.0
                task2_reward_score += 0.0
        else:
            task2_hallucination_check_score = 0.0
            task2_reward_score += 0.0
        
        reward_extra_info[f"task{task_id}_hallucination_check_score"] = task2_hallucination_check_score
        reward_extra_info[f"task{task_id}_hallucination_check_response"] = hallucination_check_response if not isinstance(hallucination_check_response, Exception) else str(hallucination_check_response)

        # Process edit_instruction response
        task2_edit_instruction_score = 0.0
        if not isinstance(edit_instruction_response, Exception) and edit_instruction_response is not None:
            try:
                reward_data = safe_json_loads(edit_instruction_response)
                task2_edit_instruction_score = float(reward_data.get("score", 0.0))
                task2_reward_score += task2_edit_instruction_score
                # task2_ans_count += 1
            except:
                task2_edit_instruction_score = 0.0
                task2_reward_score += 0.0
        else:
            task2_edit_instruction_score = 0.0
            task2_reward_score += 0.0
        
        reward_extra_info[f"task{task_id}_edit_instruction_score"] = task2_edit_instruction_score
        reward_extra_info[f"task{task_id}_edit_instruction_response"] = edit_instruction_response if not isinstance(edit_instruction_response, Exception) else str(edit_instruction_response)

        # reward_score += (task2_reward_score / task2_ans_count) if task2_ans_count > 0 else 0.0
        reward_score += task2_reward_score # not normalizing

    elif task_id == 3: # Total score: 4.0
        last = predicted_feedback
        if last is not None and "No need to generate feedback.".lower() in last.lower():
            reward_score = -100
            reward_extra_info[f"task{task_id}_vlm_reward"] = 0.0
            reward_extra_info[f"task{task_id}_reward_response"] = "None"
            reward_extra_info[f"task{task_id}_detector_reward"] = 0.0
            reward_extra_info[f"task{task_id}_detector_response"] = {"results": {}, "details": [], "errors": []}
            reward_extra_info[f"task{task_id}_detector_details"] = []
            reward_extra_info[f"task{task_id}_vlm_detector_bonus"] = 0.0
            reward_extra_info[f"task{task_id}_regeneration_followed_by_editing_reward"] = 0.0
            reward_extra_info[f"task{task_id}_regeneration_followed_by_editing_response"] = "No need to respond reward."
            return {
                "score": reward_score,
                "reward_extra_info": reward_extra_info,
            }

        # Launch all API requests in parallel
        detection_results = verify_detection_single(feedback_tuple)
        args = (prompt, gen_img, feedback_text, regen_img, ground_truth_img, summarize, feedback_tuple, predicted_summarize, predicted_tuple, predicted_answer, predicted_feedback, vqa_question, extra_info, task_id)

        vlm_task = asyncio.create_task(get_response(get_messages, *args))
        regeneration_task = asyncio.create_task(get_response(get_messages_task3_regeneration_followed_by_editing, *args))

        detector_task = None
        if detection_results:
            detector_task = asyncio.create_task(request_detector_single(detection_results, gen_img))

        # Gather results
        if detector_task:
            response, regeneration_followed_by_editing_response, detector_response = await asyncio.gather(
                vlm_task, regeneration_task, detector_task, return_exceptions=True
            )
        else:
            response, regeneration_followed_by_editing_response = await asyncio.gather(
                vlm_task, regeneration_task, return_exceptions=True
            )
            detector_response = {"results": {}, "details": [], "errors": ["No valid detection items"]}

        # Process VLM response
        task3_vlm_reward_score = 0.0
        if not isinstance(response, Exception) and response is not None:
            task3_idx_to_ans: dict = image_evaluator_parser(response)
            task3_vlm_reward_score_sum = sum(task3_idx_to_ans.values())
            task3_ans_count = len(task3_idx_to_ans)
            task3_vlm_reward_score = 1.0 if task3_vlm_reward_score_sum == task3_ans_count else 0.0
            reward_score += task3_vlm_reward_score
            reward_extra_info[f"task{task_id}_vlm_reward"] = task3_vlm_reward_score
        else:
            reward_score += 0.0
            reward_extra_info[f"task{task_id}_vlm_reward"] = 0.0
        
        reward_extra_info[f"task{task_id}_reward_response"] = response if isinstance(response, Exception) else str(response)

        # Process regeneration response
        task3_regeneration_followed_by_editing_reward_score = 0.0
        if not isinstance(regeneration_followed_by_editing_response, Exception) and regeneration_followed_by_editing_response is not None:
            try:
                reward_data = safe_json_loads(regeneration_followed_by_editing_response)
                task3_regeneration_followed_by_editing_reward_score = float(reward_data.get("score", 0.0))
                reward_score += task3_regeneration_followed_by_editing_reward_score
            except:
                task3_regeneration_followed_by_editing_reward_score = 0.0
                reward_score += 0.0
        else:
            task3_regeneration_followed_by_editing_reward_score = 0.0
            reward_score += 0.0

        reward_extra_info[f"task{task_id}_regeneration_followed_by_editing_reward"] = task3_regeneration_followed_by_editing_reward_score
        reward_extra_info[f"task{task_id}_regeneration_followed_by_editing_response"] = regeneration_followed_by_editing_response if not isinstance(regeneration_followed_by_editing_response, Exception) else str(regeneration_followed_by_editing_response)

        # Process detector response
        detector_reward = 0.0
        if detection_results and not isinstance(detector_response, Exception):
            det_results_dict = detector_response.get("results", {})
            if det_results_dict:
                all_true = all(det_results_dict.values())
                detector_reward = 1.0 if all_true else 0.0
                reward_score += detector_reward
                reward_extra_info[f"task{task_id}_detector_reward"] = detector_reward
            else:
                reward_score += 0.0
                reward_extra_info[f"task{task_id}_detector_reward"] = 0.0
        else:
            reward_score += 0.0
            reward_extra_info[f"task{task_id}_detector_reward"] = 0.0

        reward_extra_info[f"task{task_id}_detector_response"] = detector_response if not isinstance(detector_response, Exception) else str(detector_response)

        # Bonus if both perfect
        if task3_vlm_reward_score == 1 and detector_reward == 1:
            reward_score += 1.0
            reward_extra_info[f"task{task_id}_vlm_detector_bonus"] = 1.0
        else:
            reward_score += 0.0
            reward_extra_info[f"task{task_id}_vlm_detector_bonus"] = 0.0

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
    return client_manager.get_server_status()


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