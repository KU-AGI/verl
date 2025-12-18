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
from recipe.image_rl.utils import FormattingEvaluator
import asyncio
import threading
import random
from enum import Enum
import torch
import aiohttp
from recipe.image_rl.gdino_regex import _CONNECTORS, SKIP_KEYWORDS, _COMPILED_RELATIONS

# Configuration
BASE_URLS = [
    # "http://192.169.0.3:8000/v1",
    # "http://192.169.0.3:8002/v1",
    "http://192.169.0.3:8004/v1",
]
API_KEY = "EMPTY"
MAX_RETRIES = 3
BASE_DELAY = 2
MAX_CONCURRENT_REQUESTS = 4
MODEL_NAME = os.environ.get("RM_MODEL_PATH", "Qwen/Qwen3-VL-30B-A3B-Instruct")

# Health checking configuration
HEALTH_CHECK_INTERVAL = 30  # seconds
FAILURE_THRESHOLD = 3  # consecutive failures before marking as unhealthy
RECOVERY_CHECK_INTERVAL = 60  # seconds to wait before checking if unhealthy server recovered

# Detector configuration
DETECTOR_URLS = [
    "http://192.169.0.3:8084",
    "http://192.169.0.3:8085"
]
DETECTOR_TIMEOUT = 300000.0

DETECTOR_MAX_RETRIES = 2

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

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

# client manager with failover support
class ClientManager:
    def __init__(self, base_urls: List[str]):
        self.servers = []
        self.lock = threading.Lock()
        self.current_index = 0
        
        # Initialize servers
        for url in base_urls:
            client = AsyncOpenAI(api_key=API_KEY, base_url=url)
            server_info = ServerInfo(url, client)
            self.servers.append(server_info)
            
        # Start health monitoring thread
        self.health_monitor_thread = threading.Thread(target=self._health_monitor, daemon=True)
        self.health_monitor_thread.start()
        
    def get_healthy_servers(self) -> List[ServerInfo]:
        """Get list of servers that are healthy or degraded (not unhealthy)"""
        with self.lock:
            healthy_servers = []
            for server in self.servers:
                if server.status in [ServerStatus.HEALTHY, ServerStatus.DEGRADED]:
                    healthy_servers.append(server)
                elif server.should_retry_unhealthy():
                    # Give unhealthy servers a chance to recover
                    healthy_servers.append(server)
            return healthy_servers
    
    def get_best_server(self) -> Optional[ServerInfo]:
        """Get the best available server using weighted selection"""
        healthy_servers = self.get_healthy_servers()
        
        if not healthy_servers:
            print("WARNING: No healthy servers available! Using any available server...")
            return self.servers[0] if self.servers else None
            
        # Sort by status (healthy first) then by success rate
        healthy_servers.sort(key=lambda s: (
            s.status.value,  # healthy < degraded < unhealthy
            -s.success_rate,  # higher success rate first
            s.consecutive_failures  # fewer failures first
        ))
        
        return healthy_servers[0]
    
    def get_next_client_with_fallback(self) -> Optional[tuple]:
        """Get next client with automatic fallback to healthy servers"""
        server = self.get_best_server()
        if server is None:
            return None, None
            
        server_id = self.servers.index(server)
        return server.client, server_id
        
    def record_request_result(self, server_id: int, success: bool, error: Exception = None):
        """Record the result of a request for server health tracking"""
        if 0 <= server_id < len(self.servers):
            server = self.servers[server_id]
            if success:
                server.record_success()
            else:
                server.record_failure()
                if error:
                    print(f"Server {server.url} error: {repr(error)}")
    
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

# 기존 프롬프트 템플릿들은 동일하게 유지
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

TASK2_FEEDBACK_GENERATOR_SYSTEM_PROMPT_TEMPLATE = """
You are a data consistency auditor.

### Goal
Given a PROMPT describing an image, a list of QUESTIONS (semantic tuples) with corresponding ANSWERS (Yes/No),
and a FEEDBACK text describing how the image was edited,
your job is to determine whether the feedback only changes objects or attributes that were answered "No"
and leaves all "Yes" ones unchanged.

If **all ANSWERS are "Yes"**, there is **no need to generate feedback** — the correct output in that case should explicitly state:
{
  "reason": "All tuples were labeled 'Yes', meaning no edits are required."
  "label": "no_feedback_needed",
}

### Labels
- "targeted_only" : feedback changes only the No-labeled tuples.
- "non_target_touched" : feedback also modifies Yes-labeled tuples.
- "global_or_irrelevant" : feedback changes unrelated global properties (background, lighting, tone).

### Guidelines
1. Identify which objects, attributes, or relations are mentioned or affected in the FEEDBACK (e.g., banana color, cup, background).
2. Match these with the QUESTIONS and their ANSWERS:
   - If all modified targets correspond to No-labeled tuples → label as "targeted_only".
   - If any modified target corresponds to a Yes-labeled tuple → label as "non_target_touched".
   - If feedback refers only to global or environmental aspects not tied to any tuple → label as "global_or_irrelevant".
3. Output a single JSON object in this format:
{
  "targeted_entities": ["..."],
  "violations": ["..."], // if any
  "reason": "<short explanation>",
  "label": "targeted_only | non_target_touched | global_or_irrelevant"
}

### Example

PROMPT:
"A green banana and a blue cup"

QUESTIONS:
1 | entity - whole (banana)
2 | attribute - color (banana, green)
3 | entity - whole (cup)
4 | attribute - color (cup, blue)

ANSWERS:
1 | The image clearly shows a banana next to a blue cup. Answer: Yes
2 | The banana appears yellow, not green. Answer: No
3 | The image shows a ceramic mug, which is a type of cup. Answer: Yes
4 | The cup looks turquoise, which is a blue-green color. Answer: Yes

FEEDBACK:
"Change the banana's color from yellow to green, and remove one banana, positioning the remaining one leaning against the cup. Adjust the background from green to a light blue shade."

EXPECTED OUTPUT:
{
  "targeted_entities": ["banana","background"],
  "violations": [
    "background color modified even though background was not part of any No-labeled tuple"
  ],
  "reason": "Feedback correctly changes the banana (a No-labeled tuple) but also modifies the background, which was not among the No targets.",
  "label": "non_target_touched"
}
"""

TASK2_FEEDBACK_GENERATOR_USER_PROMPT_TEMPLATE = """PROMPT:
{prompt}

QUESTIONS:
{part1_tuples}

ANSWERS:
{part2_answers}

FEEDBACK:
{part3_feedback}
""".strip()

TASK2_VLM_CHECK_SYSTEM_PROMPT = """You are an AI assistant that answers a batch of yes/no questions.

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

TASK2_VLM_CHECK_INPUT_PROMPT = """
You will receive multiple questions, one per line, in the format "<index> | <question>".
For each question, follow the protocol and produce EXACTLY two lines using the template:

<index> | <one or two concise sentences for justification>
<index> | Answer: Yes or Answer: No

Questions:
{questions}""".strip()


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


def get_messages(prompt, gen_img, feedback_text, regen_img, ground_truth_img, feedback_tuple, vqa_question, extra_info, task_id):
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
        formatting_evaluator = FormattingEvaluator() # task 2 only
        part1, part2, part3 = formatting_evaluator._split_text_into_parts(feedback_text.strip())

        system_prompt = TASK2_FEEDBACK_GENERATOR_SYSTEM_PROMPT_TEMPLATE
        user_prompt = TASK2_FEEDBACK_GENERATOR_USER_PROMPT_TEMPLATE.format(prompt=prompt, part1_tuples=part1, part2_answers=part2, part3_feedback=part3)
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


def get_messages_task2_align(prompt, gen_img, feedback_text, regen_img, ground_truth_img, feedback_tuple, vqa_question, extra_info, task_id):
    system_prompt = TASK2_VLM_CHECK_SYSTEM_PROMPT
    user_prompt = TASK2_VLM_CHECK_INPUT_PROMPT.format(questions=vqa_question)
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": [
            {"type": "image_url", "image_url": {"url": convert_gen_img_to_base64(gen_img)}},
            {"type": "text", "text": user_prompt}
        ]}
    ]

    return messages


async def get_response_with_client(client, client_id, messages):
    """Get response from a specific client with improved error handling"""
    for attempt in range(MAX_RETRIES):
        try:
            response = await client.chat.completions.create(
                model=MODEL_NAME,
                messages=messages,
                max_tokens=2048,
                extra_body={"repetition_penalty": 1.2},
                timeout=300000.0  # Add timeout
            )
            # Record successful request
            client_manager.record_request_result(client_id, success=True)
            return response.choices[0].message.content
            
        except Exception as e:
            # Record failed request
            client_manager.record_request_result(client_id, success=False, error=e)
            
            if attempt < MAX_RETRIES - 1:
                print(f"Client {client_id} attempt {attempt+1} failed: {repr(e)}")
                delay = BASE_DELAY * (2**attempt) + random.uniform(0, 1)
                await asyncio.sleep(delay)
            else:
                print(f"Client {client_id} failed after {MAX_RETRIES} attempts. Error: {e}")
                return None


async def get_response_with_fallback(prompt, gen_img, feedback_text, regen_img, ground_truth_img, feedback_tuple, vqa_question, extra_info, task_id):
    """Get response with automatic server fallback"""
    messages = get_messages(prompt, gen_img, feedback_text, regen_img, ground_truth_img, feedback_tuple, vqa_question, extra_info, task_id)
    
    # Try different servers until one succeeds
    max_server_attempts = len(client_manager.servers)
    
    for server_attempt in range(max_server_attempts):
        client_info = client_manager.get_next_client_with_fallback()
        if client_info[0] is None:
            print("No available servers!")
            break
            
        client, client_id = client_info
        
        response = await get_response_with_client(client, client_id, messages)
        if response is not None:
            return response
            
        print(f"Server attempt {server_attempt + 1}/{max_server_attempts} failed, trying next server...")
    
    print("All servers failed!")
    return None


# Update the main function to use the new fallback mechanism
async def get_response(prompt, gen_img, feedback_text, regen_img, ground_truth_img, feedback_tuple, vqa_question, extra_info, task_id):
    """Get response from API with improved load balancing and failover"""
    return await get_response_with_fallback(prompt, gen_img, feedback_text, regen_img, ground_truth_img, feedback_tuple, vqa_question, extra_info, task_id)


async def get_response_with_fallback_task2_align(prompt, gen_img, feedback_text, regen_img, ground_truth_img, feedback_tuple, vqa_question, extra_info, task_id):
    """Get response with automatic server fallback"""
    messages = get_messages_task2_align(prompt, gen_img, feedback_text, regen_img, ground_truth_img, feedback_tuple, vqa_question, extra_info, task_id)
    
    # Try different servers until one succeeds
    max_server_attempts = len(client_manager.servers)
    
    for server_attempt in range(max_server_attempts):
        client_info = client_manager.get_next_client_with_fallback()
        if client_info[0] is None:
            print("No available servers!")
            break
            
        client, client_id = client_info
        
        response = await get_response_with_client(client, client_id, messages)
        if response is not None:
            return response
            
        print(f"Server attempt {server_attempt + 1}/{max_server_attempts} failed, trying next server...")
    
    print("All servers failed!")
    return None


# Update the main function to use the new fallback mechanism
async def get_response_task2_align(prompt, gen_img, feedback_text, regen_img, ground_truth_img, feedback_tuple, vqa_question, extra_info, task_id):
    """Get response from API with improved load balancing and failover"""
    return await get_response_with_fallback_task2_align(prompt, gen_img, feedback_text, regen_img, ground_truth_img, feedback_tuple, vqa_question, extra_info, task_id)


def compute_reward(response):
    """Compute reward score from response"""
    reward_score = 0.0
    try:
        boxed_result = last_boxed_only_string(response)
        if boxed_result is not None:
            result = remove_boxed(boxed_result)
            reward_score = float(result.lower() in ["true", "1", "yes"])
    except Exception as e:
        print(e)
    return reward_score


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
    Send detection requests to GDino detector API.
    
    Args:
        detection_list: List of detection info dicts with keys like:
            - type: "spatial" or "counting" (will be converted to "numeracy")
            - subject, object, relation (for spatial)
            - object, num (for counting/numeracy)
            - tuple_idx: original tuple index
        img: PIL Image or path to image
    
    Returns:
        Dict with:
            - results: Dict[tuple_idx, bool] mapping tuple indices to detection results
            - details: List of full detection responses
            - errors: List of any errors encountered
    """
    if not detection_list:
        return {"results": {}, "details": [], "errors": []}
    
    # Convert image to base64
    img_base64 = convert_gen_img_to_base64(img)
    # Remove data URL prefix if present (API expects raw base64)
    if img_base64.startswith("data:"):
        img_base64 = img_base64.split(",", 1)[1]
    
    # Prepare info_list for API request
    info_list = []
    idx_mapping = {}  # Map API index to tuple_idx
    
    for i, det_info in enumerate(detection_list):
        api_info = {}
        det_type = det_info.get("type", "")
        
        if det_type == "spatial":
            api_info = {
                "type": "spatial",
                "subject": det_info.get("subject"),
                "object": det_info.get("object"),
                "relation": det_info.get("relation")
            }
        elif det_type in ["counting", "numeracy"]:
            api_info = {
                "type": "numeracy",
                "object": det_info.get("object"),
                "num": str(det_info.get("num", ""))
            }
        else:
            continue
            
        info_list.append(api_info)
        idx_mapping[len(info_list) - 1] = det_info.get("tuple_idx", i)
    
    if not info_list:
        return {"results": {}, "details": [], "errors": ["No valid detection items"]}
    
    # Prepare request payload
    payload = {
        "info_list": info_list,
        "img_url": img_base64
    }
    
    results = {}
    details = []
    errors = []
    
    # Try each detector URL with failover
    async with aiohttp.ClientSession() as session:
        for url_idx, base_url in enumerate(DETECTOR_URLS):
            try:
                detect_url = f"{base_url}/detect"
                
                for attempt in range(DETECTOR_MAX_RETRIES):
                    try:
                        async with session.post(
                            detect_url,
                            json=payload,
                            timeout=aiohttp.ClientTimeout(total=DETECTOR_TIMEOUT)
                        ) as response:
                            if response.status == 200:
                                data = await response.json()
                                api_results = data.get("results", [])
                                
                                # Parse results
                                for api_idx, result_list in enumerate(api_results):
                                    if api_idx in idx_mapping and result_list:
                                        tuple_idx = idx_mapping[api_idx]
                                        result = result_list[0]  # First result per info item
                                        
                                        results[tuple_idx] = result.get("det_judge", False)
                                        details.append({
                                            "tuple_idx": tuple_idx,
                                            "det_judge": result.get("det_judge", False),
                                            "det_reason": result.get("det_reason", ""),
                                            "det_info": result.get("det_info", {}),
                                            "vis_data": result.get("vis_data")
                                        })
                                
                                # Success - return results
                                return {
                                    "results": results,
                                    "details": details,
                                    "errors": errors
                                }
                            else:
                                error_text = await response.text()
                                errors.append(f"Server {url_idx} returned {response.status}: {error_text[:200]}")
                                
                    except asyncio.TimeoutError:
                        errors.append(f"Server {url_idx} attempt {attempt+1} timed out")
                    except aiohttp.ClientError as e:
                        errors.append(f"Server {url_idx} attempt {attempt+1} client error: {str(e)}")
                    
                    if attempt < DETECTOR_MAX_RETRIES - 1:
                        await asyncio.sleep(0.5 * (attempt + 1))
                        
            except Exception as e:
                errors.append(f"Server {url_idx} failed: {str(e)}")
                continue
    
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


async def compute_score_single_async(prompt, gen_img, feedback_text, regen_img, ground_truth_img, feedback_tuple, vqa_question, extra_info, task_id):
    """Async version of compute_score"""
    reward_score = 0.0
    reward_extra_info = {}

    if task_id == 1: # Total score: 3.0
        # VLM based reward
        task1_vlm_reward_score = 0.0
        response = await get_response(prompt, gen_img, feedback_text, regen_img, ground_truth_img, feedback_tuple, vqa_question, extra_info, task_id)
        if response is not None:
            task1_idx_to_ans: dict = image_evaluator_parser(response)

            task1_vlm_reward_score_sum = sum(task1_idx_to_ans.values())
            task1_ans_count = len(task1_idx_to_ans)

            task1_vlm_reward_score = 1.0 if task1_vlm_reward_score_sum == task1_ans_count else 0.0

            reward_score += task1_vlm_reward_score
            reward_extra_info[f"task{task_id}_reward_response"] = response
        else:
            reward_score += 0.0
            reward_extra_info[f"task{task_id}_reward_response"] = "No response received"

        # Detector based reward - always populate for batch consistency
        detection_results = verify_detection_single(feedback_tuple)
        detector_response = {"results": {}, "details": [], "errors": ["No spatial/counting tuples found"]}
        det_details_list = []

        detector_reward = 0.0
        if detection_results:
            detector_response = await request_detector_single(detection_results, gen_img)

            # Calculate detector bonus: +1 if all detections are True
            det_results_dict = detector_response.get("results", {})
            det_details_list = detector_response.get("details", [])
            if det_results_dict:
                all_true = all(det_results_dict.values())
                detector_reward = 1.0 if all_true else 0.0
                reward_score += detector_reward
                reward_extra_info[f"task{task_id}_detector_reward"] = detector_reward
                reward_extra_info[f"task{task_id}_detector_details"] = det_details_list
            else:
                reward_score += 0.0
                reward_extra_info[f"task{task_id}_detector_reward"] = 0.0
                reward_extra_info[f"task{task_id}_detector_details"] = det_details_list
        else:
            reward_score += 0.0
            reward_extra_info[f"task{task_id}_detector_reward"] = 0.0
            reward_extra_info[f"task{task_id}_detector_details"] = det_details_list

        # Always set detector_response for batch consistency
        reward_extra_info[f"task{task_id}_detector_response"] = detector_response

        # Bonus if both perfect
        if task1_vlm_reward_score == 1 and detector_reward == 1:
            reward_score += 1.0  # bonus for both perfect
            reward_extra_info[f"task{task_id}_vlm_detector_bonus"] = 1.0
        else:
            reward_score += 0.0
            reward_extra_info[f"task{task_id}_vlm_detector_bonus"] = 0.0

    elif task_id == 2: # Total score: 1.0 (normalized)
        task2_reward_score = 0.0
        task2_ans_count = 0
        
        # Rule-based reward
        formatting_evaluator = FormattingEvaluator()
        part1, part2, part3 = formatting_evaluator._split_text_into_parts(feedback_text.strip())

        # formatting
        task2_reward_score += 1.0 if all(part is not None for part in [part1, part2, part3]) else 0.0
        task2_ans_count += 1

        # part 1 scoring
        feedback_parsed_tuple = formatting_evaluator._parse_part1(feedback_tuple)
        predict_parsed_tuple = formatting_evaluator._parse_part1(part1)
        predict_decomposed_ans = formatting_evaluator._extract_answer_paragraphs(part2)

        part1_reward_dict = formatting_evaluator._calculate_metrics_for_reward(feedback_parsed_tuple, predict_parsed_tuple, predict_decomposed_ans)
        task2_reward_score += sum(part1_reward_dict.values())
        task2_ans_count += len(part1_reward_dict)

        # VLM based reward
        # Part 0: Summarize 단계 metric 추가 (251216)
        # task2_reward_score += 1, task2_ans_count += 1
        # Part 1: 추가해야됨 (251216)
        # Part 2: 추가해야됨 (251216)
        # Part 3: feedback reward (얘는 밑에 완성됨)
        response = await get_response(prompt, gen_img, feedback_text, regen_img, ground_truth_img, feedback_tuple, vqa_question, extra_info, task_id)
        if response is not None:
            try:
                raw_json = json.loads(response)
                label_response = raw_json.get("label", "").lower()
                if label_response in ["targeted_only", "no_feedback_needed"]:
                    task2_reward_score += 1.0
                elif label_response in ["non_target_touched"]:
                    task2_reward_score += 0.0
                elif label_response in ["global_or_irrelevant"]:
                    task2_reward_score += 0.0
                else:
                    task2_reward_score += 0.0
                task2_ans_count += 1

            except:
                raw_json = {}
                task2_reward_score += 0.0
        else:
            task2_reward_score += 0.0
        
        reward_extra_info[f"task{task_id}_reward_response"] = response

        # task2 align
        align_response = await get_response_task2_align(prompt, gen_img, feedback_text, regen_img, ground_truth_img, feedback_tuple, vqa_question, extra_info, task_id)
        if align_response is not None:
            try:
                q_idx_re  = re.compile(r'^\s*(\d+)\s*\|', re.MULTILINE)
                ans_line_re = re.compile(r'^\s*(\d+)\s*\|\s*Answer:\s*(Yes|No)\s*$', re.IGNORECASE | re.MULTILINE)

                # 1) questions에서 기대 인덱스 추출 (원래 순서 유지)
                expected_indices = [int(m.group(1)) for m in q_idx_re.finditer(vqa_question)]

                # 2) 출력에서 정답 라인만 파싱 -> {idx: bool}
                idx_to_ans = {}
                for idx_str, yn in ans_line_re.findall(align_response):
                    idx = int(idx_str)
                    idx_to_ans[idx] = (yn.strip().lower() == "yes")  # 중복 있으면 마지막 값 사용

                # 3) 기대 인덱스 순서대로 answers 구성 (모두 있어야 유효)
                complete = len(idx_to_ans) == len(expected_indices) and all(idx in idx_to_ans for idx in expected_indices)
                rm_decomposed_answers = [idx_to_ans[idx] for idx in expected_indices] if complete else []

                # predicted answers와 reference answers 비교
                predict_decomposed_ans = [p.lower() == "yes" for p in predict_decomposed_ans]
                task2_reward_score += 1.0 if all(rm_decomposed_answers) == all(predict_decomposed_ans) else 0.0
                task2_ans_count += 1

            except:
                raw_json = {}
                task2_reward_score += 0.0
        else:
            task2_reward_score += 0.0

        reward_score += (task2_reward_score / task2_ans_count) if task2_ans_count > 0 else 0.0
        reward_extra_info[f"task{task_id}_reward_align_response"] = align_response

    elif task_id == 3: # Total score: 3.0
        formatting_evaluator = FormattingEvaluator()
        last = formatting_evaluator._split_text_into_parts(feedback_text)[-1]
        if last is not None and "No need to generate feedback.".lower() in last.lower():
            reward_score = -100
            reward_extra_info[f"task{task_id}_reward_response"] = "None"
            # Always populate detector_response for batch consistency
            reward_extra_info[f"task{task_id}_detector_response"] = {"results": {}, "details": [], "errors": ["Skipped: No feedback needed"]}
            reward_extra_info[f"task{task_id}_detector_reward"] = 0.0
            reward_extra_info[f"task{task_id}_detector_details"] = []

            reward_extra_info[f"task{task_id}_vlm_detector_bonus"] = 0.0
            return {
                "score": reward_score,
                "reward_extra_info": reward_extra_info,
            }

        # VLM based reward
        task3_vlm_reward_score = 0.0
        response = await get_response(prompt, gen_img, feedback_text, regen_img, ground_truth_img, feedback_tuple, vqa_question, extra_info, task_id)
        if response is not None:
            task3_idx_to_ans: dict = image_evaluator_parser(response)

            task3_vlm_reward_score_sum = sum(task3_idx_to_ans.values())
            task3_ans_count = len(task3_idx_to_ans)

            task3_vlm_reward_score = 1.0 if task3_vlm_reward_score_sum == task3_ans_count else 0.0
            reward_score += task3_vlm_reward_score
            reward_extra_info[f"task{task_id}_reward_response"] = response
        else:
            reward_score += 0.0
            reward_extra_info[f"task{task_id}_reward_response"] = "No response received"

        # Detector based reward - always populate for batch consistency
        detection_results = verify_detection_single(feedback_tuple)
        detector_response = {"results": {}, "details": [], "errors": ["No spatial/counting tuples found"]}
        det_details_list = []

        detector_reward = 0.0
        if detection_results:
            detector_response = await request_detector_single(detection_results, gen_img)

            # Calculate detector bonus: +1 if all detections are True
            det_results_dict = detector_response.get("results", {})
            det_details_list = detector_response.get("details", [])
            if det_results_dict:
                all_true = all(det_results_dict.values())
                detector_reward = 1.0 if all_true else 0.0
                reward_score += detector_reward
                reward_extra_info[f"task{task_id}_detector_reward"] = detector_reward
                reward_extra_info[f"task{task_id}_detector_details"] = det_details_list
            else:
                reward_score += 0.0
                reward_extra_info[f"task{task_id}_detector_reward"] = 0.0
                reward_extra_info[f"task{task_id}_detector_details"] = det_details_list
        else:
            reward_score += 0.0
            reward_extra_info[f"task{task_id}_detector_reward"] = 0.0
            reward_extra_info[f"task{task_id}_detector_details"] = det_details_list

        # Always set detector_response for batch consistency
        reward_extra_info[f"task{task_id}_detector_response"] = detector_response

        # Bonus if both perfect
        if task3_vlm_reward_score == 1 and detector_reward == 1:
            reward_score += 1.0  # bonus for both perfect
            reward_extra_info[f"task{task_id}_vlm_detector_bonus"] = 1.0
        else:
            reward_extra_info[f"task{task_id}_vlm_detector_bonus"] = 0.0

    return {
        "score": reward_score,
        "reward_extra_info": reward_extra_info,
    }


async def compute_score_batch_async(prompts, gen_imgs, feedback_texts, regen_imgs, ground_truth_imgs, feedback_tuples, vqa_questions, extra_infos, task_ids):
    """Async batch processing with better load balancing"""
    n = len(prompts)
    if n == 0:
        return []

    # Create semaphore for concurrency control
    semaphore = asyncio.Semaphore(MAX_CONCURRENT_REQUESTS)
    
    async def process_single_request(idx, args):
        async with semaphore:
            (prompt, gen_img, feedback_text, regen_img, ground_truth_img, feedback_tuple, vqa_question, extra_info, task_id) = args
            
            # Load ground truth image
            ground_truth_img = await asyncio.to_thread(lambda p=ground_truth_img: PIL.Image.open(p).convert("RGB")) if ground_truth_img is not None else ground_truth_img
            
            # Process the request with improved error handling
            result = await compute_score_single_async(
                prompt, gen_img, feedback_text, regen_img, ground_truth_img, feedback_tuple, vqa_question, extra_info, task_id
            )
            
            return idx, result

    # Create tasks for all requests
    tasks = []
    for idx, args in enumerate(zip(
        prompts, gen_imgs, feedback_texts, regen_imgs, ground_truth_imgs, feedback_tuples, vqa_questions, extra_infos, task_ids
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
async def compute_score_batch(prompts, gen_imgs, feedback_texts, regen_imgs, ground_truth_imgs, feedback_tuples, vqa_questions, extra_infos, task_ids):
    """Async batch processing - directly calls the async implementation"""
    return await compute_score_batch_async(prompts, gen_imgs, feedback_texts, regen_imgs, ground_truth_imgs, feedback_tuples, vqa_questions, extra_infos, task_ids)


def compute_score_batch_sync(prompts, gen_imgs, feedback_texts, regen_imgs, ground_truth_imgs, feedback_tuples, vqa_questions, extra_infos, task_ids):
    """Synchronous wrapper for non-async contexts"""
    return asyncio.run(
        compute_score_batch_async(prompts, gen_imgs, feedback_texts, regen_imgs, ground_truth_imgs, feedback_tuples, vqa_questions, extra_infos, task_ids)
    )


def get_server_health_status():
    """Get current server health status - useful for monitoring"""
    return client_manager.get_server_status()