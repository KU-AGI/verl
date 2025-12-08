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

# Configuration
BASE_URLS = [
    # "http://192.169.0.3:8000/v1",
    # "http://192.169.0.3:8002/v1",
    "http://192.169.0.3:8004/v1",
]
API_KEY = "EMPTY"
MAX_RETRIES = 3
BASE_DELAY = 2
MAX_CONCURRENT_REQUESTS = 8
MODEL_NAME = os.environ.get("RM_MODEL_PATH", "Qwen/Qwen3-VL-30B-A3B-Instruct")

# Health checking configuration
HEALTH_CHECK_INTERVAL = 30  # seconds
FAILURE_THRESHOLD = 3  # consecutive failures before marking as unhealthy
RECOVERY_CHECK_INTERVAL = 60  # seconds to wait before checking if unhealthy server recovered

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
                timeout=60.0  # Add timeout
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


async def compute_score_single_async(prompt, gen_img, feedback_text, regen_img, ground_truth_img, feedback_tuple, vqa_question, extra_info, task_id):
    """Async version of compute_score"""
    reward_score = 0.0
    reward_extra_info = {}

    if task_id == 1:
        # VLM based reward
        response = await get_response(prompt, gen_img, feedback_text, regen_img, ground_truth_img, feedback_tuple, vqa_question, extra_info, task_id)
        if response is not None:
            task1_idx_to_ans: dict = image_evaluator_parser(response)

            task1_reward_score = sum(task1_idx_to_ans.values())
            task1_ans_count = len(task1_idx_to_ans)
            
            task1_score_mean = 1.0 if task1_reward_score == task1_ans_count else 0.0

            reward_score += task1_score_mean
            reward_extra_info[f"task{task_id}_reward_response"] = response
        else:
            reward_score += 0.0
            reward_extra_info[f"task{task_id}_reward_response"] = "No response received"

    elif task_id == 2:
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

    elif task_id == 3:
        formatting_evaluator = FormattingEvaluator()
        last = formatting_evaluator._split_text_into_parts(feedback_text)[-1]
        if last is not None and "No need to generate feedback.".lower() in last.lower():
            reward_score = -100
            reward_extra_info[f"task{task_id}_reward_response"] = "None"
            return {
                "score": reward_score,
                "reward_extra_info": reward_extra_info,
            }

        # VLM based reward
        response = await get_response(prompt, gen_img, feedback_text, regen_img, ground_truth_img, feedback_tuple, vqa_question, extra_info, task_id)
        if response is not None:
            task3_idx_to_ans: dict = image_evaluator_parser(response)

            task3_reward_score = sum(task3_idx_to_ans.values())
            task3_ans_count = len(task3_idx_to_ans)
            
            task3_score_mean = 1.0 if task3_reward_score == task3_ans_count else 0.0
            reward_score += task3_score_mean
            reward_extra_info[f"task{task_id}_reward_response"] = response
        else:
            reward_score += 0.0
            reward_extra_info[f"task{task_id}_reward_response"] = "No response received"

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


def compute_score_batch(prompts, gen_imgs, feedback_texts, regen_imgs, ground_truth_imgs, feedback_tuples, vqa_questions, extra_infos, task_ids):
    """Wrapper function to run async batch processing"""
    return asyncio.run(
        compute_score_batch_async(prompts, gen_imgs, feedback_texts, regen_imgs, ground_truth_imgs, feedback_tuples, vqa_questions, extra_infos, task_ids)
    )


def get_server_health_status():
    """Get current server health status - useful for monitoring"""
    return client_manager.get_server_status()