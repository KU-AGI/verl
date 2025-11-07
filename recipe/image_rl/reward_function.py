import os
import base64
import PIL
from io import BytesIO
from typing import Optional, List, Dict, Any
import PIL.Image
from openai import OpenAI
import numpy as np
from transformers import AutoTokenizer
from concurrent.futures import ThreadPoolExecutor
import time
from collections import defaultdict
from verl.utils.reward_score.math_reward import last_boxed_only_string, remove_boxed
from recipe.image_rl.utils import FormattingEvaluator

formatter = FormattingEvaluator()

# Configuration
BASE_URLS = [
    # "http://192.169.0.3:8000/v1",
    # "http://192.169.0.3:8002/v1",
    "http://192.169.0.3:8004/v1",
]
API_KEY = "EMPTY"
MAX_RETRIES = 3
BASE_DELAY = 2
MAX_CONCURRENT_REQUESTS = 32
MODEL_NAME = os.environ.get("RM_MODEL_PATH", "OpenGVLab/InternVL3_5-38B")

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

# Create sync clients for each URL
clients = [OpenAI(api_key=API_KEY, base_url=url) for url in BASE_URLS]
client_index = 0

TASK1_FIRST_IMAGE_GENERATOR_PROMPT_TEMPLATE = """
You are a evaluator. Your task is to evaluate generated image based on the question.
You will be given a question and generated image from target model. You need to judge if the image is a good image based on the question.

[Question]
{question}

[Your Evaluation]
Give your reason of your evaluation.
Please put your final answer about the image is aligned with the question or not (i.e., '1' or '0') in \\boxed{{}}.
"""

TASK2_FEEDBACK_GENERATOR_PROMPT_TEMPLATE = """
You are a evaluator. Your task is to evaluate how well the feedback is aligned with the question.
You will be given a question and feedback from target model. You need to judge if the feedback is a good feedback based on the question.

[Question]
{question}

[Feedback]
{feedback}

[Your Evaluation]
Give your reason of your evaluation.
Please put your final answer about the feedback is aligned with the question or not (i.e., '1' or '0') in \\boxed{{}}.
"""

TASK3_REGEN_IMAGE_GENERATOR_PROMPT_TEMPLATE = """
You are a evaluator. Your task is to evaluate how well the generated image is aligned with the question.
You will be given a question and generated image from target model. You need to judge if the generated image is a good image based on the question.
First image is the generated image from target model. Second image is the ground truth image.

[Question]
{question}

[Your Evaluation]
Give your reason of your evaluation.
Please put your final answer about the generated image is aligned with the question or not (i.e., '1' or '0') in \\boxed{{}}.
"""


def convert_gen_img_to_base64(gen_img: PIL.Image.Image) -> Optional[str]:
    if isinstance(gen_img, str):
        gen_img = PIL.Image.open(gen_img)
    elif isinstance(gen_img, np.ndarray):
        gen_img = np.array(gen_img, dtype=np.uint8)
        gen_img = PIL.Image.fromarray(gen_img)

    """Convert gen_img to base64 data URL."""
    if not isinstance(gen_img, PIL.Image.Image):
        raise TypeError(f"Unsupported image type: {type(gen_img)}")

    # Convert PIL.Image → base64 data URL
    buffer = BytesIO()
    gen_img.save(buffer, format="PNG")
    img_base64 = base64.b64encode(buffer.getvalue()).decode("utf-8")

    return f"data:image/png;base64,{img_base64}"
    

def get_messages(prompt, gen_img, feedback_text, regen_img, ground_truth, task_id):
    if task_id == 1:
        system_prompt = TASK1_FIRST_IMAGE_GENERATOR_PROMPT_TEMPLATE.format(question=prompt)
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": [
                {"type": "image_url", "image_url": {"url": convert_gen_img_to_base64(gen_img)}},
            ]}
        ]
    elif task_id == 2:
        system_prompt = TASK2_FEEDBACK_GENERATOR_PROMPT_TEMPLATE.format(question=prompt, feedback=feedback_text)
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": [
                {"type": "image_url", "image_url": {"url": convert_gen_img_to_base64(gen_img)}},
                {"type": "text", "text": feedback_text},
            ]}
        ]
    elif task_id == 3:
        system_prompt = TASK3_REGEN_IMAGE_GENERATOR_PROMPT_TEMPLATE.format(question=prompt)
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": [
                {"type": "image_url", "image_url": {"url": convert_gen_img_to_base64(regen_img)}},
                {"type": "image_url", "image_url": {"url": convert_gen_img_to_base64(ground_truth)}},
            ]}
        ]
    else:
        raise ValueError(f"Invalid task: {task_id} is must be one of task1, task2, or task3.")
    
    return messages


def get_next_client():
    """Round-robin client selection"""
    global client_index
    client = clients[client_index]
    client_index = (client_index + 1) % len(clients)
    return client


def get_response(prompt, gen_img, feedback_text, regen_img, ground_truth, task_id):
    """Get response from API with retry logic"""
    messages = get_messages(prompt, gen_img, feedback_text, regen_img, ground_truth, task_id)
    
    for attempt in range(MAX_RETRIES):
        try:
            client = get_next_client()
            response = client.chat.completions.create(
                model=MODEL_NAME,
                messages=messages
            )
            return response.choices[0].message.content
        except Exception as e:
            if attempt < MAX_RETRIES - 1:
                print(f"Exception: {repr(e)}")
                delay = BASE_DELAY * (2**attempt)
                print(f"Retrying in {delay} seconds...")
                time.sleep(delay)
            else:
                print(f"Failed after {MAX_RETRIES} attempts. Error: {e}")
                return None


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


def compute_score_single(prompt, gen_img, feedback_text, regen_img, ground_truth, extra_info, task_id, **kwargs):
    """Sync version of compute_score"""
    reward_score = 0.0
    reward_extra_info = {}

    # VLM based reward
    response = get_response(prompt, gen_img, feedback_text, regen_img, ground_truth, task_id)
    if response is not None:
        reward_score += compute_reward(response)
        reward_extra_info[f"task{task_id}_reward_response"] = response
    else:
        reward_score += 0.0
    return {
        "score": reward_score,
        "reward_extra_info": reward_extra_info,
    }


def compute_score_batch_threaded(prompts, gen_imgs, feedback_texts, regen_imgs, ground_truths, extra_infos, task_ids):
    """Threaded batch processing for concurrency"""
    
    def process_single_item(args):
        prompt, gen_img, feedback_text, regen_img, ground_truth, extra_info, task_id = args
        try:
            ground_truth = PIL.Image.open(ground_truth).convert("RGB")
            return compute_score_single(prompt, gen_img, feedback_text, regen_img, ground_truth, extra_info, task_id)
        except Exception as e:
            print(f"Task failed with exception: {e}")
            return {"score": 0.0, "reward_extra_info": {}}
    
    # Prepare arguments for ThreadPoolExecutor
    args_list = list(zip(prompts, gen_imgs, feedback_texts, regen_imgs, ground_truths, extra_infos, task_ids))
    
    # Use ThreadPoolExecutor for concurrent processing
    with ThreadPoolExecutor(max_workers=MAX_CONCURRENT_REQUESTS) as executor:
        results = list(executor.map(process_single_item, args_list))
    
    return results


def compute_score_batch(prompts, gen_imgs, feedback_texts, regen_imgs, ground_truths, extra_infos, task_ids):
    """Synchronous batch processing"""
    return compute_score_batch_threaded(prompts, gen_imgs, feedback_texts, regen_imgs, ground_truths, extra_infos, task_ids)


def compute_score(prompt, gen_img, feedback_text, regen_img, ground_truth, extra_info, task_id, **kwargs):
    """Legacy single score computation"""
    results = compute_score_batch(
        [prompt], [gen_img], [feedback_text], [regen_img], [ground_truth], [extra_info], [task_id]
    )
    return results[0]