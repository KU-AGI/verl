import os
import asyncio
import base64
import PIL
from io import BytesIO
from typing import Optional, List, Dict, Any
from openai import AsyncOpenAI
import numpy as np

from verl.utils.dataset.vision_utils import process_image
from verl.utils.reward_score.math_reward import last_boxed_only_string, remove_boxed

# Configuration
BASE_URLS = [
    "http://192.169.0.2:8004/v1",
    "http://192.169.0.2:8006/v1"
]
API_KEY = "EMPTY"
MAX_RETRIES = 3
BASE_DELAY = 2
MAX_CONCURRENT_REQUESTS = 32
MODEL_NAME = os.environ.get("RM_MODEL_PATH", "OpenGVLab/InternVL3_5-38B")

# Create async clients for each URL
clients = [AsyncOpenAI(api_key=API_KEY, base_url=url) for url in BASE_URLS]
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
    ground_truth = convert_gen_img_to_base64(ground_truth)
    if task_id == 1:
        system_prompt = TASK1_FIRST_IMAGE_GENERATOR_PROMPT_TEMPLATE.format(question=prompt)
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": [
                {"type": "image_url", "image_url": {"url": convert_gen_img_to_base64(gen_img)}}
            ]}
        ]
        return messages
    elif task_id == 2:
        system_prompt = TASK2_FEEDBACK_GENERATOR_PROMPT_TEMPLATE.format(question=prompt, feedback=feedback_text)
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": [
                {"type": "image_url", "image_url": {"url": convert_gen_img_to_base64(gen_img)}},
                {"type": "text", "text": feedback_text},
            ]}
        ]
        return messages
    elif task_id == 3:
        system_prompt = TASK3_REGEN_IMAGE_GENERATOR_PROMPT_TEMPLATE.format(question=prompt)
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": [
                {"type": "image_url", "image_url": {"url": convert_gen_img_to_base64(gen_img)}},
                {"type": "image_url", "image_url": {"url": convert_gen_img_to_base64(regen_img)}}
            ]}
        ]
        return messages
    else:
        raise ValueError(f"Invalid task: {task_id} is must be one of task1, task2, or task3.")


def get_next_client():
    """Round-robin client selection"""
    global client_index
    client = clients[client_index]
    client_index = (client_index + 1) % len(clients)
    return client


async def get_response(prompt, gen_img, feedback_text, regen_img, ground_truth, task_id):
    """Get response from API with retry logic"""
    messages = get_messages(prompt, gen_img, feedback_text, regen_img, ground_truth, task_id)
    
    for attempt in range(MAX_RETRIES):
        try:
            client = get_next_client()
            response = await client.chat.completions.create(
                model=MODEL_NAME,
                messages=messages,
            )
            return response.choices[0].message.content
        except Exception as e:
            if attempt < MAX_RETRIES - 1:
                print(f"Exception: {repr(e)}")
                delay = BASE_DELAY * (2**attempt)
                print(f"Retrying in {delay} seconds...")
                await asyncio.sleep(delay)
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


async def compute_score_async(prompt, gen_img, feedback_text, regen_img, ground_truth, extra_info, task_id, **kwargs):
    """Async version of compute_score"""
    reward_score = 0.0
    reward_extra_info = {}

    # VLM based reward
    response = await get_response(prompt, gen_img, feedback_text, regen_img, ground_truth, task_id)
    if response is not None:
        reward_score += compute_reward(response)
        reward_extra_info[f"task_{task_id}"] = reward_score
    else:
        reward_score += 0.0

    return {
        "score": reward_score,
        "reward_extra_info": reward_extra_info,
    }


async def compute_score_batch_async(prompts, gen_imgs, feedback_texts, regen_imgs, ground_truths, extra_infos, task_ids):
    """Async batch processing with semaphore for concurrency control"""
    semaphore = asyncio.Semaphore(MAX_CONCURRENT_REQUESTS)
    
    async def process_with_semaphore(prompt, gen_img, feedback_text, regen_img, ground_truth, extra_info, task_id):
        async with semaphore:
            return await compute_score_async(prompt, gen_img, feedback_text, regen_img, ground_truth, extra_info, task_id)
    
    # Create tasks for all items
    tasks = []
    for prompt, gen_img, feedback_text, regen_img, ground_truth, extra_info, task_id in zip(
        prompts, gen_imgs, feedback_texts, regen_imgs, ground_truths, extra_infos, task_ids, strict=True
    ):
        task = process_with_semaphore(prompt, gen_img, feedback_text, regen_img, ground_truth, extra_info, task_id)
        tasks.append(task)
    
    # Execute all tasks concurrently
    results = await asyncio.gather(*tasks, return_exceptions=True)
    
    # Handle exceptions
    clean_results = []
    for result in results:
        if isinstance(result, Exception):
            print(f"Task failed with exception: {result}")
            clean_results.append({"score": 0.0, "reward_extra_info": {}})
        else:
            clean_results.append(result)
    
    return clean_results


def compute_score_batch(prompts, gen_imgs, feedback_texts, regen_imgs, ground_truths, extra_infos, task_ids):
    """Synchronous wrapper for backward compatibility"""
    return asyncio.run(
        compute_score_batch_async(prompts, gen_imgs, feedback_texts, regen_imgs, ground_truths, extra_infos, task_ids)
    )


def compute_score(prompt, gen_img, feedback_text, regen_img, ground_truth, extra_info, task_id, **kwargs):
    """Legacy single score computation - now uses async processing internally"""
    results = compute_score_batch(
        [prompt], [gen_img], [feedback_text], [regen_img], 
        [ground_truth], [extra_info], [task_id]
    )
    return results[0]