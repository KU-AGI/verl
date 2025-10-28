from concurrent.futures import ThreadPoolExecutor
from this import d
from time import sleep

import base64
import requests
import PIL
from io import BytesIO
from typing import Optional
from openai import OpenAI

from verl.utils.dataset.vision_utils import process_image
from verl.utils.reward_score.math_reward import last_boxed_only_string, remove_boxed

BASE_URL = "http://218.238.5.120:8000/v1"
API_KEY = "EMPTY"
MAX_RETRIES = 3
BASE_DELAY = 2
MAX_WORKERS = 32
MODEL_NAME = "OpenGVLab/InternVL3_5-38B"

client = OpenAI(api_key=API_KEY, base_url=BASE_URL)

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

TASK3_REFINED_IMAGE_GENERATOR_PROMPT_TEMPLATE = """
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

    """Convert gen_img to base64 data URL."""
    if not isinstance(gen_img, PIL.Image.Image):
        raise TypeError(f"Unsupported image type: {type(gen_img)}")

    # Convert PIL.Image → base64 data URL
    buffer = BytesIO()
    gen_img.save(buffer, format="PNG")
    img_base64 = base64.b64encode(buffer.getvalue()).decode("utf-8")

    return f"data:image/png;base64,{img_base64}"
    

def get_messages(prompt, gen_img, feedback_text, refined_gen_img, ground_truth, task_id):
    ground_truth_base64 = convert_gen_img_to_base64(ground_truth)
    if task_id == 1:
        system_prompt = TASK1_FIRST_IMAGE_GENERATOR_PROMPT_TEMPLATE.format(question=prompt)
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": [
                {"type": "image_pil", "image_pil": gen_img}
            ]}
        ]
        return messages
    elif task_id == 2:
        system_prompt = TASK2_FEEDBACK_GENERATOR_PROMPT_TEMPLATE.format(question=prompt, feedback=feedback_text)
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": [
                {"type": "image_pil", "image_pil": gen_img}, # convert_gen_img_to_base64(gen_img)}},
                {"type": "text", "text": feedback_text},
            ]}
        ]
        return messages
    elif task_id == 3:
        system_prompt = TASK3_REFINED_IMAGE_GENERATOR_PROMPT_TEMPLATE.format(question=prompt)
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": [
                {"type": "image_pil", "image_pil": gen_img},
                {"type": "image_pil", "image_pil": refined_gen_img}
            ]}
        ]
        return messages
    else:
        raise ValueError(f"Invalid task: {task_id} is must be one of task1, task2, or task3.")


def get_response(prompt, gen_img, feedback_text, refined_gen_img, ground_truth, task_id):
    messages = get_messages(prompt, gen_img, feedback_text, refined_gen_img, ground_truth, task_id)
    for attempt in range(MAX_RETRIES):
        try:
            response = client.chat.completions.create(
                model=MODEL_NAME,
                messages=messages,
            )
            return response.choices[0].message.content
        except Exception as e:
            if attempt < MAX_RETRIES - 1:
                print("Exception: ", repr(e))
                delay = BASE_DELAY * (2**attempt)
                print(f"Retrying in {delay} seconds...")
                sleep(delay)
            else:
                print(f"Failed after {MAX_RETRIES} attempts. Error: {e}")
    
    raise ConnectionRefusedError(f"Failed to run the model for {prompt}! Error: {e}")

def compute_reward(response):
    reward_score = 0.0
    try:
        boxed_result = last_boxed_only_string(response)
        if boxed_result is not None:
            result = remove_boxed(boxed_result)
            reward_score = float(result == "True")
    except Exception as e:
        print(e)
    return reward_score


def compute_score(prompt, gen_img, feedback_text, refined_gen_img, ground_truth, extra_info, **kwargs):
    reward_score = 0.0
    reward_extra_info = {}

    for task_id in range(3):
        response = get_response(prompt, gen_img, feedback_text, refined_gen_img, ground_truth, task_id+1) # task_id is 1-indexed
        if response is not None:
            reward_score += compute_reward(response)
            reward_extra_info[f"task_{task_id}"] = reward_score
            break
        else:
            reward_score += 0.0

    return {
        "score": reward_score,
        "reward_extra_info": reward_extra_info,
    }


# def compute_score_batch(prompts, gen_imgs, feedback_texts, refined_gen_imgs, ground_truths, extra_infos):
#     with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
#         futures = []
#         for prompt, gen_img, feedback_text, refined_gen_img, ground_truth, extra_info in zip(
#             prompts, gen_imgs, feedback_texts, refined_gen_imgs, ground_truths, extra_infos, strict=True
#         ):
#             future = executor.submit(compute_score, prompt, gen_img, feedback_text, refined_gen_img, ground_truth, extra_info)
#             futures.append(future)

#         results = [future.result() for future in futures]

#     return results
