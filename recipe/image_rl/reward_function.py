import os
import base64
import PIL
import re
import json
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
MODEL_NAME = os.environ.get("RM_MODEL_PATH", "Qwen/Qwen3-VL-30B-A3B-Instruct")

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

# Create sync clients for each URL
clients = [OpenAI(api_key=API_KEY, base_url=url) for url in BASE_URLS]
client_index = 0

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
    

def image_evaluator_parser(text):
    ans_line_re = re.compile(r'^\s*(\d+)\s*\|\s*Answer:\s*(Yes|No)\s*$', re.IGNORECASE | re.MULTILINE)
    
    idx_to_ans = {} # 1 | ... Answer: Yes or No -> {1: True or False}
    for idx_str, yn in ans_line_re.findall(text):
        idx = int(idx_str)
        idx_to_ans[idx] = (yn.strip().lower() == "yes")
    
    return idx_to_ans


def get_messages(prompt, gen_img, feedback_text, regen_img, ground_truth_img, feedback_tuple, vqa_question, task_id):
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


def get_next_client():
    """Round-robin client selection"""
    global client_index
    client = clients[client_index]
    client_index = (client_index + 1) % len(clients)
    return client


def get_response(prompt, gen_img, feedback_text, regen_img, ground_truth_img, feedback_tuple, vqa_question, task_id):
    """Get response from API with retry logic"""
    messages = get_messages(prompt, gen_img, feedback_text, regen_img, ground_truth_img, feedback_tuple, vqa_question, task_id)
    
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


def compute_score_single(prompt, gen_img, feedback_text, regen_img, ground_truth_img, feedback_tuple, vqa_question, extra_info, task_id, **kwargs):
    """Sync version of compute_score"""
    reward_score = 0.0
    reward_extra_info = {}

    if task_id == 1:
        # VLM based reward
        response = get_response(prompt, gen_img, feedback_text, regen_img, ground_truth_img, feedback_tuple, vqa_question, task_id)
        if response is not None:
            task1_idx_to_ans: dict = image_evaluator_parser(response)

            task1_reward_score = sum(task1_idx_to_ans.values())
            task1_ans_count = len(task1_idx_to_ans)
            
            task1_score_mean = (task1_reward_score / task1_ans_count) if task1_ans_count > 0 else 0.0

            reward_score += task1_score_mean
            reward_extra_info[f"task{task_id}_reward_response"] = response
        else:
            reward_score += 0.0

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
        feedback_parsed_tuple = formatting_evaluator._parse_part1(feedback_tuple) # gt_part1
        predict_parsed_tuple = formatting_evaluator._parse_part1(part1) # pred_part2
        predict_decomposed_ans = formatting_evaluator._extract_answer_paragraphs(part2) # pred_paragraphs

        part1_reward_dict = formatting_evaluator._calculate_metrics_for_reward(feedback_parsed_tuple, predict_parsed_tuple, predict_decomposed_ans)
        task2_reward_score += sum(part1_reward_dict.values()) # 3 of values
        task2_ans_count += len(part1_reward_dict)

        # VLM based reward
        response = get_response(prompt, gen_img, feedback_text, regen_img, ground_truth_img, feedback_tuple, vqa_question, task_id)
        if response is not None:
            try:
                raw_json = json.loads(response)
                label_response = raw_json.get("label", "").lower()
                if label_response in ["targeted_only", "no_feedback_needed"]:
                    task2_reward_score += 1.0
                elif label_response in ["non_target_touched"]:
                    task2_reward_score += 0.0 # 0.5
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
        
        reward_score += (task2_reward_score / task2_ans_count) if task2_ans_count > 0 else 0.0 # now 5 parts in total
        reward_extra_info[f"task{task_id}_reward_response"] = response

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
        response = get_response(prompt, gen_img, feedback_text, regen_img, ground_truth_img, feedback_tuple, vqa_question, task_id)
        if response is not None:
            task3_idx_to_ans: dict = image_evaluator_parser(response)

            task3_reward_score = sum(task3_idx_to_ans.values())
            task3_ans_count = len(task3_idx_to_ans)
            
            task3_score_mean = (task3_reward_score / task3_ans_count) if task3_ans_count > 0 else 0.0
            reward_score += task3_score_mean
            reward_extra_info[f"task{task_id}_reward_response"] = response
        else:
            reward_score += 0.0

    return {
        "score": reward_score,
        "reward_extra_info": reward_extra_info,
    }


def compute_score_batch_threaded(prompts, gen_imgs, feedback_texts, regen_imgs, ground_truth_imgs, feedback_tuples, vqa_questions, extra_infos, task_ids):
    """Threaded batch processing for concurrency"""
    
    def process_single_item(args):
        prompt, gen_img, feedback_text, regen_img, ground_truth_img, feedback_tuple, vqa_question, extra_info, task_id = args
        try:
            ground_truth_img = PIL.Image.open(ground_truth_img).convert("RGB")
            return compute_score_single(prompt, gen_img, feedback_text, regen_img, ground_truth_img, feedback_tuple, vqa_question, extra_info, task_id)
        except Exception as e:
            print(f"Task failed with exception: {e}")
            return {"score": 0.0, "reward_extra_info": {}}
    
    # Prepare arguments for ThreadPoolExecutor
    args_list = list(zip(prompts, gen_imgs, feedback_texts, regen_imgs, ground_truth_imgs, feedback_tuples, vqa_questions, extra_infos, task_ids))
    
    # Use ThreadPoolExecutor for concurrent processing
    with ThreadPoolExecutor(max_workers=MAX_CONCURRENT_REQUESTS) as executor:
        results = list(executor.map(process_single_item, args_list))
    
    return results


def compute_score_batch(prompts, gen_imgs, feedback_texts, regen_imgs, ground_truth_imgs, feedback_tuples, vqa_questions, extra_infos, task_ids):
    """Synchronous batch processing"""
    return compute_score_batch_threaded(prompts, gen_imgs, feedback_texts, regen_imgs, ground_truth_imgs, feedback_tuples, vqa_questions, extra_infos, task_ids)


def compute_score(prompt, gen_img, feedback_text, regen_img, ground_truth_img, feedback_tuple, vqa_question, extra_info, task_id, **kwargs):
    """Legacy single score computation"""
    results = compute_score_batch(
        [prompt], [gen_img], [feedback_text], [regen_img], [ground_truth_img], [feedback_tuple], [vqa_question], [extra_info], [task_id]
    )
    return results[0]