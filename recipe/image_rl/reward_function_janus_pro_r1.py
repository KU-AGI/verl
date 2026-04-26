"""Janus-Pro-R1-style reward functions for the 3-task image RL MDP.

Task1/task3 use the Janus-Pro-R1 reward-model rule:
    R_QA(I) = p(yes) / (p(yes) + p(no))
where p(yes) sums the reward model's first-token probabilities for
yes/Yes/YES and p(no) sums no/No/NO.

Task2 uses the local parser for the model's feedback output, then finalizes
with the Janus-Pro-R1 self-evaluation calibration reward:
    RComp = 1 - |R_QA(I) - SE(I)|.
"""

from __future__ import annotations

import asyncio
import math
from typing import Any, Dict, Optional, Tuple

import PIL.Image

from recipe.image_rl import reward_function_fine_grained as fg


JANUS_PRO_R1_VQA_PROMPT_TEMPLATE = (
    '<image>\n Does this image match the description "{prompt}", '
    "please directly respond with yes or no."
)

_YES_TOKENS = {"yes", "Yes", "YES"}
_NO_TOKENS = {"no", "No", "NO"}


def _iter_first_token_top_logprobs(choice) -> list[Any]:
    logprobs = getattr(choice, "logprobs", None)
    content = getattr(logprobs, "content", None) if logprobs is not None else None
    if not content:
        return []
    return getattr(content[0], "top_logprobs", None) or []


def _sum_token_probs(choice, token_set: set[str]) -> float:
    total = 0.0
    for item in _iter_first_token_top_logprobs(choice):
        token = str(getattr(item, "token", "")).strip()
        logprob = getattr(item, "logprob", None)
        if token in token_set and logprob is not None:
            total += math.exp(float(logprob))
    return total


def _normalize_yes_no_prob(yes_prob: float, no_prob: float) -> Optional[float]:
    denom = float(yes_prob) + float(no_prob)
    if denom <= 0.0 or not math.isfinite(denom):
        return None
    return float(yes_prob) / denom


def get_messages_janus_vqa(prompt: str, image) -> tuple[list[dict[str, Any]], str]:
    user_content = JANUS_PRO_R1_VQA_PROMPT_TEMPLATE.format(prompt=prompt or "")
    messages = [
        {
            "role": "user",
            "content": fg.content_from_prompt_with_images(
                user_content,
                [fg.convert_gen_img_to_base64(image)],
            ),
        }
    ]
    return messages, fg.RM_VLM_MODEL_PATH


async def get_janus_yes_no_reward(prompt: str, image) -> tuple[float, dict[str, Any]]:
    """Return Janus-Pro-R1 reward-model yes/no probability score."""
    if image is None:
        return 0.0, {
            "response": None,
            "yes_prob_raw": 0.0,
            "no_prob_raw": 0.0,
            "source": "missing_image",
        }

    messages, model = get_messages_janus_vqa(prompt, image)
    max_attempts = len(fg.VLM_BASE_URLS) * fg.MAX_RETRIES

    for attempt in range(max_attempts):
        client, sid, _ = await fg.borrow_rm_client(is_vlm=True)
        try:
            extra_body = {
                "top_k": -1,
                "min_p": 0.0,
                "best_of": 1,
                "repetition_penalty": 1.0,
            }
            if "qwen3.5" in model.lower():
                extra_body.update(chat_template_kwargs={"enable_thinking": False})

            response = await client.chat.completions.create(
                model=model,
                messages=messages,
                max_tokens=1,
                temperature=0.0,
                top_p=1.0,
                logprobs=True,
                top_logprobs=20,
                extra_body=extra_body,
                timeout=300000.0,
            )
            choice = response.choices[0]
            text = (choice.message.content or "").strip()
            yes_prob_raw = _sum_token_probs(choice, _YES_TOKENS)
            no_prob_raw = _sum_token_probs(choice, _NO_TOKENS)
            score = _normalize_yes_no_prob(yes_prob_raw, no_prob_raw)
            if score is not None:
                fg.vlm_client_manager.record_request_result(sid, success=True)
                return score, {
                    "response": text,
                    "yes_prob_raw": yes_prob_raw,
                    "no_prob_raw": no_prob_raw,
                    "source": "logprob",
                }

            lowered = text.lower()
            if lowered.startswith("yes") or lowered.startswith("no"):
                score = 1.0 if lowered.startswith("yes") else 0.0
                fg.vlm_client_manager.record_request_result(sid, success=True)
                return score, {
                    "response": text,
                    "yes_prob_raw": score,
                    "no_prob_raw": 1.0 - score,
                    "source": "text_fallback",
                }

            fg.vlm_client_manager.record_request_result(
                sid,
                success=False,
                error=ValueError("missing yes/no top-logprobs"),
            )
        except Exception as e:
            fg.vlm_client_manager.record_request_result(sid, success=False, error=e)
            print(
                "[JANUS-R1 REWARD] yes/no prob call failed "
                f"(attempt {attempt + 1}/{max_attempts}): {type(e).__name__}: {e}"
            )
        finally:
            await fg.release_rm_client(sid, True)

    return 0.0, {
        "response": None,
        "yes_prob_raw": 0.0,
        "no_prob_raw": 0.0,
        "source": "failed",
    }


def compute_task2_calibration_reward(
    task1_score: float,
    model_no_edit: bool,
) -> float:
    """RComp = 1 - |R_QA(I) - SE(I)|."""
    is_no_edit_binary = 1.0 if model_no_edit else 0.0
    return max(0.0, 1.0 - abs(float(task1_score) - is_no_edit_binary))


def finalize_task2_calibration_reward_extra_info(
    reward_extra_info: Dict,
    decision_vqa: float,
    decision_source: str = "missing",
) -> Tuple[float, Dict]:
    """Finalize task2 reward in Janus-Pro-R1 calibration mode."""
    finalized = dict(reward_extra_info or {})
    model_no_edit = bool(finalized.get("task2_no_feedback_needed", 0))
    target_no_edit = bool(float(decision_vqa) >= 1.0 - 1e-6)

    calibration_reward = compute_task2_calibration_reward(decision_vqa, model_no_edit)
    format_reward = float(finalized.get("task2_rule_based_format_reward", 0.0) or 0.0)
    total_reward = calibration_reward

    finalized["task2_decision_vqa_reward"] = float(decision_vqa)
    finalized["task2_decision_vqa_source"] = decision_source
    finalized["task2_target_no_edit"] = int(target_no_edit)
    finalized["task2_model_no_edit"] = int(model_no_edit)
    finalized["task2_calibration_reward"] = calibration_reward
    finalized["task2_format_reward"] = format_reward
    finalized["task2_vlm_reward"] = calibration_reward
    finalized["task2_total_reward"] = total_reward
    finalized["task2_step2_reward"] = 0.0
    finalized["task2_step3_reward"] = 0.0
    finalized["task2_step4_reward"] = total_reward
    finalized["task2_vqa_to_feedback_reward"] = calibration_reward
    finalized["task2_process"] = calibration_reward
    return calibration_reward, finalized


async def compute_score_single_async_janus(
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
    if task_id == 1:
        vqa_score, vqa_info = await get_janus_yes_no_reward(prompt, gen_img)
        return {
            "score": vqa_score,
            "reward_extra_info": {
                "task1_vqa_reward": vqa_score,
                "task1_vqa_reward_response": vqa_info.get("response"),
                "task1_vqa_yes_prob_raw": vqa_info.get("yes_prob_raw", 0.0),
                "task1_vqa_no_prob_raw": vqa_info.get("no_prob_raw", 0.0),
                "task1_vqa_score_source": vqa_info.get("source"),
                "task1_align": vqa_score,
                "task1_image_score": vqa_score,
                "task1_mdp_reward": vqa_score,
            },
        }

    if task_id == 3:
        vqa_score, vqa_info = await get_janus_yes_no_reward(prompt, regen_img)
        return {
            "score": vqa_score,
            "reward_extra_info": {
                "task3_vqa_reward": vqa_score,
                "task3_vqa_reward_response": vqa_info.get("response"),
                "task3_vqa_yes_prob_raw": vqa_info.get("yes_prob_raw", 0.0),
                "task3_vqa_no_prob_raw": vqa_info.get("no_prob_raw", 0.0),
                "task3_vqa_score_source": vqa_info.get("source"),
                "task3_align": vqa_score,
                "task3_image_score": vqa_score,
                "task3_if": 0.0,
                "task3_edit_if_reward": 0.0,
            },
        }

    if task_id == 2:
        formatting_evaluator = fg.FormattingEvaluatorV3()
        all_parts_present = all(
            part is not None
            for part in [predicted_tuple, predicted_answer, predicted_feedback]
        )
        predict_parsed_tuple = (
            formatting_evaluator._parse_tuples(predicted_tuple)
            if predicted_tuple is not None
            else []
        )
        predict_decomposed_ans = (
            formatting_evaluator._extract_verify_paragraphs(predicted_answer)
            if predicted_answer is not None
            else []
        )
        feedback_step_format_ok = (
            formatting_evaluator.check_feedback_step_format(predicted_feedback)
            if predicted_feedback is not None
            else False
        )
        no_feedback_needed = (
            predicted_feedback is not None
            and "no need to generate feedback" in predicted_feedback.lower()
        )
        tuple_format_ok = formatting_evaluator.check_tuple_schema_ok(
            predict_parsed_tuple
        )
        vqa_format_ok = len(predict_decomposed_ans) > 0
        skipped = "Skipped: Janus-Pro-R1 calibration uses self-check only"
        return {
            "score": 0.0,
            "reward_extra_info": {
                "task2_rule_based_format_reward": 1.0 if all_parts_present else 0.0,
                "task2_rule_based_decompose_reward": 0.0,
                "task2_rule_based_feedback_format_ok": int(feedback_step_format_ok),
                "task2_no_feedback_needed": int(no_feedback_needed),
                "task2_tuple_format_ok": int(tuple_format_ok),
                "task2_vqa_format_ok": int(vqa_format_ok),
                "task2_stage_raw_only": int(
                    bool((extra_info or {}).get("task2_raw_stage_only", False))
                ),
                "task2_prompt_to_tuple_reward": 0.0,
                "task2_tuple_to_vqa_reward": 0.0,
                "task2_vqa_to_feedback_reward": 0.0,
                "task2_vqa_to_feedback_content_reward": 0.0,
                "task2_step2_reward": 0.0,
                "task2_step3_reward": 0.0,
                "task2_step4_reward": 0.0,
                "task2_prompt_to_tuple_response": skipped,
                "task2_tuple_to_vqa_response": skipped,
                "task2_vqa_to_feedback_response": skipped,
            },
        }

    return await fg.compute_score_single_async(
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
    n = len(prompts)
    if n == 0:
        return []

    async def process_single_request(idx, args):
        (
            prompt,
            gen_img,
            feedback_text,
            regen_img,
            ground_truth_img,
            summarize,
            feedback_tuple,
            vqa_question,
            extra_info,
            task_id,
        ) = args

        if ground_truth_img is not None:
            ground_truth_img = await asyncio.to_thread(
                lambda p=ground_truth_img: PIL.Image.open(p).convert("RGB")
            )

        formatting_evaluator = fg.FormattingEvaluatorV3()
        predicted_tuple, predicted_answer, predicted_feedback = formatting_evaluator._split_text_into_parts(
            (feedback_text or "").strip()
        )
        predicted_summarize = None

        result = await compute_score_single_async_janus(
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

    tasks = [
        asyncio.create_task(process_single_request(idx, args))
        for idx, args in enumerate(
            zip(
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
            )
        )
    ]

    results = [None] * n
    none_indices = []
    for result in await asyncio.gather(*tasks, return_exceptions=True):
        if isinstance(result, Exception):
            print(f"[JANUS-R1 REWARD] Task failed with exception: {result}")
        else:
            idx, res = result
            results[idx] = res
            if res is None:
                none_indices.append(idx)

    if none_indices:
        print(
            f"[JANUS-R1 REWARD] Warning: {len(none_indices)}/{n} "
            f"results are None at indices: {none_indices}"
        )
    return results


async def compute_score_batch(
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
    **kwargs,
):
    if "rm_vlm_model_path" in kwargs:
        fg.RM_VLM_MODEL_PATH = kwargs["rm_vlm_model_path"]
    if "rm_llm_model_path" in kwargs:
        fg.RM_LLM_MODEL_PATH = kwargs["rm_llm_model_path"]
    if "mdp_reasoning_reward_weight" in kwargs:
        fg.MDP_REASONING_REWARD_WEIGHT = float(kwargs["mdp_reasoning_reward_weight"])
    if "mdp_reasoning_reward_cost" in kwargs:
        fg.MDP_REASONING_REWARD_COST = float(kwargs["mdp_reasoning_reward_cost"])

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
        mdp_reasoning_reward_weight=fg.MDP_REASONING_REWARD_WEIGHT,
        mdp_reasoning_reward_cost=fg.MDP_REASONING_REWARD_COST,
    )


def compute_score_batch_sync(
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
):
    return asyncio.run(
        compute_score_batch_async(
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
        )
    )


def get_server_health_status():
    return fg.get_server_health_status()
