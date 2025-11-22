# Copyright 2025 Meituan Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
import logging
import os, re, random
from typing import Any, Optional
from uuid import uuid4
from copy import deepcopy

from recipe.fully_async_policy.agent_loop.agent_loop import AgentLoopOutput, FullyAsyncAgentLoopOutput
from verl.experimental.agent_loop import AgentLoopBase
from verl.experimental.agent_loop.agent_loop import register
from verl.utils.profiler import simple_timer

from rdkit import Chem, RDLogger

RDLogger.DisableLog('rdApp.*')


logger = logging.getLogger(__file__)
logger.setLevel(os.getenv("VERL_LOGGING_LEVEL", "WARN"))


def extract_numbered_items(text: str) -> list:
    pattern = re.compile(r"^\s*\d+\.\s*(.+)$", re.MULTILINE)
    items = pattern.findall(text)
    return [item.strip().strip('`') for item in items]

def exact_match(pred_smi: str, gt_smi: str) -> bool:
    """Compare two SMILES strings for chemical equivalence."""
    try:
        mol_pred = Chem.MolFromSmiles(pred_smi)
        mol_gt = Chem.MolFromSmiles(gt_smi)
        
        if mol_pred is None or mol_gt is None:
            return False
            
        return Chem.MolToInchi(mol_pred) == Chem.MolToInchi(mol_gt)
    except Exception:
        return False

def remove_last_reflection_block(text: str) -> str:
    # 마지막 </REFLECTION> 위치 찾기
    end_tag_pos = text.rfind("</REFLECTION>")
    if end_tag_pos == -1:
        return text  # 없으면 그대로 반환

    # 마지막 <REFLECTION> 위치 찾기 (end_tag_pos 이전에서)
    start_tag_pos = text.rfind("<REFLECTION>", 0, end_tag_pos)
    if start_tag_pos == -1:
        return text  # 시작 태그가 없다면 그대로 반환

    # 마지막 reflection 블록 제거
    return text[:start_tag_pos].rstrip()


def remove_last_reflection_block_ids(token_ids: list, reflection_ids: list=[27, 5996, 28017, 29]) -> list:
    n = len(reflection_ids)
    last_idx = -1

    # 전체 token_ids에서 reflection_ids가 등장하는 모든 위치 탐색
    for i in range(len(token_ids) - n + 1):
        if token_ids[i:i+n] == reflection_ids:
            last_idx = i

    # reflection이 하나도 없으면 전체 반환
    if last_idx == -1:
        return token_ids

    # 마지막 reflection 시작 직전까지만 반환
    return token_ids[:last_idx]


def parse_steps_with_reflections(text: str):
    """
    주어진 문자열을 Step 단위로 파싱하고,
    각 Step에 포함된 <REFLECTION> 블록을 추출한다.
    
    반환 형식:
    [
        {
            "step": int,
            "content": str,        # REFLECTION 제외 Step 본문
            "reflections": [str]   # REFLECTION 블록 내용 리스트
        },
        ...
    ]
    """
    # Step 헤더 매칭
    step_pattern = re.compile(r"(## Step (\d+))")
    matches = list(step_pattern.finditer(text))
    
    steps_data = {}
    
    for i, match in enumerate(matches):
        step_header = match.group(1)
        step_num = int(match.group(2))
        
        # Step 구간의 끝 위치 계산
        start_pos = match.end()
        end_pos = matches[i + 1].start() if i + 1 < len(matches) else len(text)
        step_body = text[start_pos:end_pos].strip()
        
        # REFLECTION 블록 추출
        reflection_pattern = re.compile(r"<REFLECTION>(.*?)</REFLECTION>", re.DOTALL)
        reflections = reflection_pattern.findall(step_body)
        
        # REFLECTION 블록 제거 후 순수 Step 본문
        cleaned_body = reflection_pattern.sub("", step_body).strip()
        
        steps_data[f'step {step_num}'] = {
            "step": step_num,
            "content": cleaned_body,
            "reflections": [r.strip() for r in reflections]
        }
    
    return steps_data

def is_step_correct(step, task, d, reasoning_all):
    steps_data = parse_steps_with_reflections(reasoning_all)
    if step == 4 and task == "forward":
        reasoning_text = steps_data.get(f'step {step}', {}).get("content", "")
        correct = d[task]["reactive_atoms_smiles_str"] in reasoning_text
    elif step == 5 and task == "forward":
        reasoning_text = steps_data.get(f'step {step}', {}).get("content", "")
        for i in range(len(d[task]['reactive_atom_bonds'])):
            d[task]['reactive_atom_bonds'][i][0] = int(d[task]['reactive_atom_bonds'][i][0]) # convert to int for comparison
            d[task]['reactive_atom_bonds'][i][1] = int(d[task]['reactive_atom_bonds'][i][1]) # convert to int for comparison
            d[task]['reactive_atom_bonds'][0][2] = d[task]['reactive_atom_bonds'][0][2].replace("'", "") # remove extra quotes if any
        if len(d[task]['reactive_atom_bonds']) == 0:
            has_reactive_atom_bonds = True
            for bond_type in ['single', 'double', 'triple', 'aromatic']:
                if bond_type in reasoning_text:
                    has_reactive_atom_bonds = False
                    break
        else:
            has_reactive_atom_bonds = all(str(tuple(bond)) in reasoning_text for bond in d[task]['reactive_atom_bonds'])
        correct = has_reactive_atom_bonds
    elif step == 6 and task == "forward":
        reasoning_text = steps_data.get(f'step {step}', {}).get("content", "")
        has_tagged_smiles = d[task]["product_changes_tagged"] in reasoning_text
        correct = has_tagged_smiles
    elif step == 5 and task == "retro":
        reasoning_text = steps_data.get(f'step {step}', {}).get("content", "")
        bond_disconnection_list = []
        for bond in d[task]["bond_list"]:
            bond_str = f"{bond[0]}, {bond[1]}: {bond[2]}"
            bond_disconnection_list.append(bond_str)
        has_bond_disconnection = all(bond_str in reasoning_text for bond_str in bond_disconnection_list)
        correct = has_bond_disconnection
    elif step == 6 and task == "retro":
        reasoning_text = steps_data.get(f'step {step}', {}).get("content", "")
        has_synthons = all(synthon in reasoning_text for synthon in d[task]["synthons_list"])
        correct = has_synthons
    elif step == 7 and task == "retro":
        reasoning_text = steps_data.get(f'step {step}', {}).get("content", "")
        has_synthetic_equivalents = all(syn_equiv in reasoning_text for syn_equiv in d[task]["synthetic_equivalents"])
        correct = has_synthetic_equivalents
    elif step == 6 and task == "reagent":
        if len(steps_data.get(f"step 6", {}).get("reflections", [])) == 0:
            reagent_list = extract_numbered_items(steps_data.get(f'step 6', {}).get("content", ""))
        else:
            reagent_list = extract_numbered_items(steps_data.get(f'step 6', {}).get("reflections", [])[-1])
        # reagent_gt = ".".join(d[task]["reagents"])
        reagent_gt = d[task]['reagents']
        has_reagents = False
        for reagent_pred in reagent_list:
            if exact_match(reagent_pred, reagent_gt):
                has_reagents = True
                break
        correct = has_reagents
    elif step == 7 and task == "reagent":
        if len(steps_data.get(f"step 6", {}).get("reflections", [])) == 0:
            reagent_list = extract_numbered_items(steps_data.get(f'step 6', {}).get("content", ""))
        else:
            reagent_list = extract_numbered_items(steps_data.get(f'step 6', {}).get("reflections", [])[-1])
        reagent_gt = d[task]['reagents']
        has_reagents = False
        for reagent_pred in reagent_list:
            if exact_match(reagent_pred, reagent_gt):
                has_reagents = True
                break
        correct_reagent_number = -1
        for idx, reagent_pred in enumerate(reagent_list):
            # if exact_match(reagent_pred, ".".join(d[task]["reagents"])):
            if exact_match(reagent_pred, d[task]["reagents"]):
                correct_reagent_number = idx + 1
                break
        reasoning_text = steps_data.get(f'step {step}', {}).get("content", "")
        reagent_num = re.search(r"reagent (\d+)", reasoning_text, re.IGNORECASE)
        if reagent_num:
            predicted_reagent_number = int(reagent_num.group(1))
            has_correct_reagent_number = (predicted_reagent_number == correct_reagent_number) and has_reagents
        else:
            has_correct_reagent_number = False
        correct = has_correct_reagent_number
    else:
        raise ValueError(f"Unknown step/task combination: step {step}, task {task}")

    return correct



@register("partial_single_turn_agent")
class PartialSingleTurnAgentLoop(AgentLoopBase):
    """Naive agent loop that only do single turn chat completion."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.prompt_length = self.config.actor_rollout_ref.rollout.prompt_length
        self.response_length = self.config.actor_rollout_ref.rollout.response_length
        self.apply_chat_template_kwargs = self.config.data.get("apply_chat_template_kwargs", {})

    async def run(self, sampling_params: dict[str, Any], **kwargs) -> AgentLoopOutput:
        output: Optional[FullyAsyncAgentLoopOutput] = kwargs.get("output", None)
        messages = list(kwargs["raw_prompt"])
        param_version = kwargs.get("param_version", 0)

        metrics = {}
        request_id = uuid4().hex
        image_data = (kwargs.get("multi_modal_data") or {}).get("image", None)

        param_version_start = param_version
        param_version_end = param_version

        if not output:
            # TODO(baiyan): it is supposed to use the correct processor,
            #    but I found the async training would hang if use_correct_processor=True.
            #    so we use the tokenizer to tokenize the prompt for now.
            use_correct_processor = False
            if self.processor is not None and use_correct_processor:

                def get_prompt_ids():
                    raw_prompt = self.processor.apply_chat_template(
                        messages,
                        add_generation_prompt=True,
                        tokenize=False,
                        **self.apply_chat_template_kwargs,
                    )
                    model_inputs = self.processor(text=[raw_prompt], images=image_data, return_tensors="pt")
                    return model_inputs.pop("input_ids").squeeze(0).tolist()

                prompt_ids = await self.loop.run_in_executor(None, get_prompt_ids)
            else:
                prompt_ids = await self.loop.run_in_executor(
                    None,
                    lambda: self.tokenizer.apply_chat_template(
                        messages, add_generation_prompt=True, tokenize=True, **self.apply_chat_template_kwargs
                    ),
                )
        else:
            if output.is_cancel:
                # Resume the paused sample,
                # add the result directly after prompt_ids,
                # and reset generate_sequences metric
                prompt_ids = output.prompt_ids + output.response_ids
                metrics["generate_sequences"] = output.metrics.generate_sequences
                param_version_start = output.param_version_start
            else:
                # In the same batch of samples,
                # ome are canceled and some are not.
                # The samples without partial rollout are returned directly.
                return output
        with simple_timer("generate_sequences", metrics):
            if self.config.rollout.strategy == "naive_sampling":
                sampling_params.pop("stop", None)
                response_ids, log_probs, _, is_cancel = await self.server_manager.generate_for_partial(
                    request_id=request_id, prompt_ids=prompt_ids, sampling_params=sampling_params, image_data=image_data
                )
            elif self.config.rollout.strategy == "reflection_sampling":
                try:
                    if random.random() < self.config.rollout.strategy_ratio:
                        prompt_origin_ids = deepcopy(prompt_ids)
                        task = kwargs['task']
                        if task == "forward":
                            refl_steps = [4, 5, 6]
                        elif task == "retro":
                            refl_steps = [5, 6, 7]
                        elif task == "reagent":
                            refl_steps = [6, 7]
                        else:
                            raise ValueError(f"Unknown task: {task}")
                        for step_i, step in enumerate(refl_steps):
                            stop_strs = [f"## Step {step + 1}", "<REFLECTION", "<ANSWER", "</think>"]
                            stop_ids = [self.tokenizer.encode(s) for s in stop_strs]
                            sampling_params["stop"] = stop_strs
                            response_ids, log_probs_tmp, _, is_cancel = await self.server_manager.generate_for_partial(
                                request_id=uuid4().hex, prompt_ids=prompt_ids, sampling_params=sampling_params, image_data=image_data
                            )
                            # Remove the stop_ids from response_ids if generated
                            for stop_id in stop_ids:
                                if response_ids[-len(stop_id):] == stop_id:
                                    response_ids = response_ids[:-len(stop_id)]
                                    log_probs_tmp = log_probs_tmp[:-len(stop_id)]
                                    break
                            prompt_ids += response_ids
                            # prompt_ids = remove_last_reflection_block_ids(prompt_ids, reflection_ids=[27, 5996, 28017, 29])
                            raw_prompt = self.tokenizer.decode(prompt_ids)
                            step_correct = is_step_correct(step, task, kwargs['extra_info']['supporting_info'], raw_prompt)
                            if step_correct:
                                if step_i == len(refl_steps) - 1:
                                    next_ids = [151668] # self.tokenizer.encode("</think>")
                                else:
                                    next_ids = self.tokenizer.encode(f"## Step {step + 1}")
                            else:
                                if step_i == len(refl_steps) - 1:
                                    next_ids = [27, 5996, 28017] # self.tokenizer.encode("<REFLECTION")
                                else:
                                    next_ids = [27, 5996, 28017] # self.tokenizer.encode("<REFLECTION")
                                prompt_ids += next_ids
                                break
                            prompt_ids += next_ids

                        sampling_params.pop("stop", None)
                        response_ids, log_probs_tmp, prompt_logprobs, is_cancel = await self.server_manager.generate_for_partial(
                            request_id=uuid4().hex, prompt_ids=prompt_ids, sampling_params=sampling_params, image_data=image_data
                        )
                        prompt_logprobs += log_probs_tmp
                        prompt_ids += response_ids
                        # response_ids are the newly generated tokens after all steps. Remove prompt_origin_ids from prompt_ids
                        response_ids = prompt_ids[len(prompt_origin_ids):]
                        log_probs = prompt_logprobs[len(prompt_origin_ids):]
                        prompt_ids = prompt_origin_ids
                        # response_text = self.tokenizer.decode(response_ids)
                    else:
                        prompt_ids = self.tokenizer.apply_chat_template(
                            messages,
                            add_generation_prompt=True,
                            tokenize=True,
                            **self.apply_chat_template_kwargs,
                        )
                        sampling_params.pop("stop", None)
                        response_ids, log_probs, _, is_cancel = await self.server_manager.generate_for_partial(
                            request_id=request_id, prompt_ids=prompt_ids, sampling_params=sampling_params, image_data=image_data
                        )
                except Exception as e:
                    logger.error(f"Reflection sampling failed with error: {e}")
                    # In case of any error, fall back to normal generation
                    prompt_ids = self.tokenizer.apply_chat_template(
                        messages,
                        add_generation_prompt=True,
                        tokenize=True,
                        **self.apply_chat_template_kwargs,
                    )
                    sampling_params.pop("stop", None)
                    response_ids, log_probs, _, is_cancel = await self.server_manager.generate_for_partial(
                        request_id=request_id, prompt_ids=prompt_ids, sampling_params=sampling_params, image_data=image_data
                    )
        if not output:
            response_mask = [1] * len(response_ids)
        else:
            # Pause the sample to be resumed, add the output result to response_ids, and reset response_mask
            prompt_ids = output.prompt_ids
            log_probs = output.log_probs + log_probs
            response_ids = output.response_ids + response_ids
            response_mask = [1] * len(response_ids)

        assert len(response_ids) == len(response_mask) == len(log_probs), f"response_ids: {len(response_ids)}, response_mask: {len(response_mask)}, log_probs: {len(log_probs)} are not equal."
        return FullyAsyncAgentLoopOutput(
            prompt_ids=prompt_ids,
            response_ids=response_ids[: self.response_length],
            response_mask=response_mask[: self.response_length],
            num_turns=2,
            metrics=metrics,
            is_cancel=is_cancel,
            log_probs=log_probs,
            param_version_start=param_version_start,
            param_version_end=param_version_end,
            # multi_modal_data={"image": image_data} if image_data is not None else {},
        )
