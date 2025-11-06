# Copyright 2024 Bytedance Ltd. and/or its affiliates
# Copyright 2022 EleutherAI and the HuggingFace Inc. team. All rights reserved.
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
# Adapted from https://github.com/EleutherAI/lm-evaluation-harness/blob/main/lm_eval/tasks/hendrycks_math/utils.py

import re
from typing import Optional, Dict, Any, Tuple, Set, List, Union
from dataclasses import dataclass, field
from abc import ABC, abstractmethod
from enum import Enum
from rdkit import Chem, RDLogger
from collections import defaultdict

RDLogger.DisableLog('rdApp.*')

# Constants
class Tags:
    ANSWER = r"<ANSWER>(.*?)</ANSWER>"
    THINK = r"<think>(.*?)</think>"
    STEP = r"(## Step(?: [0-9]+(?:-[0-9]+)?):)([\s\S]*?)(?=(?:## Step|\Z))"
    STEP_ID = r"Step ([0-9]+(?:-[0-9]+)?)"
    REFLECTION = r"<REFLECTION>([\s\S]*?)</REFLECTION>"

class TaskType(Enum):
    FORWARD = "forward"
    RETRO = "retro" 
    REAGENT = "reagent"

@dataclass
class StepData:
    step_id: str
    original: str
    reflection: Optional[str] = None
    verify: bool = False
    metadata: Optional[Dict[str, Any]] = None

@dataclass
class EvaluationResult:
    score: float
    acc: bool
    pred: Optional[str]
    metrics: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary format for compatibility."""
        result = {
            "score": self.score,
            "acc": self.acc,
            "pred": self.pred,
        }
        # Add metrics to the main dict for flat structure
        if self.metrics:
            result.update(self.metrics)
        return result

class SMILESValidator:
    @staticmethod
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

class TextProcessor:
    @staticmethod
    def extract_answer_tag(text: str) -> str:
        """Extract content from ANSWER tags."""
        match = re.search(Tags.ANSWER, text, re.DOTALL)
        return match.group(1).strip() if match else text.strip()
    
    @staticmethod
    def parse_step_content(step_text: str, step_id: str, metadata: Optional[Dict[str, Any]] = None) -> StepData:
        """Parse step content including reflections."""
        reflection_match = re.search(Tags.REFLECTION, step_text)
        reflection = reflection_match.group(1).strip() if reflection_match else None

        return StepData(
            step_id=step_id,
            original=step_text.strip(),
            reflection=reflection,
            verify=reflection is not None,
            metadata=metadata
        )

class StepParser:
    @staticmethod
    def decompose_steps(rationale: str) -> Dict[str, StepData]:
        """Extract and parse all steps from the rationale."""
        decomposed_dict = {}
        
        # Extract reasoning part (before answer tag)
        reasoning_steps, answer = rationale.split("<ANSWER>")[0], rationale.split("<ANSWER>")[-1].replace("</ANSWER>", "").strip()
        
        # extract <think> tag
        match =re.search(r'<think>(.*?)</think>', reasoning_steps, re.DOTALL)
        if match:
            reasoning_steps = match.group(1).strip()

        # Find all step matches
        matches = re.findall(Tags.STEP, reasoning_steps)
        
        for step_title, step_content in matches:
            step_id_match = re.search(Tags.STEP_ID, step_title.strip())
            if not step_id_match:
                continue
                
            step_id_str = step_id_match.group(1)
            step_data = TextProcessor.parse_step_content(step_content, step_id_str, metadata={"answer": answer})
            decomposed_dict[f"step_{step_id_str}"] = step_data
        
        return decomposed_dict

# Base classes for step evaluators
class BaseStepEvaluator(ABC):
    """Base class for step-specific evaluators."""
    
    def __init__(self, step_id: str):
        self.step_id = step_id
    
    @abstractmethod
    def parse_step(self, step: StepData) -> Dict[str, Any]:
        """Parse step content and extract relevant data."""
        pass
    
    @abstractmethod
    def evaluate_step(self, pred_parsed: Dict[str, Any], gt_parsed: Dict[str, Any]) -> Dict[str, int]:
        """Compare parsed data and return metrics."""
        pass
    
    def evaluate(self, pred_step: StepData, gt_step: StepData) -> Dict[str, int]:
        """Main evaluation method."""
        pred_parsed = self.parse_step(pred_step)
        gt_parsed = self.parse_step(gt_step)
        return self.evaluate_step(pred_parsed, gt_parsed)
    
    def _get_content(self, step: StepData) -> str:
        """Get appropriate content from step (original or reflection)."""
        return step.reflection if (step.verify and step.reflection) else step.original

# Forward step evaluators
class ForwardStep4Evaluator(BaseStepEvaluator):
    """Step 4: Reactive atom identification."""
    
    def parse_step(self, step: StepData) -> Dict[str, Any]:
        content = self._get_content(step)
        
        if step.verify and step.reflection:
            highlighted_smiles = content.split("It should be")[-1].split("\n")[-1].strip().rstrip(".")
        else:
            highlighted_smiles = content.split("`**[atom]:[number]**` as follows.")[-1].split("\n")[-1].strip()
        
        return {"reactive_atoms_smiles_str": highlighted_smiles}
    
    def evaluate_step(self, pred_parsed: Dict[str, Any], gt_parsed: Dict[str, Any]) -> Dict[str, int]:
        return {
            "forward/step4/has_reactive_atoms_smiles": 1 if pred_parsed["reactive_atoms_smiles_str"] == gt_parsed["reactive_atoms_smiles_str"] else 0
        }

class ForwardStep5Evaluator(BaseStepEvaluator):
    """Step 5: Bond formation."""

    def parse_step(self, step: StepData) -> Dict[str, Any]:
        content = self._get_content(step)
        
        if step.verify and step.reflection:
            bond_pairs = content.split("It should be")[-1].split("\n")[-1].strip().split(".")[0].strip()
        else:
            bond_pairs = content.split("the following bonds will be formed.")[-1].strip().split("\n")[-1].strip().split(".")[0].strip()
        
        bonds = set()
        if bond_pairs:
            for line in bond_pairs.split("\n"):
                bond_pair = line.strip().strip("(").strip(")").strip()
                if bond_pair:
                    parts = [p.strip() for p in bond_pair.split(",")]
                    if len(parts) >= 3:
                        try:
                            atom1, atom2, bond_type = int(parts[0]), int(parts[1]), parts[2]
                            bonds.add((atom1, atom2, bond_type))
                        except (ValueError, IndexError):  # Skip lines that can't be parsed
                            continue
        
        return {"reactive_atom_bonds_set": bonds}
    
    def evaluate_step(self, pred_parsed: Dict[str, Any], gt_parsed: Dict[str, Any]) -> Dict[str, int]:
        return {
            # pred_parsed set is a subset of gt_parsed set
            "forward/step5/has_reactive_atom_bonds": 1 if pred_parsed["reactive_atom_bonds_set"] <= gt_parsed["reactive_atom_bonds_set"] else 0
        }

class ForwardStep6Evaluator(BaseStepEvaluator):
    """Step 6: Toward product SMILES."""

    def apply_edit_tags(self, smiles: str) -> str:
        """
        DEL, REP_OLD/REP_NEW, ADD tags to SMILES string
        """

        # 1. REP_OLD ... REP_NEW 
        # REP_OLD is ignored and only REP_NEW is kept
        def repl_rep(match):
            old = match.group("old")
            new = match.group("new")
            return new  # old is ignored and only new is kept

        smiles = re.sub(
            r"<REP_OLD>(?P<old>.*?)</REP_OLD>\s*<REP_NEW>(?P<new>.*?)</REP_NEW>",
            repl_rep,
            smiles,
            flags=re.DOTALL,
        )

        # 2. DEL tag is deleted
        smiles = re.sub(r"<DEL>.*?</DEL>", "", smiles, flags=re.DOTALL)

        # 3. ADD tag is only kept
        smiles = re.sub(r"<ADD>(.*?)</ADD>", r"\1", smiles, flags=re.DOTALL)

        return smiles

    def parse_step(self, step: StepData) -> Dict[str, Any]:
        content = self._get_content(step)   
        
        if step.verify and step.reflection:
            product_smiles = content.split("The product should be")[-1].split("\n")[-1].strip().rstrip(".")
        else:
            tagged_smiles = content.split("we get the following")[-1].strip().split("\n")[-1].strip()
            product_smiles = self.apply_edit_tags(tagged_smiles)

        return {"product_smiles": product_smiles}
    
    def evaluate_step(self, pred_parsed: Dict[str, Any], gt_parsed: Dict[str, Any]) -> Dict[str, int]:
        smiles_match = SMILESValidator.exact_match(pred_parsed["product_smiles"], gt_parsed["product_smiles"])
        return {
            "forward/step6/matched_product_smiles": 1 if smiles_match else 0,
        }

# Retrosynthesis step evaluators
class RetroStep5Evaluator(BaseStepEvaluator):
    """Step 5: Identify bond disconnections."""
    
    def parse_step(self, step: StepData) -> Dict[str, Any]:
        content = self._get_content(step)
        
        if step.verify and step.reflection:
            bond_disconnections = content.split("It should be")[-1].split("\n")[-1].strip().split(".")[0].strip()
        else:
            if "it does not appear that" in content:
                return {"bond_disconnections_set": set()}
            else:
                bond_disconnections = content.split("it is expected that the following bonds would have been formed.")[-1].strip()

        bond_disconnections_set = set()
        if bond_disconnections:
            for line in bond_disconnections.split("\n"):
                line = line.strip()
                if not line:  # Skip empty lines
                    continue
                try:
                    parts = re.split(",|:", line)
                    if len(parts) != 3:  # Skip lines that don't have exactly 3 parts
                        continue
                    atom1, atom2, bond_type = parts
                    atom1, atom2, bond_type = int(atom1), int(atom2), bond_type.strip()
                    bond_disconnections_set.add((atom1, atom2, bond_type))
                except (ValueError, IndexError):  # Skip lines that can't be parsed
                    continue

        return {"bond_disconnections_set": bond_disconnections_set}
    
    def evaluate_step(self, pred_parsed: Dict[str, Any], gt_parsed: Dict[str, Any]) -> Dict[str, int]:
        return {
            # pred_parsed set is a subset of gt_parsed set
            "retro/step5/has_bond_disconnections": 1 if pred_parsed["bond_disconnections_set"] <= gt_parsed["bond_disconnections_set"] else 0
        }


class RetroStep6Evaluator(BaseStepEvaluator):
    """Step 6: Identify synthons."""
    
    def parse_step(self, step: StepData) -> Dict[str, Any]:
        content = self._get_content(step)
        
        # Like 1. Synthons
        # or 1. Synthons, 2. Synthons
        pattern = re.compile(r"\b\d+\.\s*([^\n\r]+)")
        synthons = [m.group(1).strip() for m in pattern.finditer(content)]
        return {"synthons_set": set(synthons)}

    def evaluate_step(self, pred_parsed: Dict[str, Any], gt_parsed: Dict[str, Any]) -> Dict[str, int]:
        return {
            # pred_parsed set is a subset of gt_parsed set
            "retro/step6/has_synthons": 1 if pred_parsed["synthons_set"] <= gt_parsed["synthons_set"] else 0,
        }


class RetroStep7Evaluator(BaseStepEvaluator):
    """Step 7: Identify synthetic equivalents."""
    
    def parse_step(self, step: StepData) -> Dict[str, Any]:
        content = self._get_content(step)
        
        # Like 1. Synthetic equivalents
        # or 1. Synthetic equivalents, 2. Synthetic equivalents
        pattern = re.compile(r"\b\d+\.\s*([^\n\r]+)")
        synthetic_equivalents = [m.group(1).strip() for m in pattern.finditer(content)]
        return {"synthetic_equivalents_set": set(synthetic_equivalents)}
    
    def evaluate_step(self, pred_parsed: Dict[str, Any], gt_parsed: Dict[str, Any]) -> Dict[str, int]:
        return {
            # pred_parsed set is a subset of gt_parsed set
            "retro/step7/has_synthetic_equivalents": 1 if pred_parsed["synthetic_equivalents_set"] <= gt_parsed["synthetic_equivalents_set"] else 0,
        }
    

# Reagent prediction step evaluators
class ReagentStep6Evaluator(BaseStepEvaluator):
    """Step 6: Analyze starting materials and products."""
    
    def parse_step(self, step: StepData) -> Dict[str, Any]:
        content = self._get_content(step)
        
        if step.verify and step.reflection:
            candidate_blocks = content.split("It should be")[-1].split("\n")[-1].strip()
        else:
            candidate_blocks = content.split("The candidate reagents are as follows:")[-1].strip()

        candidate_blocks_set = set()
        for line in candidate_blocks.splitlines():
            candidate_block = line.strip()
            if not candidate_block:
                continue
            # Match "number. SMILES"
            m = re.match(r"^\d+\.\s*(.+)$", line)
            if m:
                candidate_blocks_set.add(m.group(1).strip())
        
        return {"candidate_blocks_set": candidate_blocks_set, "answer": set(step.metadata["answer"])}
    
    def evaluate_step(self, pred_parsed: Dict[str, Any], gt_parsed: Dict[str, Any]) -> Dict[str, int]:
        return {
            # pred_parsed set is a subset of gt_parsed set
            "reagent/step6/has_correct_reagent_numberials": 1 if gt_parsed["answer"] <= pred_parsed["candidate_blocks_set"] else 0,
        }


# Task-specific evaluator factories
class TaskEvaluatorFactory:
    """Factory to create appropriate evaluators for each task type."""
    
    @staticmethod
    def create_evaluators(task_type: TaskType) -> Dict[str, BaseStepEvaluator]:
        """Create step evaluators for the given task type."""
        
        if task_type == TaskType.FORWARD:
            return {
                "step_4": ForwardStep4Evaluator("step_4"),
                "step_5": ForwardStep5Evaluator("step_5"),
                "step_6": ForwardStep6Evaluator("step_6"),
                # Add more forward steps as needed
            }
        
        elif task_type == TaskType.RETRO:
            return {
                "step_5": RetroStep5Evaluator("step_5"),
                "step_6": RetroStep6Evaluator("step_6"),
                "step_7": RetroStep7Evaluator("step_7"),
                # Add more retro steps as needed
            }
        
        elif task_type == TaskType.REAGENT:
            return {
                "step_6": ReagentStep6Evaluator("step_6"),
                # Add more reagent steps as needed
            }
        
        else:
            raise ValueError(f"Unknown task type: {task_type}")

class TaskEvaluator:
    """Main evaluator that coordinates different task types."""
    
    def __init__(self):
        self.evaluator_cache = {}
    
    def _get_evaluators(self, task_type: TaskType) -> Dict[str, BaseStepEvaluator]:
        """Get or create evaluators for the task type."""
        if task_type not in self.evaluator_cache:
            self.evaluator_cache[task_type] = TaskEvaluatorFactory.create_evaluators(task_type)
        return self.evaluator_cache[task_type]
    
    def evaluate_task(self, task_type: TaskType, pred_steps: Dict[str, StepData], gt_steps: Dict[str, StepData]) -> Dict[str, Any]:
        """Evaluate any task type with its specific steps."""
        metrics = {}
        evaluators = self._get_evaluators(task_type)
        
        for step_key, evaluator in evaluators.items():
            if step_key in pred_steps and step_key in gt_steps:
                step_metrics = evaluator.evaluate(pred_steps[step_key], gt_steps[step_key])
                metrics.update(step_metrics)
        
        return metrics


class StepEvaluator():
    def __init__(self):
        pass

    def parse_reactant_reagent_in_forward(self, input_str):
        """
        Parses a string of the format "... reactant XX ... reagent YY" and extracts XX and YY.

        Returns:
            (reactant, reagent): tuple of strings, or (None, None) if not found.
        """
        reactant_match = re.search(r"reactant\s+([^\s]+)", input_str, re.IGNORECASE)
        reagent_match = re.search(r"reagent\s+([^\s]+)", input_str, re.IGNORECASE)
        reactant = reactant_match.group(1) if reactant_match else None
        reagent = reagent_match.group(1) if reagent_match else None
        
        reactant = reactant.replace(",", "") if reactant else None
        reagent = reagent.replace(",", "") if reagent else None
        reactant = reactant.replace("`", "") if reactant else None
        reagent = reagent.replace("`", "") if reagent else None
        reactant = reactant.replace("**", "") if reactant else None
        reagent = reagent.replace("**", "") if reagent else None
        reactant = reactant.replace("<", "") if reactant else None
        reagent = reagent.replace(">", "") if reagent else None
        reactant = reactant.replace("*", "") if reactant else None
        reagent = reagent.replace("*", "") if reagent else None

        reactant = reactant.strip(".") if reactant else None
        reagent = reagent.strip(".") if reagent else None
        return reactant, reagent


    def parse_reactant_product_in_reagent(self, input_str):
        """
        Parses a string of the format "... reactant is XX ... product is YY" and extracts XX and YY.

        Returns:
            (reactant, product): tuple of strings, or (None, None) if not found.
        """
        reactant_match = re.search(r"reactant\s+is\s+([^\s]+)", input_str, re.IGNORECASE)
        product_match = re.search(r"product\s+is\s+([^\s]+)", input_str, re.IGNORECASE)
        reactant = reactant_match.group(1) if reactant_match else None
        product = product_match.group(1) if product_match else None
        
        reactant = reactant.replace(",", "") if reactant else None
        product = product.replace(",", "") if product else None
        reactant = reactant.replace("`", "") if reactant else None
        product = product.replace("`", "") if product else None
        reactant = reactant.replace("**", "") if reactant else None
        product = product.replace("**", "") if product else None
        reactant = reactant.replace("<", "") if reactant else None
        product = product.replace(">", "") if product else None
        reactant = reactant.replace("*", "") if reactant else None
        product = product.replace("*", "") if product else None

        reactant = reactant.strip(".") if reactant else None
        product = product.strip(".") if product else None
        return reactant, product


    def extract_numbered_items(self, text: str) -> list:
        pattern = re.compile(r"^\s*\d+\.\s*(.+)$", re.MULTILINE)
        items = pattern.findall(text)
        return [item.strip().strip('`') for item in items]


    def parse_steps_with_reflections(self, text: str) -> List[Dict]:
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


    def calculate_forward_rationale_metrics(self, info, predicted_rationale):
        predicted_step4_rationale = ""
        predicted_step5_rationale = ""
        predicted_step6_rationale = ""
        has_tagged_smiles = False
        steps_data = self.parse_steps_with_reflections(predicted_rationale)
        reflection_bonus = 0.
        for step_key, step_info in steps_data.items():
            if step_info["step"] == 4:
                has_reflection = len(step_info["reflections"]) > 0
                if has_reflection:
                    predicted_step4_rationale = step_info["reflections"][-1]
                    reflection_bonus += 0.5 / 3
                else:
                    predicted_step4_rationale = step_info["content"]
            elif step_info["step"] == 5:
                has_reflection = len(step_info["reflections"]) > 0
                if has_reflection:
                    predicted_step5_rationale = step_info["reflections"][-1]
                    reflection_bonus += 0.5 / 3
                else:
                    predicted_step5_rationale = step_info["content"]
            elif step_info["step"] == 6:
                has_reflection = len(step_info["reflections"]) > 0
                if has_reflection:
                    predicted_step6_rationale = step_info["reflections"][-1]
                    has_tagged_smiles = ".".join(info["products"]) in predicted_step6_rationale
                    reflection_bonus += 0.5 / 3
                else:
                    predicted_step6_rationale = step_info["content"]
                    has_tagged_smiles = info["product_changes_tagged"] in predicted_step6_rationale

        # Metric 3: SMILES highlighting bonding atoms
        has_reactive_atoms_smiles = info["reactive_atoms_smiles_str"] in predicted_step4_rationale

        # Metric 4: Reactive atom bonds
        # Check all of the str(tuple(info['reactive_atom_bonds'][0]) in predicted_reasoning
        has_reactive_atom_bonds = all(str(tuple(bond)) in predicted_step5_rationale for bond in info['reactive_atom_bonds'])

        return {
            "forward/step4/has_reactive_atoms_smiles": int(has_reactive_atoms_smiles),
            "forward/step5/has_reactive_atom_bonds": int(has_reactive_atom_bonds),
            "forward/step6/has_tagged_smiles": int(has_tagged_smiles),
        }, reflection_bonus


    def calculate_retro_rationale_metrics(self, info, predicted_rationale):
        predicted_step5_rationale = ""
        predicted_step6_rationale = ""
        predicted_step7_rationale = ""

        steps_data = self.parse_steps_with_reflections(predicted_rationale)
        reflection_bonus = 0.
        for step_key, step_info in steps_data.items():
            if step_info["step"] == 5:
                has_reflection = len(step_info["reflections"]) > 0
                if has_reflection:
                    predicted_step5_rationale = step_info["reflections"][-1]
                    reflection_bonus += 0.5 / 3
                else:
                    predicted_step5_rationale = step_info["content"]
            elif step_info["step"] == 6:
                has_reflection = len(step_info["reflections"]) > 0
                if has_reflection:
                    predicted_step6_rationale = step_info["reflections"][-1]
                    reflection_bonus += 0.5 / 3
                else:
                    predicted_step6_rationale = step_info["content"]
            elif step_info["step"] == 7:
                has_reflection = len(step_info["reflections"]) > 0
                if has_reflection:
                    predicted_step7_rationale = step_info["reflections"][-1]
                    reflection_bonus += 0.5 / 3
                else:
                    predicted_step7_rationale = step_info["content"]


        # Metric 4: Bond disconnected
        bond_disconnection_list = []
        for bond in info["bond_list"]:
            bond_str = f"{bond[0]}, {bond[1]}: {bond[2]}"
            bond_disconnection_list.append(bond_str)
        has_bond_disconnection = all(bond_str in predicted_step5_rationale for bond_str in bond_disconnection_list)

        # Metric 5: Synthons
        has_synthons = all(synthon in predicted_step6_rationale for synthon in info["synthons_list"])

        # Metric 6: Synthetic equivalents
        has_synthetic_equivalents = all(syn_equiv in predicted_step7_rationale for syn_equiv in info["synthetic_equivalents"])

        return {
            "retro/step5/has_bond_disconnection": int(has_bond_disconnection),
            "retro/step6/has_synthons": int(has_synthons),
            "retro/step7/has_synthetic_equivalents": int(has_synthetic_equivalents),
        }, reflection_bonus


    def calculate_reagent_rationale_metrics(self, info, predicted_rationale):
        predicted_step6_rationale = ""
        predicted_step7_rationale = ""

        steps_data = self.parse_steps_with_reflections(predicted_rationale)
        for step_key, step_info in steps_data.items():
            if step_info["step"] == 6:
                has_reflection = len(step_info["reflections"]) > 0
                if has_reflection:
                    predicted_step6_rationale = step_info["reflections"][-1]
                    reflection_bonus += 0.5 / 1
                else:
                    predicted_step6_rationale = step_info["content"]
            elif step_info["step"] == 7:
                has_reflection = len(step_info["reflections"]) > 0
                if has_reflection:
                    predicted_step7_rationale = step_info["reflections"][-1]
                else:
                    predicted_step7_rationale = step_info["content"]

        # Metric 3: Has reagents
        reagent_list = self.extract_numbered_items(predicted_step6_rationale)
        reagent_gt = ".".join(info["reagents"])
        has_reagents = False
        for reagent_pred in reagent_list:
            if SMILESValidator.exact_match(reagent_pred, reagent_gt):
                has_reagents = True
                break

        # Metric 4: Correct reagent number
        correct_reagent_number = -1
        for idx, reagent_pred in enumerate(reagent_list):
            if SMILESValidator.exact_match(reagent_pred, ".".join(info["reagents"])):
                correct_reagent_number = idx + 1
                break
        reagent_num = re.search(r"reagent (\d+)", predicted_step7_rationale, re.IGNORECASE)
        if reagent_num:
            predicted_reagent_number = int(reagent_num.group(1))
            has_correct_reagent_number = (predicted_reagent_number == correct_reagent_number) and has_reagents
        else:
            has_correct_reagent_number = False

        return {
            "reagent/step6/has_reagents": int(has_reagents),
            # "reagent/step7/has_correct_reagent_number": int(has_correct_reagent_number),
        }, reflection_bonus

    """
    def evaluate(self, info_list, GT_rationale_list, predicted_reasoning_list, task):
        if "forward" in task:
            forward_metrics_dict = defaultdict(list)
            for info, GT_rationale, predicted_reasoning in zip(info_list, GT_rationale_list, predicted_reasoning_list):
                forward_metrics = self.calculate_forward_rationale_metrics(info, GT_rationale, predicted_reasoning)
                for key, value in forward_metrics.items():
                    forward_metrics_dict[key].append(value)
            metric_dict = {}
            for key, values in forward_metrics_dict.items():
                metric_dict[key] = sum(values) / len(values) if values else 0.0
            return metric_dict

        elif "retro" in task:
            retro_metrics_dict = defaultdict(list)
            for info, GT_rationale, predicted_reasoning in zip(info_list, GT_rationale_list, predicted_reasoning_list):
                retro_metrics = self.calculate_retro_rationale_metrics(info, GT_rationale, predicted_reasoning)
                for key, value in retro_metrics.items():
                    retro_metrics_dict[key].append(value)
            metric_dict = {}
            for key, values in retro_metrics_dict.items():
                metric_dict[key] = sum(values) / len(values) if values else 0.0
            return metric_dict

        elif "reagent" in task:
            reagent_metrics_dict = defaultdict(list)
            for info, GT_rationale, predicted_reasoning in zip(info_list, GT_rationale_list, predicted_reasoning_list):
                reagent_metrics = self.calculate_reagent_rationale_metrics(info, GT_rationale, predicted_reasoning)
                for key, value in reagent_metrics.items():
                    reagent_metrics_dict[key].append(value)
            metric_dict = {}
            for key, values in reagent_metrics_dict.items():
                metric_dict[key] = sum(values) / len(values) if values else 0.0
            return metric_dict
    """

class ChemistryEvaluator:
    """Main evaluation interface."""
    
    def __init__(self):
        self.task_evaluator = TaskEvaluator()
        self.validator = SMILESValidator()
        self.parser = StepParser()
        self.step_evaluator = StepEvaluator()
    
    def is_correct_strict_tag(self, pred: str, gt: str) -> Tuple[int, Optional[str]]:
        """Check prediction correctness using strict ANSWER tag criteria."""
        extracted_pred = TextProcessor.extract_answer_tag(pred) if pred else None
        extracted_gt = TextProcessor.extract_answer_tag(gt) if gt else None
        
        is_correct = self.validator.exact_match(extracted_pred, extracted_gt) if extracted_pred and extracted_gt else False
        return (1 if is_correct else 0), extracted_pred
    
    def verify(self, solution_str: str, ground_truth: str, extra_info: Optional[Dict] = None) -> Tuple[bool, Optional[str]]:
        """Verify if the solution is correct."""
        correct_score, pred = self.is_correct_strict_tag(solution_str, ground_truth)
        return correct_score == 1, pred
    
    def compute_score(self, solution_str: str, ground_truth: str, extra_info: Optional[Dict] = None) -> Union[EvaluationResult, Dict[str, Any]]:
        """Compute comprehensive evaluation score."""
        correct, pred = self.verify(solution_str, ground_truth, extra_info)

        task = extra_info['task']
        info = {
            "products": extra_info['products'],
            "reactants": extra_info['reactants'],
            "reagents": extra_info['reagents'],
        }
        if "reagents" in extra_info["supporting_info"]['reagent']:
            del extra_info["supporting_info"]['reagent']['reagents']
        # extra_info["supporting_info"]['reagent']['reagent_list'] = extra_info["supporting_info"]['reagent']['reagents']
        # del extra_info["supporting_info"]['reagent']['reagents']
        info.update(extra_info["supporting_info"][task])
        match = re.search(r'<think>(.*?)</think>', solution_str, re.DOTALL)
        predicted_rationale = ""
        if match:
            predicted_rationale = match.group(1).strip()
        if task == "forward":
            step_eval_results, reflection_bonus = self.step_evaluator.calculate_forward_rationale_metrics(info, predicted_rationale)
        elif task == "retro":
            step_eval_results, reflection_bonus = self.step_evaluator.calculate_retro_rationale_metrics(info, predicted_rationale)
        elif task == "reagent":
            step_eval_results, reflection_bonus = self.step_evaluator.calculate_reagent_rationale_metrics(info, predicted_rationale)
        else:
            step_eval_results = {}

        reward = correct + (sum(step_eval_results.values()) / len(step_eval_results) if step_eval_results else 0.0) + reflection_bonus

        result = EvaluationResult(
            score=reward,
            acc=correct,
            pred=pred,
            metrics=step_eval_results
        )

        # Return dict format for compatibility with existing code
        return result.to_dict()


def compute_score(solution_str: str, ground_truth: str, extra_info: Optional[dict] = None) -> Dict[str, Any]:
    evaluator = ChemistryEvaluator()
    result = evaluator.compute_score(solution_str, ground_truth, extra_info)
    # result is already a dict from compute_score method
    return result
