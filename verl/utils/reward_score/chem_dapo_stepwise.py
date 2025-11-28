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
from pprint import pprint

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
        step4_initial_rationale = ""
        step4_has_reflection = False
        predicted_step5_rationale = ""
        step5_initial_rationale = ""
        step5_has_reflection = False
        predicted_step6_rationale = ""
        step6_initial_rationale = ""
        step6_has_reflection = False
        has_tagged_smiles = False
        steps_data = self.parse_steps_with_reflections(predicted_rationale)
        reflection_bonus = {
            "step4": 0,
            "step5": 0,
            "step6": 0,
        }
        for step_key, step_info in steps_data.items():
            if step_info["step"] == 4:
                has_reflection = len(step_info["reflections"]) > 0
                if has_reflection:
                    predicted_step4_rationale = step_info["reflections"][-1]
                    reflection_bonus["step4"] = 1
                    step4_initial_rationale = step_info["content"]
                    step4_has_reflection = True
                else:
                    predicted_step4_rationale = step_info["content"]
                    step4_initial_rationale = step_info["content"]
                    step4_has_reflection = False
            elif step_info["step"] == 5:
                has_reflection = len(step_info["reflections"]) > 0
                if has_reflection:
                    predicted_step5_rationale = step_info["reflections"][-1]
                    reflection_bonus["step5"] = 1
                    step5_initial_rationale = step_info["content"]
                    step5_has_reflection = True
                else:
                    predicted_step5_rationale = step_info["content"]
                    step5_initial_rationale = step_info["content"]
                    step5_has_reflection = False
            elif step_info["step"] == 6:
                has_reflection = len(step_info["reflections"]) > 0
                if has_reflection:
                    predicted_step6_rationale = step_info["reflections"][-1]
                    has_tagged_smiles = ".".join(info["products"]) in predicted_step6_rationale
                    reflection_bonus["step6"] = 1
                    step6_initial_rationale = step_info["content"]
                    step6_has_reflection = True
                else:
                    predicted_step6_rationale = step_info["content"]
                    has_tagged_smiles = info["product_changes_tagged"] in predicted_step6_rationale
                    step6_initial_rationale = step_info["content"]
                    step6_has_reflection = False

        # Metric 3: SMILES highlighting bonding atoms
        has_reactive_atoms_smiles = info["reactive_atoms_smiles_str"] in predicted_step4_rationale
        has_reactive_atom_smiles_initial = info["reactive_atoms_smiles_str"] in step4_initial_rationale

        # Metric 4: Reactive atom bonds
        # Check all of the str(tuple(info['reactive_atom_bonds'][0]) in predicted_reasoning
        for i in range(len(info['reactive_atom_bonds'])):
            info['reactive_atom_bonds'][i][0] = int(info['reactive_atom_bonds'][i][0]) # convert to int for comparison
            info['reactive_atom_bonds'][i][1] = int(info['reactive_atom_bonds'][i][1]) # convert to int for comparison
            info['reactive_atom_bonds'][0][2] = info['reactive_atom_bonds'][0][2].replace("'", "") # remove extra quotes if any
        if len(info['reactive_atom_bonds']) == 0:
            has_reactive_atom_bonds = True
            for bond_type in ['single', 'double', 'triple', 'aromatic']:
                if bond_type in predicted_step5_rationale:
                    has_reactive_atom_bonds = False
                    break
        else:
            has_reactive_atom_bonds = all(str(tuple(bond)) in predicted_step5_rationale for bond in info['reactive_atom_bonds'])
        
        if len(info['reactive_atom_bonds']) == 0:
            has_reactive_atom_bonds_initial = True
            for bond_type in ['single', 'double', 'triple', 'aromatic']:
                if bond_type in step5_initial_rationale:
                    has_reactive_atom_bonds_initial = False
                    break
        else:
            has_reactive_atom_bonds_initial = all(str(tuple(bond)) in step5_initial_rationale for bond in info['reactive_atom_bonds'])
        has_tagged_smiles_initial = info["product_changes_tagged"] in step6_initial_rationale

        step4_initial_correct = has_reactive_atom_smiles_initial
        step5_initial_correct = has_reactive_atom_bonds_initial
        step6_initial_correct = has_tagged_smiles_initial
        step4_initial_incorrect = not has_reactive_atom_smiles_initial
        step5_initial_incorrect = not has_reactive_atom_bonds_initial
        step6_initial_incorrect = not has_tagged_smiles_initial

        step4_has_not_reflection = not step4_has_reflection
        step5_has_not_reflection = not step5_has_reflection
        step6_has_not_reflection = not step6_has_reflection

        # Positive: Model generates reflection
        # Negative: Model does not generate reflection
        # Calculate TP, TN, FP, FN for each step. Only check reflection is correct
        step4_TP = step4_initial_incorrect and step4_has_reflection
        step4_TN = step4_initial_correct and step4_has_not_reflection
        step4_FP = step4_initial_correct and step4_has_reflection
        step4_FN = step4_initial_incorrect and step4_has_not_reflection
        step5_TP = step5_initial_incorrect and step5_has_reflection
        step5_TN = step5_initial_correct and step5_has_not_reflection
        step5_FP = step5_initial_correct and step5_has_reflection
        step5_FN = step5_initial_incorrect and step5_has_not_reflection
        step6_TP = step6_initial_incorrect and step6_has_reflection
        step6_TN = step6_initial_correct and step6_has_not_reflection
        step6_FP = step6_initial_correct and step6_has_reflection
        step6_FN = step6_initial_incorrect and step6_has_not_reflection

        # Reflection accuracy reward
        step4_reflection_correct = step4_TP - step4_FP
        step5_reflection_correct = step5_TP - step5_FP
        step6_reflection_correct = step6_TP - step6_FP
        total_reflection_correct_list = [step4_reflection_correct, step5_reflection_correct, step6_reflection_correct]
        reflection_ratio = sum(total_reflection_correct_list) / len(total_reflection_correct_list)

        content_reward_dict = {
            "forward/step4/has_reactive_atoms_smiles": int(has_reactive_atoms_smiles),
            "forward/step5/has_reactive_atom_bonds": int(has_reactive_atom_bonds),
            "forward/step6/has_tagged_smiles": int(has_tagged_smiles),
        }

        reflection_decision_reward_dict = {
            "forward/step4/correct_reflection": int(step4_reflection_correct),
            "forward/step5/correct_reflection": int(step5_reflection_correct),
            "forward/step6/correct_reflection": int(step6_reflection_correct),
        }

        return {
            "forward/step4/has_reactive_atoms_smiles": int(has_reactive_atoms_smiles),
            "forward/step5/has_reactive_atom_bonds": int(has_reactive_atom_bonds),
            "forward/step6/has_tagged_smiles": int(has_tagged_smiles),
            "forward/step4/TP": step4_TP,
            "forward/step4/TN": step4_TN,
            "forward/step4/FP": step4_FP,
            "forward/step4/FN": step4_FN,
            "forward/step5/TP": step5_TP,
            "forward/step5/TN": step5_TN,
            "forward/step5/FP": step5_FP,
            "forward/step5/FN": step5_FN,
            "forward/step6/TP": step6_TP,
            "forward/step6/TN": step6_TN,
            "forward/step6/FP": step6_FP,
            "forward/step6/FN": step6_FN,
            "forward/step4/correct_reflection": int(step4_reflection_correct),
            "forward/step5/correct_reflection": int(step5_reflection_correct),
            "forward/step6/correct_reflection": int(step6_reflection_correct),
            "forward/total_reflection_ratio": reflection_ratio,
        }, reflection_bonus, content_reward_dict, reflection_decision_reward_dict


    def calculate_retro_rationale_metrics(self, info, predicted_rationale):
        predicted_step5_rationale = ""
        step5_initial_rationale = ""
        step5_has_reflection = False
        predicted_step6_rationale = ""
        step6_initial_rationale = ""
        step6_has_reflection = False
        predicted_step7_rationale = ""
        step7_initial_rationale = ""
        step7_has_reflection = False

        steps_data = self.parse_steps_with_reflections(predicted_rationale)
        reflection_bonus = {
            "step5": 0,
            "step6": 0,
            "step7": 0,
        }
        for step_key, step_info in steps_data.items():
            if step_info["step"] == 5:
                has_reflection = len(step_info["reflections"]) > 0
                if has_reflection:
                    predicted_step5_rationale = step_info["reflections"][-1]
                    reflection_bonus["step5"] = 1
                    step5_initial_rationale = step_info["content"]
                    step5_has_reflection = True
                else:
                    predicted_step5_rationale = step_info["content"]
                    step5_initial_rationale = step_info["content"]
                    step5_has_reflection = False
            elif step_info["step"] == 6:
                has_reflection = len(step_info["reflections"]) > 0
                if has_reflection:
                    predicted_step6_rationale = step_info["reflections"][-1]
                    reflection_bonus["step6"] = 1
                    step6_initial_rationale = step_info["content"]
                    step6_has_reflection = True
                else:
                    predicted_step6_rationale = step_info["content"]
                    step6_initial_rationale = step_info["content"]
                    step6_has_reflection = False
            elif step_info["step"] == 7:
                has_reflection = len(step_info["reflections"]) > 0
                if has_reflection:
                    predicted_step7_rationale = step_info["reflections"][-1]
                    reflection_bonus["step7"] = 1
                    step7_initial_rationale = step_info["content"]
                    step7_has_reflection = True
                else:
                    predicted_step7_rationale = step_info["content"]
                    step7_initial_rationale = step_info["content"]
                    step7_has_reflection = False

        # Metric 4: Bond disconnected
        bond_disconnection_list = []
        for bond in info["bond_list"]:
            bond_str = f"{bond[0]}, {bond[1]}: {bond[2]}"
            bond_disconnection_list.append(bond_str)
        has_bond_disconnection = all(bond_str in predicted_step5_rationale for bond_str in bond_disconnection_list)
        has_bond_disconnection_initial = all(bond_str in step5_initial_rationale for bond_str in bond_disconnection_list)

        # Metric 5: Synthons
        has_synthons = all(synthon in predicted_step6_rationale for synthon in info["synthons_list"])
        has_synthons_initial = all(synthon in step6_initial_rationale for synthon in info["synthons_list"])

        # Metric 6: Synthetic equivalents
        has_synthetic_equivalents = all(syn_equiv in predicted_step7_rationale for syn_equiv in info["synthetic_equivalents"])
        has_synthetic_equivalents_initial = all(syn_equiv in step7_initial_rationale for syn_equiv in info["synthetic_equivalents"])

        step5_initial_correct = has_bond_disconnection_initial
        step6_initial_correct = has_synthons_initial
        step7_initial_correct = has_synthetic_equivalents_initial
        step5_initial_incorrect = not has_bond_disconnection_initial
        step6_initial_incorrect = not has_synthons_initial
        step7_initial_incorrect = not has_synthetic_equivalents_initial

        step5_has_not_reflection = not step5_has_reflection
        step6_has_not_reflection = not step6_has_reflection
        step7_has_not_reflection = not step7_has_reflection

        # Positive: Model generates reflection
        # Negative: Model does not generate reflection
        # Calculate TP, TN, FP, FN for each step. Only check reflection is correct
        step5_TP = step5_initial_incorrect and step5_has_reflection
        step5_TN = step5_initial_correct and step5_has_not_reflection
        step5_FP = step5_initial_correct and step5_has_reflection
        step5_FN = step5_initial_incorrect and step5_has_not_reflection
        step6_TP = step6_initial_incorrect and step6_has_reflection
        step6_TN = step6_initial_correct and step6_has_not_reflection
        step6_FP = step6_initial_correct and step6_has_reflection
        step6_FN = step6_initial_incorrect and step6_has_not_reflection
        step7_TP = step7_initial_incorrect and step7_has_reflection
        step7_TN = step7_initial_correct and step7_has_not_reflection
        step7_FP = step7_initial_correct and step7_has_reflection
        step7_FN = step7_initial_incorrect and step7_has_not_reflection

        # Reflection accuracy reward
        step5_reflection_correct = step5_TP - step5_FP
        step6_reflection_correct = step6_TP - step6_FP
        step7_reflection_correct = step7_TP - step7_FP
        total_reflection_correct_list = [step5_reflection_correct, step6_reflection_correct, step7_reflection_correct]
        reflection_ratio = sum(total_reflection_correct_list) / len(total_reflection_correct_list)

        content_reward_dict = {
            "retro/step5/has_bond_disconnection": int(has_bond_disconnection),
            "retro/step6/has_synthons": int(has_synthons),
            "retro/step7/has_synthetic_equivalents": int(has_synthetic_equivalents),
        }

        reflection_decision_reward_dict = {
            "retro/step5/correct_reflection": int(step5_reflection_correct),
            "retro/step6/correct_reflection": int(step6_reflection_correct),
            "retro/step7/correct_reflection": int(step7_reflection_correct),
        }

        return {
            "retro/step5/has_bond_disconnection": int(has_bond_disconnection),
            "retro/step6/has_synthons": int(has_synthons),
            "retro/step7/has_synthetic_equivalents": int(has_synthetic_equivalents),
            "retro/step5/TP": step5_TP,
            "retro/step5/TN": step5_TN,
            "retro/step5/FP": step5_FP,
            "retro/step5/FN": step5_FN,
            "retro/step6/TP": step6_TP,
            "retro/step6/TN": step6_TN,
            "retro/step6/FP": step6_FP,
            "retro/step6/FN": step6_FN,
            "retro/step7/TP": step7_TP,
            "retro/step7/TN": step7_TN,
            "retro/step7/FP": step7_FP,
            "retro/step7/FN": step7_FN,
            "retro/step5/correct_reflection": int(step5_reflection_correct),
            "retro/step6/correct_reflection": int(step6_reflection_correct),
            "retro/step7/correct_reflection": int(step7_reflection_correct),
            "retro/total_reflection_ratio": reflection_ratio,
        }, reflection_bonus, content_reward_dict, reflection_decision_reward_dict


    def calculate_reagent_rationale_metrics(self, info, predicted_rationale):
        predicted_step6_rationale = ""
        step6_initial_rationale = ""
        step6_has_reflection = False
        predicted_step7_rationale = ""
        step7_initial_rationale = ""
        step7_has_reflection = False

        steps_data = self.parse_steps_with_reflections(predicted_rationale)
        reflection_bonus = {
            "step6": 0,
            "step7": 0,
        }
        for step_key, step_info in steps_data.items():
            if step_info["step"] == 6:
                has_reflection = len(step_info["reflections"]) > 0
                if has_reflection:
                    predicted_step6_rationale = step_info["reflections"][-1]
                    reflection_bonus["step6"] = 1
                    step6_initial_rationale = step_info["content"]
                    step6_has_reflection = True
                else:
                    predicted_step6_rationale = step_info["content"]
                    step6_initial_rationale = step_info["content"]
                    step6_has_reflection = False
            elif step_info["step"] == 7:
                has_reflection = len(step_info["reflections"]) > 0
                if has_reflection:
                    predicted_step7_rationale = step_info["reflections"][-1]
                    reflection_bonus["step7"] = 1
                    step7_initial_rationale = step_info["content"]
                    step7_has_reflection = True
                else:
                    predicted_step7_rationale = step_info["content"]
                    step7_initial_rationale = step_info["content"]
                    step7_has_reflection = False

        # Metric 3: Has reagents
        reagent_list = self.extract_numbered_items(predicted_step6_rationale)
        reagent_gt = ".".join(info["reagents"])
        has_reagents = False
        for reagent_pred in reagent_list:
            if SMILESValidator.exact_match(reagent_pred, reagent_gt):
                has_reagents = True
                break

        reagent_list_initial = self.extract_numbered_items(step6_initial_rationale)
        reagent_gt = ".".join(info["reagents"])
        has_reagents_initial = False
        for reagent_pred in reagent_list_initial:
            if SMILESValidator.exact_match(reagent_pred, reagent_gt):
                has_reagents_initial = True
                break

        if step6_has_reflection:
            # any of the initial reagents overlaps with reflection reagents, has_reagents=False
            if len(set(reagent_list) & set(reagent_list_initial)) > 0:
                has_reagents = False


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

        correct_reagent_number_initial = -1
        for idx, reagent_pred in enumerate(reagent_list_initial):
            if SMILESValidator.exact_match(reagent_pred, ".".join(info["reagents"])):
                correct_reagent_number_initial = idx + 1
                break
        reagent_num_initial = re.search(r"reagent (\d+)", step7_initial_rationale, re.IGNORECASE)
        if reagent_num_initial:
            predicted_reagent_number_initial = int(reagent_num_initial.group(1))
            has_correct_reagent_number_initial = (predicted_reagent_number_initial == correct_reagent_number_initial) and has_reagents_initial
        else:
            has_correct_reagent_number_initial = False

        step6_initial_correct = has_reagents_initial
        step7_initial_correct = has_correct_reagent_number_initial
        step6_initial_incorrect = not has_reagents_initial
        step7_initial_incorrect = not has_correct_reagent_number_initial

        step6_has_not_reflection = not step6_has_reflection
        step7_has_not_reflection = not step7_has_reflection

        # Positive: Model generates reflection
        # Negative: Model does not generate reflection
        # Calculate TP, TN, FP, FN for each step. Only check reflection is correct
        step6_TP = step6_initial_incorrect and step6_has_reflection
        step6_TN = step6_initial_correct and step6_has_not_reflection
        step6_FP = step6_initial_correct and step6_has_reflection
        step6_FN = step6_initial_incorrect and step6_has_not_reflection
        step7_TP = step7_initial_incorrect and step7_has_reflection
        step7_TN = step7_initial_correct and step7_has_not_reflection
        step7_FP = step7_initial_correct and step7_has_reflection
        step7_FN = step7_initial_incorrect and step7_has_not_reflection

        # Reflection accuracy reward
        # step6_reflection_correct = step6_TP + step6_TN - step6_FP - step6_FN
        # step7_reflection_correct = step7_TP + step7_TN - step7_FP - step7_FN
        step6_reflection_correct = step6_TP - step6_FP
        step7_reflection_correct = step7_TP - step7_FP
        total_reflection_correct_list = [step6_reflection_correct, step7_reflection_correct]
        reflection_ratio = sum(total_reflection_correct_list) / len(total_reflection_correct_list)

        content_reward_dict = {
            "reagent/step6/has_reagents": int(has_reagents),
            "reagent/step7/has_correct_reagent_number": int(has_correct_reagent_number),
        }

        reflection_decision_reward_dict = {
            "reagent/step6/correct_reflection": int(step6_reflection_correct),
            "reagent/step7/correct_reflection": int(step7_reflection_correct),
        }

        return {
            "reagent/step6/has_reagents": int(has_reagents),
            "reagent/step7/has_correct_reagent_number": int(has_correct_reagent_number),
            "reagent/step6/TP": step6_TP,
            "reagent/step6/TN": step6_TN,
            "reagent/step6/FP": step6_FP,
            "reagent/step6/FN": step6_FN,
            "reagent/step7/TP": step7_TP,
            "reagent/step7/TN": step7_TN,
            "reagent/step7/FP": step7_FP,
            "reagent/step7/FN": step7_FN,
            "reagent/step6/correct_reflection": int(step6_reflection_correct),
            "reagent/step7/correct_reflection": int(step7_reflection_correct),
            "reagent/total_reflection_ratio": reflection_ratio,
        }, reflection_bonus, content_reward_dict, reflection_decision_reward_dict


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

    def validate_structure(self, text: str, task: str) -> bool:
        # 1. <think> ... </think> 존재 여부 확인
        think_match = re.search(r"<think>(.*?)</think>", text, re.DOTALL)
        if not think_match:
            return False
        think_content = think_match.group(1)

        # 2. <ANSWER> ... </ANSWER> 존재 여부 확인
        answer_match = re.findall(r"<ANSWER>(.*?)</ANSWER>", text, re.DOTALL)
        if len(answer_match) != 1:
            return False

        # 3. Step 패턴 찾기
        step_pattern = r"## Step (\d+):"
        steps = re.findall(step_pattern, think_content)

        if not steps:
            return False

        # Step 번호가 1부터 N까지 순서대로인지 확인
        steps = list(map(int, steps))
        N = len(steps)
        if steps != list(range(1, N + 1)):
            return False

        # Step별 블록을 분할
        step_blocks = re.split(step_pattern, think_content)[1:]  # [num1, block1, num2, block2 ...]
        # step_blocks 구조: [step_num1, step_content1, step_num2, step_content2, ...]

        # 4. 각 step에서 reflection 개수 확인 (0 또는 1개)
        reflections_by_step = {}
        for i in range(0, len(step_blocks), 2):
            step_num = int(step_blocks[i])
            step_content = step_blocks[i + 1]
            ref_blocks = re.findall(r"<REFLECTION>(.*?)</REFLECTION>", step_content, re.DOTALL)
            if len(ref_blocks) > 1:
                return False  # 하나 step에 reflection 2개 이상 금지
            reflections_by_step[step_num] = len(ref_blocks)

        # ---------------------------------------------------------------
        # 5. task 규칙에 따른 reflection step 존재 검증
        # ---------------------------------------------------------------

        if task == "forward":
            allowed_steps = {4, 5, 6}
        elif task == "retro":
            allowed_steps = {5, 6, 7}
        elif task == "reagent":
            allowed_steps = {6, 7}
        else:
            raise ValueError(f"Unknown task type: {task}")

        for step, count in reflections_by_step.items():
            if count == 1 and step not in allowed_steps:
                return False

        return True 


    def compute_score(self,
                      solution_str: str,
                      ground_truth: str,
                      extra_info: Optional[Dict] = None,
                      use_content_reward=False,
                      use_decision_reward=False,
                      use_reflection_bonus=False,
                      reflection_bonus_weight=0.0
                      ) -> Union[EvaluationResult, Dict[str, Any]]:
        """Compute comprehensive evaluation score."""
        task = extra_info['task']
        correct_structure = self.validate_structure(solution_str, task)
        correct, pred = self.verify(solution_str, ground_truth, extra_info)

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
            step_eval_results, reflection_bonus_dict, content_reward_dict, reflection_decision_reward_dict = self.step_evaluator.calculate_forward_rationale_metrics(info, predicted_rationale)
        elif task == "retro":
            step_eval_results, reflection_bonus_dict, content_reward_dict, reflection_decision_reward_dict = self.step_evaluator.calculate_retro_rationale_metrics(info, predicted_rationale)
        elif task == "reagent":
            step_eval_results, reflection_bonus_dict, content_reward_dict, reflection_decision_reward_dict = self.step_evaluator.calculate_reagent_rationale_metrics(info, predicted_rationale)
        else:
            step_eval_results = {}
            reflection_bonus_dict = {
                "DUMMY": 0,
            }

        reward = correct

        step_rewards = {
            "step1": 0.0,
            "step2": 0.0,
            "step3": 0.0,
            "step4": 0.0,
            "step5": 0.0,
            "step6": 0.0,
            "step7": 0.0,
            "answer": float(correct),
        }

        reward_metric_dict = {
            f"{task}/reward/answer_correct": correct,
            f"{task}/reward/valid_structure": int(correct_structure),
        }
        if not correct_structure:
            reward -= 2.0
        if use_content_reward:
            content_reward = sum(content_reward_dict.values()) / len(content_reward_dict)
            reward += content_reward
            reward_metric_dict[f"{task}/reward/content_reward"] = content_reward
            for step_key in content_reward_dict:
                if "step1" in step_key:
                    step_rewards["step1"] += content_reward_dict[step_key] / len(content_reward_dict)
                elif "step2" in step_key:
                    step_rewards["step2"] += content_reward_dict[step_key] / len(content_reward_dict)
                elif "step3" in step_key:
                    step_rewards["step3"] += content_reward_dict[step_key] / len(content_reward_dict)
                elif "step4" in step_key:
                    step_rewards["step4"] += content_reward_dict[step_key] / len(content_reward_dict)
                elif "step5" in step_key:
                    step_rewards["step5"] += content_reward_dict[step_key] / len(content_reward_dict)
                elif "step6" in step_key:
                    step_rewards["step6"] += content_reward_dict[step_key] / len(content_reward_dict)
                elif "step7" in step_key:
                    step_rewards["step7"] += content_reward_dict[step_key] / len(content_reward_dict)
        if use_decision_reward:
            decision_reward = sum(reflection_decision_reward_dict.values()) / len(reflection_decision_reward_dict)
            reward += decision_reward
            reward_metric_dict[f"{task}/reward/decision_reward"] = decision_reward
            for step_key in reflection_decision_reward_dict:
                if "step1" in step_key:
                    step_rewards["step1"] += reflection_decision_reward_dict[step_key] / len(reflection_decision_reward_dict)
                elif "step2" in step_key:
                    step_rewards["step2"] += reflection_decision_reward_dict[step_key] / len(reflection_decision_reward_dict)
                elif "step3" in step_key:
                    step_rewards["step3"] += reflection_decision_reward_dict[step_key] / len(reflection_decision_reward_dict)
                elif "step4" in step_key:
                    step_rewards["step4"] += reflection_decision_reward_dict[step_key] / len(reflection_decision_reward_dict)
                elif "step5" in step_key:
                    step_rewards["step5"] += reflection_decision_reward_dict[step_key] / len(reflection_decision_reward_dict)
                elif "step6" in step_key:
                    step_rewards["step6"] += reflection_decision_reward_dict[step_key] / len(reflection_decision_reward_dict)
                elif "step7" in step_key:
                    step_rewards["step7"] += reflection_decision_reward_dict[step_key] / len(reflection_decision_reward_dict)
        if use_reflection_bonus:
            if len(reflection_bonus_dict) > 0:
                reflection_bonus_reward = (sum(reflection_bonus_dict.values()) / len(reflection_bonus_dict)) * reflection_bonus_weight
                for step_key in reflection_bonus_dict:
                    if "step1" in step_key:
                        step_rewards["step1"] += reflection_bonus_dict[step_key] * reflection_bonus_weight / len(reflection_bonus_dict)
                    elif "step2" in step_key:
                        step_rewards["step2"] += reflection_bonus_dict[step_key] * reflection_bonus_weight / len(reflection_bonus_dict)
                    elif "step3" in step_key:
                        step_rewards["step3"] += reflection_bonus_dict[step_key] * reflection_bonus_weight / len(reflection_bonus_dict)
                    elif "step4" in step_key:
                        step_rewards["step4"] += reflection_bonus_dict[step_key] * reflection_bonus_weight / len(reflection_bonus_dict)
                    elif "step5" in step_key:
                        step_rewards["step5"] += reflection_bonus_dict[step_key] * reflection_bonus_weight / len(reflection_bonus_dict)
                    elif "step6" in step_key:
                        step_rewards["step6"] += reflection_bonus_dict[step_key] * reflection_bonus_weight / len(reflection_bonus_dict)
                    elif "step7" in step_key:
                        step_rewards["step7"] += reflection_bonus_dict[step_key] * reflection_bonus_weight / len(reflection_bonus_dict)
            else:
                reflection_bonus_reward = 0.0
            reward += reflection_bonus_reward
            reward_metric_dict[f"{task}/reward/reflection_bonus_reward"] = reflection_bonus_reward

        # print("="*100)
        # print("=== predicted_rational ===\n", solution_str)
        # print("-"*100)
        # pprint(step_eval_results)
        # print("-"*100)
        # pprint(reflection_bonus_dict)
        # print("-"*100)
        # pprint(content_reward_dict)
        # print("-"*100)
        # pprint(reflection_decision_reward_dict)
        # print("-"*100)
        # pprint(step_rewards)
        # print("-"*100)
        # pprint(extra_info['reagents'])
        # print("="*100)

        
        step_eval_results.update(reward_metric_dict)

        result = EvaluationResult(
            score=reward,
            acc=correct,
            pred=pred,
            metrics=step_eval_results
        )
        result = result.to_dict()
        result["step_rewards"] = step_rewards
        result["reflection_decision_reward_dict"] = reflection_decision_reward_dict

        # Return dict format for compatibility with existing code
        return result


def compute_score(solution_str: str,
                  ground_truth: str,
                  extra_info: Optional[dict] = None,
                  use_content_reward=False,
                  use_decision_reward=False,
                  use_reflection_bonus=False,
                  reflection_bonus_weight=0.0
                 ) -> Dict[str, Any]:
    evaluator = ChemistryEvaluator()
    result = evaluator.compute_score(
        solution_str,
        ground_truth,
        extra_info,
        use_content_reward,
        use_decision_reward,
        use_reflection_bonus,
        reflection_bonus_weight
    )
    # result is already a dict from compute_score method
    return result
