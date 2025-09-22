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
    def parse_step_content(step_text: str, step_id: str) -> StepData:
        """Parse step content including reflections."""
        reflection_match = re.search(Tags.REFLECTION, step_text)
        reflection = reflection_match.group(1).strip() if reflection_match else None
        
        return StepData(
            step_id=step_id,
            original=step_text.strip(),
            reflection=reflection,
            verify=reflection is not None
        )

class StepParser:
    @staticmethod
    def decompose_steps(rationale: str) -> Dict[str, StepData]:
        """Extract and parse all steps from the rationale."""
        decomposed_dict = {}
        
        # Extract reasoning part (before answer tag)
        reasoning_steps = rationale.split("<ANSWER>")[0]
        
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
            step_data = TextProcessor.parse_step_content(step_content, step_id_str)
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
            highlighted_smiles = content.split("It should be")[-1].split("\n")[-1].strip()
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
            product_smiles = content.split("The product should be")[-1].split("\n")[-1].strip()
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
                bond_disconnections = content

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
                    atom1, atom2, bond_type = int(atom1), int(atom2), bond_type
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
        
        return {"candidate_blocks_set": candidate_blocks_set}
    
    def evaluate_step(self, pred_parsed: Dict[str, Any], gt_parsed: Dict[str, Any]) -> Dict[str, int]:
        return {
            # pred_parsed set is a subset of gt_parsed set
            "reagent/step6/has_correct_reagent_numberials": 1 if pred_parsed["candidate_blocks_set"] <= gt_parsed["candidate_blocks_set"] else 0,
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

class ChemistryEvaluator:
    """Main evaluation interface."""
    
    def __init__(self):
        self.task_evaluator = TaskEvaluator()
        self.validator = SMILESValidator()
        self.parser = StepParser()
    
    def is_correct_strict_tag(self, pred: str, gt: str) -> Tuple[int, Optional[str]]:
        """Check prediction correctness using strict ANSWER tag criteria."""
        extracted_pred = TextProcessor.extract_answer_tag(pred) if pred else None
        extracted_gt = TextProcessor.extract_answer_tag(gt) if gt else None
        
        is_correct = self.validator.exact_match(extracted_pred, extracted_gt) if extracted_pred and extracted_gt else False
        return (1 if is_correct else -1), extracted_pred
    
    def verify(self, solution_str: str, ground_truth: str, extra_info: Optional[Dict] = None) -> Tuple[bool, Optional[str]]:
        """Verify if the solution is correct."""
        correct_score, pred = self.is_correct_strict_tag(solution_str, ground_truth)
        return correct_score == 1, pred
    
    def compute_score(self, solution_str: str, ground_truth: str, extra_info: Optional[Dict] = None) -> Union[EvaluationResult, Dict[str, Any]]:
        """Compute comprehensive evaluation score."""
        correct, pred = self.verify(solution_str, ground_truth, extra_info)
        
        metrics = {}
        if extra_info and "task" in extra_info:
            task_type = TaskType(extra_info["task"])
            pred_steps = self.parser.decompose_steps(solution_str)
            gt_steps = self.parser.decompose_steps(ground_truth)
            metrics = self.task_evaluator.evaluate_task(task_type, pred_steps, gt_steps)
        
        reward = correct + sum(metrics.values()) if correct else -1.0

        result = EvaluationResult(
            score=reward,
            acc=reward > 0,
            pred=pred,
            metrics=metrics
        )
        
        # Return dict format for compatibility with existing code
        return result.to_dict()


def compute_score(solution_str: str, ground_truth: str, extra_info: Optional[dict] = None) -> Dict[str, Any]:
    evaluator = ChemistryEvaluator()
    result = evaluator.compute_score(solution_str, ground_truth, extra_info)
    # result is already a dict from compute_score method
    return result
