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

import inspect, re, json, ast
from typing import Optional, Dict, Any, Tuple, Set, List, Union
from dataclasses import dataclass, field
from abc import ABC, abstractmethod
from enum import Enum
from rdkit import Chem, DataStructs, RDLogger
from rdkit.Chem import MACCSkeys, AllChem, Fragments
from collections import defaultdict
from pprint import pprint
from nltk.translate.bleu_score import corpus_bleu
from openai import OpenAI

RDLogger.DisableLog('rdApp.*')

frag_to_name = {
    "fr_Al_COO": "aliphatic carboxylic acids",
    "fr_Al_OH": "aliphatic hydroxyl groups",
    "fr_Al_OH_noTert": "aliphatic hydroxyl groups excluding tert-OH",
    "fr_ArN": "N functional groups attached to aromatics",
    "fr_Ar_COO": "Aromatic carboxylic acide",
    "fr_Ar_N": "aromatic nitrogens",
    "fr_Ar_NH": "aromatic amines",
    "fr_Ar_OH": "aromatic hydroxyl groups",
    "fr_COO": "carboxylic acids",
    "fr_COO2": "carboxylic acids",
    "fr_C_O": "carbonyl O",
    "fr_C_O_noCOO": "carbonyl O, excluding COOH",
    "fr_C_S": "thiocarbonyl",
    "fr_HOCCN": "C(OH)CCN-Ctert-alkyl or C(OH)CCNcyclic",
    "fr_Imine": "Imines",
    "fr_NH0": "Tertiary amines",
    "fr_NH1": "Secondary amines",
    "fr_NH2": "Primary amines",
    "fr_N_O": "hydroxylamine groups",
    "fr_Ndealkylation1": "XCCNR groups",
    "fr_Ndealkylation2": "tert-alicyclic amines (no heteroatoms, not quinine-like bridged N)",
    "fr_Nhpyrrole": "H-pyrrole nitrogens",
    "fr_SH": "thiol groups",
    "fr_aldehyde": "aldehydes",
    "fr_alkyl_carbamate": "alkyl carbamates (subject to hydrolysis)",
    "fr_alkyl_halide": "alkyl halides",
    "fr_allylic_oxid": "allylic oxidation sites excluding steroid dienone",
    "fr_amide": "amides",
    "fr_amidine": "amidine groups",
    "fr_aniline": "anilines",
    "fr_aryl_methyl": "aryl methyl sites for hydroxylation",
    "fr_azide": "azide groups",
    "fr_azo": "azo groups",
    "fr_barbitur": "barbiturate groups",
    "fr_benzene": "benzene rings",
    "fr_benzodiazepine": "benzodiazepines with no additional fused rings",
    "fr_bicyclic": "Bicyclic",
    "fr_diazo": "diazo groups",
    "fr_dihydropyridine": "dihydropyridines",
    "fr_epoxide": "epoxide rings",
    "fr_ester": "esters",
    "fr_ether": "ether oxygens (including phenoxy)",
    "fr_furan": "furan rings",
    "fr_guanido": "guanidine groups",
    "fr_halogen": "halogens",
    "fr_hdrzine": "hydrazine groups",
    "fr_hdrzone": "hydrazone groups",
    "fr_imidazole": "imidazole rings",
    "fr_imide": "imide groups",
    "fr_isocyan": "isocyanates",
    "fr_isothiocyan": "isothiocyanates",
    "fr_ketone": "ketones",
    "fr_ketone_Topliss": "ketones excluding diaryl, a,b-unsat. dienones, heteroatom on Calpha",
    "fr_lactam": "beta lactams",
    "fr_lactone": "cyclic esters (lactones)",
    "fr_methoxy": "methoxy groups",
    "fr_morpholine": "morpholine rings",
    "fr_nitrile": "nitriles",
    "fr_nitro": "nitro groups",
    "fr_nitro_arom": "nitro benzene ring substituents",
    "fr_nitro_arom_nonortho": "non-ortho nitro benzene ring substituents",
    "fr_nitroso": "nitroso groups, excluding NO2",
    "fr_oxazole": "oxazole rings",
    "fr_oxime": "oxime groups",
    "fr_para_hydroxylation": "para-hydroxylation sites",
    "fr_phenol": "phenols",
    "fr_phenol_noOrthoHbond": "phenolic OH excluding ortho intramolecular Hbond substituents",
    "fr_phos_acid": "phosphoric acid groups",
    "fr_phos_ester": "phosphoric ester groups",
    "fr_piperdine": "piperdine rings",
    "fr_piperzine": "piperzine rings",
    "fr_priamide": "primary amides",
    "fr_prisulfonamd": "primary sulfonamides",
    "fr_pyridine": "pyridine rings",
    "fr_quatN": "quaternary nitrogens",
    "fr_sulfide": "thioether",
    "fr_sulfonamd": "sulfonamides",
    "fr_sulfone": "sulfone groups",
    "fr_term_acetylene": "terminal acetylenes",
    "fr_tetrazole": "tetrazole rings",
    "fr_thiazole": "thiazole rings",
    "fr_thiocyan": "thiocyanates",
    "fr_thiophene": "thiophene rings",
    "fr_unbrch_alkane": "unbranched alkanes of at least 4 members (excludes halogenated alkanes)",
    "fr_urea": "urea groups"
}


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


class StepEvaluatorV10():
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

        # if step6_has_reflection:
        #     # any of the initial reagents overlaps with reflection reagents, has_reagents=False
        #     if len(set(reagent_list) & set(reagent_list_initial)) > 0:
        #         has_reagents = False


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
        # step6_reflection_correct = step6_TP and has_reagents
        # step7_reflection_correct = step7_TP and has_correct_reagent_number
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



class StepEvaluator():
    def __init__(self):
        pass

    def _set_atom_mapping(self, smiles: str):
        if smiles == "":
            return smiles
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            raise ValueError("Invalid SMILES (RDKit failed to parse).")
        for atom in mol.GetAtoms():
            atom.SetAtomMapNum(atom.GetIdx() + 1)
        mapped_smiles = Chem.MolToSmiles(mol, canonical=False)
        return mapped_smiles


    def _masked_smiles(
        self,
        mol: Chem.Mol,
        keep_atoms: Tuple[int, ...]
    ) -> str:
        """
        Return a masked SMILES whose length is identical to the
        RDKit-generated (non-canonical) SMILES for `mol`.
        • 매치되지 않은 원자  → '*'
        • 그 원자와 붙어 있는 결합문자·기호·링번호 → '*'
        *문자 삭제는 일절 하지 않는다.*
        """
        _aux_chars = set("-=#:/\\.@[]") | set("()0123456789")
        # ─ 1) non-matched 원자를 dummy('*')로 변환
        rw = Chem.RWMol(mol)
        keep = set(keep_atoms)
        for idx in range(rw.GetNumAtoms()):
            if idx not in keep:
                a = rw.GetAtomWithIdx(idx)
                a.SetAtomicNum(0)          # dummy atom (*)
                a.SetIsotope(0)
                a.SetFormalCharge(0)
                a.SetNoImplicit(True)
                a.SetNumExplicitHs(0)

        # ─ 2) 입력 순서 유지(canonical=False)로 SMILES 생성
        smi = Chem.MolToSmiles(
            rw.GetMol(),
            canonical=False,
            isomericSmiles=True
        ).replace(':', '')

        # ─ 3) 결합/괄호/링번호 치환(삭제 X)
        chars = list(smi)
        n = len(chars)

        for i, ch in enumerate(chars):
            if ch in _aux_chars:
                left_x  = (i > 0     and chars[i - 1] == '*')
                right_x = (i < n - 1 and chars[i + 1] == '*')
                if left_x or right_x:      # ‘*’와 인접하면 마스킹
                    chars[i] = '*'

        return ''.join(chars).replace("*", "_")

    def _maccs_pattern_and_thr(self, bit_id: int) -> Tuple[str | None, int]:
        smarts, thr = MACCSkeys.smartsPatts.get(bit_id, (None, 0))
        return (None if smarts in (None, "?") else smarts), thr

    def _dedupe(self, matches: Tuple[Tuple[int, ...], ...]) -> List[Tuple[int, ...]]:
        seen, out = set(), []
        for atoms in matches:
            key = tuple(sorted(atoms))
            if key not in seen:
                seen.add(key)
                out.append(atoms)
        return out

    def _fragement_subs_info(
        self,
        smiles: str,
        *,
        include_rdkit: bool = True,
        include_maccs: bool = True,
        include_no_pattern: bool = False,         # MACCS only
        return_indices: bool = True,
        return_subsmiles: bool = True,
        return_masked_smiles: bool = True,        # ★ NEW ★
    ) -> Dict[str, Any]:
        """
        Combine RDKit fragment counters and MACCS key explanations.

        Each match entry may now contain:
            * atom_ids          – 1-based indices         (optional)
            * smiles            – substructure SMILES     (optional)
            * masked_smiles     – outside atoms → 'X'     (optional)
        """
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            raise ValueError(f"Invalid SMILES: {smiles}")

        results: Dict[str, Any] = {}

        # ─ RDKit fragment counters ──────────────────────────
        if include_rdkit:
            rd_out = {}
            need_details = return_indices or return_subsmiles or return_masked_smiles

            for name, fn in self._iter_fragment_funcs():
                count = fn(mol)
                if not count:
                    continue

                entry = {"count": count}
                if need_details:
                    patt = self._frag_pattern(fn)
                    if patt is not None:
                        hits = mol.GetSubstructMatches(patt, uniquify=True)[:count]
                        entry["matches"] = [
                            {
                                **({"atom_ids": [i + 1 for i in atoms]}
                                if return_indices else {}),
                                **({"smiles": Chem.MolFragmentToSmiles(
                                        mol, atomsToUse=atoms,
                                        canonical=True, isomericSmiles=True)
                                } if return_subsmiles else {}),
                                **({"masked_smiles": self._masked_smiles(mol, atoms)}
                                if return_masked_smiles else {}),
                            }
                            for atoms in hits
                        ]
                rd_out[name] = entry
            results["rdkit"] = rd_out

        # ─ MACCS keys ───────────────────────────────────────
        if include_maccs:
            mc_out = {}
            max_hits = mol.GetNumAtoms() ** 2  # generous upper bound

            for bid in range(1, 167):
                smarts, thr = self._maccs_pattern_and_thr(bid)
                if smarts is None and not include_no_pattern:
                    continue

                matches = []
                if smarts:
                    q = Chem.MolFromSmarts(smarts)
                    if q:
                        raw = mol.GetSubstructMatches(q, uniquify=False,
                                                    maxMatches=max_hits)
                        matches = self._dedupe(raw)

                if len(matches) <= thr:
                    if smarts is None and include_no_pattern:
                        mc_out[bid] = {"count": 0, "matches": []}
                    continue

                mc_out[bid] = {
                    "count": len(matches),
                    "matches": [
                        {
                            **({"atom_ids": [i + 1 for i in atoms]}
                            if return_indices else {}),
                            **({"smiles": Chem.MolFragmentToSmiles(
                                    mol, atomsToUse=atoms,
                                    canonical=True, isomericSmiles=True)
                            } if return_subsmiles else {}),
                            **({"masked_smiles": self._masked_smiles(mol, atoms)}
                            if return_masked_smiles else {}),
                        }
                        for atoms in matches
                    ],
                }
            results["maccs"] = mc_out

        return results

    def _get_rdkit_subs(self, smiles: str):
        info = self._fragement_subs_info(smiles)
        rdkit_info = info.get('rdkit', {})
        related_info = []
        for name, entry in rdkit_info.items():
            for match in entry['matches']:
                related_info.append({
                    "name": frag_to_name[name],
                    "related_atom_ids": match['atom_ids'],
                    "masked_smiles": self._compress_underbar(match['masked_smiles']),
                })
        return related_info

    def _iter_fragment_funcs(self):
        for name in dir(Fragments):
            if name.startswith("fr_"):
                fn = getattr(Fragments, name)
                if callable(fn):
                    yield name, fn

    def _compress_underbar(self, s: str) -> str:
        return re.sub(r'\_+', '_', s).strip('_')

    def _frag_pattern(self, fn):
        sig = inspect.signature(fn)
        p = sig.parameters.get("pattern")
        if p and isinstance(p.default, Chem.Mol):      # ≥ 2024.09
            return p.default
        smarts_dict = getattr(Fragments, "_fragFuncSMARTS", None) \
                    or getattr(Fragments, "_fragmentSmarts", None)
        return smarts_dict.get(fn.__name__) if smarts_dict else None

    def _get_minimal_funcgroup_info(self, funcgroup_info: List[Dict[str, Any]]):
        minimal_info = []
        for entry in funcgroup_info:
            name = entry['name']
            ids_str = str(sorted(entry['related_atom_ids']))
            masked_smiles = entry['masked_smiles']
            minimal_info.append(f"{name} | {masked_smiles} | {ids_str}")
        return minimal_info


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


    def _exact_match(self, ot_smi, gt_smi):
        if ot_smi == "" and gt_smi == "":
            return 1
        m_out = Chem.MolFromSmiles(ot_smi)
        m_gt = Chem.MolFromSmiles(gt_smi)

        try:
            if Chem.MolToInchi(m_out) == Chem.MolToInchi(m_gt):
                return 1
        except:
            pass
        return 0


    def extract_json_in_tags(self, rationale: str, tag: str) -> str:
        matches = re.findall(fr"<{tag}>(.*?)</{tag}>", rationale, re.DOTALL)
        if matches:
            for match in matches:
                try:
                    json_dict = json.loads(match.strip())
                    return json_dict
                except json.JSONDecodeError:
                    continue
            return None
        else:
            return None


    def calcualte_forward_rationale_metrics(self, info, GT_rationale, predicted_rationale):
        molecular_role_dict = self.extract_json_in_tags(predicted_rationale, "MOLECULAR_ROLE")
        precursor_stat_dict = self.extract_json_in_tags(predicted_rationale, "PRECURSOR_STAT")
        reactant_funcgroup_dict = self.extract_json_in_tags(predicted_rationale, "REACTANT_FUNCGROUP")
        template_dict = self.extract_json_in_tags(predicted_rationale, "TEMPLATE")


        # Metric 1: Molecular role
        try:
            gt_molecular_role_dict = {
                "reactant": info["reactant_str"],
                "reagent": info["reagent_str"],
                "solvent": info["solvent_str"],
            }
            correct_molecular_role = True
            for molecule_type, gt_smiles in gt_molecular_role_dict.items():
                pred_smiles = molecular_role_dict.get(molecule_type, "")
                if not self._exact_match(pred_smiles, gt_smiles):
                    correct_molecular_role = False
                    break
        except:
            correct_molecular_role = False


        # Metric 2: Precursor statistics
        try:
            gt_precursor_stat_dict = info["precursor_with_solvent_smiles_stat"]
            correct_precursor_stat = True
            for key, gt_value in gt_precursor_stat_dict.items():
                pred_value = precursor_stat_dict.get(key, None)
                if pred_value != gt_value:
                    correct_precursor_stat = False
                    break
            if set(gt_precursor_stat_dict.keys()) != set(precursor_stat_dict.keys()):
                correct_precursor_stat = False
        except:
            correct_precursor_stat = False


        # Metric 3: Reactant functional groups
        try:
            gt_reactant_funcgroup_dict = info["reactant_funcgroup_and_count"]
            correct_reactant_funcgroup = True
            for funcgroup, gt_count in gt_reactant_funcgroup_dict.items():
                pred_count = reactant_funcgroup_dict.get(funcgroup, None)
                if pred_count != gt_count:
                    correct_reactant_funcgroup = False
                    break
            if set(gt_reactant_funcgroup_dict.keys()) != set(reactant_funcgroup_dict.keys()):
                correct_reactant_funcgroup = False
        except:
            correct_reactant_funcgroup = False

        # Metric 4: Template
        try:
            gt_template = info["template"]
            if template_dict['template'] == gt_template:
                correct_template = True
            else:
                correct_template = False
        except:
            correct_template = False

        return {
            "forward/correct_molecular_role": int(correct_molecular_role),
            "forward/correct_precursor_stat": int(correct_precursor_stat),
            "forward/correct_reactant_funcgroup": int(correct_reactant_funcgroup),
            "forward/correct_template": int(correct_template),            
        }


    def calcualte_retro_rationale_metrics(self, info, GT_rationale, predicted_rationale): ## EDITED ##
        product_info_dict = self.extract_json_in_tags(predicted_rationale, "PRODUCT_INFO")
        candidate_structure_list = self.extract_json_in_tags(predicted_rationale, "CANDIDATE_STRUCTURE")
        stratetic_bond_disconnection_dict = self.extract_json_in_tags(predicted_rationale, "STRATEGIC_BOND_DISCONNECTION")
        synthetic_equivalents_list = self.extract_json_in_tags(predicted_rationale, "SYNTHETIC_EQUIVALENT")
        if product_info_dict is None:
            product_info_dict = {}
        if candidate_structure_list is None:
            candidate_structure_list = []
        if stratetic_bond_disconnection_dict is None:
            stratetic_bond_disconnection_dict = {}
        if synthetic_equivalents_list is None:
            synthetic_equivalents_list = []

        # Metric 1: Product info
        gt_product_info_dict = {
            "Atom mapped SMILES": info['product_mapping'],
            "Functional groups": info['product_minimal_funcgroup_info'],
            "SMILES statistics": info['product_smiles_stat'],
        }
        # Metric 1-1: product atom mapped SMILES
        try:
            correct_product_atom_mapped_smiles = product_info_dict.get("Atom mapped SMILES", "") == gt_product_info_dict["Atom mapped SMILES"]
        except:
            correct_product_atom_mapped_smiles = False
        # Metric 1-2: product functional groups
        try:
            predict_funcgroups = product_info_dict.get("Functional groups", [])
            gt_funcgroups = gt_product_info_dict["Functional groups"]
            correct_product_funcgroup = set(predict_funcgroups) == set(gt_funcgroups)
        except:
            correct_product_funcgroup = False
        # Metric 1-3: product SMILES statistics
        try:
            predict_smiles_stat = product_info_dict.get("SMILES statistics", {})
            gt_smiles_stat = gt_product_info_dict["SMILES statistics"]
            correct_product_smiles_stat = predict_smiles_stat == gt_smiles_stat
        except:
            correct_product_smiles_stat = False
        
        correct_product_info = all([
            correct_product_atom_mapped_smiles,
            correct_product_funcgroup,
            correct_product_smiles_stat,
        ])

        # Metric 2: Candidate structure
        try:
            gt_candidate_pair = [info["bond_list"][0][0], info["bond_list"][0][1]]
            predicted_candidate_atoms = candidate_structure_list[0].split("|")[-1].strip()
            correct_candidate_structure = all(str(g) in predicted_candidate_atoms for g in gt_candidate_pair)
        except:
            correct_candidate_structure = False
        
        # Metric 3: Strategic bond disconnection
        try:
            gt_strategic_bond_disconnection = {
                "Disconnect bonds": info["bond_list"],
                "Synthons": info["synthons_list"],
            }
            predicted_disconnect_bonds = stratetic_bond_disconnection_dict.get("Disconnect bonds", [])
            predicted_synthons = stratetic_bond_disconnection_dict.get("Synthons", [])
            gt_disconnect_bonds = [tuple(bond) for bond in gt_strategic_bond_disconnection["Disconnect bonds"]]
            predicted_disconnect_bonds = [tuple(bond) for bond in predicted_disconnect_bonds]
            correct_disconnect_bonds = set(predicted_disconnect_bonds) == set(gt_disconnect_bonds)
            correct_synthons = set(predicted_synthons) == set(gt_strategic_bond_disconnection["Synthons"])
        except Exception:
            correct_disconnect_bonds = False
            correct_synthons = False

        # Metric 4: Synthetic equivalents
        try:
            gt_synthetic_equivalents = info["synthetic_equivalents_list"]
            predicted_synthetic_equivalents = synthetic_equivalents_list
            correct_synthetic_equivalents = set(predicted_synthetic_equivalents) == set(gt_synthetic_equivalents)
        except Exception:
            correct_synthetic_equivalents = False

        return {
            "retro/correct_product_atom_mapped_smiles": int(correct_product_atom_mapped_smiles),
            "retro/correct_product_funcgroup": int(correct_product_funcgroup),
            "retro/correct_product_smiles_stat": int(correct_product_smiles_stat),
            "retro/correct_product_info": int(correct_product_info),
            "retro/correct_candidate_structure": int(correct_candidate_structure),
            "retro/correct_disconnect_bonds": int(correct_disconnect_bonds),
            "retro/correct_synthons": int(correct_synthons),
            "retro/correct_synthetic_equivalents": int(correct_synthetic_equivalents),
        }

        # return {
        #     "retro/correct_product_funcgroup": int(correct_product_funcgroup),
        #     "retro/correct_product_stat": int(correct_product_stat),
        #     "retro/correct_bond_disconnect": int(correct_bond_disconnect),
        #     "retro/correct_synthon": int(correct_synthon),
        #     "retro/correct_synthetic_equivalent": int(correct_synthetic_equivalent),
        # }


    def calcualte_condition_rationale_metrics(self, info, GT_rationale, predicted_rationale):
        reactant_removed_funcgroup_dict = self.extract_json_in_tags(predicted_rationale, "REACTANT_REMOVED_FUNCGROUP")
        product_added_funcgroup_dict = self.extract_json_in_tags(predicted_rationale, "PRODUCT_ADDED_FUNCGROUP")
        reactant_stat_dict = self.extract_json_in_tags(predicted_rationale, "REACTANT_STAT")
        product_stat_dict = self.extract_json_in_tags(predicted_rationale, "PRODUCT_STAT")
        condition_dict = self.extract_json_in_tags(predicted_rationale, "CONDITION")

        # Metric 1: Reactant removed functional groups
        try:
            gt_reactant_removed_funcgroup_dict = info["disappeared_from_reactant_to_product_funcgroup_and_count"]
            correct_reactant_removed_funcgroup = True
            for funcgroup, gt_count in gt_reactant_removed_funcgroup_dict.items():
                pred_count = reactant_removed_funcgroup_dict.get(funcgroup, None)
                if pred_count != gt_count:
                    correct_reactant_removed_funcgroup = False
                    break
            if set(gt_reactant_removed_funcgroup_dict.keys()) != set(reactant_removed_funcgroup_dict.keys()):
                correct_reactant_removed_funcgroup = False
        except:
            correct_reactant_removed_funcgroup = False

        # Metric 2: Product added functional groups
        try:
            gt_product_added_funcgroup_dict = info["new_from_reactant_to_product_funcgroup_and_count"]
            correct_product_added_funcgroup = True
            for funcgroup, gt_count in gt_product_added_funcgroup_dict.items():
                pred_count = product_added_funcgroup_dict.get(funcgroup, None)
                if pred_count != gt_count:
                    correct_product_added_funcgroup = False
                    break
            if set(gt_product_added_funcgroup_dict.keys()) != set(product_added_funcgroup_dict.keys()):
                correct_product_added_funcgroup = False
        except:
            correct_product_added_funcgroup = False

        # Metric 3: Reactant statistics
        try:
            gt_reactant_stat_dict = info["reactant_smiles_stat"]
            correct_reactant_stat = True
            for key, gt_value in gt_reactant_stat_dict.items():
                pred_value = reactant_stat_dict.get(key, None)
                if pred_value != gt_value:
                    correct_reactant_stat = False
                    break
            if set(gt_reactant_stat_dict.keys()) != set(reactant_stat_dict.keys()):
                correct_reactant_stat = False
        except:
            correct_reactant_stat = False

        # Metric 4: Product statistics
        try:
            gt_product_stat_dict = info["product_smiles_stat"]
            correct_product_stat = True
            for key, gt_value in gt_product_stat_dict.items():
                pred_value = product_stat_dict.get(key, None)
                if pred_value != gt_value:
                    correct_product_stat = False
                    break
            if set(gt_product_stat_dict.keys()) != set(product_stat_dict.keys()):
                correct_product_stat = False
        except:
            correct_product_stat = False

        # Metric 5: Step 2 rationale BLEU score
        try:
            step2_match = re.search(r"##\s*Step\s*2\b(.*?)##\s*Step\s*3\b", predicted_rationale, re.DOTALL)
            predicted_step2_rationale = step2_match.group(1).strip() if step2_match else ""
            step2_match = re.search(r"##\s*Step\s*2\b(.*?)##\s*Step\s*3\b", GT_rationale, re.DOTALL)
            GT_step2_rationale = step2_match.group(1).strip() if step2_match else ""        
            step2_bleu = corpus_bleu([[GT_step2_rationale.split()]], [predicted_step2_rationale.split()])
        except:
            step2_bleu = 0.0

        # Metric 6: Condition
        try:
            gt_condition_dict = {
                "reagents": info["reagent_str"],
                "solvents": info["solvent_str"],
            }
            correct_condition = True
            for key, gt_value in gt_condition_dict.items():
                pred_value = condition_dict.get(key, "")
                if not self._exact_match(pred_value, gt_value):
                    correct_condition = False
                    break
        except:
            correct_condition = False

        return {
            "correct_reactant_removed_funcgroup": int(correct_reactant_removed_funcgroup),
            "correct_product_added_funcgroup": int(correct_product_added_funcgroup),
            "correct_reactant_stat": int(correct_reactant_stat),
            "correct_product_stat": int(correct_product_stat),
            "step2_bleu": step2_bleu,
            "correct_condition": int(correct_condition),
        }

    def evaluate(self, info_list, GT_rationale_list, predicted_reasoning_list, task):
        if "forward" in task:
            forward_metrics_dict = defaultdict(list)
            for info, GT_rationale, predicted_reasoning in zip(info_list, GT_rationale_list, predicted_reasoning_list):
                forward_metrics = self.calcualte_forward_rationale_metrics(info, GT_rationale, predicted_reasoning)
                for key, value in forward_metrics.items():
                    forward_metrics_dict[key].append(value)
            metric_dict = {}
            for key, values in forward_metrics_dict.items():
                metric_dict[key] = sum(values) / len(values) if values else 0.0
            return metric_dict

        elif "retro" in task:
            retro_metrics_dict = defaultdict(list)
            for info, GT_rationale, predicted_reasoning in zip(info_list, GT_rationale_list, predicted_reasoning_list):
                retro_metrics = self.calcualte_retro_rationale_metrics(info, GT_rationale, predicted_reasoning)
                for key, value in retro_metrics.items():
                    retro_metrics_dict[key].append(value)
            metric_dict = {}
            for key, values in retro_metrics_dict.items():
                metric_dict[key] = sum(values) / len(values) if values else 0.0
            return metric_dict

        elif "condition" in task:
            condition_metrics_dict = defaultdict(list)
            for info, GT_rationale, predicted_reasoning in zip(info_list, GT_rationale_list, predicted_reasoning_list):
                condition_metrics = self.calcualte_condition_rationale_metrics(info, GT_rationale, predicted_reasoning)
                for key, value in condition_metrics.items():
                    condition_metrics_dict[key].append(value)
            metric_dict = {}
            for key, values in condition_metrics_dict.items():
                metric_dict[key] = sum(values) / len(values) if values else 0.0
            return metric_dict





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
        # 1. <think> ... </think> 태그 페어 및 개수 검증
        open_think_tags = re.findall(r"<think>", text)
        close_think_tags = re.findall(r"</think>", text)
        # 각각 정확히 1회만 등장해야 함
        if len(open_think_tags) != 1 or len(close_think_tags) != 1:
            return False
        open_pos = text.find("<think>")
        close_pos = text.find("</think>")
        # 순서 및 위치 검증
        if open_pos == -1 or close_pos == -1 or close_pos < open_pos:
            return False
        # 추가적인 잘못된 패턴(예: 중첩 시작 태그 후 단일 종료 태그) 방지:
        # "<think>" 이후 다시 "<think>" 가 나오면 잘못된 구조
        if text.find("<think>", open_pos + len("<think>")) != -1:
            return False
        # 추출
        think_content = text[open_pos + len("<think>"):close_pos]
        if think_content.strip() == "":
            return False

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

    def parse_raw_response(self, raw_response: str):
        """
        Parse the raw response from the model to extract reasoning steps and SMILES.
        raw_response format:
        <think>{reasoning steps}</think>\n\n<ANSWER>{SMILES}</ANSWER>
        """
        reasoning_pattern = r"<think>(.*?)</think>"
        smiles_pattern = r"<ANSWER>(.*?)</ANSWER>"

        reasoning_match = re.search(reasoning_pattern, raw_response, re.DOTALL)
        smiles_match = re.search(smiles_pattern, raw_response, re.DOTALL)

        reasoning_steps = reasoning_match.group(1).strip() if reasoning_match else ""
        smiles = smiles_match.group(1).strip() if smiles_match else ""
        return reasoning_steps, smiles

    def maccs_similarity(self, ot_m, gt_m):
        return DataStructs.FingerprintSimilarity(
            MACCSkeys.GenMACCSKeys(gt_m), 
            MACCSkeys.GenMACCSKeys(ot_m), 
            metric=DataStructs.TanimotoSimilarity
        )

    def morgan_similarity(self, ot_m, gt_m, radius=2):
        return DataStructs.TanimotoSimilarity(
            AllChem.GetMorganFingerprint(gt_m, radius), 
            AllChem.GetMorganFingerprint(ot_m, radius)
        )

    def rdk_similarity(self, ot_m, gt_m):
        return DataStructs.FingerprintSimilarity(
            Chem.RDKFingerprint(gt_m), 
            Chem.RDKFingerprint(ot_m), 
            metric=DataStructs.TanimotoSimilarity
        )

    def roundtrip(self, reactant: str, product: str, roundtrip_client):
        system_prompt = "You are a chemist."
        user_prompt = f"{reactant} Considering the given starting materials, what might be the resulting product in a chemical reaction?"
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ]
        response = roundtrip_client.chat.completions.create(
            model="/models/roundtrip",
            messages=messages,
            max_tokens=500,
            temperature=0.0,
            n=1,
        )
        raw_response = response.choices[0].message.content.strip()
        reasoning_pattern = r"<think>(.*?)</think>"
        smiles_pattern = r"<ANSWER>(.*?)</ANSWER>"

        reasoning_match = re.search(reasoning_pattern, raw_response, re.DOTALL)
        smiles_match = re.search(smiles_pattern, raw_response, re.DOTALL)

        reasoning_steps = reasoning_match.group(1).strip() if reasoning_match else ""
        roundtrip_product = smiles_match.group(1).strip() if smiles_match else ""
        is_correct = self.validator.exact_match(roundtrip_product, product)
        
        return roundtrip_product

    def compute_score(self,
                      solution_str: str,
                      ground_truth: str,
                      extra_info: Optional[Dict] = None,
                      use_roundtrip_reward=False,
                      use_content_reward=False,
                      use_decision_reward=False,
                      use_reflection_bonus=False,
                      reflection_bonus_weight=0.0,
                      roundtrip_client=None,
                      roundtrip_cache=None,
                      ) -> Union[EvaluationResult, Dict[str, Any]]:
        """Compute comprehensive evaluation score."""
        task = extra_info['task']
        # correct_structure = self.validate_structure(solution_str, task)
        correct, pred = self.verify(solution_str, ground_truth, extra_info)

        pred_rationale, pred_smiles = self.parse_raw_response(solution_str)
        gt_rationale, gt_smiles = self.parse_raw_response(ground_truth)

        info = json.loads(extra_info['info_json_str'])
        info['products'] = extra_info['products']
        info['reactants'] = extra_info['reactants']

        if task == "forward":
            info['precursor_with_solvent_smiles_stat'] = ast.literal_eval(info['precursor_with_solvent_smiles_stat'])
            info['reactant_funcgroup_and_count'] = ast.literal_eval(info['reactant_funcgroup_and_count'])
            step_eval_results = self.step_evaluator.calcualte_forward_rationale_metrics(info, gt_rationale, pred_rationale)
            answer_correct = self.step_evaluator._exact_match(pred_smiles, gt_smiles)
        elif task == "retro":

            ## EDITED ##
            info['bond_list'] = ast.literal_eval(info['bond_list'])
            info['product_mapping'] = info['product_mapping']
            info['product_minimal_funcgroup_info'] = ast.literal_eval(info['product_minimal_funcgroup_info'])
            # info['product_reactive_atoms'] = ast.literal_eval(info['product_reactive_atoms'])
            info['product_smiles_stat'] = ast.literal_eval(info['product_smiles_stat'])
            info['reactant_str'] = info['reactant_str']
            info['product_str'] = info['product_str']
            info['synthons_list'] = ast.literal_eval(info['synthons_list'])
            info['synthetic_equivalents_list'] = ast.literal_eval(info['synthetic_equivalents_list'])
            info['template'] = ast.literal_eval(info['template'])
            ############


            # info['bond_funcgroup_and_count'] = ast.literal_eval(info['bond_funcgroup_and_count'])
            # info['product_smiles_stat'] = ast.literal_eval(info['product_smiles_stat'])
            # info['bond_list'] = ast.literal_eval(info['bond_list'])
            # info['synthons_list_new'] = ast.literal_eval(info['synthons_list_new'])
            # info['synthetic_equivalents_list'] = ast.literal_eval(info['synthetic_equivalents_list'])
            step_eval_results = self.step_evaluator.calcualte_retro_rationale_metrics(info, gt_rationale, pred_rationale)
            if use_roundtrip_reward:
                cached_product = roundtrip_cache.get(pred_smiles)
                if cached_product is not None:
                    roundtrip_product = cached_product
                    # print(f"Using cached roundtrip product")
                else:
                    roundtrip_product = self.roundtrip(reactant=pred_smiles, product=info['product_str'], roundtrip_client=roundtrip_client)
                    roundtrip_cache.set(pred_smiles, roundtrip_product)
                    # print(f"Caching roundtrip product")
                is_roundtrip_correct = self.validator.exact_match(roundtrip_product, info['product_str'])
                # print(f"Roundtrip prediction correct: {is_roundtrip_correct}")
                answer_correct = int(is_roundtrip_correct)
            else:
                answer_correct = self.step_evaluator._exact_match(pred_smiles, gt_smiles)
        elif task == "condition":
            step_eval_results = self.step_evaluator.calcualte_condition_rationale_metrics(info, gt_rationale, pred_rationale)
        else:
            raise ValueError(f"Unknown task type: {task}")

        try:
            maccs_sim = self.maccs_similarity(Chem.MolFromSmiles(pred_smiles), Chem.MolFromSmiles(gt_smiles))
            morgan_sim = self.morgan_similarity(Chem.MolFromSmiles(pred_smiles), Chem.MolFromSmiles(gt_smiles))
            rdk_sim = self.rdk_similarity(Chem.MolFromSmiles(pred_smiles), Chem.MolFromSmiles(gt_smiles))
        except:
            maccs_sim = 0.0
            morgan_sim = 0.0
            rdk_sim = 0.0

        reward = answer_correct
        # reward = maccs_sim



        """
        info = {
            "products": extra_info['products'],
            "reactants": extra_info['reactants'],
            # "reagents": extra_info['reagents'],
        }
        # if "reagents" in extra_info["supporting_info"]['reagent']:
        #     del extra_info["supporting_info"]['reagent']['reagents']
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
        """

        reward_metric_dict = {
            f"{task}/reward/answer_correct": answer_correct,
            f"{task}/reward/maccs_similarity": maccs_sim,
            f"{task}/reward/morgan_similarity": morgan_sim,
            f"{task}/reward/rdk_similarity": rdk_sim,
            # f"{task}/reward/valid_structure": int(correct_structure),
        }
        # if not correct_structure:
        #     reward -= 2.0
        if use_content_reward:
            content_reward = sum(step_eval_results.values()) / len(step_eval_results)
            reward += content_reward
            reward_metric_dict[f"{task}/reward/content_reward"] = content_reward
        
        step_eval_results.update(reward_metric_dict)

        result = EvaluationResult(
            score=reward,
            acc=correct,
            pred=pred,
            metrics=step_eval_results
        )
        result = result.to_dict()
        # result["step_rewards"] = step_rewards
        # result["reflection_decision_reward_dict"] = reflection_decision_reward_dict

        # Return dict format for compatibility with existing code
        return result


def compute_score(solution_str: str,
                  ground_truth: str,
                  extra_info: Optional[dict] = None,
                  use_roundtrip_reward=False,
                  use_content_reward=False,
                  use_decision_reward=False,
                  use_reflection_bonus=False,
                  reflection_bonus_weight=0.0,
                  roundtrip_client=None,
                  roundtrip_cache=None,
                 ) -> Dict[str, Any]:
    evaluator = ChemistryEvaluator()
    assert roundtrip_client is not None, "Roundtrip client must be provided if use_roundtrip_reward is True."
    result = evaluator.compute_score(
        solution_str,
        ground_truth,
        extra_info,
        use_roundtrip_reward,
        use_content_reward,
        use_decision_reward,
        use_reflection_bonus,
        reflection_bonus_weight,
        roundtrip_client=roundtrip_client,
        roundtrip_cache=roundtrip_cache,
    )
    # result is already a dict from compute_score method
    return result


if __name__ == "__main__":
    solution_str = """<think>
## Step 1: Understanding molecular roles  
The reactant is Sc1cccc(Br)c1.CC(C)=CCBr and the product is CC(C)=CCSc1cccc(Br)c1. The reaction involves the formation of a thioether bond between the aryl group (Sc1cccc(Br)c1) and the alkyl chain (CC(C)=CCBr), indicating a coupling between a sulfur-containing nucleophile and an electrophilic carbon center.

## Step 2: Analysis of main functional groups  
The reactant contains a bromoaryl group (Br attached to an aromatic ring) and a bromoalkyl group (Br attached to a carbon chain). The product contains a thioether linkage (S–C), suggesting that a sulfur nucleophile has reacted with an electrophilic carbon center. The presence of bromine in both reactants indicates that it may act as a leaving group in the reaction.

## Step 3: Predict possible reaction mechanisms  
The formation of a thioether bond suggests an S_N2 mechanism or a transition-metal-catalyzed coupling. Given the presence of bromine, which is a good leaving group, the reaction could proceed via nucleophilic substitution or through a palladium- or copper-catalyzed cross-coupling. The introduction of sulfur implies that a thiolate nucleophile is involved, likely generated from a thiol by deprotonation.

## Step 4: Δ analysis  
The functional group/substructure change (Δ) between reactant and product involves the introduction of a quaternary nitrogen (masked SMILES `_CSc_`) and the removal of H-pyrrole nitrogens (masked SMILES `S_`). This indicates that the sulfur atom from the thiolate nucleophile has been incorporated into the product, forming a new S–C bond.

## Step 5: Derivation of required functions  
The reaction requires a strong base to deprotonate a thiol, generating a thiolate nucleophile. It also requires an electrophilic carbon center with a good leaving group, such as bromide. Additionally, if transition-metal catalysis is involved, the reagent must provide a suitable catalyst like palladium or copper.

## Step 6: Generation of candidate reagents  
The candidate reagents are as follows:  
1. [Al].[Al]  
2. CC(=O)[O-].C[O-]  
3. O=C([O-])[O-].[K+]

<REFLECTION>
Wait, considering that the reaction requires a sulfur nucleophile (or a thiolate) and possibly a transition‑metal catalyst, the candidate reagents listed ([Al].[Al], CC(=O)[O-].C[O-], O=C([O-])[O-].[K+]) do not provide sulfur or an appropriate base or catalyst for forming the thioether bond.
It should be
1. CCCC[SnH](CCCC)CCCC.C[O-]
2. [Pd+2].O=P([O-])([O-])[O-]
3. [Na+].[OH-]
</REFLECTION>

## Step 7: Selection of reagents  
The reagent 3 is the most suitable reagent for the reaction, since it provides both a strong base ([OH-]) to generate the thiolate nucleophile and a cation ([Na+]) to stabilize the anion and enhance solubility. This combination supports the formation of the thioether bond through nucleophilic substitution.
</think>

<ANSWER>
[Na+].[OH-]
</ANSWER>"""


    ground_truth = "O=C([O-])[O-].[K+]"
    extra_info = {
        'class_name': 'Carboxylic acid + amine reaction',
        'products': ['CC(C)N(C)c1nc2cc(C(=O)NC(C)(C)C)ccc2[nH]c1=O'],
        'reactants': ['CC(C)N(C)c1nc2cc(C(=O)O)ccc2[nH]c1=O', 'CC(C)(C)N'],
        'reagents': ['O=C([O-])[O-]', '[K+]'],
        'rxn_str': 'CC(C)(C)N.CC(C)N(C)c1nc2cc(C(=O)O)ccc2[nH]c1=O.CCCP1(=O)OP(=O)(CCC)OP(=O)(CCC)O1.ClCCl>>CC(C)N(C)c1nc2cc(C(=O)NC(C)(C)C)ccc2[nH]c1=O',
        'supporting_info': {
            'forward': {
                'canonical_intermediate': '',
                'generated_substructure_from_precursor': ['DUMMY'],
                'intermediate_smiles': '',
                'masked_smiles': ['DUMMY'],
                'product_changes_tagged': '',
                'reactive_atom_bonds': [['', '', '']],
                'reactive_atoms_smiles_str': ''
            },
            'reagent': {
                'correct_reagent_number': '2',
                'generated_substructure_list': ['DUMMY'],
                'masked_smiles': ['DUMMY'],
                'removed_substructure_list': ['DUMMY']
            },
            'retro': {
                'atom_mapping': 'DUMMY',
                'bond_list': [['', '', '']],
                'masked_smiles': ['DUMMY'],
                'synthetic_equivalents': [''],
                'synthons_list': [''],
                'synthons_list_new': ['']
            }
        },
        'task': 'reagent',
        'yields': [],
        'rollout_reward_scores': {}
    }
    score = compute_score(
        solution_str,
        ground_truth,
        extra_info,
        use_roundtrip_reward=True,
        use_content_reward=True,
        use_decision_reward=True,
        use_reflection_bonus=True,
        reflection_bonus_weight=0.5
    )
    pprint(score)