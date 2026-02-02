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


    def calculate_retro_rationale_metrics(self, info, GT_rationale, predicted_rationale): ## EDITED ##
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

    def evaluate(self, info_list, GT_rationale_list, predicted_reasoning_list, task):
       if "retro" in task:
            retro_metrics_dict = defaultdict(list)
            for info, GT_rationale, predicted_reasoning in zip(info_list, GT_rationale_list, predicted_reasoning_list):
                retro_metrics = self.calculate_retro_rationale_metrics(info, GT_rationale, predicted_reasoning)
                for key, value in retro_metrics.items():
                    retro_metrics_dict[key].append(value)
            metric_dict = {}
            for key, values in retro_metrics_dict.items():
                metric_dict[key] = sum(values) / len(values) if values else 0.0
            return metric_dict




class ChemistryEvaluator:
    """Main evaluation interface."""
    
    def __init__(self):
        self.validator = SMILESValidator()
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
    
    def convert_to_canonical_smiles(self, smiles):
        if not smiles:
            return None
        molecule = Chem.MolFromSmiles(smiles)
        if molecule is not None:
            canonical_smiles = Chem.MolToSmiles(molecule, isomericSmiles=True, canonical=True)
            return canonical_smiles
        else:
            return None

    def roundtrip(self, reactant: str, product: str, roundtrip_client):
        system_prompt = "You are a chemist."
        user_prompt = f"{self.convert_to_canonical_smiles(reactant)} Considering the given starting materials, what might be the resulting product in a chemical reaction?"
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

        if task == "retro":

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
            step_eval_results = self.step_evaluator.calculate_retro_rationale_metrics(info, gt_rationale, pred_rationale)
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
