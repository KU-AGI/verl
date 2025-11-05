import re
from typing import List, Dict
from collections import defaultdict


class ReflectionEvaluator():
    def __init__(self):
        pass

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

    def extract_numbered_items(self, text: str) -> list:
        pattern = re.compile(r"^\s*\d+\.\s*(.+)$", re.MULTILINE)
        items = pattern.findall(text)
        return [item.strip().strip('`') for item in items]

    def evaluate(self, info_list, GT_rationale_list, predicted_reasoning_list, task):
        results = defaultdict(list)
        for info, GT_rationale, predicted_reasoning in zip(info_list, GT_rationale_list, predicted_reasoning_list):
            gt_steps = self.parse_steps_with_reflections(GT_rationale)
            pred_steps = self.parse_steps_with_reflections(predicted_reasoning)
            for step_key, step_dict in pred_steps.items():
                content = step_dict["content"]
                reflections = step_dict["reflections"]
                if len(reflections) > 0:
                    results[f"{step_key}_has_reflection"].append(1)
                else:
                    results[f"{step_key}_has_reflection"].append(0)

        avg_results = {}
        for result_key, result_list in results.items():
            avg_results[result_key] = sum(result_list) / len(result_list) if result_list else 0.0

        return avg_results




if __name__ == "__main__":
    evaluator = ReflectionEvaluator()
    info_list = [{
        "reagents": [],
        "solvents": [],
        "products": [
            "CC(=O)Nc1nc2c(Oc3cc(-c4ccc(C(F)(F)F)cc4)nc(CO)n3)cccc2s1"
        ],
        "generated_substructure_from_precursor": [
            "DUMMY"
        ],
        "masked_smiles": [
            "DUMMY"
        ],
        "reactive_atoms_smiles_str": "CCC[C][Sn]([**C:6**]O)([C]CCC)[C]CCC.CC(=O)Nc1nc2c(Oc3cc(-c4ccc(C(F)(F)F)cc4)n[**c:39**](Cl)n3)cccc2s1",
        "reactive_atom_bonds": [
            [
                6,
                39,
                "'single'"
            ]
        ],
        "product_changes_tagged": "CC<DEL>C[CH2][Sn]</DEL>(<REP_OLD>[CH2]CCC)([CH2]CCC)[CH](</REP_OLD><REP_NEW>=</REP_NEW>O)<ADD>N</ADD>c1<DEL>(Cl)</DEL>n<ADD>c2</ADD>c(Oc<DEL>2cccc</DEL>3<DEL>sc(NC(C)=O)nc23)</DEL>cc(-c<REP_OLD>2</REP_OLD><REP_NEW>4</REP_NEW>ccc(C(F)(F)F)cc<REP_OLD>2</REP_OLD><REP_NEW>4</REP_NEW>)n<ADD>c(CO)n3)cccc2s</ADD>1",
        "intermediate_smiles": "CCC[CH2][Sn]([CH](O)c1(Cl)nc(-c2ccc(C(F)(F)F)cc2)cc(Oc2c3nc(NC(C)=O)sc3ccc2)n1)([CH2]CCC)[CH2]CCC",
        "canonical_intermediate": "CCC[CH2][Sn]([CH2]CCC)([CH2]CCC)[CH](O)c1(Cl)nc(Oc2cccc3sc(NC(C)=O)nc23)cc(-c2ccc(C(F)(F)F)cc2)n1"
    }]
    GT_rationale_list = [
        "## Step 1: Understanding molecular roles  \nThe precursor contains only reactant `CCCC[Sn](CO)(CCCC)CCCC.CC(=O)Nc1nc2c(Oc3cc(-c4ccc(C(F)(F)F)cc4)nc(Cl)n3)cccc2s1.`\n\n## Step 2: Analysis of main functional groups  \nThe precursor contains the substructure `_Sn]([CH2]O)(_`, which corresponds to a methylene group between a tin atom and an oxygen atom. This substructure is likely to participate in a nucleophilic or electrophilic substitution reaction due to the presence of the tin atom, which is a versatile organometallic center. Additionally, the substructure `_CH2][Sn]([CH2_CH2_CH2_` indicates a quaternary heteroatom (tin) attached to three carbons and one additional atom, suggesting potential for coordination or bond formation with other functional groups.\n\n## Step 3: Predict possible reaction mechanisms  \nBy considering the substructure involving the tin atom and its connectivity, it is likely that the reaction will occur through a nucleophilic substitution mechanism. The tin center may act as a nucleophile or electrophile, depending on the reaction conditions, and could facilitate the transfer of a methyl group or other substituent to an adjacent carbon or oxygen atom.\n\n## Step 4: Highlight the atoms that participate in chemical reactions\nConsidering above information, the atoms that will participate in the chemical reaction in precursor SMILES can be expressed as `**[atom]:[number]**` as follows.  \nCCC[C][**Sn:5**]([C]O)([C]CCC)[C]CCC.CC(=O)Nc1nc2c(Oc3cc(-c4ccc(C(F)(F)F)cc4)n[**c:38**](Cl)n3)cccc2s1\n<REFLECTION>\nWait, considering that the tin atom is bonded to a methylene group bearing oxygen, the reaction involves that methylene carbon (not Sn itself), so assigning a mapping number to Sn and using 38 for the aromatic carbon misidentifies the reacting atoms; they should be C:6 and c:39 instead.\nIt should be CCC[C][Sn]([**C:6**]O)([C]CCC)[C]CCC.CC(=O)Nc1nc2c(Oc3cc(-c4ccc(C(F)(F)F)cc4)n[**c:39**](Cl)n3)cccc2s1.\n</REFLECTION>\n\n## Step 5: Predict bond pairs\nBased on the above atomic mapping numbers, if the bonds to be formed by the highlighted atoms are expressed in the form of `([atom mapping number], [atom mapping number], [bond type])`, the following bonds will be formed.  \n(6, 39, 'single')\n\n## Step 6: SMILES change prediction  \nIf the above bond pairs are applied to the precursor, without considering the atomic rules or valence rules, it can be expressed as CCC[CH2][Sn]([CH](O)c1(Cl)nc(-c2ccc(C(F)(F)F)cc2)cc(Oc2c3nc(NC(C)=O)sc3ccc2)n1)([CH2]CCC)[CH2]CCC. If this is canonicalized, it becomes CCC[CH2][Sn]([CH2]CCC)([CH2]CCC)[CH](O)c1(Cl)nc(Oc2cccc3sc(NC(C)=O)nc23)cc(-c2ccc(C(F)(F)F)cc2)n1. From the canonicalized SMILES, the product is formed by modifying the SMILES string to remove the tin atom and replace it with a carbon-carbon bond between the electrophilic partner and the organotin reagent. This modification ensures compliance with valence rules and atomic connectivity.  \nIf we represent the changes from canonicalized SMILES as <DEL> for deleted parts, <ADD> for added parts, and <REP_OLD> and <REP_NEW> for modified parts, we get the following: \nCCC<DEL>[</DEL>C<DEL>H2]</DEL>[Sn](<DEL>[</DEL>C<DEL>H2]CCC)([CH2]CCC)[CH](O)</DEL>c1<DEL>(Cl)</DEL>nc(Oc2cccc3sc(NC(C)=O)nc23)cc(-c2ccc(C(F)(F)F)cc2)n1<ADD>)(CCCC)CCCC</ADD>\n<REFLECTION>\nThe predicted SMILES incorrectly deletes the tin atom and misassigns bonding, violating valence rules and producing an unphysical structure.\nThe product should be CC(=O)Nc1nc2c(Oc3cc(-c4ccc(C(F)(F)F)cc4)nc(CO)n3)cccc2s1.\n</REFLECTION>"
    ]
    predicted_rationale_list = [
        "## Step 1: Understanding molecular roles  \nThe precursor contains only reactant `CCCC[Sn](CO)(CCCC)CCCC.CC(=O)Nc1nc2c(Oc3cc(-c4ccc(C(F)(F)F)cc4)nc(Cl)n3)cccc2s1.`\n\n## Step 2: Analysis of main functional groups  \nThe precursor contains the substructure `_Sn]([CH2]O)(_`, which corresponds to a methylene group between a tin atom and an oxygen atom. This substructure is likely to participate in a nucleophilic or electrophilic substitution reaction due to the presence of the tin atom, which is a versatile organometallic center. Additionally, the substructure `_CH2][Sn]([CH2_CH2_CH2_` indicates a quaternary heteroatom (tin) attached to three carbons and one additional atom, suggesting potential for coordination or bond formation with other functional groups.\n\n## Step 3: Predict possible reaction mechanisms  \nBy considering the substructure involving the tin atom and its connectivity, it is likely that the reaction will occur through a nucleophilic substitution mechanism. The tin center may act as a nucleophile or electrophile, depending on the reaction conditions, and could facilitate the transfer of a methyl group or other substituent to an adjacent carbon or oxygen atom.\n\n## Step 4: Highlight the atoms that participate in chemical reactions\nConsidering above information, the atoms that will participate in the chemical reaction in precursor SMILES can be expressed as `**[atom]:[number]**` as follows.  \nCCC[C][**Sn:5**]([C]O)([C]CCC)[C]CCC.CC(=O)Nc1nc2c(Oc3cc(-c4ccc(C(F)(F)F)cc4)n[**c:38**](Cl)n3)cccc2s1\n<REFLECTION>\nWait, considering that the tin atom is bonded to a methylene group bearing oxygen, the reaction involves that methylene carbon (not Sn itself), so assigning a mapping number to Sn and using 38 for the aromatic carbon misidentifies the reacting atoms; they should be C:6 and c:39 instead.\nIt should be CCC[C][Sn]([**C:6**]O)([C]CCC)[C]CCC.CC(=O)Nc1nc2c(Oc3cc(-c4ccc(C(F)(F)F)cc4)n[**c:39**](Cl)n3)cccc2s1.\n</REFLECTION>\n\n## Step 5: Predict bond pairs\nBased on the above atomic mapping numbers, if the bonds to be formed by the highlighted atoms are expressed in the form of `([atom mapping number], [atom mapping number], [bond type])`, the following bonds will be formed.  \n(6, 39, 'single')\n\n## Step 6: SMILES change prediction  \nIf the above bond pairs are applied to the precursor, without considering the atomic rules or valence rules, it can be expressed as CCC[CH2][Sn]([CH](O)c1(Cl)nc(-c2ccc(C(F)(F)F)cc2)cc(Oc2c3nc(NC(C)=O)sc3ccc2)n1)([CH2]CCC)[CH2]CCC. If this is canonicalized, it becomes CCC[CH2][Sn]([CH2]CCC)([CH2]CCC)[CH](O)c1(Cl)nc(Oc2cccc3sc(NC(C)=O)nc23)cc(-c2ccc(C(F)(F)F)cc2)n1. From the canonicalized SMILES, the product is formed by modifying the SMILES string to remove the tin atom and replace it with a carbon-carbon bond between the electrophilic partner and the organotin reagent. This modification ensures compliance with valence rules and atomic connectivity.  \nIf we represent the changes from canonicalized SMILES as <DEL> for deleted parts, <ADD> for added parts, and <REP_OLD> and <REP_NEW> for modified parts, we get the following: \nCCC<DEL>[</DEL>C<DEL>H2]</DEL>[Sn](<DEL>[</DEL>C<DEL>H2]CCC)([CH2]CCC)[CH](O)</DEL>c1<DEL>(Cl)</DEL>nc(Oc2cccc3sc(NC(C)=O)nc23)cc(-c2ccc(C(F)(F)F)cc2)n1<ADD>)(CCCC)CCCC</ADD>\n<REFLECTION>\nThe predicted SMILES incorrectly deletes the tin atom and misassigns bonding, violating valence rules and producing an unphysical structure.\nThe product should be CC(=O)Nc1nc2c(Oc3cc(-c4ccc(C(F)(F)F)cc4)nc(CO)n3)cccc2s1.\n</REFLECTION>"
    ]
    task = "reflect_forward_trainsubset_step46"
    
    results = evaluator.evaluate(info_list, GT_rationale_list, predicted_rationale_list, task)
    for k, v in results.items():
        print(f"{k}: {v}")
        print("-----")