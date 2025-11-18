import json, re
import time
import traceback
# import threading
# import concurrent.futures
import multiprocessing
from concurrent.futures import ProcessPoolExecutor, as_completed
from evaluator.smiles_evaluator import MoleculeSMILESEvaluator
from prompt_templates import *
from openai import OpenAI
from tqdm import tqdm
from transformers import AutoTokenizer

from rdkit import Chem, RDLogger

RDLogger.DisableLog('rdApp.*')

# ---- 설정 ----
model = "ReactionReasoner"  # "predictiononly" or "ReactionReasoner"
tasks = ["forward", "retro", "reagent"]
# tasks = ["reagent"]
max_samples = 10000  # 각 task별 최대 샘플 수


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
        correct = d['info']["reactive_atoms_smiles_str"] in reasoning_text
    elif step == 5 and task == "forward":
        reasoning_text = steps_data.get(f'step {step}', {}).get("content", "")
        for i in range(len(d['info']['reactive_atom_bonds'])):
            d['info']['reactive_atom_bonds'][i][0] = int(d['info']['reactive_atom_bonds'][i][0]) # convert to int for comparison
            d['info']['reactive_atom_bonds'][i][1] = int(d['info']['reactive_atom_bonds'][i][1]) # convert to int for comparison
            d['info']['reactive_atom_bonds'][0][2] = d['info']['reactive_atom_bonds'][0][2].replace("'", "") # remove extra quotes if any
        if len(d['info']['reactive_atom_bonds']) == 0:
            has_reactive_atom_bonds = True
            for bond_type in ['single', 'double', 'triple', 'aromatic']:
                if bond_type in reasoning_text:
                    has_reactive_atom_bonds = False
                    break
        else:
            has_reactive_atom_bonds = all(str(tuple(bond)) in reasoning_text for bond in d['info']['reactive_atom_bonds'])
        correct = has_reactive_atom_bonds
    elif step == 6 and task == "forward":
        reasoning_text = steps_data.get(f'step {step}', {}).get("content", "")
        has_tagged_smiles = d['info']["product_changes_tagged"] in reasoning_text
        correct = has_tagged_smiles
    elif step == 5 and task == "retro":
        reasoning_text = steps_data.get(f'step {step}', {}).get("content", "")
        bond_disconnection_list = []
        for bond in d['info']["bond_list"]:
            bond_str = f"{bond[0]}, {bond[1]}: {bond[2]}"
            bond_disconnection_list.append(bond_str)
        has_bond_disconnection = all(bond_str in reasoning_text for bond_str in bond_disconnection_list)
        correct = has_bond_disconnection
    elif step == 6 and task == "retro":
        reasoning_text = steps_data.get(f'step {step}', {}).get("content", "")
        has_synthons = all(synthon in reasoning_text for synthon in d['info']["synthons_list"])
        correct = has_synthons
    elif step == 7 and task == "retro":
        reasoning_text = steps_data.get(f'step {step}', {}).get("content", "")
        has_synthetic_equivalents = all(syn_equiv in reasoning_text for syn_equiv in d['info']["synthetic_equivalents"])
        correct = has_synthetic_equivalents
    elif step == 6 and task == "reagent":
        if len(steps_data.get(f"step 6", {}).get("reflections", [])) == 0:
            reagent_list = extract_numbered_items(steps_data.get(f'step 6', {}).get("content", ""))
        else:
            reagent_list = extract_numbered_items(steps_data.get(f'step 6', {}).get("reflections", [])[-1])
        # reagent_gt = ".".join(d['info']["reagents"])
        reagent_gt = d['info']['reagents']
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
        reagent_gt = d['info']['reagents']
        has_reagents = False
        for reagent_pred in reagent_list:
            if exact_match(reagent_pred, reagent_gt):
                has_reagents = True
                break
        correct_reagent_number = -1
        for idx, reagent_pred in enumerate(reagent_list):
            # if exact_match(reagent_pred, ".".join(d['info']["reagents"])):
            if exact_match(reagent_pred, d['info']["reagents"]):
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




all_results = {}
all_predictions = {}
all_ground_truths = {}
# for temp in [0.2, 0.4, 0.6, 0.8, 1.0, 1.2, 1.4, 1.6, 1.8, 2.0]:
for temp in [1.0]:
    # for strategy in ["naive_sampling", "force_reflection"]:
    for strategy in ["force_reflection"]:
        # for n_generations in [4, 8, 12, 16, 20]:
        for n_generations in [8]:
        # for strategy in ["naive_sampling", "step123greedy_then_naive_sampling", "step123sampling_then_greedy_sampling", "step12345greedy_then_naive_sampling", "step123456greedy_then_naive_sampling", "step123456greedy_then_naive_sampling"]:
            TEMPERATURE = temp
            STRATEGY = strategy
            if model == "predictiononly":
                MAX_TOKENS = 500
            elif model == "ReactionReasoner":
                MAX_TOKENS = 1700
            else:
                raise ValueError(f"Unknown model: {model}")
            
            # 한 입력당 생성할 후보 수
            if temp == 0.0:
                NUM_GENERATIONS = 1
                STRATEGY = "greedy"
                MAX_TOKENS = 1700
                if f"0.0, {STRATEGY}" in all_results:
                    # 이미 처리됨
                    continue
            else:
                NUM_GENERATIONS = n_generations

            all_results[f"{TEMPERATURE}, {STRATEGY}"] = {}
            all_predictions[f"{TEMPERATURE}, {STRATEGY}"] = {}
            all_ground_truths[f"{TEMPERATURE}, {STRATEGY}"] = {}

            # 서버별 동시 요청 수
            PER_SERVER_CONCURRENCY = 1

            if model == "predictiononly":
                IP_PORTs = [
                    "114.110.130.181:8000",
                    # "114.110.130.181:8001",
                    # "114.110.130.181:8002",
                    # "114.110.130.181:8003",
                    # "114.110.130.181:8004",
                    # "114.110.130.181:8005",
                    # "114.110.130.181:8006",
                    # "114.110.130.181:8007",
                ]
            elif model == "ReactionReasoner":
                IP_PORTs = [
                    # "192.169.0.2:8000",
                    # "192.169.0.2:8001",
                    # "192.169.0.2:8002",
                    # "192.169.0.2:8003",
                    # "192.169.0.2:8004",
                    # "192.169.0.2:8005",
                    # "192.169.0.2:8006",
                    # "192.169.0.2:8007",
                    "192.169.0.3:8000",
                    # "192.169.0.3:8001",
                    # "192.169.0.3:8002",
                    # "192.169.0.3:8003",
                    # "192.169.0.3:8004",
                    # "192.169.0.3:8005",
                    # "192.169.0.3:8006",
                    # "192.169.0.3:8007",
                ]
            NUM_SERVERS = len(IP_PORTs)

            # OpenAI 클라이언트(서버별)
            clients = [
                OpenAI(base_url=f"http://{ip_port}/v1", api_key="EMPTY")
                for ip_port in IP_PORTs
            ]
            tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen3-8B", trust_remote_code=True)

            # # 서버별 세마포어 (동시성 제어)
            # server_semaphores = [threading.Semaphore(PER_SERVER_CONCURRENCY) for _ in range(NUM_SERVERS)]

            evaluator = MoleculeSMILESEvaluator()

            def _extract_answer_from_content(content):
                """응답에서 <ANSWER>...</ANSWER> 태그로 감싸진 부분을 안전하게 추출"""
                try:
                    return content.split("<ANSWER>")[-1].split("</ANSWER>")[0].strip()
                except Exception:
                    return content.strip()

            def call_model_collect_n(client, messages, d, model_path="/models/ReactionReasoner", **kwargs):
                """
                한 입력(messages)에 대해 총 `num_required`개의 생성 결과를 수집하여 리스트로 반환.
                """
                collected = []
                if temp == 0.0:
                    completion = client.chat.completions.create(
                        model=model_path,
                        messages=messages,
                        n=1,
                        temperature=0.0,
                        max_tokens=1700,
                    )
                    choices = getattr(completion, "choices", None) or completion.get("choices", [])
                    for ch in choices:
                        # 여러 포맷 대비 안전한 접근
                        content = ""
                        try:
                            content = ch.message.content
                        except Exception:
                            # dict-like 혹은 다른 구조에 대비
                            content = ch.get("message", {}).get("content") if isinstance(ch, dict) else getattr(ch, "text", None) or str(ch)
                        # if "<REFLECTION>" in content:
                        #     print(f"REFLECTION found")
                        ans = _extract_answer_from_content(content or "")
                        collected.append(ans)
                elif STRATEGY == "naive_sampling":
                    completion = client.chat.completions.create(
                        model=model_path,
                        messages=messages,
                        n=NUM_GENERATIONS,
                        temperature=kwargs.get("temperature", 0.0),
                        max_tokens=kwargs.get("max_tokens", 1700),
                    )
                    choices = getattr(completion, "choices", None) or completion.get("choices", [])
                    for ch in choices:
                        # 여러 포맷 대비 안전한 접근
                        content = ""
                        try:
                            content = ch.message.content
                        except Exception:
                            # dict-like 혹은 다른 구조에 대비
                            content = ch.get("message", {}).get("content") if isinstance(ch, dict) else getattr(ch, "text", None) or str(ch)
                        # if "<REFLECTION>" in content:
                        #     print(f"REFLECTION found")
                        ans = _extract_answer_from_content(content or "")
                        collected.append(ans)
                elif STRATEGY == "force_reflection":
                    if task == "forward":
                        refl_steps = [4, 5, 6]
                    elif task == "retro":
                        refl_steps = [5, 6, 7]
                    elif task == "reagent":
                        refl_steps = [6, 7]
                    else:
                        raise ValueError(f"Unknown task: {task}")

                    for request_i in range(NUM_GENERATIONS):
                        raw_prompt = tokenizer.apply_chat_template(
                            messages,
                            tokenize=False,
                            add_generation_prompt=True,
                            enable_thinking=True,
                        )
                        for step_i, step in enumerate(refl_steps):
                            stop_strs = [f"## Step {step + 1}", "<REFLECTION>", "<ANSWER>", "</think>"]
                            response = client.completions.create(
                                model=model_path,
                                prompt=raw_prompt,
                                n=1,
                                temperature=kwargs.get("temperature", 0.0),
                                max_tokens=kwargs.get("max_tokens", 1700),
                                stop=stop_strs,
                            )
                            choices = getattr(response, "choices", None) or response.get("choices", [])
                            response = choices[0].text
                            raw_prompt += response
                            raw_prompt = remove_last_reflection_block(raw_prompt).strip()
                            step_correct = is_step_correct(step, task, d, raw_prompt)
                            if step_correct:
                                if step_i == len(refl_steps) - 1:
                                    next_tag = "\n</think>"
                                else:
                                    next_tag = f"\n\n## Step {step + 1}"
                            else:
                                if step_i == len(refl_steps) - 1:
                                    next_tag = "\n<REFLECTION>"
                                else:
                                    next_tag = "\n\n<REFLECTION>"
                            raw_prompt += next_tag
                        response = client.completions.create(
                            model=model_path,
                            prompt=raw_prompt,
                            n=1,
                            temperature=kwargs.get("temperature", 0.0),
                            max_tokens=kwargs.get("max_tokens", 1700),
                        )
                        choices = getattr(response, "choices", None) or response.get("choices", [])
                        response = choices[0].text
                        raw_prompt += response
                        ans = _extract_answer_from_content(raw_prompt or "")
                        collected.append(ans)
                else:
                    raise ValueError(f"Unknown STRATEGY: {STRATEGY}")


                return collected

            def worker_call(server_idx, messages, d, global_idx):
                """
                server_idx: which server to use (0..NUM_SERVERS-1)
                messages: chat messages
                global_idx: original data index (for debugging)
                """
                # sem = server_semaphores[server_idx]
                # client = clients[server_idx]
                # sem.acquire()
                # try:
                #     return call_model_collect_n(client, messages, d, num_required=NUM_GENERATIONS, model_path="/models/ReactionReasoner", temperature=TEMPERATURE, max_tokens=MAX_TOKENS)
                # finally:
                #     sem.release()

                ip_port = IP_PORTs[server_idx]
                client = OpenAI(base_url=f"http://{ip_port}/v1", api_key="EMPTY")
                return call_model_collect_n(
                    client,
                    messages,
                    d,
                    model_path="/models/ReactionReasoner",
                    temperature=TEMPERATURE,
                    max_tokens=MAX_TOKENS,
                )

            # ---- 작업 루프 (task별) ----
            for task in tasks:
                system_prompt = pretraining_system_prompt
                gt_molecules = []
                pred_molecules = []

                with open(f"/data/llm-reaction-reasoning/data/orderly/excluded_test/excluded_{task}_test_v10.json", "r") as f:
                    data = json.load(f)[:max_samples]
                # with open(f"/data/data/orderly/excluded_test/excluded_{task}_test_v10_required.jsonl", "r") as f:
                #     data = [json.loads(line) for line in f.readlines()][:max_samples]

                if task == "forward":
                    user_prompt_template = forward_user_prompt_templates[0]
                elif task == "retro":
                    user_prompt_template = retro_user_prompt_templates[0]
                elif task == "reagent":
                    user_prompt_template = reagent_user_prompt_templates[0]
                else:
                    raise ValueError(f"Unknown task: {task}")

                n = len(data)
                preds_by_index = [None] * n
                gts_by_index = [None] * n

                # Thread pool의 총 워커 수: 서버 수 * 서버별 동시성
                max_workers = NUM_SERVERS * PER_SERVER_CONCURRENCY
                with ProcessPoolExecutor(max_workers=max_workers) as executor:
                    futures = {}
                    for idx, d in enumerate(data):
                        # prepare input
                        if task == "forward":
                            input_smiles = f"{'.'.join(d['reactants'] + d['reagents'])}"
                            gt_keys = ["products"]
                        elif task == "retro":
                            input_smiles = ".".join(d['products'])
                            gt_keys = ["reactants"]
                        elif task == "reagent":
                            input_smiles = f"{'.'.join(d['reactants'])}>>{'.'.join(d['products'])}"
                            gt_keys = ["reagents"]

                        user_prompt = user_prompt_template.replace("[SMILES]", input_smiles)
                        gt = ""
                        for gt_key in gt_keys:
                            gt += ".".join(d[gt_key]) + "."
                        gt = gt.strip(".")
                        messages = [
                            {"role": "system", "content": system_prompt},
                            {"role": "user", "content": user_prompt}
                        ]

                        # 라운드로빈으로 서버 선택
                        server_idx = idx % NUM_SERVERS

                        fut = executor.submit(worker_call, server_idx, messages, d, idx)
                        futures[fut] = idx
                        gts_by_index[idx] = gt

                    # 진행바: futures 완료를 기다리며 업데이트
                    for fut in tqdm(
                        as_completed(futures),
                        total=len(futures),
                        desc=f"Task {task}"
                    ):
                        idx = futures[fut]
                        try:
                            preds_list = fut.result()  # 이제 list[str] (길이 NUM_GENERATIONS)
                        except Exception:
                            print(f"[ERROR] request idx={idx} failed. Filling with {NUM_GENERATIONS} empty predictions. Exception:")
                            traceback.print_exc()
                            preds_list = [""] * NUM_GENERATIONS
                        # evaluator expects e.g. [["pred1","pred2",...], ["pred1",...], ...]
                        preds_by_index[idx] = preds_list

                # 정렬된 결과 사용
                gt_molecules = gts_by_index
                pred_molecules = preds_by_index

                eval_results_passk = evaluator.evaluate_top_m(
                    predictions=pred_molecules,
                    references=gt_molecules,
                    # metrics=["exact_match"],
                )
                eval_results_passk = {k: v.item() for k, v in eval_results_passk.items()}
                eval_results_majority_vote = evaluator.evaluate_majority_vote(
                    predictions=pred_molecules,
                    references=gt_molecules,
                    metrics=["exact_match"],
                )
                eval_results_majority_vote = {k: v.item() for k, v in eval_results_majority_vote.items()}

                print("=" * 100)
                print(f"Temperature: {TEMPERATURE}, Strategy: {STRATEGY}")
                print("Evaluation Results (Pass@K):")
                print(f"Task: {task}, Model: {model}, Generations per input: {NUM_GENERATIONS}")
                print(eval_results_passk)
                print("=" * 100)
                print()
                print("=" * 100)
                print(f"Temperature: {TEMPERATURE}, Strategy: {STRATEGY}")
                print("Evaluation Results (Majority Vote):")
                print(f"Task: {task}, Model: {model}, Generations per input: {NUM_GENERATIONS}")
                print(eval_results_majority_vote)
                print("=" * 100)
                print()
                all_results[f"{TEMPERATURE}, {STRATEGY}"][task] = {
                    "pass@k": eval_results_passk,
                    "majority_vote": eval_results_majority_vote,
                }
                all_predictions[f"{TEMPERATURE}, {STRATEGY}"][task] = pred_molecules
                all_ground_truths[f"{TEMPERATURE}, {STRATEGY}"][task] = gt_molecules

        # # 결과 저장
        # with open(f"/llm-reaction-reasoning/Analysis/sampling_analysis_data/{model}_all_results.json", "w") as f:
        #     json.dump(all_results, f, indent=4)
        # with open(f"/llm-reaction-reasoning/Analysis/sampling_analysis_data/{model}_all_predictions.json", "w") as f:
        #     json.dump(all_predictions, f, indent=4)
        # with open(f"/llm-reaction-reasoning/Analysis/sampling_analysis_data/{model}_all_ground_truths.json", "w") as f:
        #     json.dump(all_ground_truths, f, indent=4)