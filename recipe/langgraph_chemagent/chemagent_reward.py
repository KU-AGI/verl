# Copyright 2024 Bytedance Ltd. and/or its affiliates
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
"""
Chemistry Molecular Design Reward.

Evaluates SMILES predictions against molecular property and fragment constraints
by calling the pre-launched tool servers (start_tool_servers.sh).

Tool servers used:
  - MoleculePropertyAnalyzer  (port 9000): molecular properties (MW, logP, rings, ...)
  - FunctionalGroups          (port 9008): substructure / functional group counts

Usage as a custom_reward_function (YAML config):
    reward:
      custom_reward_function:
        path: recipe/langgraph_agent/chemagent_reward.py
        name: compute_score
        reward_kwargs:
          base_url: http://localhost
          analyze_molecule_port: 9000
          func_group_port: 9008

The ground_truth field in the dataset's reward_model should be a JSON string (or
Python dict/list) in one of the two formats produced by chemistry_tool_agent:

  v3 (dict):  {"properties": [...], "fragments": [...]}
  v2 (list):  [{"properties": [...]}, {"fragments": [...]}]

Infeasible instances:
  Set extra_info["infeasible"] = True (or include "infeasible": true in the
  ground_truth dict) to indicate that no valid molecule exists for the task.
  The model should respond with "None" in that case.
"""

import os
import json
import logging
import re
from typing import Any

import requests
import aiohttp

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Port / URL defaults  (match start_tool_servers.sh)
# ---------------------------------------------------------------------------
_DEFAULT_BASE_URL = os.getenv("TOOL_SERVER_BASE_URL", "http://localhost")
_DEFAULT_ANALYZE_PORT = 9000  # MoleculePropertyAnalyzer
_DEFAULT_FUNC_GROUP_PORT = 9008  # FunctionalGroups

# ---------------------------------------------------------------------------
# Key mapping: dataset constraint name → tool output key
#
# benchmark_v4 property names already match MoleculePropertyAnalyzer output
# keys exactly (MW, logP, HBD, HBA, TPSA, logD, logS, rotB, rings_total,
# formal_charge, heavy_atoms, BBBP, HIA, QED, Mutag, ...).
# The fallback PROP_KEY_MAP.get(name, name) handles all unregistered names.
# Old-format aliases are kept for backward compatibility.
# ---------------------------------------------------------------------------
PROP_KEY_MAP = {
    # legacy benchmark aliases → tool key
    "Atom Count": "heavy_atoms",
    "Charge": "formal_charge",
    "Mutagenicity": "Mutag",
    "Rings": "rings_total",
    "RotB": "rotB",
}

# benchmark_v4 fragment names are already the common names returned by
# FunctionalGroups.exchange_fragment_counts_to_names() via FR_DISPLAY_NAME_OVERRIDES.
# No mapping needed — the dict below is kept empty intentionally.
# (Old evaluate_benchmark.py used "methoxy groups -och3" etc., but the current
# tool returns short common names: "methoxy", "nitroso", "alkyl carbamate", "benzodiazepine")
FRAG_KEY_MAP: dict[str, str] = {}

# Error prefixes returned by tool servers on failure
_ERROR_PREFIXES = (
    "Initialization Error:",
    "Execution Error:",
    "SMILES Syntax Error:",
    "Input Argument Error:",
)


# ---------------------------------------------------------------------------
# SMILES utilities
# ---------------------------------------------------------------------------

def validate_smiles(smiles: str) -> bool:
    """Return True iff *smiles* is a non-empty, parseable SMILES string."""
    if not smiles or smiles.strip().lower() == "none" or not isinstance(smiles, str):
        return False
    try:
        from rdkit import Chem
        mol = Chem.MolFromSmiles(smiles.strip())
        return mol is not None
    except Exception:
        return False


def extract_smiles_from_response(response_str: str) -> str:
    """
    Extract the final SMILES prediction from the agent response string.

    Priority:
      1. Content inside the last <ANSWER>...</ANSWER> tag.
      2. The raw response string (stripped).

    Returns an empty string if nothing useful is found.
    """
    if not isinstance(response_str, str):
        return ""

    # Look for <ANSWER>...</ANSWER> tags (case-insensitive)
    matches = re.findall(r"<ANSWER>(.*?)</ANSWER>", response_str, re.IGNORECASE | re.DOTALL)
    if matches:
        return matches[-1].strip()

    return response_str.strip()


# ---------------------------------------------------------------------------
# Ground-truth parsing  (mirrors evaluate_benchmark.py::_parse_answer)
# ---------------------------------------------------------------------------

def _parse_answer(answer) -> tuple[dict, dict]:
    """
    Normalize *answer* to (props_source, frags_source).

    Handles v2 list format and v3 dict format.
    """
    if isinstance(answer, dict):
        return answer, answer
    if isinstance(answer, list):
        props = answer[0] if len(answer) > 0 else {}
        frags = answer[1] if len(answer) > 1 else {}
        return props, frags
    return {}, {}


def _load_ground_truth(ground_truth) -> dict | list:
    """Parse ground_truth from JSON string or pass through if already structured."""
    if isinstance(ground_truth, (dict, list)):
        return ground_truth
    if isinstance(ground_truth, str):
        try:
            return json.loads(ground_truth)
        except json.JSONDecodeError:
            pass
    return {}


# ---------------------------------------------------------------------------
# Property / fragment satisfaction checks
# (mirrors evaluate_benchmark.py)
# ---------------------------------------------------------------------------

def check_properties(target_props: list[dict], measured_props: dict) -> dict[str, bool]:
    """Return {prop_name: satisfied} for each property constraint."""
    results = {}
    for prop_item in target_props:
        prop_name = prop_item["property"]
        tool_key = PROP_KEY_MAP.get(prop_name, prop_name)
        satisfied = False
        if tool_key in measured_props:
            val = measured_props[tool_key]
            if "range" in prop_item:
                p_min = prop_item["range"].get("min", float("-inf"))
                p_max = prop_item["range"].get("max", float("inf"))
                satisfied = p_min <= val <= p_max
            elif "min" in prop_item or "max" in prop_item:
                p_min = prop_item.get("min", float("-inf"))
                p_max = prop_item.get("max", float("inf"))
                satisfied = p_min <= val <= p_max
            elif "value" in prop_item:
                satisfied = val == prop_item["value"]
        results[prop_name] = satisfied
    return results


def check_fragments(target_frags: list[dict], measured_frags: dict) -> dict[str, bool]:
    """Return {frag_name: satisfied} for each fragment constraint."""
    results = {}
    for frag_dict in target_frags:
        for name, count in frag_dict.items():
            tool_key = FRAG_KEY_MAP.get(name, name)
            results[name] = measured_frags.get(tool_key) == count
    return results


# ---------------------------------------------------------------------------
# HTTP calls to tool servers
# ---------------------------------------------------------------------------

def _load_tool_classes() -> dict:
    """Try to import chemagent.tools.TOOLS_CLASS for in-process fallback.

    TOOLS_CLASS is keyed by tool function name (e.g. 'analyze_overall_molecule_properties'),
    so we build a reverse mapping keyed by class name (e.g. 'MoleculePropertyAnalyzer').
    """
    try:
        import importlib
        tools_module = importlib.import_module("chemagent.tools")
        return {cls.__name__: cls for cls in tools_module.TOOLS_CLASS.values()}
    except Exception:
        pass
    return {}


# Lazy-loaded in-process tool instances (class name → instance)
_fallback_instances: dict[str, Any] = {}
_tools_class: dict | None = None


def _get_fallback_instance(class_name: str):
    """Get or create a fallback in-process tool instance."""
    global _tools_class
    if _tools_class is None:
        _tools_class = _load_tool_classes()
    if class_name not in _tools_class:
        return None
    if class_name not in _fallback_instances:
        _fallback_instances[class_name] = _tools_class[class_name]()
    return _fallback_instances[class_name]


class ChemistryToolServerClient:
    """Thin HTTP client for the pre-launched chemistry tool servers.
    Falls back to in-process execution via chemagent.tools when server is unreachable."""

    def __init__(
        self,
        base_url: str = _DEFAULT_BASE_URL,
        analyze_molecule_port: int = _DEFAULT_ANALYZE_PORT,
        func_group_port: int = _DEFAULT_FUNC_GROUP_PORT,
        timeout: int = 60,
        use_http: bool = True,
    ):
        self.analyze_url = f"{base_url}:{analyze_molecule_port}/call"
        self.func_group_url = f"{base_url}:{func_group_port}/call"
        self.timeout = timeout
        self.use_http = use_http

    def _parse_analyze_result(self, result, smiles_list: list[str]) -> dict[str, dict]:
        if isinstance(result, str) and any(result.startswith(e) for e in _ERROR_PREFIXES):
            raise RuntimeError(f"MoleculePropertyAnalyzer error: {result.splitlines()[0]}")
        res_map: dict[str, dict] = {}
        if isinstance(result, list):
            for entry in result:
                if isinstance(entry, dict):
                    res_map.update(entry)
        elif isinstance(result, dict):
            res_map = {smiles_list[0]: result}
        return res_map

    def analyze_molecules(self, smiles_list: list[str]) -> dict[str, dict]:
        """
        Call MoleculePropertyAnalyzer (port 9000) for a batch of SMILES.
        Falls back to in-process execution if HTTP fails or use_http=False.
        """
        payload = {"mol_smiles_list": smiles_list}
        http_err = None

        if self.use_http:
            try:
                resp = requests.post(self.analyze_url, json={**payload, "return_text": False}, timeout=self.timeout)
                resp.raise_for_status()
                result = resp.json().get("result", {})
                return self._parse_analyze_result(result, smiles_list)
            except Exception as http_exc:
                http_err = http_exc
                logger.warning(f"MoleculePropertyAnalyzer HTTP failed ({http_exc}); trying in-process fallback")

        # In-process fallback
        instance = _get_fallback_instance("MoleculePropertyAnalyzer")
        if instance is None:
            raise RuntimeError(
                f"MoleculePropertyAnalyzer HTTP failed and no in-process fallback available: {http_err}"
            )
        try:
            result = instance._run_base(query=payload, return_text=False)
            if isinstance(result, str):
                result = json.loads(result)
            return self._parse_analyze_result(result, smiles_list)
        except Exception as fallback_exc:
            raise RuntimeError(f"MoleculePropertyAnalyzer in-process fallback failed: {fallback_exc}") from fallback_exc

    def get_functional_groups(self, smiles: str) -> dict:
        """
        Call FunctionalGroups (port 9008) for a single SMILES.
        Falls back to in-process execution if HTTP fails or use_http=False.
        """
        payload = {"mol_smiles": smiles}
        http_err = None

        if self.use_http:
            try:
                resp = requests.post(self.func_group_url, json={**payload, "return_text": False}, timeout=self.timeout)
                resp.raise_for_status()
                result = resp.json().get("result", {})
                if isinstance(result, str) and any(result.startswith(e) for e in _ERROR_PREFIXES):
                    raise RuntimeError(f"FunctionalGroups error: {result.splitlines()[0]}")
                return result if isinstance(result, dict) else {}
            except Exception as http_exc:
                http_err = http_exc
                logger.warning(f"FunctionalGroups HTTP failed ({http_exc}); trying in-process fallback")

        # In-process fallback
        instance = _get_fallback_instance("FunctionalGroups")
        if instance is None:
            raise RuntimeError(
                f"FunctionalGroups HTTP failed and no in-process fallback available: {http_err}"
            )
        try:
            result = instance._run_base(query=payload, return_text=False)
            if isinstance(result, str):
                result = json.loads(result)
            return result if isinstance(result, dict) else {}
        except Exception as fallback_exc:
            raise RuntimeError(f"FunctionalGroups in-process fallback failed: {fallback_exc}") from fallback_exc


# ---------------------------------------------------------------------------
# Tool-level reward
# ---------------------------------------------------------------------------

def _validate_args_against_schema(args: dict, schema: dict) -> bool:
    """Return True iff args satisfies the JSON schema (required fields + types)."""
    try:
        import jsonschema
        jsonschema.validate(instance=args, schema=schema)
        return True
    except Exception:
        return False


def _compute_tool_reward(
    turn_tool_calls: list[list[dict]],
    turn_invalid_tool_calls: list[list[dict]],
    tool_schemas: dict[str, dict],
) -> dict[str, float]:
    """
    Compute tool-call quality reward in [0, 1].

    Per-call scoring:
      - invalid JSON (in turn_invalid_tool_calls)  → 0
      - valid JSON but schema mismatch             → 0
      - valid JSON and schema satisfied            → 1

    tool_reward = mean score across all calls.
    Returns 1.0 when there are no tool calls.

    Returns:
      tool_reward      – mean per-call score
      tool_calls       – total calls attempted
      tool_schema_fail – calls that failed schema validation
      tool_json_fail   – calls with invalid JSON
    """
    scores = []

    # invalid JSON calls — one entry per bad call
    for turn_invalids in turn_invalid_tool_calls:
        for _ in turn_invalids:
            scores.append(0.0)

    # valid JSON calls — check schema
    schema_fails = 0
    for turn_valids in turn_tool_calls:
        for call in turn_valids:
            schema = tool_schemas.get(call["name"]) if tool_schemas else None
            if schema is None:
                # unknown tool or no schema available → skip schema check
                scores.append(1.0)
            elif _validate_args_against_schema(call["args"], schema):
                scores.append(1.0)
            else:
                scores.append(0.0)
                schema_fails += 1

    total_calls = len(scores)
    json_fails = sum(len(t) for t in turn_invalid_tool_calls)
    tool_reward = sum(scores) / total_calls if total_calls > 0 else 0.0

    return {
        "tool_reward":      tool_reward,
        "tool_calls":       total_calls,
        "tool_json_fail":   json_fails,
        "tool_schema_fail": schema_fails,
    }


# ---------------------------------------------------------------------------
# Main reward class
# ---------------------------------------------------------------------------

class ChemistryMolDesignReward:
    """
    Reward evaluator for chemistry molecular design tasks.

    Calls pre-launched HTTP tool servers to measure whether a predicted SMILES
    satisfies the molecular property and fragment constraints in ground_truth.

    Score breakdown:
      - Infeasible task, model outputs "None"  → 1.0
      - Infeasible task, model outputs SMILES  → 0.0
      - Feasible task, invalid SMILES          → 0.0
      - Feasible task, valid SMILES            →  1.0 # (n_prop_ok + n_frag_ok) / (n_prop + n_frag)
                                                  (partial credit; 1.0 iff all constraints met)
    """

    def __init__(
        self,
        base_url: str = _DEFAULT_BASE_URL,
        analyze_molecule_port: int = _DEFAULT_ANALYZE_PORT,
        func_group_port: int = _DEFAULT_FUNC_GROUP_PORT,
        timeout: int = 60,
        use_http: bool = True,
    ):
        self.client = ChemistryToolServerClient(
            base_url=base_url,
            analyze_molecule_port=analyze_molecule_port,
            func_group_port=func_group_port,
            timeout=timeout,
            use_http=use_http,
        )

    def _is_none_prediction(self, pred: str) -> bool:
        return pred.strip().lower() in ("none", "")

    def __call__(
        self,
        solution_str: str,
        ground_truth: Any,
        extra_info: dict | None = None,
    ) -> dict[str, float]:
        """
        Evaluate a single prediction.

        Args:
            solution_str: Raw decoded response from the agent.
            ground_truth: JSON string or dict/list with property/fragment constraints.
            extra_info:   Optional dict; may contain ``infeasible: True``.

        Returns:
            dict with keys:
              score       – combined reward in [0, 1]
              prop_score  – fraction of property constraints satisfied
              frag_score  – fraction of fragment constraints satisfied (NaN if none)
              valid_smiles – 1 if predicted SMILES is valid, 0 otherwise
        """
        extra_info = extra_info or {}

        # --- Tool call data from agent loop (set by convert_to_agent_output) ---
        turn_tool_calls         = extra_info.get("turn_tool_calls", [])
        turn_invalid_tool_calls = extra_info.get("turn_invalid_tool_calls", [])
        tool_schemas            = extra_info.get("tool_schemas", {})

        # --- Parse ground truth ---
        gt = _load_ground_truth(ground_truth)

        # Infeasible flag can come from extra_info or from the gt dict itself
        infeasible = extra_info.get("infeasible", False)
        if isinstance(gt, dict):
            infeasible = infeasible or gt.get("infeasible", False)

        props_src, frags_src = _parse_answer(gt)
        target_props: list[dict] = props_src.get("properties", [])
        target_frags: list[dict] = frags_src.get("fragments", [])

        # --- Tool-level reward (shared across all branches) ---
        tool_info = _compute_tool_reward(turn_tool_calls, turn_invalid_tool_calls, tool_schemas)

        # --- Extract prediction SMILES ---
        pred = extract_smiles_from_response(solution_str)

        # --- Infeasible branch ---
        if infeasible:
            correct = self._is_none_prediction(pred)
            s = 1.0 if correct else 0.0
            return {"score": s, "prop_score": s, "frag_score": s, "valid_smiles": 0.0, **tool_info}

        # --- Feasible branch: validate SMILES ---
        if not validate_smiles(pred):
            return {"score": 0.0, "prop_score": 0.0, "frag_score": float("nan"), "valid_smiles": 0.0, **tool_info}

        # --- Call tool servers ---
        n_prop_ok = 0
        n_prop_total = len(target_props)
        n_frag_ok = 0
        n_frag_total = sum(len(fd) for fd in target_frags)

        # Property evaluation
        prop_score = 0.0
        if n_prop_total > 0:
            try:
                res_map = self.client.analyze_molecules([pred])
                p_res = check_properties(target_props, res_map.get(pred, {}))
                n_prop_ok = sum(p_res.values())
                prop_score = n_prop_ok / n_prop_total
            except RuntimeError as exc:
                logger.warning(f"Property analysis failed for {pred}: {exc}")
                prop_score = 0.0
        else:
            prop_score = 1.0  # no property constraint → full credit

        # Fragment evaluation
        frag_score = float("nan")
        if n_frag_total > 0:
            try:
                fg_res = self.client.get_functional_groups(pred)
                f_res = check_fragments(target_frags, fg_res)
                n_frag_ok = sum(f_res.values())
                frag_score = n_frag_ok / n_frag_total
            except RuntimeError as exc:
                logger.warning(f"Fragment analysis failed for {pred}: {exc}")
                frag_score = 0.0
                n_frag_total = 0  # skip in combined score

        # --- Combined score ---
        total_constraints = n_prop_total + (n_frag_total if n_frag_total > 0 else 0)
        if total_constraints > 0:
            score = int(n_prop_ok == n_frag_ok == 1) # (n_prop_ok + n_frag_ok) / total_constraints
        else:
            # No constraints at all: reward for producing a valid SMILES
            score = 1.0

        return {
            "score": score,
            "prop_score": prop_score,
            "frag_score": frag_score,
            "valid_smiles": 1.0,
            **tool_info,
        }


# ---------------------------------------------------------------------------
# Turn-consistency reward via LLM judge
# ---------------------------------------------------------------------------

_TURN_JUDGE_SYSTEM_PROMPT = """\
You are an expert chemistry agent evaluator.
Given consecutive reasoning steps and tool calls of a chemistry problem-solving agent, \
evaluate whether the agent's reasoning and tool usage is improving and non-redundant across turns.

Respond with JSON only:
{
  "score": <float 0.0–1.0>,
  "reason": "<brief explanation>"
}

Scoring guide:
- 1.0: reasoning builds on previous turn, tool calls are diverse and purposeful
- 0.5: some progress but partially redundant reasoning or repeated tool calls
- 0.0: identical or circular reasoning/tool calls, no progress
"""


def _format_turn_pair(
    turn_idx: int,
    prev_tools: list[dict],
    prev_reason: str,
    curr_tools: list[dict],
    curr_reason: str,
) -> str:
    return (
        f"=== Turn {turn_idx} → Turn {turn_idx + 1} ===\n"
        f"[Turn {turn_idx}] Reasoning:\n{prev_reason or '(none)'}\n"
        f"[Turn {turn_idx}] Tool calls: {json.dumps(prev_tools, ensure_ascii=False)}\n\n"
        f"[Turn {turn_idx + 1}] Reasoning:\n{curr_reason or '(none)'}\n"
        f"[Turn {turn_idx + 1}] Tool calls: {json.dumps(curr_tools, ensure_ascii=False)}\n"
    )


async def _query_turn_judge(
    session: aiohttp.ClientSession,
    judge_url: str,
    judge_model: str,
    user_content: str,
    timeout: int = 30,
) -> float:
    """Send a single turn-pair to the judge server and return a score in [0, 1]."""
    payload = {
        "model": judge_model,
        "messages": [
            {"role": "system", "content": _TURN_JUDGE_SYSTEM_PROMPT},
            {"role": "user", "content": user_content},
        ],
        "temperature": 0.0,
        "max_tokens": 256,
    }
    try:
        async with session.post(judge_url, json=payload, timeout=aiohttp.ClientTimeout(total=timeout)) as resp:
            resp.raise_for_status()
            data = await resp.json()
            text = data["choices"][0]["message"]["content"]
            result = json.loads(text)
            return float(result.get("score", 0.5))
    except Exception as exc:
        logger.warning(f"Turn judge request failed: {exc}")
        return 0.5  # neutral fallback


async def compute_turn_consistency_reward(
    turn_tool_calls: list[list[dict]],
    turn_reasoning: list[str],
    judge_url: str = "http://localhost:8000/v1/chat/completions",
    judge_model: str = "judge",
    timeout: int = 30,
) -> dict[str, float]:
    """
    Compare consecutive turns' tool_calls and reasoning_content via LLM judge.

    Args:
        turn_tool_calls: Per-turn tool call list, from extra_info["turn_tool_calls"].
        turn_reasoning:  Per-turn reasoning text, from extra_info["turn_reasoning"].
        judge_url:       OpenAI-compatible chat completions endpoint.
        judge_model:     Model name to pass in the request.

    Returns:
        {
          "turn_consistency_reward": float,   # mean score across all consecutive pairs
          "turn_consistency_scores": list[float],
        }
    """
    n = min(len(turn_tool_calls), len(turn_reasoning))
    if n < 2:
        return {"turn_consistency_reward": 1.0, "turn_consistency_scores": []}

    async with aiohttp.ClientSession() as session:
        tasks = []
        for i in range(n - 1):
            content = _format_turn_pair(
                i,
                prev_tools=turn_tool_calls[i],
                prev_reason=turn_reasoning[i],
                curr_tools=turn_tool_calls[i + 1],
                curr_reason=turn_reasoning[i + 1],
            )
            tasks.append(_query_turn_judge(session, judge_url, judge_model, content, timeout))

        import asyncio
        scores = await asyncio.gather(*tasks)

    mean_score = sum(scores) / len(scores) if scores else 1.0
    return {
        "turn_consistency_reward": mean_score,
        "turn_consistency_scores": list(scores),
    }


# ---------------------------------------------------------------------------
# Module-level compute_score (plug-in for verl custom_reward_function)
# ---------------------------------------------------------------------------

# Lazily initialized singleton to avoid re-creating HTTP session objects per call.
_reward_instance: ChemistryMolDesignReward | None = None


def _get_reward_instance(
    base_url: str = _DEFAULT_BASE_URL,
    analyze_molecule_port: int = _DEFAULT_ANALYZE_PORT,
    func_group_port: int = _DEFAULT_FUNC_GROUP_PORT,
    timeout: int = 60,
    use_http: bool = True,
) -> ChemistryMolDesignReward:
    global _reward_instance
    if _reward_instance is None:
        _reward_instance = ChemistryMolDesignReward(
            base_url=base_url,
            analyze_molecule_port=analyze_molecule_port,
            func_group_port=func_group_port,
            timeout=timeout,
            use_http=use_http,
        )
    return _reward_instance


def compute_score(
    data_source: str,
    solution_str: str,
    ground_truth: Any,
    extra_info: dict | None = None,
    # verl passes these as keyword args from reward_kwargs
    base_url: str = _DEFAULT_BASE_URL,
    analyze_molecule_port: int = _DEFAULT_ANALYZE_PORT,
    func_group_port: int = _DEFAULT_FUNC_GROUP_PORT,
    timeout: int = 60,
    use_http: bool = os.getenv("TOOLS_USE_HTTP", "true").lower() not in ("false", "0"),
    tool_weight: float = 0.1,
    # turn consistency judge
    turn_consistency_weight: float = 0.0,
    judge_url: str = "http://localhost:8000/v1/chat/completions",
    judge_model: str = "judge",
    judge_timeout: int = 30,
    **kwargs,
) -> dict[str, float]:
    """
    Entry point for verl ``custom_reward_function``.

    Compatible signature:
        compute_score(data_source, solution_str, ground_truth, extra_info=None, **reward_kwargs)

    Args:
        tool_weight: Weight given to tool-quality reward when blending with
                     the answer-correctness score.
                     final_score = answer_score * (1 - tool_weight) + tool_reward * tool_weight
                     Set to 0.0 to disable tool reward.

    Returns a dict so that reward_extra_info in NaiveRewardManager is populated
    with prop_score, frag_score, valid_smiles, and tool_* keys.
    """
    reward_fn = _get_reward_instance(
        base_url=base_url,
        analyze_molecule_port=analyze_molecule_port,
        func_group_port=func_group_port,
        timeout=timeout,
        use_http=use_http,
    )
    result = reward_fn(solution_str=solution_str, ground_truth=ground_truth, extra_info=extra_info)

    if tool_weight > 0.0:
        answer_score = result["score"]
        tool_reward = result.get("tool_reward", 1.0)
        result["score"] = answer_score * (1.0 - tool_weight) + tool_reward * tool_weight

    # # Turn consistency reward (optional)
    # if turn_consistency_weight > 0.0 and extra_info:
    #     turn_tool_calls = extra_info.get("turn_tool_calls", [])
    #     turn_reasoning = extra_info.get("turn_reasoning", [])
    #     if len(turn_tool_calls) >= 2:
    #         import asyncio
    #         turn_info = asyncio.run(
    #             compute_turn_consistency_reward(
    #                 turn_tool_calls=turn_tool_calls,
    #                 turn_reasoning=turn_reasoning,
    #                 judge_url=judge_url,
    #                 judge_model=judge_model,
    #                 timeout=judge_timeout,
    #             )
    #         )
    #         result.update(turn_info)
    #         result["score"] = (
    #             result["score"] * (1.0 - turn_consistency_weight)
    #             + turn_info["turn_consistency_reward"] * turn_consistency_weight
    #         )

    return result
