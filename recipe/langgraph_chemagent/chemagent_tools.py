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
Chemistry Tools Adapter for LangChain/LangGraph.

This module converts chemistry tools from BaseTool format to LangChain tool format,
enabling their use in ReactAgentLoop for RL training.
"""

import importlib
import logging
import sys
from pathlib import Path
from typing import Any

import requests as _requests
from langchain_core.tools import StructuredTool
from pydantic import BaseModel, Field, create_model

logger = logging.getLogger(__name__)

# Characters that cannot appear in a valid SMILES string.
# If a model-generated argument contains these, it is almost certainly a hallucinated
# JSON/HTML fragment rather than a real SMILES, and passing it to RDKit in-process
# can cause SIGSEGV that kills the Ray worker.
_SMILES_FORBIDDEN_CHARS = frozenset('<>{}"\x00\x01\x02\x03\x04\x05\x06\x07\x08')


def _is_plausible_smiles(value) -> bool:
    """Lightweight SMILES sanity check that does NOT call RDKit.

    Returns False if *value* is obviously not a SMILES string (e.g. a JSON
    fragment like ``<tool_call>{"name": ...}``).  Used as a fast pre-filter
    before the more thorough RDKit validation in ``_validate_smiles_rdkit``.
    """
    if isinstance(value, list):
        return all(_is_plausible_smiles(v) for v in value)
    if not isinstance(value, str):
        return True  # non-string args are not SMILES, skip check
    s = value.strip()
    if not s or len(s) > 2000:
        return False
    return not any(c in _SMILES_FORBIDDEN_CHARS for c in s)


def _validate_smiles_rdkit(value) -> bool:
    """Validate SMILES with RDKit (safe: MolFromSmiles never SIGSEGVs).

    Used to catch syntactically-invalid SMILES before they reach downstream
    ML models (e.g. ADMET property predictors) that may SIGSEGV on bad input.
    Returns True for non-string/non-list values (not our job to validate).
    """
    try:
        from rdkit import Chem
        if isinstance(value, list):
            return all(_validate_smiles_rdkit(v) for v in value)
        if not isinstance(value, str):
            return True
        s = value.strip()
        if not s:
            return False
        return Chem.MolFromSmiles(s) is not None
    except Exception:
        return False


# Mapping from JSON schema tool names to tool server class names + ports.
# The JSON schema uses descriptive function names while the tool servers
# use class-style names.  This mapping lets us dispatch HTTP calls to the
# correct server regardless of which naming convention was used.
_SCHEMA_NAME_TO_SERVER = {
    "analyze_overall_molecule_properties": "MoleculePropertyAnalyzer",
    "convert_iupac_to_smiles": "IUPAC2SMILES",
    "calculate_tanimoto_similarity": "MolSimilarity",
    "canonicalize_smiles": "CanonicalizeSMILES",
    "count_molecule_atoms": "CountMolAtoms",
    "calculate_molecular_weight": "SMILES2Weight",
    "convert_chemical_name_to_smiles": "Name2SMILES",
    "convert_selfies_to_smiles": "SELFIES2SMILES",
    "get_functional_groups": "FunctionalGroups",
    "query_moses_by_fg_counts_prop_ranges": "DBSearch",
    "generate_molecule_with_fragments": "MoleculeGenerator",
    "edit_molecules_by_functional_groups": "MoleculeEditor",
    "search_wikipedia": "WikipediaSearch",
    "search_pubchem": "PubchemSearch",
    "generate_molecules_by_property_ranges": "MolMIMPropertyRangeGenerator",
    "optimize_molecule_by_prop_ranges": "MolMIMPropertyRangeOptimizer",
    "run_python_code": "PythonShell",
}


def load_chemistry_tools(tools_path: str) -> tuple[dict, dict, dict]:
    """
    Load chemistry tools from the specified path.

    Tries to import chemagent.tools first. If that fails (e.g. rdkit not
    installed in the training env), falls back to a pre-generated JSON schema
    file so that HTTP-dispatch tools can still be created without rdkit.

    Args:
        tools_path: Path to chemistry tools directory (e.g., '/path/to/chemagent/tools')

    Returns:
        tuple: (TOOLS_JSON_SCHEMA dict, TOOLS_CLASS dict, TOOL_SERVER_PORTS dict)
               TOOLS_CLASS is {} when loaded from JSON fallback (HTTP-only mode).
    """
    tools_path = Path(tools_path)

    # --- Try direct import first ---
    if tools_path.exists():
        # tools_path = .../chemistry_tool_agent_clean/chemagent/tools
        # repo root  = .../chemistry_tool_agent_clean/
        repo_root = str(tools_path.parent.parent)
        if repo_root not in sys.path:
            sys.path.insert(0, repo_root)

        try:
            tools_module = importlib.import_module('chemagent.tools')
            TOOLS_JSON_SCHEMA = tools_module.TOOLS_JSON_SCHEMA
            TOOLS_CLASS = tools_module.TOOLS_CLASS
            TOOL_SERVER_PORTS = getattr(tools_module, 'TOOL_SERVER_PORTS', {})
            print(f"[load_chemistry_tools] imported chemagent.tools: {len(TOOLS_CLASS)} tools")
            return TOOLS_JSON_SCHEMA, TOOLS_CLASS, TOOL_SERVER_PORTS
        except Exception as e:
            print(f"[load_chemistry_tools] chemagent import failed ({e}), trying JSON fallback")

    # --- JSON fallback: schemas pre-generated from chemagent env ---
    json_path = Path(__file__).parent / "chemagent_tool_schemas.json"
    if json_path.exists():
        import json as _json
        data = _json.loads(json_path.read_text())
        TOOLS_JSON_SCHEMA = data["schemas"]
        TOOL_SERVER_PORTS = data["ports"]
        print(f"[load_chemistry_tools] loaded {len(TOOLS_JSON_SCHEMA)} tools from {json_path}")
        return TOOLS_JSON_SCHEMA, {}, TOOL_SERVER_PORTS

    raise FileNotFoundError(
        f"Chemistry tools not found: chemagent import failed and no JSON fallback at {json_path}. "
        "Generate it with: conda run -n chemagent python -c \"...\""
    )


def create_pydantic_model_from_properties(tool_name: str, properties: dict) -> type[BaseModel]:
    """
    Create a Pydantic model from OpenAI function properties schema.

    Args:
        tool_name: Name of the tool (used for model naming)
        properties: Properties dict from tool schema

    Returns:
        Pydantic BaseModel class
    """
    fields = {}
    for prop_name, prop_info in properties.items():
        # Map JSON schema types to Python types
        type_mapping = {
            'string': str,
            'integer': int,
            'number': float,
            'boolean': bool,
            'array': list,
            'object': dict,
        }

        prop_type = prop_info.get('type', 'string')
        python_type = type_mapping.get(prop_type, str)

        # Create Field with description
        field_description = prop_info.get('description', '')
        fields[prop_name] = (python_type, Field(description=field_description))

    # Create dynamic Pydantic model
    model = create_model(f"{tool_name}Input", **fields)
    return model


def convert_chemistry_tool_to_langchain(
    tool_key: str,
    tool_class: type,
    tool_schema: dict,
    port: int = None,
    base_url: str = "http://localhost",
    use_http: bool = True,
) -> StructuredTool:
    """
    Convert a chemistry BaseTool to LangChain StructuredTool.

    If *use_http* is True and *port* is given, tool calls are dispatched via
    HTTP POST to the pre-launched tool server (start_tool_servers.sh).
    Falls back to in-process execution when the HTTP call fails.

    If *use_http* is False, tool calls run in-process directly (no HTTP).

    Args:
        tool_key: Tool identifier key
        tool_class: BaseTool class
        tool_schema: OpenAI function schema
        port: TCP port of the running tool server (None -> in-process only)
        base_url: Base URL of the tool server (default: http://localhost)
        use_http: If True, dispatch via HTTP; if False, run in-process directly

    Returns:
        LangChain StructuredTool
    """
    # Extract schema info
    function_schema = tool_schema.get('function', {})
    func_name = function_schema.get('name', tool_key)
    description = function_schema.get('description', '')
    properties = function_schema.get('parameters', {}).get('properties', {})

    # Create Pydantic model for args
    args_schema = create_pydantic_model_from_properties(func_name, properties)

    # Determine the HTTP endpoint.
    # PythonShell uses /execute with different payload keys.
    is_python_shell = (
        _SCHEMA_NAME_TO_SERVER.get(tool_key) == "PythonShell"
        or tool_key == "PythonShell"
        or func_name == "run_python_code"
    )

    if use_http and port is not None:
        if is_python_shell:
            server_url = f"{base_url}:{port}/execute"
        else:
            server_url = f"{base_url}:{port}/call"
    else:
        server_url = None

    # Lazy-loaded in-process instance (only created if HTTP is unavailable or disabled)
    _fallback_instance: list = []  # use list as mutable container for nonlocal

    def tool_func(**kwargs) -> str:
        """Dispatch to HTTP tool server (use_http=True) or run in-process (use_http=False)."""
        if server_url is not None:
            try:
                if is_python_shell:
                    # PythonShell expects {convid, code} at /execute
                    payload = {
                        "convid": kwargs.get("conv_id", "default"),
                        "code": kwargs.get("python_code", ""),
                    }
                else:
                    payload = {**kwargs, "return_text": True}

                response = _requests.post(
                    server_url,
                    json=payload,
                    timeout=60,
                )
                response.raise_for_status()
                return str(response.json().get("result", ""))
            except Exception as e:
                if use_http:
                    # When HTTP mode is enabled, do not fall back to in-process execution.
                    # In-process RDKit/chemagent code can SIGSEGV on malformed inputs,
                    # killing the Ray worker and the entire trainer.
                    logger.error(f"HTTP call to {server_url} failed ({e}); returning error (in-process fallback disabled for use_http=True)")
                    return f"Error: tool server at {server_url} unreachable: {e}"
                logger.warning(
                    f"HTTP call to {server_url} failed ({e}); falling back to in-process execution"
                )

        # In-process execution (use_http=False only)
        if tool_class is None:
            return (
                f"Error: in-process execution unavailable (no tool class) and "
                f"HTTP server at {server_url} unreachable"
            )

        # Guard against malformed model outputs that can SIGSEGV in-process tools
        # (e.g. ADMET ML models crash on invalid molecular graphs).
        # Two-stage: fast char filter first, then full RDKit parse.
        for key, val in kwargs.items():
            if "smiles" not in key.lower():
                continue
            if not _is_plausible_smiles(val):
                logger.error(
                    f"Rejecting in-process call to {func_name}: "
                    f"argument '{key}' contains non-SMILES characters: {repr(str(val))[:120]}"
                )
                return f"Error: argument '{key}' is not a valid SMILES string"
            if not _validate_smiles_rdkit(val):
                logger.error(
                    f"Rejecting in-process call to {func_name}: "
                    f"argument '{key}' failed RDKit validation: {repr(str(val))[:120]}"
                )
                return f"Error: argument '{key}' is not a valid SMILES string"

        if not _fallback_instance:
            _fallback_instance.append(tool_class())
        try:
            result = _fallback_instance[0]._run_base(query=kwargs)
            return str(result)
        except Exception as e:
            logger.error(f"Error executing {func_name}: {e}")
            return f"Error: {str(e)}"

    # Create LangChain StructuredTool
    langchain_tool = StructuredTool(
        name=func_name,
        description=description,
        func=tool_func,
        args_schema=args_schema,
    )

    return langchain_tool


def get_chemistry_tools(
    tools_path: str,
    selected_tools: list[str] = None,
    base_url: str = "http://localhost",
    use_http: bool = True,
) -> list[StructuredTool]:
    """
    Load and convert chemistry tools to LangChain format.

    If *use_http* is True (default), tool calls are dispatched via HTTP to
    pre-launched tool servers, with automatic in-process fallback on failure.
    If *use_http* is False, tools run in-process directly (no HTTP).

    Args:
        tools_path: Path to chemistry tools directory
        selected_tools: Optional list of tool keys to load (if None, loads all)
        base_url: Base URL of the tool servers (default: http://localhost)
        use_http: If True, use HTTP tool servers; if False, run in-process

    Returns:
        List of LangChain StructuredTools
    """
    TOOLS_JSON_SCHEMA, TOOLS_CLASS, TOOL_SERVER_PORTS = load_chemistry_tools(tools_path)

    langchain_tools = []

    # Determine which tools to load
    # Fall back to TOOLS_JSON_SCHEMA keys when TOOLS_CLASS is empty (JSON schema mode)
    tools_to_load = selected_tools if selected_tools else list(TOOLS_JSON_SCHEMA.keys())

    # PythonShell always communicates with an external HTTP server internally
    # (localhost:8888). Running it without that server blocks for the full
    # requests timeout on every call, stalling all agent workers.
    _HTTP_ONLY_TOOLS = {"run_python_code"}

    for tool_key in tools_to_load:
        if tool_key not in TOOLS_JSON_SCHEMA:
            logger.warning(f"Tool {tool_key} not found in registered tools")
            continue

        if not use_http and tool_key in _HTTP_ONLY_TOOLS:
            logger.warning(
                f"Skipping tool '{tool_key}': requires an external HTTP server "
                f"(port 8888) which is not available in use_http=False mode."
            )
            continue

        try:
            tool_class = TOOLS_CLASS.get(tool_key)  # None in JSON schema mode
            tool_schema = TOOLS_JSON_SCHEMA[tool_key]
            port = TOOL_SERVER_PORTS.get(tool_key)

            langchain_tool = convert_chemistry_tool_to_langchain(
                tool_key, tool_class, tool_schema, port=port, base_url=base_url, use_http=use_http
            )
            langchain_tools.append(langchain_tool)

            mode = f"HTTP port {port}" if (use_http and port) else "in-process"
            logger.info(f"Converted chemistry tool: {tool_key} ({mode})")

        except Exception as e:
            logger.error(f"Failed to convert tool {tool_key}: {e}")
            continue

    logger.info(f"Successfully converted {len(langchain_tools)} chemistry tools")
    return langchain_tools
