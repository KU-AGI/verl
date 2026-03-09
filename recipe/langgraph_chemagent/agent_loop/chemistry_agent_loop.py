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
Chemistry Agent Loop using LangGraph ReactAgent with Chemistry Tools.

This implementation extends ReactAgentLoop to work with chemistry tools
for molecular discovery and chemical analysis tasks.
"""

import logging
import os
from typing import Optional

from recipe.langgraph_chemagent.chemagent_tools import get_chemistry_tools
from recipe.langgraph_chemagent.agent_loop.react_agent_loop import ReactAgentLoop

logger = logging.getLogger(__name__)


class ChemistryReactAgentLoop(ReactAgentLoop):
    """
    ReactAgent loop equipped with chemistry tools for molecular tasks.

    Supports chemistry tools including:
    - Tanimoto similarity calculation
    - SMILES canonicalization
    - Molecular property calculations (ADMET, functional groups, etc.)
    - Molecule editing and structure manipulation
    """

    _class_initialized = False

    @classmethod
    def init_class(cls, config, tokenizer, **kwargs):
        """
        Initialize chemistry agent with tools loaded from chemistry_tool_agent.

        Args:
            config: Agent configuration
            tokenizer: Tokenizer for the model
            **kwargs: Additional arguments including:
                - chemistry_tools_path: Path to chemistry tools directory
                - selected_tools: List of tool keys to load (optional, loads all if None)
        """
        if cls._class_initialized:
            return

        # Get chemistry tools path from config or kwargs
        chemistry_tools_path = kwargs.get(
            'chemistry_tools_path',
            os.getenv('CHEMISTRY_TOOLS_PATH', '/data/users/pimang62/chemistry_tool_agent_clean/chemagent/tools')
        )

        # Get selected tools (optional)
        selected_tools = kwargs.get('selected_tools', None)

        # Base URL for tool servers (default: http://localhost)
        base_url = kwargs.get('base_url', os.getenv('TOOL_SERVER_BASE_URL', 'http://localhost'))

        # HTTP vs in-process toggle: TOOLS_USE_HTTP=true (default) or false
        _use_http_env = os.getenv('TOOLS_USE_HTTP', 'true').lower()
        use_http = kwargs.get('use_http', _use_http_env != 'false')

        mode_str = "HTTP tool servers" if use_http else "in-process (local)"
        print(f"[ChemistryReactAgentLoop] Loading chemistry tools from: {chemistry_tools_path}")
        print(f"[ChemistryReactAgentLoop] Tool execution mode: {mode_str}")

        try:
            # Load chemistry tools
            cls.tools = get_chemistry_tools(
                chemistry_tools_path, selected_tools=selected_tools, base_url=base_url, use_http=use_http
            )
            tool_names = [tool.name for tool in cls.tools]
            print(f"[ChemistryReactAgentLoop] Loaded {len(cls.tools)} tools: {tool_names}")

        except Exception as e:
            import traceback
            print(f"[ChemistryReactAgentLoop] FAILED to load chemistry tools: {e}")
            traceback.print_exc()
            cls.tools = []

        # Call parent init_class to build graph
        super().init_class(config, tokenizer, **kwargs)

    def _get_tool_schemas(self) -> dict | None:
        """Return tool argument schemas keyed by tool name for reward computation."""
        return {
            tool.name: tool.args_schema.model_json_schema()
            for tool in self.tools
            if getattr(tool, "args_schema", None) is not None
        }


class ChemistryWithSandboxReactAgentLoop(ChemistryReactAgentLoop):
    """
    Chemistry Agent with both chemistry tools and sandbox code execution.

    This combines:
    1. Chemistry tools (SMILES, molecular properties, etc.)
    2. Sandbox Fusion tool for general Python code execution

    Useful for complex tasks requiring both chemical calculations and
    general computational capabilities.
    """

    _class_initialized = False

    @classmethod
    def init_class(cls, config, tokenizer, **kwargs):
        """
        Initialize chemistry agent with both chemistry tools and sandbox execution.

        Args:
            config: Agent configuration
            tokenizer: Tokenizer for the model
            **kwargs: Additional arguments including:
                - chemistry_tools_path: Path to chemistry tools directory
                - selected_tools: List of chemistry tool keys to load
                - sandbox_config: Sandbox Fusion configuration dict
        """
        if cls._class_initialized:
            return

        # First, load chemistry tools using parent class
        super().init_class(config, tokenizer, **kwargs)

        # Then, add sandbox tool if configured
        sandbox_config = kwargs.get('sandbox_config', None)

        if sandbox_config:
            try:
                from langchain_core.tools import StructuredTool
                from pydantic import BaseModel, Field

                from verl.tools.sandbox_fusion_tools import SandboxFusionTool

                # Create sandbox tool schema
                class CodeInput(BaseModel):
                    code: str = Field(description="The Python code to execute")

                # Import and initialize sandbox tool
                tool_schema = {
                    "type": "function",
                    "function": {
                        "name": "code_interpreter",
                        "description": "Execute Python code in a sandboxed environment. "
                        "Useful for computations, data analysis, and when chemistry "
                        "tools are insufficient.",
                        "parameters": {
                            "type": "object",
                            "properties": {"code": {"type": "string", "description": "The code to execute."}},
                            "required": ["code"],
                        },
                    },
                }

                sandbox_tool_instance = SandboxFusionTool(config=sandbox_config, tool_schema=tool_schema)

                # Create wrapper function
                async def execute_code(code: str) -> str:
                    """Execute Python code in sandbox."""
                    try:
                        result, _, _ = await sandbox_tool_instance.execute(
                            instance_id="sandbox", parameters={"code": code}
                        )
                        return str(result)
                    except Exception as e:
                        return f"Execution error: {str(e)}"

                # Create LangChain tool
                sandbox_langchain_tool = StructuredTool(
                    name="code_interpreter",
                    description=tool_schema["function"]["description"],
                    func=execute_code,
                    args_schema=CodeInput,
                    coroutine=execute_code,  # Support async
                )

                # Add to tools list
                cls.tools.append(sandbox_langchain_tool)
                logger.info("Added sandbox code_interpreter tool")

            except Exception as e:
                logger.error(f"Failed to add sandbox tool: {e}")
                logger.warning("Continuing with chemistry tools only")

        # Rebuild graph with updated tools
        cls.graph = cls.build_graph()
        cls._class_initialized = True
        logger.info(f"ChemistryWithSandboxReactAgentLoop initialized with {len(cls.tools)} tools")
