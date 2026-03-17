# LangGraph ChemAgent

LangGraph ChemAgent is an RL training recipe for chemistry tool-using agents. It trains LLMs to solve molecular design tasks (generation, optimization, etc.) using 16 specialized chemistry tools via a multi-turn ReAct agent loop, powered by GRPO on verl's distributed Ray + FSDP2/Megatron backend.

## Architecture Overview

```
┌───────────────────────────────────────────────────────────┐
│  RL Training Loop (Ray + FSDP2 / Megatron)                │
│  ┌───────────┐  ┌───────────┐  ┌──────────────────────┐   │
│  │  Actor    │  │  Ref      │  │  Rollout (e.g. vLLM) │   │
│  │  (Policy) │  │  (KL Ref) │  │  + Agent Loop        │   │
│  └───────────┘  └───────────┘  └──────────┬───────────┘   │
│                                           │               │
│                               ┌───────────▼───────────┐   │
│                               │  ReAct Agent Loop     │   │
│                               │  (LangGraph)          │   │
│                               └───────────┬───────────┘   │
└───────────────────────────────────────────┼────────────── ┘
                                            │
                               ┌────────────▼────────────┐
                               │  Chemistry Tool Servers │
                               │  (FastAPI, ports 9000–  │
                               │   9015 + Jupyter 8888)  │
                               └─────────────────────────┘
```

## Prerequisites

- Python 3.12
- CUDA-compatible GPUs (minimum 2)
- Docker (optional, for containerized setup)

## Installation

Please refer to the official verl [installation docs](https://verl.readthedocs.io/en/latest/start/install.html#install-from-custom-environment).

### Option A: Docker

Pull a pre-built image from the verl [Docker Hub](https://hub.docker.com/r/verlai/verl):

```bash
make init-container
```

### Option B: Conda

```bash
conda create -n verl python==3.12
conda activate verl

# With Megatron support
bash scripts/install_vllm_sglang_mcore.sh

# Or FSDP only (no Megatron)
USE_MEGATRON=0 bash scripts/install_vllm_sglang_mcore.sh

# (Optional) For Recent Models, upgrade transformers
pip install transformers==5.3.0
```

## Setup Chemistry Tools

Clone the chemistry tool agent repository and install dependencies:

```bash
cd verl/third_party
git clone https://github.com/KU-AGI/chemistry_tool_agent_clean.git

cd chemistry_tool_agent_clean
git checkout langgraph_short
git submodule update --init --recursive

# Install all dependencies (torch, peft, admet, jupyter, pymatgen)
pip install --ignore-installed blinker -e ".[all]"
```

## Tool Servers

Each chemistry tool runs as a dedicated FastAPI server. Heavy models (ADMET, MolMIM, etc.) are loaded once and shared across agent processes.

| Tool | Port | Description |
|------|------|-------------|
| MoleculePropertyAnalyzer | 9000 | ADMET property prediction |
| IUPAC2SMILES | 9001 | IUPAC name to SMILES conversion |
| MolSimilarity | 9002 | Tanimoto similarity calculation |
| CanonicalizeSMILES | 9003 | SMILES canonicalization |
| CountMolAtoms | 9004 | Atom counting |
| SMILES2Weight | 9005 | Molecular weight calculation |
| Name2SMILES | 9006 | Common name to SMILES |
| SELFIES2SMILES | 9007 | SELFIES to SMILES conversion |
| FunctionalGroups | 9008 | Functional group detection |
| DBSearch | 9009 | Chemical database search |
| MoleculeGenerator | 9010 | ML-based molecule generation |
| MoleculeEditor | 9011 | Molecular structure editing |
| WikipediaSearch | 9012 | Wikipedia lookup |
| PubchemSearch | 9013 | PubChem lookup |
| MolMIMPropertyRangeGenerator | 9014 | MolMIM guided generation |
| MolMIMPropertyRangeOptimizer | 9015 | MolMIM guided optimization |
| PythonShell | 8888 | Jupyter-based code execution |

### Starting Tool Servers

```bash
cd verl/third_party/chemistry_tool_agent_clean

# Start all tool servers
bash scripts/start_tool_servers.sh

# Start and wait until all servers are healthy
bash scripts/start_tool_servers.sh --wait
```

### Health Check

```bash
bash scripts/check_tool_servers.sh               # health + sample call
bash scripts/check_tool_servers.sh --health-only  # health only
bash scripts/check_tool_servers.sh --tool MoleculePropertyAnalyzer  # single tool
```

### Stopping Tool Servers

```bash
bash scripts/start_molmim_local.sh --stop
kill $(cat /tmp/tool_server_pids.txt)
```

### MolMIM Model Serving

MolMIMPropertyRangeGenerator (9014) and MolMIMPropertyRangeOptimizer (9015) require a separately served MolMIM model.

**Docker:**

```bash
# Set $NGC_CLI_API_KEY in Makefile first
make start-molmim
```

**Local:**

```bash
# Download model from NGC Catalog
# See: https://catalog.ngc.nvidia.com/orgs/nvidia/teams/clara/models/molmim/files?version=1.3
CUDA_VISIBLE_DEVICES=0 MOLMIM_NUM_BACKENDS=8 NUM_GPUS=1 bash scripts/start_molmim_local.sh
```

For detailed tool server documentation, see [docs/tool-server.md](../../verl/third_party/chemistry_tool_agent_clean/docs/tool-server.md).

## Data Preparation

Generate training and test datasets:

```bash
python recipe/langgraph_chemagent/data_preprocess/create_chemagent_dataset.py
```

This produces parquet files at `$DATA_ROOT/data/chemistry_agent/`:
- `train_generation.parquet`
- `test_generation.parquet`

## Training

### Tool Execution Mode

Choose how chemistry tools are executed during training:

```bash
# HTTP mode: tools run as FastAPI servers (requires start_tool_servers.sh)
export TOOLS_USE_HTTP=true

# In-process mode (default): tools run within the training process
export TOOLS_USE_HTTP=false
```

### Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `CHEMISTRY_TOOLS_PATH` | - | Path to `chemagent/tools` directory |
| `TOOL_SERVER_BASE_URL` | `http://localhost` | Base URL for tool servers |
| `TOOLS_USE_HTTP` | `false` | Tool execution mode (`true`/`false`) |
| `ANALYZE_MOLECULE_PORT` | `9000` | MoleculePropertyAnalyzer server port |
| `FUNC_GROUP_PORT` | `9008` | FunctionalGroups server port |

### Run Scripts

Training scripts are provided for different models under `scripts/`:

```bash
# GPT-OSS 20B (8 GPUs, SGLang TP=8, FSDP2)
bash recipe/langgraph_chemagent/scripts/run_chemistry_agent_gpt_oss.sh

# Qwen
bash recipe/langgraph_chemagent/scripts/run_chemistry_agent_qwen.sh

# EXAONE
bash recipe/langgraph_chemagent/scripts/run_chemistry_agent_exaone.sh
```

### Configuration

Hydra configs are in `config/`:

- `chemagent_trainer.yaml` - Main trainer config (imports `ppo_trainer` defaults)
- `chemagent.yaml` - Agent loop config (defines ReAct agent loop target)
- `chemagent_megatron_trainer.yaml` - Megatron backend variant

## Project Structure

```
recipe/langgraph_chemagent/
├── scripts/
│   ├── run_chemistry_agent_gpt_oss.sh    # GPT-OSS training script
│   ├── run_chemistry_agent_qwen.sh       # Qwen training script
│   └── run_chemistry_agent_exaone.sh     # EXAONE training script
├── config/
│   ├── chemagent_trainer.yaml            # Main Hydra config
│   └── chemagent.yaml                    # Agent loop config
│   └── chemagent_megatron_trainer.yaml   # Megatron variant (Not use)
├── agent_loop/
│   ├── chemistry_agent_loop.py           # Chemistry-specific ReAct loop
│   ├── react_agent_loop.py               # Base ReAct loop (LangGraph)
│   ├── agent_loop.py                     # Core agent loop
│   ├── tool_parser.py                    # Tool call parsing
│   └── utils.py                          # Helpers
├── workers/
│   ├── fsdp_workers.py                   # FSDP worker config
│   └── megatron_workers.py               # Megatron worker config (Not use)
├── data_preprocess/
│   └── create_chemagent_dataset.py       # Dataset creation
├── utils/
│   ├── metric_utils.py                   # Evaluation metrics
│   ├── core_algos.py                     # Algorithm utilities
│   └── fsdp_utils.py                     # FSDP utilities
├── main_ppo.py                           # Training entry point
├── ray_trainer.py                        # Ray PPO trainer
├── chat_model.py                         # Chat model wrapper
├── chemagent_tools.py                    # Tool adapter (BaseTool -> LangChain)
├── chemagent_tool_schemas.json           # Tool JSON schemas
├── chemagent_reward.py                   # Reward function
└── config.py                             # Config utilities
```
