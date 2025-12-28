# Copyright 2025 Bytedance Ltd. and/or its affiliates
# Copyright 2025 Meituan Ltd. and/or its affiliates
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

from typing import Optional, Union

import torch
import torch.distributed as dist
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
from torch.distributed.fsdp import StateDictType, FullStateDictConfig, ShardedStateDictConfig

try:
    from torch.distributed.tensor import DTensor
    from torch.distributed.tensor._dtensor_spec import DTensorSpec
    HAS_DTENSOR = True
except ImportError:
    HAS_DTENSOR = False
    DTensor = None
    DTensorSpec = None


def is_fsdp2_model(model: torch.nn.Module) -> bool:
    """Check if model is wrapped with FSDP2 (has DTensor parameters)."""
    if not HAS_DTENSOR:
        return False
    for param in model.parameters():
        if isinstance(param, DTensor):
            return True
    return False


def is_fsdp1_model(model: torch.nn.Module) -> bool:
    """Check if model is wrapped with FSDP1."""
    return isinstance(model, FSDP)


def fsdp_sharded_save_to_cpu(
    model: torch.nn.Module,
) -> Union[
    tuple[dict[str, tuple[torch.Tensor, "DTensorSpec"]], "DTensorSpec"],  # FSDP2 return
    dict[str, torch.Tensor],  # FSDP1 return
]:
    """
    Sharded Save: Each process only saves the local shard from its own GPU to CPU memory.
    Automatically detects FSDP1 vs FSDP2 and handles accordingly.

    Args:
        model: FSDP-wrapped model (either FSDP1 or FSDP2).

    Returns:
        For FSDP2: (cpu_sharded_state dict, global_spec)
        For FSDP1: sharded state dict
    """
    if is_fsdp2_model(model):
        return _fsdp2_sharded_save_to_cpu(model)
    elif is_fsdp1_model(model):
        return _fsdp1_sharded_save_to_cpu(model)
    else:
        # Fallback: regular model without FSDP
        return _regular_save_to_cpu(model)


def fsdp_sharded_load_from_cpu(
    model: torch.nn.Module,
    cpu_sharded_state: Union[
        tuple[dict[str, tuple[torch.Tensor, Optional["DTensorSpec"]]], "DTensorSpec"],
        dict[str, torch.Tensor],
    ],
    target_spec: Optional["DTensorSpec"] = None,
) -> None:
    """
    Sharded Load: Each process only loads the CPU shard it is responsible for to the GPU.
    Automatically detects FSDP1 vs FSDP2 and handles accordingly.

    Args:
        model: FSDP model to be restored.
        cpu_sharded_state: Shard data from fsdp_sharded_save_to_cpu.
        target_spec: For FSDP2 only - global DTensorSpec from saving.
    """
    if is_fsdp2_model(model):
        assert target_spec is not None, "target_spec required for FSDP2 models"
        _fsdp2_sharded_load_from_cpu(model, cpu_sharded_state, target_spec)
    elif is_fsdp1_model(model):
        _fsdp1_sharded_load_from_cpu(model, cpu_sharded_state)
    else:
        # Fallback: regular model without FSDP
        _regular_load_from_cpu(model, cpu_sharded_state)


# ============== FSDP2 Implementation ==============

def _fsdp2_sharded_save_to_cpu(
    model: torch.nn.Module,
) -> tuple[dict[str, tuple[torch.Tensor, "DTensorSpec"]], "DTensorSpec"]:
    """FSDP2 sharded save implementation."""
    cpu_sharded_state = {}
    global_spec = None

    for param_name, param in model.named_parameters():
        if not isinstance(param, DTensor):
            cpu_tensor = param.detach().cpu()
            cpu_sharded_state[param_name] = (cpu_tensor, None)
            continue

        if global_spec is None:
            global_spec = param._spec
            assert hasattr(global_spec, "device_mesh"), "DTensorSpec must contain 'device_mesh' attribute"
            assert hasattr(global_spec, "placements"), "DTensorSpec must contain 'placements' attribute"

        local_gpu_tensor = param._local_tensor
        local_cpu_tensor = local_gpu_tensor.detach().cpu()
        cpu_sharded_state[param_name] = (local_cpu_tensor, param._spec)

    assert global_spec is not None, "No DTensor-type parameters found in the model. FSDP2 sharding may not be enabled."
    return cpu_sharded_state, global_spec


def _fsdp2_sharded_load_from_cpu(
    model: torch.nn.Module,
    cpu_sharded_state: dict[str, tuple[torch.Tensor, Optional["DTensorSpec"]]],
    target_spec: "DTensorSpec",
) -> None:
    """FSDP2 sharded load implementation."""
    current_device_mesh = None
    for param in model.parameters():
        if isinstance(param, DTensor):
            current_device_mesh = param._spec.device_mesh
            break
    assert current_device_mesh is not None, "DTensor parameters not initialized in the model to be loaded"
    assert current_device_mesh == target_spec.device_mesh, (
        f"device_mesh mismatch during loading! Original: {target_spec.device_mesh}, Current: {current_device_mesh}"
    )

    for param_name, param in model.named_parameters():
        if param_name not in cpu_sharded_state:
            continue

        local_cpu_tensor, saved_spec = cpu_sharded_state[param_name]

        if isinstance(param, DTensor):
            assert saved_spec is not None, f"DTensorSpec missing in saved state for parameter {param_name}"
            assert saved_spec.placements == target_spec.placements, (
                f"Sharding strategy mismatch for parameter {param_name} (conflicts with global rules)!"
            )

            target_device = param._local_tensor.device
            local_gpu_tensor = local_cpu_tensor.to(target_device)
            param._local_tensor.copy_(local_gpu_tensor)
        else:
            target_device = param.device
            param.data.copy_(local_cpu_tensor.to(target_device))

    dist.barrier()


# ============== FSDP1 Implementation ==============

def _fsdp1_sharded_save_to_cpu(
    model: FSDP,
) -> dict[str, torch.Tensor]:
    """
    FSDP1 sharded save implementation.
    Uses SHARDED_STATE_DICT to save local shards only.
    """
    # Use sharded state dict - each rank saves only its shard
    sharded_cfg = ShardedStateDictConfig(offload_to_cpu=True)
    
    with FSDP.state_dict_type(model, StateDictType.SHARDED_STATE_DICT, sharded_cfg):
        state_dict = model.state_dict()
    
    # Ensure all tensors are on CPU
    cpu_state_dict = {}
    for key, value in state_dict.items():
        if isinstance(value, torch.Tensor):
            cpu_state_dict[key] = value.cpu() if value.device.type != 'cpu' else value
        else:
            cpu_state_dict[key] = value
    
    return cpu_state_dict


def _fsdp1_sharded_load_from_cpu(
    model: FSDP,
    cpu_sharded_state: dict[str, torch.Tensor],
) -> None:
    """
    FSDP1 sharded load implementation.
    Uses SHARDED_STATE_DICT to load local shards only.
    """
    sharded_cfg = ShardedStateDictConfig(offload_to_cpu=True)
    
    with FSDP.state_dict_type(model, StateDictType.SHARDED_STATE_DICT, sharded_cfg):
        model.load_state_dict(cpu_sharded_state)
    
    dist.barrier()


# ============== Regular Model (No FSDP) ==============

def _regular_save_to_cpu(
    model: torch.nn.Module,
) -> dict[str, torch.Tensor]:
    """Save regular model state dict to CPU."""
    cpu_state_dict = {}
    for name, param in model.named_parameters():
        cpu_state_dict[name] = param.detach().cpu()
    return cpu_state_dict


def _regular_load_from_cpu(
    model: torch.nn.Module,
    cpu_state_dict: dict[str, torch.Tensor],
) -> None:
    """Load regular model state dict from CPU."""
    for name, param in model.named_parameters():
        if name in cpu_state_dict:
            target_device = param.device
            param.data.copy_(cpu_state_dict[name].to(target_device))
    
    if dist.is_initialized():
        dist.barrier()


# ============== Legacy API (backward compatibility) ==============

# Keep old function names for backward compatibility
fsdp2_sharded_save_to_cpu = fsdp_sharded_save_to_cpu
fsdp2_sharded_load_from_cpu = fsdp_sharded_load_from_cpu