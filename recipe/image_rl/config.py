from dataclasses import dataclass
from typing import Optional

from verl.workers.config.rollout import RolloutConfig

__all__ = [
    "ImageGenerationRolloutConfig",
]

@dataclass
class ImageGenerationRolloutConfig(RolloutConfig):
    cfg_weight: float = 1.0
    feedback_system_prompt : Optional[str] = None
    refine_system_prompt : Optional[str] = None
    saving: bool = False
    save_dir: Optional[str] = None