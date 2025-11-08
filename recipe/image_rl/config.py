from dataclasses import dataclass
from typing import Optional

from dataclasses import dataclass, field
from typing import Any, Optional

from omegaconf import MISSING
from transformers import AutoConfig

from verl.base_config import BaseConfig
from verl.utils.fs import copy_to_local
from verl.utils.import_utils import import_external_libs
from verl.utils.model import get_generation_config, update_model_config

from verl.workers.config.rollout import RolloutConfig

__all__ = [
    "ImageGenerationRolloutConfig",
    "ImageGenerationHFModelConfig",
]

@dataclass
class ImageGenerationRolloutConfig(RolloutConfig):
    cfg_weight: float = 1.0
    temperature: float = 1.0
    txt_top_k: int = 50
    txt_top_p: float = 1.0
    img_top_k: int = 4096
    img_top_p: float = 1.0
    image_token_num_per_image: int = 576


@dataclass
class ImageGenerationHFModelConfig(BaseConfig):
    # note that we separate model_path, model_config_path and tokenizer_path in case they are different
    _mutable_fields = {
        "hf_config_path",
        "tokenizer_path",
        "hf_config",
        "generation_config",
        "tokenizer",
        "processor",
        "local_path",
        "architectures",
        "local_hf_config_path",
        "local_tokenizer_path",
    }

    path: str = MISSING
    local_path: Optional[str] = None
    hf_config_path: Optional[str] = None
    local_hf_config_path: Optional[str] = None
    tokenizer_path: Optional[str] = None
    local_tokenizer_path: Optional[str] = None

    # whether to load tokenizer. This is useful when we only want to load model config
    load_tokenizer: bool = True

    hf_config: Any = None
    generation_config: Any = None
    tokenizer: Any = None
    processor: Any = None

    # whether to use shared memory
    use_shm: bool = False
    trust_remote_code: bool = False

    # custom chat template for the model
    custom_chat_template: Optional[str] = None

    external_lib: Optional[str] = None

    override_config: dict = field(default_factory=dict)

    enable_gradient_checkpointing: bool = True
    enable_activation_offload: bool = False

    use_remove_padding: bool = False

    # lora related. We may setup a separate config later
    lora_rank: int = 0
    lora_alpha: int = 16
    target_modules: Optional[str] = "all-linear"

    exclude_modules: Optional[str] = None
    # path to pre-trained LoRA adapter to load for continued training
    lora_adapter_path: Optional[str] = None
    use_liger: bool = False

    use_fused_kernels: bool = False
    fused_kernel_options: dict = field(default_factory=dict)

    architectures: Optional[list[str]] = None

    def __post_init__(self):
        import_external_libs(self.external_lib)

        if self.hf_config_path is None:
            self.hf_config_path = self.path
        if self.tokenizer_path is None:
            self.tokenizer_path = self.path

        self.local_path = copy_to_local(self.path, use_shm=self.use_shm)

        # === Modified ===
        # # constuct tokenizer
        # if self.load_tokenizer:
        #     self.local_tokenizer_path = copy_to_local(self.tokenizer_path, use_shm=self.use_shm)
        #     self.tokenizer = hf_tokenizer(self.local_tokenizer_path, trust_remote_code=self.trust_remote_code)
        #     self.processor = hf_processor(self.local_tokenizer_path, trust_remote_code=self.trust_remote_code)
        # === Modified ===

        from janus.models import VLChatProcessor
        # patch for janus
        self.processor = VLChatProcessor.from_pretrained(self.local_path)
        self.tokenizer = self.processor.tokenizer

        # === Modified ===
        # if self.custom_chat_template is not None:
        #     if self.processor is not None:
        #         self.processor.chat_template = self.custom_chat_template
        #     else:
        #         self.tokenizer.chat_template = self.custom_chat_template
        # === Modified ===

        self.local_hf_config_path = copy_to_local(self.hf_config_path, use_shm=self.use_shm)
        self.generation_config = get_generation_config(
            self.local_hf_config_path, trust_remote_code=self.trust_remote_code
        )

        # constuct hf_config
        attn_implementation = self.override_config.get("attn_implementation", "flash_attention_2")
        self.hf_config = AutoConfig.from_pretrained(
            self.local_hf_config_path, trust_remote_code=self.trust_remote_code, attn_implementation=attn_implementation
        )
        self.hf_config.language_config.use_flash_attention_2 = True
        self.hf_config.language_config._attn_implementation = "flash_attention_2"

        override_config_kwargs = {}

        if self.tokenizer is not None:
            override_config_kwargs.update(
                {
                    "boi_token_id": self.processor.tokenizer.convert_tokens_to_ids('<begin_of_image>'),
                    "eoi_token_id": self.processor.tokenizer.convert_tokens_to_ids('<end_of_image>'),
                    "bos_token_id": self.tokenizer.bos_token_id,
                    "eos_token_id": self.tokenizer.eos_token_id,
                    "pad_token_id": self.tokenizer.pad_token_id,
                }
            )

        # TODO: (vermouth1992). self.config.model in megatron differs from that of fsdp in the override_config.
        override_config = (
            self.override_config["model_config"] if "model_config" in self.override_config else self.override_config
        )
        override_config_kwargs.update(override_config)
        update_model_config(self.hf_config, override_config_kwargs=override_config_kwargs)

        # === Modified ===
        # self.share_embeddings_and_output_weights = getattr(self.hf_config, "tie_word_embeddings", False)

        # # get model architectures
        # self.architectures = getattr(self.hf_config, "architectures", None)
        # assert self.architectures is not None and len(self.architectures) == 1, (
        #     "Expect only one architecture, got {}".format(self.architectures)
        # )

        # # per model patch
        # if getattr(self.hf_config, "model_type", None) == "kimi_vl":
        #     self.hf_config.text_config.topk_method = "greedy"
        # === Modified ===

    def get_processor(self):
        return self.processor if self.processor is not None else self.tokenizer
