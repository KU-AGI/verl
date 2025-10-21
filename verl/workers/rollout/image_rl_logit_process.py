import inspect
import math
from collections.abc import Iterable
from typing import TYPE_CHECKING, Callable, Optional, Union, Tuple

import numpy as np
import torch

import inspect
import math
from collections.abc import Iterable
from typing import TYPE_CHECKING, Callable, Optional, Union, Tuple

import numpy as np
import torch

from transformers.pytorch_utils import isin_mps_friendly
from transformers.utils import add_start_docstrings
from transformers.utils.logging import get_logger


# TODO (joao): We shouldn't need this, but there would be a circular import
if TYPE_CHECKING:
    from transformers.generation.configuration_utils import GenerationConfig

logger = get_logger(__name__)


LOGITS_PROCESSOR_INPUTS_DOCSTRING = r"""
    Args:
        input_ids (`torch.LongTensor` of shape `(batch_size, sequence_length)`):
            Indices of input sequence tokens in the vocabulary. [What are input IDs?](../glossary#input-ids)
        scores (`torch.FloatTensor` of shape `(batch_size, config.vocab_size)`):
            Prediction scores of a language modeling head. These can be logits for each vocabulary when not using beam
            search or log softmax for each vocabulary token when using beam search

    Return:
        `torch.FloatTensor` of shape `(batch_size, config.vocab_size)`: The processed prediction scores.

"""


class LogitsProcessor:
    """Abstract base class for all logit processors that can be applied during generation."""

    @add_start_docstrings(LOGITS_PROCESSOR_INPUTS_DOCSTRING)
    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor) -> torch.FloatTensor:
        raise NotImplementedError(
            f"{self.__class__} is an abstract class. Only classes inheriting this class can be called."
        )


# Customized Logits Processor for Classifier-Free Guidance with Embeddings
class CFGEmbeddingLogitsProcessor(LogitsProcessor):
    r"""
    [`LogitsProcessor`] that implements Classifier-Free Guidance by preparing conditional and 
    unconditional embeddings and applying CFG during generation. This processor follows the 
    approach used in multimodal generation where embeddings are manipulated rather than input tokens.

    Args:
        task (`int`):
            Task identifier that determines masking behavior:
            - task 1: Masks content between <|User|> and <|Assistant|> tokens for unconditional
            - task 2: No CFG applied (bypassed) 
            - task 3: Masks content between <eoi>\n and <|Assistant|> tokens for unconditional
        cfg_weight (`float`, *optional*, defaults to 5.0):
            CFG guidance weight. Higher values increase adherence to conditional prompt.
        pad_token_id (`int`):
            Token ID used for padding/masking.
        device (`str`, *optional*, defaults to "cpu"):
            Device to allocate tensors on.
        model (`torch.nn.Module`, *optional*):
            Reference to the model for accessing embedding layers.
        
    Examples:

    ```python
    >>> from transformers import AutoTokenizer, AutoModelForCausalLM

    >>> model = AutoModelForCausalLM.from_pretrained("your-model")
    >>> tokenizer = AutoTokenizer.from_pretrained("your-model")
    
    >>> # Task 1: Text-to-Image CFG
    >>> cfg_processor = CFGEmbeddingLogitsProcessor(
    ...     task=1,
    ...     pad_token_id=tokenizer.pad_token_id,
    ...     cfg_weight=5.0,
    ...     model=model
    ... )
    
    >>> # Task 3: Regeneration CFG  
    >>> cfg_processor = CFGEmbeddingLogitsProcessor(
    ...     task=3,
    ...     pad_token_id=tokenizer.pad_token_id,
    ...     cfg_weight=3.0,
    ...     model=model
    ... )
    
    >>> inputs = tokenizer("Your input text", return_tensors="pt")
    >>> outputs = model.generate(**inputs, logits_processor=[cfg_processor])
    ```
    """

    def __init__(
        self,
        task: int,
        pad_token_id: int,
        cfg_weight: float = 5.0,
        device: str = "cpu",
        model: Optional[torch.nn.Module] = None
    ):
        if not isinstance(task, int) or task not in [1, 2, 3]:
            raise ValueError(f"`task` must be 1, 2, or 3, but got {task}")
        
        if not isinstance(cfg_weight, (int, float)) or cfg_weight < 0:
            raise ValueError(f"`cfg_weight` must be a non-negative number, but got {cfg_weight}")
        
        if not isinstance(pad_token_id, int):
            raise ValueError(f"`pad_token_id` must be an integer, but got {type(pad_token_id)}")

        self.task = task
        self.pad_token_id = pad_token_id
        self.cfg_weight = cfg_weight
        self.device = device
        self.model = model
        
        # Task-specific markers - create with proper dtype
        if task == 1:
            # Task 1: <|User|> to <|Assistant|> masking
            self.start_marker = torch.tensor([100601], device=device, dtype=torch.long)  # <|User|>
            self.end_marker = torch.tensor([100602], device=device, dtype=torch.long)    # <|Assistant|>
            self.mask_offset = (0, 2)  # content_start_index, content_end_index offset
        elif task == 3:
            # Task 3: <end_of_image>\n to <|Assistant|> masking
            self.start_marker = torch.tensor([100593], device=device, dtype=torch.long)  # <end_of_image>
            self.end_marker = torch.tensor([100602], device=device, dtype=torch.long)    # <|Assistant|>
            self.mask_offset = (1, 2)  # content_start_index, content_end_index offset
        
        # State management
        self.cfg_prepared = False
        self.conditional_embeds = None
        self.unconditional_embeds = None
        self.attention_mask = None
        self.batch_size = None

    def find_sequence(self, tensor: torch.Tensor, sequence: torch.Tensor) -> int:
        """
        Find the starting position of a sequence within a tensor.
        
        Args:
            tensor (`torch.Tensor`): Input tensor to search in
            sequence (`torch.Tensor`): Sequence to find
            
        Returns:
            `int`: Starting position of the sequence, or -1 if not found
        """
        if len(sequence) > len(tensor):
            return -1
        
        # Ensure both tensors are on the same device and dtype
        if tensor.device != sequence.device:
            sequence = sequence.to(tensor.device)
        if tensor.dtype != sequence.dtype:
            sequence = sequence.to(dtype=tensor.dtype)
            
        len_needle = sequence.shape[0]
        for i in range(tensor.shape[0] - len_needle + 1):
            if torch.equal(tensor[i:i+len_needle], sequence):
                return i
        return -1

    def prepare_cfg_embeds(
        self, 
        input_ids: torch.LongTensor, 
        attention_mask: torch.LongTensor,
        input_embeds: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Prepare conditional and unconditional embeddings for CFG.
        
        Args:
            input_ids (`torch.LongTensor`): Input token IDs
            attention_mask (`torch.LongTensor`): Attention mask for input_ids
            input_embeds (`torch.Tensor`, *optional*): Pre-computed input embeddings
        """
        # Task 2: No CFG
        if self.task == 2:
            if input_embeds is not None:
                return input_embeds, attention_mask
            else:
                return input_ids, attention_mask
        
        batch_size, seq_len = input_ids.shape
        
        # Get embeddings if not provided
        if input_embeds is None and self.model is not None:
            if hasattr(self.model, 'language_model'):
                embed_layer = self.model.language_model.get_input_embeddings()
            elif hasattr(self.model, 'get_input_embeddings'):
                embed_layer = self.model.get_input_embeddings()
            else:
                raise ValueError("Cannot find embedding layer in model")
            input_embeds = embed_layer(input_ids)
        elif input_embeds is None:
            raise ValueError("Either input_embeds or model must be provided")
        
        # Prepare conditional and unconditional embeddings
        cond_embeds = input_embeds.clone()
        uncond_embeds = input_embeds.clone()
        
        # Get pad embedding
        pad_id = torch.tensor(self.pad_token_id, device=input_ids.device)
        if hasattr(self.model, 'language_model'):
            pad_embed = self.model.language_model.get_input_embeddings()(pad_id).unsqueeze(0)
        else:
            pad_embed = torch.zeros(1, input_embeds.shape[-1], device=input_ids.device, dtype=input_embeds.dtype)
        
        # Apply masking for unconditional embeddings
        for i, row in enumerate(input_ids):
            start_pos = self.find_sequence(row, self.start_marker)
            end_pos = self.find_sequence(row, self.end_marker)
            
            if start_pos != -1 and end_pos != -1 and start_pos < end_pos:
                content_start_index = start_pos + self.mask_offset[0]
                content_end_index = end_pos + self.mask_offset[1]
                
                if content_start_index < content_end_index:
                    mask_len = content_end_index - content_start_index
                    pad_embeds = pad_embed.expand(mask_len, -1)
                    uncond_embeds[i, content_start_index:content_end_index] = pad_embeds
        
        # Create final embeddings with doubled batch size (interleaved)
        embed_dim = input_embeds.shape[-1]
        final_embeds = pad_embed.expand(batch_size * 2, seq_len, embed_dim).clone()
        final_attention_mask = torch.zeros((batch_size * 2, seq_len), dtype=torch.long, device=input_ids.device)
        
        # Interleave conditional and unconditional
        final_embeds[0:batch_size] = cond_embeds
        final_embeds[batch_size:] = uncond_embeds
        
        # Create attention mask (assuming all tokens are valid initially)
        final_attention_mask[0:batch_size] = attention_mask
        final_attention_mask[batch_size:] = attention_mask
        
        # Store for later use
        self.conditional_embeds = cond_embeds
        self.unconditional_embeds = uncond_embeds
        self.attention_mask = final_attention_mask
        self.batch_size = batch_size
        self.cfg_prepared = True
        
        return final_embeds, final_attention_mask

    def apply_cfg_to_logits(self, logits: torch.FloatTensor) -> torch.FloatTensor:
        """
        Apply CFG to logits by combining conditional and unconditional predictions.
        
        Args:
            logits (`torch.FloatTensor`): Model logits with doubled batch size
            
        Returns:
            `torch.FloatTensor`: CFG-adjusted logits with original batch size
        """
        if self.task == 2 or not self.cfg_prepared:
            return logits
        
        # Split logits into conditional and unconditional
        cond_logits, uncond_logits = logits.split(self.batch_size, dim=0)
        # Apply CFG formula: uncond + weight * (cond - uncond)
        cfg_logits = uncond_logits + self.cfg_weight * (cond_logits - uncond_logits)
        
        return cfg_logits

    @add_start_docstrings(LOGITS_PROCESSOR_INPUTS_DOCSTRING)
    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor) -> torch.FloatTensor:
        """
        Process logits with CFG. Note that input_ids should already be doubled if CFG is active.
        
        Args:
            input_ids (`torch.LongTensor`): Input token IDs (possibly doubled batch size)
            scores (`torch.FloatTensor`): Model prediction scores (possibly doubled batch size)
            
        Returns:
            `torch.FloatTensor`: CFG-processed prediction scores
        """
        # Task 2: No CFG processing
        if self.task == 2:
            return scores
        
        # Check if this is a CFG setup (doubled batch size)
        if self.cfg_prepared and scores.shape[0] == self.batch_size * 2:
            # Apply CFG to logits
            processed_scores = self.apply_cfg_to_logits(scores)
            
            # Log CFG application
            logger.debug(f"Applied CFG with weight {self.cfg_weight} for task {self.task}")
            
            return processed_scores
        else:
            # Not in CFG mode, return original scores
            return scores

    def reset_state(self):
        """Reset processor state for new generation."""
        self.cfg_prepared = False
        self.conditional_embeds = None
        self.unconditional_embeds = None  
        self.attention_mask = None
        self.batch_size = None

    def get_cfg_embeds(self) -> Optional[Tuple[torch.Tensor, torch.Tensor]]:
        """
        Get prepared CFG embeddings if available.
        
        Returns:
            Optional tuple of (conditional_embeds, unconditional_embeds)
        """
        if self.cfg_prepared:
            return self.conditional_embeds, self.unconditional_embeds
        return None