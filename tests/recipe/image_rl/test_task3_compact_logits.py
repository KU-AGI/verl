from types import SimpleNamespace

import torch
from torch import nn

from janus.models.modeling_vlm import MultiModalityCausalLM


class _CountingLinear(nn.Linear):
    def __init__(self, in_features: int, out_features: int):
        super().__init__(in_features, out_features)
        self.calls = 0

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        self.calls += 1
        return super().forward(inputs)


class _LanguageModel(nn.Module):
    def __init__(self, hidden_size: int, text_vocab_size: int):
        super().__init__()
        self.lm_head = _CountingLinear(hidden_size, text_vocab_size)
        self.config = SimpleNamespace(vocab_size=text_vocab_size)


class _HeadHarness(nn.Module):
    def __init__(self, hidden_size: int, text_vocab_size: int, image_vocab_size: int):
        super().__init__()
        self.language_model = _LanguageModel(hidden_size, text_vocab_size)
        self.gen_head = nn.Linear(hidden_size, image_vocab_size)
        self.guidance_scale = 2.0

    def dense(self, hidden_states, output_starts):
        return MultiModalityCausalLM._forward_with_split_heads(
            self, hidden_states, output_starts
        )

    def compact(self, hidden_states, output_starts, output_mask):
        return MultiModalityCausalLM._forward_with_compact_image_head(
            self, hidden_states, output_starts, output_mask
        )


def _active_dense_image_logits(dense_logits, output_starts, response_len, image_vocab_size):
    rows = []
    for row, start in enumerate(output_starts[0::2]):
        logit_start = start - 1
        rows.append(dense_logits[row, logit_start : logit_start + response_len, :image_vocab_size])
    return torch.stack(rows)


def test_task3_compact_logits_match_dense_logprob_entropy_and_gradient():
    torch.manual_seed(7)
    hidden_size = 5
    text_vocab_size = 11
    image_vocab_size = 7
    response_len = 4
    output_starts = [4, 4, 3, 3]
    output_mask = torch.ones(4, response_len)

    dense_model = _HeadHarness(hidden_size, text_vocab_size, image_vocab_size)
    compact_model = _HeadHarness(hidden_size, text_vocab_size, image_vocab_size)
    compact_model.load_state_dict(dense_model.state_dict())

    dense_hidden = torch.randn(4, 8, hidden_size, requires_grad=True)
    compact_hidden = dense_hidden.detach().clone().requires_grad_(True)

    dense_logits, dense_starts = dense_model.dense(dense_hidden, output_starts)
    compact_logits, compact_starts = compact_model.compact(
        compact_hidden, output_starts, output_mask
    )
    dense_image_logits = _active_dense_image_logits(
        dense_logits, output_starts, response_len, image_vocab_size
    )

    assert dense_starts == compact_starts == output_starts[0::2]
    torch.testing.assert_close(compact_logits, dense_image_logits)
    assert dense_model.language_model.lm_head.calls == 1
    assert compact_model.language_model.lm_head.calls == 0

    token_ids = torch.tensor([[0, 2, 4, 6], [1, 3, 5, 0]])
    compact_log_probs = torch.log_softmax(compact_logits, dim=-1).gather(
        -1, token_ids.unsqueeze(-1)
    ).squeeze(-1)
    dense_log_probs = torch.log_softmax(
        _active_dense_image_logits(dense_logits, output_starts, response_len, text_vocab_size),
        dim=-1,
    ).gather(-1, token_ids.unsqueeze(-1)).squeeze(-1)
    torch.testing.assert_close(compact_log_probs, dense_log_probs)

    compact_probs = torch.softmax(compact_logits, dim=-1)
    compact_entropy = -(compact_probs * torch.log_softmax(compact_logits, dim=-1)).sum(-1)
    dense_active_logits = torch.stack(
        [
            dense_logits[row, start - 1 : start - 1 + response_len]
            for row, start in enumerate(output_starts[0::2])
        ]
    )
    dense_probs = torch.softmax(dense_active_logits, dim=-1)
    dense_entropy = -(dense_probs * torch.log_softmax(dense_active_logits, dim=-1)).sum(-1)
    torch.testing.assert_close(compact_entropy, dense_entropy)

    dense_image_logits.sum().backward()
    compact_logits.sum().backward()
    torch.testing.assert_close(compact_hidden.grad, dense_hidden.grad)
    torch.testing.assert_close(compact_model.gen_head.weight.grad, dense_model.gen_head.weight.grad)
    torch.testing.assert_close(compact_model.gen_head.bias.grad, dense_model.gen_head.bias.grad)


def test_task3_compact_logits_keep_fixed_shape_for_all_invalid_local_rows():
    model = _HeadHarness(hidden_size=3, text_vocab_size=9, image_vocab_size=5)
    hidden_states = torch.randn(4, 6, 3, requires_grad=True)
    output_mask = torch.zeros(4, 4)

    logits, starts = model.compact(hidden_states, [6, 6, 6, 6], output_mask)

    assert logits.shape == (2, 4, 5)
    assert starts == [6, 6]
    assert model.language_model.lm_head.calls == 0
    # This mirrors the actor's all-invalid dummy scalar: it keeps the FSDP
    # graph connected while contributing exactly zero gradient.
    (logits.flatten()[0] * 0.0).backward()
    assert model.gen_head.weight.grad is not None
    assert torch.count_nonzero(model.gen_head.weight.grad) == 0

