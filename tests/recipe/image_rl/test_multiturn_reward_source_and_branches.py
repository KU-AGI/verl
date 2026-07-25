import asyncio
from types import SimpleNamespace

import numpy as np
import pytest
import torch
from PIL import Image

from recipe.fully_async_policy_image_rl.agent_loop.agent_loop_hf import FullyAsyncAgentLoopManager
from recipe.fully_async_policy_image_rl.sglang_rollout.janus_sglang_server import JanusSGLangAsyncServer
from verl import DataProto
from verl.workers.reward_manager.image_generation import ImageGenerationRewardManager


def _object_array(values):
    result = np.empty(len(values), dtype=object)
    for i, value in enumerate(values):
        result[i] = value
    return result


def _reward_batch(**non_tensors):
    size = len(non_tensors["prompt"])
    defaults = {
        "task1_gen_imgs_pil_list": _object_array([Image.new("RGB", (2, 2), "red")] * size),
        "task2_feedback_texts": np.array(["Step 1: edit"] * size, dtype=object),
        "task3_regen_imgs_pil_list": _object_array([Image.new("RGB", (2, 2), "blue")] * size),
        "reward_model": _object_array([{} for _ in range(size)]),
    }
    defaults.update(non_tensors)
    return DataProto.from_dict(
        tensors={"dummy": torch.zeros((size, 1))},
        non_tensors=defaults,
    )


def _capturing_reward_manager(captured):
    def compute_score(
        prompts,
        gen_imgs,
        feedback_texts,
        regen_imgs,
        ground_truth_imgs,
        summarizes,
        feedback_tuples,
        vqa_questions,
        extra_infos,
        task_ids,
    ):
        captured["gen_imgs"] = gen_imgs
        captured["regen_imgs"] = regen_imgs
        return [{"score": 0.0, "reward_extra_info": {}} for _ in prompts]

    return ImageGenerationRewardManager(
        tokenizer=None,
        processor=None,
        num_examine=0,
        compute_score=compute_score,
    )


@pytest.mark.parametrize(
    ("task_id", "source_key", "source_color"),
    [
        (1, "task1_gen_imgs_pil_list", "red"),
        (2, "task2_input_imgs_pil_list", "green"),
        (3, "task3_input_imgs_pil_list", "green"),
    ],
)
def test_reward_manager_uses_task_specific_source_image(task_id, source_key, source_color):
    captured = {}
    source = Image.new("RGB", (2, 2), source_color)
    batch = _reward_batch(
        prompt=np.array(["prompt"], dtype=object),
        **{source_key: _object_array([source])},
    )

    _capturing_reward_manager(captured).verify(batch, task_id)

    assert captured["gen_imgs"][0].getpixel((0, 0)) == source.getpixel((0, 0))
    assert captured["regen_imgs"][0].getpixel((0, 0)) == (0, 0, 255)


def test_reward_manager_legacy_fallback_and_length_validation(capsys):
    captured = {}
    batch = _reward_batch(prompt=np.array(["prompt"], dtype=object))
    manager = _capturing_reward_manager(captured)

    manager.verify(batch, task_id=2)

    assert captured["gen_imgs"][0].getpixel((0, 0)) == (255, 0, 0)
    assert "falling back to task1_gen_imgs_pil_list" in capsys.readouterr().out

    batch.non_tensor_batch["task2_input_imgs_pil_list"] = _object_array([])
    with pytest.raises(ValueError, match="Source image count mismatch for task2"):
        manager.verify(batch, task_id=2)


def test_task2_and_task3_snapshot_the_same_turn_source(monkeypatch):
    import recipe.fully_async_policy_image_rl.sglang_rollout.janus_sglang_server as rollout_module

    server_class = JanusSGLangAsyncServer.__ray_metadata__.modified_class
    server = server_class.__new__(server_class)
    server.processor = SimpleNamespace(pad_id=0)
    server.tokenizer = SimpleNamespace(encode=lambda text, add_special_tokens=False: [1, 2])
    server.response_length = 4
    server._task2_training_text = lambda prompt: prompt
    server._task2_request_text = lambda prompt: prompt
    server._task3_training_text = lambda prompt, feedback: f"{prompt}:{feedback}"
    server._task3_system_prompt = lambda: "system"
    server._tokenize_left = lambda texts: (
        torch.ones((len(texts), 2), dtype=torch.long),
        torch.ones((len(texts), 2), dtype=torch.long),
    )
    server._expand_image_placeholders_ids = lambda input_ids: (
        input_ids,
        torch.ones_like(input_ids),
    )
    server._sglang_top_k = lambda value: value
    server._pixel_values_from_pils = lambda images: torch.zeros((len(images), 3, 2, 2))
    monkeypatch.setattr(
        rollout_module,
        "build_segment_response_mask",
        lambda feedback_ids, tokenizer, **kwargs: torch.full_like(feedback_ids, 2),
    )

    source_image = Image.new("RGB", (2, 2), "green")
    edited_image = Image.new("RGB", (2, 2), "blue")
    source_uri = server._pil_to_data_uri(source_image)
    edited_uri = server._pil_to_data_uri(edited_image)

    async def post_json(path, payload):
        if path == "/generate":
            return {"text": "Step 1: edit", "output_ids": [1, 2], "meta_info": {}}
        return {
            "images": [{
                "image_token_ids": [7, 8],
                "image_base64": edited_uri,
                "image_token_logprobs": None,
            }]
        }

    server._post_json = post_json
    batch = DataProto.from_dict(
        tensors={
            "current_imgs_pixel_values": torch.zeros((1, 3, 2, 2)),
            "current_img_tokens": torch.tensor([[3, 4]], dtype=torch.long),
        },
        non_tensors={
            "prompt": np.array(["prompt"], dtype=object),
            "current_imgs_base64": np.array([source_uri], dtype=object),
            "task1_gen_imgs_base64": np.array([source_uri], dtype=object),
        },
    )
    gen_config = {
        "is_validate": True,
        "temperature": 1.0,
        "txt_top_p": 1.0,
        "txt_top_k": -1,
        "cfg_weight": 1.0,
        "img_top_p": 1.0,
        "img_top_k": -1,
    }

    batch = asyncio.run(server._generate_task2(batch, gen_config))
    batch = asyncio.run(server._generate_task3(batch, gen_config))

    assert batch.non_tensor_batch["task2_input_imgs_pil_list"][0].getpixel((0, 0)) == (0, 128, 0)
    assert batch.non_tensor_batch["task3_input_imgs_pil_list"][0].getpixel((0, 0)) == (0, 128, 0)
    assert tuple(batch.non_tensor_batch["task3_regen_imgs_pil_list"][0, 0, 0]) == (0, 0, 255)


def _branch_batch(size):
    return DataProto.from_dict(tensors={"dummy": torch.zeros((size, 1))})


def test_branch_ids_are_sample_local_when_allocations_interleave():
    manager = FullyAsyncAgentLoopManager.__new__(FullyAsyncAgentLoopManager)

    sample_a_turn0, next_a = manager._spawn_child_branch_ids(_branch_batch(2), 0)
    sample_b_turn0, next_b = manager._spawn_child_branch_ids(_branch_batch(3), 0)
    sample_a_turn1, next_a = manager._spawn_child_branch_ids(
        _branch_batch(2),
        next_a,
        parent_ids=sample_a_turn0.non_tensor_batch["branch_id"],
    )

    assert sample_a_turn0.non_tensor_batch["branch_id"].tolist() == [0, 1]
    assert sample_b_turn0.non_tensor_batch["branch_id"].tolist() == [0, 1, 2]
    assert sample_a_turn1.non_tensor_batch["branch_id"].tolist() == [2, 3]
    assert sample_a_turn1.non_tensor_batch["parent_branch_id"].tolist() == [0, 1]
    assert next_a == 4
    assert next_b == 3
    assert not hasattr(manager, "_branch_id_counter")


def test_branch_chain_detects_cycles_instead_of_looping():
    assert FullyAsyncAgentLoopManager._resolve_branch_chain({2: 1, 1: -1}, 2, 9) == [1, 2]

    with pytest.raises(ValueError, match="trajectory_id=9"):
        FullyAsyncAgentLoopManager._resolve_branch_chain({1: 2, 2: 1}, 2, 9)
