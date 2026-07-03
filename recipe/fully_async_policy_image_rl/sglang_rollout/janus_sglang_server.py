import asyncio
import base64
import concurrent.futures
import json
import logging
import os
import signal
import subprocess
import sys
import time
import urllib.error
import urllib.request
from io import BytesIO
from pathlib import Path
from typing import Any, Optional

import numpy as np
import ray
import torch
from PIL import Image
from ray.actor import ActorHandle

from verl import DataProto
from verl.utils.config import omega_conf_to_dataclass
from verl.workers.config import HFModelConfig, RolloutConfig
from verl.workers.rollout.replica import RolloutMode
from recipe.image_rl.config import ImageGenerationHFModelConfig, ImageGenerationRolloutConfig
from recipe.image_rl.utils import FormattingEvaluatorV3, build_segment_response_mask


def _ensure_vendored_sglang_path() -> None:
    repo_root = Path(__file__).resolve().parents[3]
    sglang_python = repo_root / "sglang" / "python"
    if sglang_python.exists():
        path = str(sglang_python)
        if path not in sys.path:
            sys.path.insert(0, path)
        os.environ["PYTHONPATH"] = f"{path}:{os.environ.get('PYTHONPATH', '')}"


def _resolve_sglang_python() -> str:
    env_python = os.environ.get("JANUS_SGLANG_PYTHON")
    if env_python:
        return env_python
    if sys.executable and os.path.exists(sys.executable):
        return sys.executable
    fallback = "/data/anaconda3/envs/sglang_diffusion/bin/python"
    if os.path.exists(fallback):
        return fallback
    return "python"


def _vendored_sglang_python_path() -> str | None:
    sglang_python = Path(__file__).resolve().parents[3] / "sglang" / "python"
    return str(sglang_python) if sglang_python.exists() else None


# _ensure_vendored_sglang_path()

from sglang.srt.configs.janus_pro import VLChatProcessor  # noqa: E402
from verl.workers.rollout.utils import get_free_port  # noqa: E402


logger = logging.getLogger(__file__)
logger.setLevel(logging.INFO)


EDIT_SYSTEM_PROMPT = """You are a strict image editing assistant.
Your task is to revise a *failed* generated image according to the user's instruction and the original generation intent.

INPUT FORMAT:
1. The source image is located between {image_start_tag} and {image_end_tag}.
2. The original text-to-image generation prompt will be provided after the keyword 'INPUT_PROMPT:'
3. Step-by-step feedback will be provided after the keyword 'FEEDBACK:'.
   - The feedback will be a sequence of instructions, each starting with 'Step X:' (e.g., 'Step 1:', 'Step 2:', ...).
   - You MUST follow ALL steps in order and produce a final image that satisfies the entire sequence, not just an intermediate step.

CRITICAL RULES:
1. You MUST Look at the image between {image_start_tag} and {image_end_tag} as the ground truth.
2. Preserve the background, objects, and style from the input image unless explicitly asked to change them.
3. Do NOT generate a completely new image from scratch.
4. **You MUST strictly maintain the spatial layout and composition of the source image.**
5. You MUST also reference the INPUT_PROMPT as the original intended content of the image, but the visible source image remains the primary ground truth.
6. When applying FEEDBACK, carefully execute each step one by one while keeping previous changes consistent, and ensure the final result reflects all steps combined."""


@ray.remote(num_cpus=1)
class JanusSGLangAsyncServer:
    """SGLang-backed rollout server with the same DataProto contract as ImageUnifiedRollout."""

    def __init__(
        self,
        config: ImageGenerationRolloutConfig | RolloutConfig,
        model_config: ImageGenerationHFModelConfig | HFModelConfig,
        rollout_mode: RolloutMode,
        workers: list[ActorHandle],
        replica_rank: int,
        node_rank: int,
        nnodes: int,
        cuda_visible_devices: str,
    ):
        os.environ["CUDA_VISIBLE_DEVICES"] = cuda_visible_devices
        os.environ.setdefault("SGLANG_SKIP_SGL_KERNEL_VERSION_CHECK", "1")
        assert torch.cuda.is_available(), "Janus SGLang server should run on a GPU node"

        self.config: RolloutConfig = omega_conf_to_dataclass(config)
        self.model_config: HFModelConfig = omega_conf_to_dataclass(model_config, dataclass_type=HFModelConfig)
        self.config.max_model_len = self.config.prompt_length + self.config.response_length
        self.rollout_mode = rollout_mode
        self.workers = workers
        self.replica_rank = replica_rank
        self.node_rank = node_rank
        self.nnodes = nnodes

        if self.rollout_mode != RolloutMode.HYBRID and self.config.load_format == "dummy":
            self.config.load_format = "auto"

        self._server_address = ray.util.get_node_ip_address().strip("[]")
        self._server_port = None
        self._master_address = self._server_address if self.node_rank == 0 else None
        self._master_port = None
        self._server_process: Optional[subprocess.Popen] = None
        self._server_log_file = None

        self.paused = False
        self.lock = asyncio.Lock()
        self.cancel_event: dict[str, asyncio.Event] = {}
        self.generation_tasks: dict[str, asyncio.Task] = {}
        self.executor = concurrent.futures.ThreadPoolExecutor(max_workers=8)

        self.pending_weight_version: Optional[int] = None
        self.weight_update_lock = asyncio.Lock()
        self.weight_update_worker_task: Optional[asyncio.Task] = None
        self.ongoing_generations = 0
        self.latest_available_version = -1
        self.applied_version = -1

        self.processor = None
        self.tokenizer = None
        self.formatter_v3 = FormattingEvaluatorV3()

    def get_master_address(self):
        return self._master_address, self._master_port

    def get_server_address(self):
        assert self._server_port is not None, "SGLang HTTP server is not launched yet"
        return self._server_address, self._server_port

    async def launch_server(self, master_address: str = None, master_port: int = None):
        if self.node_rank != 0:
            raise RuntimeError("janus_sglang currently launches one single-node SGLang server per replica")

        engine_kwargs = dict(self.config.get("engine_kwargs", {}).get("sglang", {}) or {})
        attention_backend = engine_kwargs.pop("attention_backend", "triton")
        sampling_backend = engine_kwargs.pop("sampling_backend", "pytorch")
        chat_template = engine_kwargs.pop("chat_template", "janus-pro")
        disable_radix_cache = engine_kwargs.pop("disable_radix_cache", True)
        enable_janus_image_cuda_graph = engine_kwargs.pop("enable_janus_image_cuda_graph", True)

        self._server_port, server_sock = get_free_port(self._server_address)
        server_sock.close()

        sglang_python = _resolve_sglang_python()
        cmd = [
            sglang_python,
            "-m",
            "sglang.launch_server",
            "--model-path",
            self.model_config.local_path,
            "--host",
            self._server_address,
            "--port",
            str(self._server_port),
            "--chat-template",
            str(chat_template),
            "--dtype",
            str(self.config.dtype),
            "--mem-fraction-static",
            str(self.config.gpu_memory_utilization),
            "--attention-backend",
            str(attention_backend),
            "--sampling-backend",
            str(sampling_backend),
            "--tp-size",
            str(self.config.tensor_model_parallel_size),
            "--dp-size",
            str(self.config.data_parallel_size),
            "--log-level",
            "error",
            "--skip-server-warmup",
        ]
        if self.model_config.trust_remote_code:
            cmd.append("--trust-remote-code")
        if disable_radix_cache:
            cmd.append("--disable-radix-cache")
        if enable_janus_image_cuda_graph:
            cmd.append("--enable-janus-image-cuda-graph")
        if self.config.enforce_eager:
            cmd.append("--disable-cuda-graph")
        if self.config.load_format and self.config.load_format != "dummy":
            cmd.extend(["--load-format", str(self.config.load_format)])
        max_running_requests = self.config.get("max_num_seqs", None)
        if max_running_requests is not None:
            cmd.extend(["--max-running-requests", str(max_running_requests)])
        for key, value in engine_kwargs.items():
            cli_key = "--" + key.replace("_", "-")
            if isinstance(value, bool):
                if value:
                    cmd.append(cli_key)
            elif value is not None:
                cmd.extend([cli_key, str(value)])

        env = os.environ.copy()
        env["CUDA_VISIBLE_DEVICES"] = os.environ["CUDA_VISIBLE_DEVICES"]
        env["SGLANG_SKIP_SGL_KERNEL_VERSION_CHECK"] = "1"
        env.setdefault("CUDA_HOME", "/data/anaconda3/envs/sglang_diffusion")

        log_path = f"/tmp/janus_sglang_server_{self.replica_rank}_{self.node_rank}_{os.getpid()}.log"
        self._server_log_file = open(log_path, "a", buffering=1)
        logger.info("Launching Janus SGLang server: %s", " ".join(cmd))
        self._server_process = subprocess.Popen(
            cmd,
            env=env,
            stdout=self._server_log_file,
            stderr=subprocess.STDOUT,
            text=True,
            preexec_fn=os.setsid,
        )

        await self._wait_for_http_server(log_path)

        self.processor = VLChatProcessor.from_pretrained(
            self.model_config.local_path,
            trust_remote_code=self.model_config.trust_remote_code,
        )
        self.tokenizer = self.processor.tokenizer
        if self.processor is None or self.tokenizer is None:
            raise RuntimeError("Janus SGLang rollout requires tokenizer_manager.processor/tokenizer")
        if getattr(self.tokenizer, "pad_token_id", None) is None:
            self.tokenizer.pad_token_id = self.processor.pad_id

        self.image_start_tag = self.processor.image_start_tag
        self.image_end_tag = self.processor.image_end_tag
        self.image_tag = self.processor.image_tag
        self.image_token_num_per_image = int(getattr(self.config, "image_token_num_per_image", 576))
        self.prompt_length = int(self.config.prompt_length)
        self.response_length = int(self.config.response_length)

        self.weight_update_worker_task = asyncio.create_task(self._weight_update_worker())
        logger.info(
            f"Janus SGLang server launched on {self._server_address}:{self._server_port} "
            f"replica={self.replica_rank} node={self.node_rank}"
        )

    async def _wait_for_http_server(self, log_path: str):
        health_url = self._server_url("/health")
        for _ in range(600):
            if self._server_process is not None and self._server_process.poll() is not None:
                tail = ""
                try:
                    with open(log_path, "r") as f:
                        tail = "".join(f.readlines()[-80:])
                except OSError:
                    pass
                raise RuntimeError(
                    f"Janus SGLang server exited early with code {self._server_process.returncode}.\n{tail}"
                )
            try:
                await asyncio.to_thread(urllib.request.urlopen, health_url, timeout=2)
                return
            except Exception:
                await asyncio.sleep(1.0)
        raise TimeoutError(f"Timed out waiting for Janus SGLang server health at {health_url}")

    async def init_model(self):
        await self.launch_server()

    async def set_latest_available_version(self, version: int):
        async with self.weight_update_lock:
            if version > self.latest_available_version:
                self.latest_available_version = version
                logger.info(f"[JanusSGLang {self.replica_rank}] v{version} available in SHM")

    async def ensure_weights_updated(self) -> int:
        async with self.weight_update_lock:
            if self.applied_version >= self.latest_available_version:
                return self.applied_version

        while True:
            async with self.weight_update_lock:
                if self.ongoing_generations == 0:
                    break
            await asyncio.sleep(0.05)

        async with self.weight_update_lock:
            target_v = self.latest_available_version
            if self.applied_version >= target_v:
                return self.applied_version
            await self._apply_weights_from_shm(target_v)
            self.applied_version = target_v
            return self.applied_version

    async def _apply_weights_from_shm(self, target_v: int):
        if int(self.config.tensor_model_parallel_size) != 1:
            raise RuntimeError("janus_sglang SHM flat weight update currently supports tensor_model_parallel_size=1")

        file_path = f"/dev/shm/weights_v{target_v}.pt"
        metadata_path = "/dev/shm/rollout_weight_metadata.pt"
        wait_count = 0
        while not os.path.exists(file_path):
            await asyncio.sleep(0.05)
            wait_count += 1
            if wait_count > 200:
                raise FileNotFoundError(f"{file_path} not found after 10s")
        if not os.path.exists(metadata_path):
            raise FileNotFoundError(
                f"{metadata_path} not found. Actor export must write rollout weight metadata for janus_sglang."
            )

        t0 = time.time()
        result = await self._post_json(
            "/janus/update_weights_from_flat_file",
            {
                "file_path": file_path,
                "metadata_path": metadata_path,
                "load_format": None,
                "weight_version": f"v{target_v}",
                "bucket_bytes": int(getattr(self.config, "update_weights_bucket_megabytes", 512)) << 20,
            },
        )
        if result.get("success") is False:
            raise RuntimeError(result.get("message", "unknown SGLang weight update failure"))
        logger.info(f"[JanusSGLang {self.replica_rank}] applied v{target_v} in {time.time() - t0:.2f}s")

    async def _weight_update_worker(self):
        while True:
            try:
                await asyncio.sleep(0.1)
                async with self.weight_update_lock:
                    pending = self.pending_weight_version
                    idle = self.ongoing_generations == 0
                if pending is not None and idle:
                    await self.set_latest_available_version(pending)
                    await self.ensure_weights_updated()
                    async with self.weight_update_lock:
                        if self.pending_weight_version == pending:
                            self.pending_weight_version = None
            except asyncio.CancelledError:
                break
            except Exception as exc:
                logger.error(f"Janus SGLang background weight update failed: {exc}", exc_info=True)
                await asyncio.sleep(1.0)

    async def queue_weight_update(self, version: int):
        async with self.weight_update_lock:
            if version == -1:
                self.pending_weight_version = None
            elif self.pending_weight_version is None or version > self.pending_weight_version:
                self.pending_weight_version = version

    async def get_weight_update_status(self) -> dict:
        async with self.weight_update_lock:
            return {
                "replica_rank": self.replica_rank,
                "node_rank": self.node_rank,
                "pending_weight_version": self.pending_weight_version,
                "ongoing_generations": self.ongoing_generations,
                "has_pending_update": self.pending_weight_version is not None,
                "latest_available_version": self.latest_available_version,
                "applied_version": self.applied_version,
            }

    def _server_url(self, path: str) -> str:
        assert self._server_port is not None, "SGLang HTTP server is not ready"
        return f"http://{self._server_address}:{self._server_port}{path}"

    @staticmethod
    def _post_json_sync(url: str, payload: dict[str, Any], timeout: float = 600.0):
        data = json.dumps(payload).encode("utf-8")
        request = urllib.request.Request(
            url,
            data=data,
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        try:
            with urllib.request.urlopen(request, timeout=timeout) as response:
                return json.loads(response.read().decode("utf-8"))
        except urllib.error.HTTPError as exc:
            body = exc.read().decode("utf-8", errors="replace")
            raise RuntimeError(f"SGLang request failed {exc.code}: {body}") from exc

    async def _post_json(self, path: str, payload: dict[str, Any]):
        return await asyncio.to_thread(self._post_json_sync, self._server_url(path), payload)

    def _get_generation_config(self, prompt: DataProto) -> dict[str, Any]:
        is_validate = bool(prompt.meta_info.get("validate", False))
        if is_validate:
            val_kwargs = self.config.val_kwargs
            return {
                "is_validate": True,
                "cfg_weight": getattr(val_kwargs, "val_cfg_weight", 5.0),
                "temperature": getattr(val_kwargs, "val_temperature", 1.0),
                "txt_top_k": getattr(val_kwargs, "val_txt_top_k", 50),
                "txt_top_p": getattr(val_kwargs, "val_txt_top_p", 1.0),
                "img_top_k": getattr(val_kwargs, "val_img_top_k", 4096),
                "img_top_p": getattr(val_kwargs, "val_img_top_p", 1.0),
            }
        else:
            return {
                "is_validate": False,
                "cfg_weight": getattr(self.config, "cfg_weight", 5.0),
                "temperature": getattr(self.config, "temperature", 1.0),
                "txt_top_k": getattr(self.config, "txt_top_k", 50),
                "txt_top_p": getattr(self.config, "txt_top_p", 1.0),
                "img_top_k": getattr(self.config, "img_top_k", 4096),
                "img_top_p": getattr(self.config, "img_top_p", 1.0),
            }

    def _get_sft_format(self, prompt: str, system_prompt: str = "", append_image_start: bool = True) -> str:
        formatted_system = (
            system_prompt.format(image_start_tag=self.image_start_tag, image_end_tag=self.image_end_tag)
            if system_prompt
            else ""
        )
        conversation = [{"role": "<|User|>", "content": prompt}, {"role": "<|Assistant|>", "content": ""}]
        if hasattr(self.processor, "apply_sft_template_for_multi_turn_prompts"):
            sft_format = self.processor.apply_sft_template_for_multi_turn_prompts(
                conversations=conversation,
                sft_format=self.processor.sft_format,
                system_prompt=formatted_system,
            )
        else:
            seps = ["\n\n", "<｜end▁of▁sentence｜>"]
            sft_format = (formatted_system + seps[0]) if formatted_system else ""
            for i, message in enumerate(conversation):
                role = message.get("role", "")
                content = message.get("content", "")
                if role in ("<|User|>", "User", "user"):
                    role = "<|User|>"
                elif role in ("<|Assistant|>", "Assistant", "assistant"):
                    role = "<|Assistant|>"
                if content:
                    sft_format += f"{role}: {content.strip()}{seps[i % 2]}"
                else:
                    sft_format += f"{role}:"
            sft_format = sft_format.strip()
        return sft_format + self.image_start_tag if append_image_start else sft_format

    def _task2_request_text(self, prompt: str) -> str:
        return (
            self._get_sft_format(prompt, append_image_start=False)
            + self.image_tag
            + "\nFirst, decompose the input prompt into explicit prompt contents that are visually verifiable.\n"
            + "Exclude subjective, inferential, or non-verifiable content.\n"
        )

    def _task2_training_text(self, prompt: str) -> str:
        return (
            self._get_sft_format(prompt)
            + self.image_tag
            + self.image_end_tag
            + "\nFirst, decompose the input prompt into explicit prompt contents that are visually verifiable.\n"
            + "Exclude subjective, inferential, or non-verifiable content.\n"
        )

    def _task3_training_text(self, prompt: str, feedback: str) -> str:
        content = (
            f"{self.image_start_tag}{self.image_tag}{self.image_end_tag}\n"
            "Please edit the image as instructed.\n"
            "The FEEDBACK will be given as multiple steps (Step 1, Step 2, ...). "
            "You MUST apply all steps in order and produce a final image reflecting all changes.\n"
            f"INPUT_PROMPT: {prompt}\n"
            f"FEEDBACK: \n{feedback or 'No need to generate feedback.'}"
        )
        return self._get_sft_format(content, system_prompt=EDIT_SYSTEM_PROMPT)

    @staticmethod
    def _as_str_list(values) -> list[str]:
        return [str(x) for x in np.asarray(values, dtype=object).reshape(-1).tolist()]

    def _tokenize_left(self, texts: list[str]) -> tuple[torch.Tensor, torch.Tensor]:
        old_padding_side = getattr(self.tokenizer, "padding_side", "right")
        self.tokenizer.padding_side = "left"
        self.tokenizer.pad_token_id = self.processor.pad_id
        try:
            encoded = self.tokenizer(texts, padding=True, return_tensors="pt")
        finally:
            self.tokenizer.padding_side = old_padding_side
        return encoded["input_ids"].long(), encoded["attention_mask"].long()

    def _expand_image_placeholders_ids(self, input_ids_tensor: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        image_id = int(self.processor.image_id)
        pad_id = int(self.processor.pad_id)
        k = int(self.image_token_num_per_image)
        sequences = []
        lengths = []
        for input_ids in input_ids_tensor:
            mask = input_ids == image_id
            counts = torch.where(mask, torch.full_like(input_ids, k), torch.ones_like(input_ids))
            seq = input_ids.repeat_interleave(counts)
            sequences.append(seq)
            lengths.append(seq.numel())
        max_len = max(lengths)
        out = torch.full((len(sequences), max_len), pad_id, dtype=torch.long)
        for i, seq in enumerate(sequences):
            out[i, max_len - seq.numel() :] = seq
        return out, (out != pad_id).long()

    def _pad_tensor_left(self, tensor: torch.Tensor, target_length: int, pad_value: int | float = 0) -> torch.Tensor:
        current_length = tensor.size(1)
        if current_length >= target_length:
            return tensor[:, -target_length:]
        shape = list(tensor.shape)
        shape[1] = target_length - current_length
        padding = torch.full(shape, pad_value, dtype=tensor.dtype, device=tensor.device)
        return torch.cat([padding, tensor], dim=1)

    def _pad_tensor_right(self, tensor: torch.Tensor, target_length: int, pad_value: int | float = 0) -> torch.Tensor:
        current_length = tensor.size(1)
        if current_length >= target_length:
            return tensor[:, :target_length]
        shape = list(tensor.shape)
        shape[1] = target_length - current_length
        padding = torch.full(shape, pad_value, dtype=tensor.dtype, device=tensor.device)
        return torch.cat([tensor, padding], dim=1)

    def _apply_padding_to_dataproto(self, data_proto: DataProto) -> DataProto:
        for key in ("task1_input_ids", "task1_attention_mask", "task2_input_ids", "task2_attention_mask", "task3_input_ids", "task3_attention_mask"):
            if key in data_proto.batch:
                pad_value = 0 if "mask" in key else self.processor.pad_id
                data_proto.batch[key] = self._pad_tensor_left(data_proto.batch[key], self.prompt_length, pad_value)
        for key in ("task2_feedback_ids", "task2_response_mask", "task2_segment_mask", "task2_rollout_log_probs"):
            if key in data_proto.batch:
                pad_value = 0 if ("mask" in key or "log_probs" in key) else self.tokenizer.eos_token_id
                data_proto.batch[key] = self._pad_tensor_right(data_proto.batch[key], self.response_length, pad_value)
        return data_proto

    @staticmethod
    def _image_base64_to_pil(image_base64: str) -> Image.Image:
        if image_base64.startswith("data:image"):
            image_base64 = image_base64.split(",", 1)[1]
        return Image.open(BytesIO(base64.b64decode(image_base64))).convert("RGB")

    @staticmethod
    def _pil_to_data_uri(image: Image.Image) -> str:
        buffer = BytesIO()
        image.save(buffer, format="PNG")
        return "data:image/png;base64," + base64.b64encode(buffer.getvalue()).decode("utf-8")

    def _pixel_values_from_pils(self, images: list[Image.Image]) -> torch.Tensor:
        pixel_values = self.processor.image_processor(images).pixel_values
        if not isinstance(pixel_values, torch.Tensor):
            pixel_values = torch.tensor(pixel_values)
        return pixel_values

    @staticmethod
    def _extract_token_logprobs(raw_logprobs, token_ids: list[int]) -> list[float]:
        values: list[float] = []
        for item in raw_logprobs or []:
            if isinstance(item, dict):
                value = item.get("logprob", item.get("log_prob", 0.0))
            elif isinstance(item, (int, float)):
                value = item
            elif isinstance(item, (list, tuple)) and item:
                value = item[0]
            else:
                value = 0.0
            values.append(float(value))
        if len(values) < len(token_ids):
            values.extend([0.0] * (len(token_ids) - len(values)))
        return values[: len(token_ids)]

    @staticmethod
    def _sglang_top_k(value: int | float) -> int:
        value = int(value)
        return value if value > 0 else -1

    @staticmethod
    def _stack_2d(values: list[list[int]], pad_value: int = 0, dtype: torch.dtype = torch.long) -> torch.Tensor:
        max_len = max((len(v) for v in values), default=0)
        out = torch.full((len(values), max_len), pad_value, dtype=dtype)
        for i, value in enumerate(values):
            if value:
                out[i, : len(value)] = torch.tensor(value, dtype=dtype)
        return out

    async def _generate_task1(self, data_proto: DataProto, gen_config: dict[str, Any]) -> DataProto:
        prompts = self._as_str_list(data_proto.non_tensor_batch["prompt"])
        input_ids, attention_mask = self._tokenize_left([self._get_sft_format(prompt) for prompt in prompts])
        data_proto.batch["task1_input_ids"] = input_ids.cpu()
        data_proto.batch["task1_attention_mask"] = attention_mask.cpu()

        async def request_one(prompt: str):
            return await self._post_json(
                "/janus/generate_image",
                {
                    "prompt": prompt,
                    "cfg_weight": float(gen_config["cfg_weight"]),
                    "temperature": float(gen_config["temperature"]),
                    "top_p": float(gen_config["img_top_p"]),
                    "top_k": self._sglang_top_k(gen_config["img_top_k"]),
                    "n": 1,
                    "return_logprob": not gen_config["is_validate"],
                },
            )

        results = await asyncio.gather(*[request_one(prompt) for prompt in prompts])
        image_infos = [result["images"][0] for result in results]
        token_lists = [[int(x) for x in info["image_token_ids"]] for info in image_infos]
        image_base64s = [info["image_base64"] for info in image_infos]
        images = [self._image_base64_to_pil(image_base64) for image_base64 in image_base64s]
        image_tokens = self._stack_2d(token_lists, pad_value=0, dtype=torch.long)
        pixel_values = self._pixel_values_from_pils(images)

        data_proto.non_tensor_batch["task1_gen_imgs_pil_list"] = np.array(images, dtype=object)
        data_proto.non_tensor_batch["task1_gen_imgs_base64"] = np.array(image_base64s, dtype=object)
        data_proto.non_tensor_batch["current_imgs_base64"] = np.array(image_base64s, dtype=object)
        data_proto.batch["task1_gen_imgs_pixel_values"] = pixel_values.cpu()
        data_proto.batch["task1_gen_img_tokens"] = image_tokens.cpu()
        data_proto.batch["task1_response_mask"] = torch.ones_like(image_tokens, dtype=torch.long).cpu()
        data_proto.batch["current_imgs_pixel_values"] = pixel_values.cpu().clone()
        data_proto.batch["current_img_tokens"] = image_tokens.cpu().clone()
        if not gen_config["is_validate"]:
            logprobs = [
                self._extract_token_logprobs(info.get("image_token_logprobs"), tokens)
                for info, tokens in zip(image_infos, token_lists)
            ]
            data_proto.batch["task1_rollout_log_probs"] = self._stack_2d(logprobs, pad_value=0, dtype=torch.float32)
        return data_proto

    async def _generate_task2(self, data_proto: DataProto, gen_config: dict[str, Any]) -> DataProto:
        prompts = self._as_str_list(data_proto.non_tensor_batch["prompt"])
        training_texts = [self._task2_training_text(prompt) for prompt in prompts]
        input_ids, _ = self._tokenize_left(training_texts)
        expanded_input_ids, expanded_attention_mask = self._expand_image_placeholders_ids(input_ids)
        data_proto.batch["task2_input_ids"] = expanded_input_ids.cpu()
        data_proto.batch["task2_attention_mask"] = expanded_attention_mask.cpu()

        current_base64 = data_proto.non_tensor_batch.get("current_imgs_base64")
        if current_base64 is None:
            current_base64 = data_proto.non_tensor_batch.get("task1_gen_imgs_base64")
        if current_base64 is None:
            pil_list = data_proto.non_tensor_batch.get("task1_gen_imgs_pil_list")
            if pil_list is None:
                raise ValueError("task2 requires current image base64 or PIL image from task1/task3")
            current_base64 = [self._pil_to_data_uri(img) for img in np.asarray(pil_list, dtype=object).reshape(-1)]
        current_base64 = self._as_str_list(current_base64)

        async def request_one(prompt: str, image_uri: str):
            return await self._post_json(
                "/generate",
                {
                    "text": self._task2_request_text(prompt),
                    "image_data": image_uri,
                    "sampling_params": {
                        "max_new_tokens": int(self.response_length),
                        "temperature": float(gen_config["temperature"]),
                        "top_p": float(gen_config["txt_top_p"]),
                        "top_k": self._sglang_top_k(gen_config["txt_top_k"]),
                        "skip_special_tokens": True,
                    },
                    "return_logprob": not gen_config["is_validate"],
                },
            )

        outputs = await asyncio.gather(*[request_one(prompt, image_uri) for prompt, image_uri in zip(prompts, current_base64)])
        token_lists = []
        feedback_texts = []
        logprob_lists = []
        for output in outputs:
            if isinstance(output, list):
                output = output[0]
            text = str(output.get("text", ""))
            token_ids = output.get("output_ids")
            if token_ids is None:
                token_ids = self.tokenizer.encode(text, add_special_tokens=False)
            token_ids = [int(x) for x in token_ids]
            feedback_texts.append(text)
            token_lists.append(token_ids)
            if not gen_config["is_validate"]:
                raw_logprobs = (output.get("meta_info") or {}).get("output_token_logprobs")
                logprob_lists.append(self._extract_token_logprobs(raw_logprobs, token_ids))

        feedback_ids = self._stack_2d(token_lists, pad_value=int(self.processor.pad_id), dtype=torch.long)
        segment_mask = build_segment_response_mask(feedback_ids, self.tokenizer)
        response_mask = (segment_mask > 0).long()
        data_proto.non_tensor_batch["task2_feedback_texts"] = np.array(feedback_texts, dtype=object)
        data_proto.batch["task2_feedback_ids"] = feedback_ids.cpu()
        data_proto.batch["task2_response_mask"] = response_mask.cpu()
        data_proto.batch["task2_segment_mask"] = segment_mask.cpu()
        if not gen_config["is_validate"]:
            log_probs = self._stack_2d(logprob_lists, pad_value=0, dtype=torch.float32)
            if log_probs.size(1) < response_mask.size(1):
                log_probs = self._pad_tensor_right(log_probs, response_mask.size(1), 0)
            data_proto.batch["task2_rollout_log_probs"] = log_probs[:, : response_mask.size(1)].masked_fill(response_mask == 0, 0.0).cpu()
        return data_proto

    async def _generate_task3(self, data_proto: DataProto, gen_config: dict[str, Any]) -> DataProto:
        prompts = self._as_str_list(data_proto.non_tensor_batch["prompt"])
        current_pixels = data_proto.batch.get("current_imgs_pixel_values", None)
        if current_pixels is None or len(current_pixels) == 0:
            current_pixels = data_proto.batch.get("task1_gen_imgs_pixel_values", None)
        if current_pixels is None or len(current_pixels) == 0:
            raise ValueError("task3 requires current image pixel values")
        current_tokens = data_proto.batch.get("current_img_tokens", None)
        if current_tokens is None or len(current_tokens) == 0:
            current_tokens = data_proto.batch.get("task1_gen_img_tokens", None)
        if current_tokens is None or len(current_tokens) == 0:
            raise ValueError("task3 requires current image tokens")

        data_proto.batch["task3_input_imgs_pixel_values"] = current_pixels.detach().cpu().clone()
        data_proto.batch["task3_input_img_tokens"] = current_tokens.detach().cpu().clone()

        raw_feedbacks = self._as_str_list(data_proto.non_tensor_batch.get("task2_feedback_texts", []))
        if len(raw_feedbacks) != len(prompts):
            raise ValueError("task3 requires one task2_feedback_text per prompt")
        feedbacks = [self.formatter_v3._split_text_into_parts(feedback)[-1] for feedback in raw_feedbacks]

        training_texts = [self._task3_training_text(prompt, feedback) for prompt, feedback in zip(prompts, feedbacks)]
        input_ids, _ = self._tokenize_left(training_texts)
        expanded_input_ids, expanded_attention_mask = self._expand_image_placeholders_ids(input_ids)
        data_proto.batch["task3_input_ids"] = expanded_input_ids.cpu()
        data_proto.batch["task3_attention_mask"] = expanded_attention_mask.cpu()

        token_lists_for_request = current_tokens.detach().cpu().long().tolist()

        async def request_one(prompt: str, feedback: str, image_tokens: list[int]):
            return await self._post_json(
                "/janus/generate_image",
                {
                    "mode": "edit",
                    "input_prompt": prompt,
                    "feedback": feedback or "No need to generate feedback.",
                    "input_image_token_ids": image_tokens,
                    "cfg_weight": float(gen_config["cfg_weight"]),
                    "temperature": float(gen_config["temperature"]),
                    "top_p": float(gen_config["img_top_p"]),
                    "top_k": self._sglang_top_k(gen_config["img_top_k"]),
                    "n": 1,
                    "return_logprob": not gen_config["is_validate"],
                },
            )

        results = await asyncio.gather(
            *[
                request_one(prompt, feedback, image_tokens)
                for prompt, feedback, image_tokens in zip(prompts, feedbacks, token_lists_for_request)
            ]
        )
        image_infos = [result["images"][0] for result in results]
        regen_token_lists = [[int(x) for x in info["image_token_ids"]] for info in image_infos]
        regen_base64s = [info["image_base64"] for info in image_infos]
        regen_images = [self._image_base64_to_pil(image_base64) for image_base64 in regen_base64s]
        regen_tokens = self._stack_2d(regen_token_lists, pad_value=0, dtype=torch.long)
        regen_pixels = self._pixel_values_from_pils(regen_images)

        data_proto.non_tensor_batch["task3_regen_imgs_pil_list"] = np.array(regen_images, dtype=object)
        data_proto.non_tensor_batch["task3_regen_imgs_base64"] = np.array(regen_base64s, dtype=object)
        data_proto.non_tensor_batch["current_imgs_base64"] = np.array(regen_base64s, dtype=object)
        data_proto.batch["task3_regen_imgs_pixel_values"] = regen_pixels.cpu()
        data_proto.batch["task3_regen_img_tokens"] = regen_tokens.cpu()
        data_proto.batch["task3_response_mask"] = torch.ones_like(regen_tokens, dtype=torch.long).cpu()
        data_proto.batch["current_imgs_pixel_values"] = regen_pixels.cpu().clone()
        data_proto.batch["current_img_tokens"] = regen_tokens.cpu().clone()
        if not gen_config["is_validate"]:
            logprobs = [
                self._extract_token_logprobs(info.get("image_token_logprobs"), tokens)
                for info, tokens in zip(image_infos, regen_token_lists)
            ]
            data_proto.batch["task3_rollout_log_probs"] = self._stack_2d(logprobs, pad_value=0, dtype=torch.float32)
        return data_proto

    async def _generate_step(
        self,
        prompt_data: DataProto,
        sampling_params: dict[str, Any],
        request_id: str,
    ) -> DataProto:
        async with self.weight_update_lock:
            self.ongoing_generations += 1
        try:
            prompt_data.meta_info.update(sampling_params)
            gen_config = self._get_generation_config(prompt_data)
            prompt_data.meta_info.update(
                temperature=gen_config["temperature"],
                cfg_weight=gen_config["cfg_weight"],
                txt_top_k=gen_config["txt_top_k"],
                txt_top_p=gen_config["txt_top_p"],
                img_top_k=gen_config["img_top_k"],
                img_top_p=gen_config["img_top_p"],
            )
            task_funcs = {
                1: self._generate_task1,
                2: self._generate_task2,
                3: self._generate_task3,
            }
            task_id_tensor = prompt_data.batch.get("task_id", None)
            if task_id_tensor is not None:
                task_id = int(task_id_tensor.view(-1)[0].item())
                selected = [task_funcs.get(task_id)]
            else:
                rollout_task_ids = prompt_data.meta_info.get("rollout_task_ids", None)
                selected = [task_funcs[i] for i in rollout_task_ids if i in task_funcs] if rollout_task_ids else list(task_funcs.values())

            gen_start = time.perf_counter()
            for func in selected:
                if func is not None:
                    prompt_data = await func(prompt_data, gen_config)
            elapsed = time.perf_counter() - gen_start
            prompt_data.meta_info.setdefault("metrics", {})["generate_sequences"] = [elapsed] * len(prompt_data)
            return self._apply_padding_to_dataproto(prompt_data)
        finally:
            async with self.weight_update_lock:
                self.ongoing_generations -= 1

    async def generate_for_partial(
        self,
        prompt_data: DataProto,
        sampling_params: dict[str, Any],
        request_id: str,
    ) -> tuple[Optional[DataProto], bool]:
        async with self.lock:
            if self.paused:
                return None, True
            self.cancel_event[request_id] = asyncio.Event()
            cancel_handle = asyncio.create_task(self.cancel_event[request_id].wait())
            generation_handle = asyncio.create_task(self._generate_step(prompt_data, sampling_params, request_id))
            self.generation_tasks[request_id] = generation_handle

        done, pending = await asyncio.wait([generation_handle, cancel_handle], return_when=asyncio.FIRST_COMPLETED)
        result = None
        is_cancel = False
        for task in done:
            if task == generation_handle:
                result = await task
            elif task == cancel_handle:
                is_cancel = True
        for task in pending:
            task.cancel()
            try:
                await task
            except asyncio.CancelledError:
                pass
        async with self.lock:
            self.cancel_event.pop(request_id, None)
            self.generation_tasks.pop(request_id, None)
            if generation_handle not in done:
                is_cancel = True
        return result, is_cancel

    async def generate(self, prompt_data: DataProto, sampling_params: dict[str, Any], request_id: str) -> DataProto:
        result, is_cancel = await self.generate_for_partial(prompt_data, sampling_params, request_id)
        if is_cancel or result is None:
            raise RuntimeError(f"Janus SGLang generation cancelled or failed for {request_id}")
        return result

    async def cancel(self):
        async with self.lock:
            self.paused = True
            for event in list(self.cancel_event.values()):
                event.set()
            for task in list(self.generation_tasks.values()):
                if not task.done():
                    task.cancel()

    async def resume(self):
        async with self.lock:
            self.paused = False

    async def wake_up(self):
        return None

    async def sleep(self):
        return None

    async def reset_prefix_cache(self):
        try:
            await self._post_json("/flush_cache", {})
        except Exception as exc:
            logger.warning(f"Janus SGLang flush_cache failed: {exc}")
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    async def clear_kv_cache(self):
        await self.reset_prefix_cache()

    async def shutdown(self):
        if self.weight_update_worker_task is not None:
            self.weight_update_worker_task.cancel()
        if self._server_process is not None and self._server_process.poll() is None:
            try:
                os.killpg(os.getpgid(self._server_process.pid), signal.SIGTERM)
            except Exception:
                self._server_process.terminate()
            await asyncio.sleep(2)
            if self._server_process.poll() is None:
                try:
                    os.killpg(os.getpgid(self._server_process.pid), signal.SIGKILL)
                except Exception:
                    self._server_process.kill()
        if self._server_log_file is not None:
            self._server_log_file.close()

    def __del__(self):
        proc = getattr(self, "_server_process", None)
        if proc is not None and proc.poll() is None:
            try:
                os.killpg(os.getpgid(proc.pid), signal.SIGTERM)
            except Exception:
                try:
                    proc.terminate()
                except Exception:
                    pass
