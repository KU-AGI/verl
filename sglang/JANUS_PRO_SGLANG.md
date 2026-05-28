# Janus-Pro SGLang 사용법

이 문서는 Janus/Janus-Pro 계열 모델을 SGLang 서버 한 개에 올려서
`image generation`, `image understanding`, `image edit/regen`을 처리하는 방법을 정리한다.

현재 구현의 기준은 `verl/workers/rollout/image_unified_rollout.py`의 task1, task2, task3 흐름이다.

- Task1: text prompt -> image token 576개 -> PNG/base64 image
- Task2: generated image + prompt -> text critique/feedback
- Task3: source image token 576개 + prompt + feedback -> edited image token 576개 -> PNG/base64 image

## 서버 실행

Janus-Pro-7B 기본 checkpoint:

```bash
cd /home/gpuuser/yongjin/research/sglang

CUDA_VISIBLE_DEVICES=0 \
CUDA_HOME=/data/anaconda3/envs/sglang_diffusion \
SGLANG_SKIP_SGL_KERNEL_VERSION_CHECK=1 \
PYTHONPATH=/home/gpuuser/yongjin/research/sglang/python \
/data/anaconda3/envs/sglang_diffusion/bin/python -m sglang.launch_server \
  --model-path /data/mllm/checkpoints/Janus-Pro-7B \
  --host 127.0.0.1 \
  --port 31000 \
  --trust-remote-code \
  --chat-template janus-pro \
  --dtype bfloat16 \
  --mem-fraction-static 0.95 \
  --attention-backend triton \
  --sampling-backend pytorch \
  --enable-janus-image-cuda-graph \
  --disable-radix-cache
```

SFT rollout checkpoint를 사용할 때는 `--model-path`만 바꾼다.

```bash
--model-path /data/mllm/experements/ckpt/janus_sft/0423_v10_sft_no_summarize/version_0/step=014000.ckpt/hf_model
```

옵션 기준:

- `--disable-radix-cache`: task3/edit까지 한 서버에서 안전하게 쓰려면 켠다. edit은 동일한 placeholder token 위치에 source image VQ embedding을 주입하므로, prefix cache가 source image 차이를 모르면 잘못된 cache를 재사용할 수 있다.
- `--enable-janus-image-cuda-graph`: image-generation-only decode batch에 Janus image CUDA graph를 사용한다. text/image-understanding이 같은 continuous batch에 섞이면 그 step은 eager로 fallback하고, 다시 image-only가 되면 graph를 탄다.
- `--attention-backend triton`, `--sampling-backend pytorch`: 현재 검증한 조합이다.

상태 확인:

```bash
curl -sS http://127.0.0.1:31000/health
```

## 공통 규칙

- image token 수는 항상 576개다. `384 / 16 = 24`, `24 * 24 = 576`.
- image generation CFG는 `guided = uncond + cfg_weight * (cond - uncond)`로 적용한다.
- `/janus/generate_image` 응답의 `image_token_ids`는 생성된 576개 VQ token만 담는다. `<begin_of_image>`는 포함하지 않는다.
- `n`은 지원한다. `n=16`이면 사용자에게는 image 16개를 반환하고, 내부적으로는 CFG 때문에 cond/uncond 32개 sequence가 돈다.
- task2 request text에는 `<image_placeholder>`만 넣는다. `<begin_of_image>`와 `<end_of_image>`를 직접 넣지 않는다.
- Janus tokenizer는 HF/verl rollout과 같은 fast tokenizer 경로를 사용해야 한다. SGLang이 slow Llama tokenizer로 떨어지면 공백/개행이 사라져 task2 text가 깨진다.

## Task1: 이미지 생성

Task1은 `/janus/generate_image`를 사용한다.

요청은 prompt만 넣으면 된다. 서버가 내부적으로 SFT template을 만들고 마지막에 `<begin_of_image>`를 붙인다.

내부 prompt 형태:

```text
<|User|>: {prompt}

<|Assistant|>:<begin_of_image>
```

요청 예시:

```bash
curl -sS http://127.0.0.1:31000/janus/generate_image \
  -H 'Content-Type: application/json' \
  -d '{
    "prompt": "a photo of three baseball bats",
    "cfg_weight": 5,
    "temperature": 1,
    "top_p": 1,
    "top_k": 4096,
    "n": 16
  }' > /tmp/task1_image_gen.json
```

응답:

```json
{
  "images": [
    {
      "image_base64": "...",
      "image_url": "data:image/png;base64,...",
      "image_token_ids": [576개의 VQ image token],
      "finish_reason": {"type": "length", "length": 576}
    }
  ]
}
```

Task3에서 edit/regen을 하려면 여기서 받은 `image_token_ids` 576개를 보관한다.

## Task2: 이미지 이해 / critique

Task2는 일반 `/generate`를 사용한다. `image_data`에는 Task1에서 만든 PNG를 base64 data URI로 넣거나, SGLang multimodal loader가 받을 수 있는 이미지 payload를 넣는다.

중요: SGLang 요청 text에는 `<image_placeholder>`만 넣는다. Janus processor가 내부에서 이 placeholder를 `<begin_of_image>` + 576 image tokens + `<end_of_image>`로 확장한다.

rollout prompt 형태:

```text
<|User|>: {prompt}

<|Assistant|>:<image_placeholder>
First, decompose the input prompt into explicit prompt contents that are visually verifiable.
Exclude subjective, inferential, or non-verifiable content.
```

요청 예시:

```bash
curl -sS http://127.0.0.1:31000/generate \
  -H 'Content-Type: application/json' \
  -d '{
    "text": "<|User|>: a photo of three baseball bats\n\n<|Assistant|>:<image_placeholder>\nFirst, decompose the input prompt into explicit prompt contents that are visually verifiable.\nExclude subjective, inferential, or non-verifiable content.\n",
    "image_data": "data:image/png;base64,...",
    "sampling_params": {
      "max_new_tokens": 1024,
      "temperature": 1,
      "top_p": 1,
      "top_k": -1,
      "skip_special_tokens": true
    }
  }' > /tmp/task2_image_und.json
```

Batch로 여러 이미지를 처리할 때는 `text`와 `image_data`를 같은 길이의 list로 보낸다.

```json
{
  "text": [
    "<|User|>: ...\n\n<|Assistant|>:<image_placeholder>\nFirst, ...\n",
    "<|User|>: ...\n\n<|Assistant|>:<image_placeholder>\nFirst, ...\n"
  ],
  "image_data": [
    "data:image/png;base64,...",
    "data:image/png;base64,..."
  ],
  "sampling_params": {
    "max_new_tokens": 1024,
    "temperature": 1,
    "top_p": 1,
    "top_k": -1,
    "skip_special_tokens": true
  }
}
```

Task2 output에서 `Third, Generate corrective feedback.` 뒤의 내용을 Task3 `feedback`으로 넘긴다. 해당 구간을 찾지 못하면 전체 output을 feedback으로 쓰거나, rollout 정책에 맞춰 `No need to generate feedback.`을 넣는다.

## Task3: 이미지 edit / regen

Task3도 `/janus/generate_image`를 사용하되 `mode: "edit"`와 `input_image_token_ids`를 넣는다.

서버는 내부적으로 source image를 raw pixel image로 다시 encode하지 않는다. Task1 또는 이전 Task3 응답의 576개 `image_token_ids`를 source image VQ token으로 사용한다.

요청 예시:

```bash
curl -sS http://127.0.0.1:31000/janus/generate_image \
  -H 'Content-Type: application/json' \
  -d '{
    "mode": "edit",
    "input_prompt": "a photo of three baseball bats",
    "feedback": "Step 1: Remove the extra baseball bat so that exactly three baseball bats remain.",
    "input_image_token_ids": [576개의 VQ image token],
    "cfg_weight": 5,
    "temperature": 1,
    "top_p": 1,
    "top_k": 4096,
    "n": 1
  }' > /tmp/task3_edit.json
```

`input_image_token_ids` 형식:

- 576개짜리 list 하나: 같은 source image로 `n`개 edit 생성
- 576개짜리 list-of-list: output마다 다른 source image 사용

예시:

```json
{
  "mode": "edit",
  "input_prompt": "a photo of three baseball bats",
  "feedback": "No need to generate feedback.",
  "input_image_token_ids": [
    [576개의 VQ image token],
    [576개의 VQ image token]
  ],
  "cfg_weight": 5,
  "temperature": 1,
  "top_p": 1,
  "top_k": 4096,
  "n": 2
}
```

Task3 CFG는 Task1 text-to-image CFG와 다르다. 조건 row와 비조건 row는 다음처럼 만든다.

- source image VQ embedding 구간은 cond/uncond 모두 보존한다.
- `<end_of_image>` 뒤 newline부터 `<|Assistant|>` 직후까지의 instruction 구간만 uncond에서 pad embedding으로 바꾼다.
- 마지막 output `<begin_of_image>`는 cond/uncond 모두 보존한다.
- 이후 576개 image token을 생성한다.

## PNG 저장

`image_base64`를 파일로 저장하는 간단한 예시:

```bash
/data/anaconda3/envs/sglang_diffusion/bin/python - <<'PY'
import base64
import json

with open("/tmp/task1_image_gen.json") as f:
    out = json.load(f)

for i, item in enumerate(out["images"]):
    with open(f"/tmp/janus_{i:02d}.png", "wb") as f:
        f.write(base64.b64decode(item["image_base64"]))
PY
```

## 권장 rollout 순서

1. Task1 `/janus/generate_image`
   - 입력: `prompt`, `cfg_weight=5`, `temperature=1`, `top_p=1`, `top_k=4096`, `n`
   - 저장: `image_base64`, `image_token_ids`

2. Task2 `/generate`
   - 입력: task1 PNG를 `image_data`로 넣고, text에는 `<image_placeholder>`만 넣는다.
   - sampling: `max_new_tokens=1024`, `temperature=1`, `top_p=1`, `top_k=-1`
   - 저장: critique text, feedback text

3. Task3 `/janus/generate_image`
   - 입력: `mode="edit"`, task1 또는 이전 task3의 `image_token_ids`, `input_prompt`, `feedback`
   - sampling: `cfg_weight=5`, `temperature=1`, `top_p=1`, `top_k=4096`, `n`
   - 조건: 서버는 `--disable-radix-cache`로 실행

## 검증한 상태

2026-05-10 기준으로 다음을 확인했다.

- Task1 image generation: `n=16`, 모든 output이 576 image token 반환.
- Task2 image understanding: HF/verl tokenizer와 같은 prompt tokenization으로 정상 critique 형식 반환.
- Task3 edit/regen: source image token 576개를 gen embedding으로 주입하고 576 image token 반환.
- SGLang tokenizer가 slow Llama tokenizer로 떨어지면 공백/개행이 사라져 task2 output이 깨진다. 현재 Janus `multi_modality` checkpoint는 `tokenizer.json` fast tokenizer를 직접 로드하도록 수정했다.
