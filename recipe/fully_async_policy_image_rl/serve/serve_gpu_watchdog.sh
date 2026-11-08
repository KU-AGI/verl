#!/usr/bin/env bash
set -uo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../../.." && pwd)"
cd "${REPO_ROOT}"

GPU_LIST="${GPU_LIST:-5 6 7}"
read -r -a GPUS <<< "${GPU_LIST}"

MODEL_PATH="${MODEL_PATH:-/home/work/AGILAB/mllm_reasoning/data/checkpoints/Qwen3.5-35B-A3B}"
SERVED_MODEL_NAME="${SERVED_MODEL_NAME:-Qwen/Qwen3.5-35B-A3B}"
MEM_FRACTION_STATIC="${MEM_FRACTION_STATIC:-0.94}"
CONTEXT_LENGTH="${CONTEXT_LENGTH:-16384}"
MAX_RUNNING_REQUESTS="${MAX_RUNNING_REQUESTS:-16}"
MAX_MAMBA_CACHE_SIZE="${MAX_MAMBA_CACHE_SIZE:-80}"
CHUNKED_PREFILL_SIZE="${CHUNKED_PREFILL_SIZE:-4096}"
MAX_PREFILL_TOKENS="${MAX_PREFILL_TOKENS:-16384}"
CUDA_GRAPH_MAX_BS_DECODE="${CUDA_GRAPH_MAX_BS_DECODE:-16}"
NUM_CONTINUOUS_DECODE_STEPS="${NUM_CONTINUOUS_DECODE_STEPS:-4}"
ATTENTION_BACKEND="${ATTENTION_BACKEND:-triton}"
SAMPLING_BACKEND="${SAMPLING_BACKEND:-pytorch}"

RM_PORT_BASE="${RM_PORT_BASE:-8000}"
CHECK_INTERVAL_SECONDS="${CHECK_INTERVAL_SECONDS:-10}"
HEALTH_TIMEOUT_SECONDS="${HEALTH_TIMEOUT_SECONDS:-5}"
STARTUP_GRACE_SECONDS="${STARTUP_GRACE_SECONDS:-600}"
RM_HEALTH_FAILURE_THRESHOLD="${RM_HEALTH_FAILURE_THRESHOLD:-6}"
RESTART_DELAY_SECONDS="${RESTART_DELAY_SECONDS:-10}"
MAX_RESTART_DELAY_SECONDS="${MAX_RESTART_DELAY_SECONDS:-60}"
TERM_GRACE_SECONDS="${TERM_GRACE_SECONDS:-20}"
START_STAGGER_SECONDS="${START_STAGGER_SECONDS:-10}"
LOG_DIR="${LOG_DIR:-/tmp/fully_async_policy_image_rl_gpu567}"
LOCK_FILE="${LOCK_FILE:-/tmp/fully_async_policy_image_rl_gpu567.lock}"
SKIP_PORT_CHECK="${SKIP_PORT_CHECK:-0}"
PORT_CHECK_PYTHON="${PORT_CHECK_PYTHON:-/home/work/AGILAB/conda/envs/sglang/bin/python}"
RM_SCRIPT="${SCRIPT_DIR}/serve_rm_sglang.sh"
SUPERVISOR_LOG="${LOG_DIR}/watchdog.log"

mkdir -p "${LOG_DIR}"

log() {
    printf '[%(%Y-%m-%d %H:%M:%S)T] %s\n' -1 "$*" | tee -a "${SUPERVISOR_LOG}"
}

if (("${#GPUS[@]}" == 0)); then
    log "ERROR: GPU_LIST is empty"
    exit 1
fi

for required_command in curl setsid flock; do
    if ! command -v "${required_command}" >/dev/null 2>&1; then
        log "ERROR: required command not found: ${required_command}"
        exit 1
    fi
done
if [[ ! -f "${RM_SCRIPT}" ]]; then
    log "ERROR: required script not found: ${RM_SCRIPT}"
    exit 1
fi

exec 9>"${LOCK_FILE}"
if ! flock -n 9; then
    log "ERROR: another watchdog already holds ${LOCK_FILE}"
    exit 1
fi

declare -A RM_PID RM_STARTED_AT RM_FAILURES RM_RESTARTS

port_is_in_use() {
    local port="$1"
    "${PORT_CHECK_PYTHON}" -c \
        'import socket, sys; s = socket.socket(); s.settimeout(1); sys.exit(0 if s.connect_ex(("127.0.0.1", int(sys.argv[1]))) == 0 else 1)' \
        "${port}"
}

preflight_ports() {
    local gpu port conflict=0
    if [[ "${SKIP_PORT_CHECK}" == "1" ]]; then
        log "WARNING: initial port conflict check is disabled"
        return 0
    fi
    for gpu in "${GPUS[@]}"; do
        port=$((RM_PORT_BASE + gpu))
        if port_is_in_use "${port}"; then
            log "ERROR: GPU ${gpu} RM port ${port} is already in use"
            conflict=1
        fi
    done
    if ((conflict)); then
        log "Stop existing services or choose a different RM_PORT_BASE."
        return 1
    fi
}

start_service() {
    local gpu="$1"
    local port=$((RM_PORT_BASE + gpu))
    local log_file="${LOG_DIR}/gpu${gpu}_rm.log"
    local pid

    setsid env \
        GPU_ID="${gpu}" \
        PORT="${port}" \
        MODEL_PATH="${MODEL_PATH}" \
        SERVED_MODEL_NAME="${SERVED_MODEL_NAME}" \
        MEM_FRACTION_STATIC="${MEM_FRACTION_STATIC}" \
        CONTEXT_LENGTH="${CONTEXT_LENGTH}" \
        MAX_RUNNING_REQUESTS="${MAX_RUNNING_REQUESTS}" \
        MAX_MAMBA_CACHE_SIZE="${MAX_MAMBA_CACHE_SIZE}" \
        CHUNKED_PREFILL_SIZE="${CHUNKED_PREFILL_SIZE}" \
        MAX_PREFILL_TOKENS="${MAX_PREFILL_TOKENS}" \
        CUDA_GRAPH_MAX_BS_DECODE="${CUDA_GRAPH_MAX_BS_DECODE}" \
        NUM_CONTINUOUS_DECODE_STEPS="${NUM_CONTINUOUS_DECODE_STEPS}" \
        ATTENTION_BACKEND="${ATTENTION_BACKEND}" \
        SAMPLING_BACKEND="${SAMPLING_BACKEND}" \
        bash "${RM_SCRIPT}" >>"${log_file}" 2>&1 </dev/null &

    pid=$!
    RM_PID["${gpu}"]="${pid}"
    RM_STARTED_AT["${gpu}"]="$(date +%s)"
    RM_FAILURES["${gpu}"]=0
    log "Started RM on GPU ${gpu}, port ${port}, pid ${pid}, log ${log_file}"
}

stop_service() {
    local gpu="$1"
    local pid="${RM_PID[${gpu}]:-}"
    local attempt
    [[ -n "${pid}" ]] || return 0

    kill -TERM -- "-${pid}" 2>/dev/null || true
    for ((attempt = 0; attempt < TERM_GRACE_SECONDS; attempt++)); do
        if ! kill -0 -- "-${pid}" 2>/dev/null; then
            break
        fi
        sleep 1
    done
    if kill -0 -- "-${pid}" 2>/dev/null; then
        log "Force killing RM process group ${pid} on GPU ${gpu}"
        kill -KILL -- "-${pid}" 2>/dev/null || true
    fi
    wait "${pid}" 2>/dev/null || true
    unset 'RM_PID['"${gpu}"']'
}

health_ok() {
    local gpu="$1"
    local port=$((RM_PORT_BASE + gpu))
    local response
    response="$(curl -fsS --max-time "${HEALTH_TIMEOUT_SECONDS}" \
        "http://127.0.0.1:${port}/v1/models" 2>/dev/null)" || return 1
    grep -Fq "${SERVED_MODEL_NAME}" <<< "${response}"
}

restart_service() {
    local gpu="$1"
    local restarts delay
    stop_service "${gpu}"
    restarts=$(("${RM_RESTARTS[${gpu}]:-0}" + 1))
    RM_RESTARTS["${gpu}"]="${restarts}"
    delay=$((RESTART_DELAY_SECONDS * restarts))
    if ((delay > MAX_RESTART_DELAY_SECONDS)); then
        delay="${MAX_RESTART_DELAY_SECONDS}"
    fi
    log "Reloading RM on GPU ${gpu} after ${delay}s (restart ${restarts})"
    sleep "${delay}"
    start_service "${gpu}"
}

check_service() {
    local gpu="$1"
    local pid="${RM_PID[${gpu}]:-}"
    local started_at="${RM_STARTED_AT[${gpu}]:-0}"
    local failures="${RM_FAILURES[${gpu}]:-0}"
    local now age

    if [[ -z "${pid}" ]] || ! kill -0 "${pid}" 2>/dev/null; then
        [[ -z "${pid}" ]] || wait "${pid}" 2>/dev/null || true
        log "Detected exited RM process on GPU ${gpu}"
        restart_service "${gpu}"
        return
    fi

    now="$(date +%s)"
    age=$((now - started_at))
    if ((age < STARTUP_GRACE_SECONDS)); then
        return
    fi

    if health_ok "${gpu}"; then
        if ((failures > 0)); then
            log "RM on GPU ${gpu} recovered after ${failures} failed health checks"
        fi
        RM_FAILURES["${gpu}"]=0
        RM_RESTARTS["${gpu}"]=0
        return
    fi

    failures=$((failures + 1))
    RM_FAILURES["${gpu}"]="${failures}"
    log "Health check failed for RM on GPU ${gpu} (${failures}/${RM_HEALTH_FAILURE_THRESHOLD})"
    if ((failures >= RM_HEALTH_FAILURE_THRESHOLD)); then
        log "Detected unhealthy RM on GPU ${gpu}"
        restart_service "${gpu}"
    fi
}

shutdown() {
    local exit_code=$?
    local gpu
    trap - INT TERM EXIT
    log "Stopping watchdog and all managed RM servers"
    for gpu in "${GPUS[@]}"; do
        stop_service "${gpu}"
    done
    exit "${exit_code}"
}

trap shutdown INT TERM EXIT

if ! preflight_ports; then
    exit 1
fi

log "Starting ${SERVED_MODEL_NAME} on GPUs: ${GPUS[*]}"
log "Profile: context=${CONTEXT_LENGTH}, requests=${MAX_RUNNING_REQUESTS}, mamba_cache=${MAX_MAMBA_CACHE_SIZE}, cuda_graph_bs=${CUDA_GRAPH_MAX_BS_DECODE}"
for gpu in "${GPUS[@]}"; do
    RM_RESTARTS["${gpu}"]=0
    start_service "${gpu}"
    sleep "${START_STAGGER_SECONDS}"
done

log "All RM services launched; entering monitor loop"
while true; do
    for gpu in "${GPUS[@]}"; do
        check_service "${gpu}"
    done
    sleep "${CHECK_INTERVAL_SECONDS}"
done
