#!/usr/bin/env bash
set -uo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../../.." && pwd)"
cd "${REPO_ROOT}"

GPU_LIST="${GPU_LIST:-5 6 7}"
DETECTOR_GPU_LIST="${DETECTOR_GPU_LIST:-6 7}"
read -r -a GPUS <<< "${GPU_LIST}"
read -r -a DETECTOR_GPUS <<< "${DETECTOR_GPU_LIST}"

# GPU 5 has no detector, so it can use a larger SGLang cache/concurrency profile.
HIGH_CAPACITY_GPU="${HIGH_CAPACITY_GPU:-5}"
HIGH_CAPACITY_MEM_FRACTION_STATIC="${HIGH_CAPACITY_MEM_FRACTION_STATIC:-0.92}"
HIGH_CAPACITY_MAX_RUNNING_REQUESTS="${HIGH_CAPACITY_MAX_RUNNING_REQUESTS:-32}"
HIGH_CAPACITY_MAX_TOTAL_TOKENS="${HIGH_CAPACITY_MAX_TOTAL_TOKENS:-65536}"
HIGH_CAPACITY_CHUNKED_PREFILL_SIZE="${HIGH_CAPACITY_CHUNKED_PREFILL_SIZE:-4096}"
HIGH_CAPACITY_MAX_PREFILL_TOKENS="${HIGH_CAPACITY_MAX_PREFILL_TOKENS:-8192}"
HIGH_CAPACITY_CUDA_GRAPH_MAX_BS="${HIGH_CAPACITY_CUDA_GRAPH_MAX_BS:-32}"
HIGH_CAPACITY_MAX_MAMBA_CACHE_SIZE="${HIGH_CAPACITY_MAX_MAMBA_CACHE_SIZE:-96}"

RM_PORT_BASE="${RM_PORT_BASE:-8000}"
DET_PORT_BASE="${DET_PORT_BASE:-8080}"
CHECK_INTERVAL_SECONDS="${CHECK_INTERVAL_SECONDS:-10}"
HEALTH_TIMEOUT_SECONDS="${HEALTH_TIMEOUT_SECONDS:-5}"
STARTUP_GRACE_SECONDS="${STARTUP_GRACE_SECONDS:-600}"
RM_HEALTH_FAILURE_THRESHOLD="${RM_HEALTH_FAILURE_THRESHOLD:-6}"
DET_HEALTH_FAILURE_THRESHOLD="${DET_HEALTH_FAILURE_THRESHOLD:-18}"
RESTART_DELAY_SECONDS="${RESTART_DELAY_SECONDS:-10}"
MAX_RESTART_DELAY_SECONDS="${MAX_RESTART_DELAY_SECONDS:-60}"
TERM_GRACE_SECONDS="${TERM_GRACE_SECONDS:-20}"
START_STAGGER_SECONDS="${START_STAGGER_SECONDS:-10}"
LOG_DIR="${LOG_DIR:-/tmp/fully_async_policy_image_rl_gpu567}"
LOCK_FILE="${LOCK_FILE:-/tmp/fully_async_policy_image_rl_gpu567.lock}"
SKIP_PORT_CHECK="${SKIP_PORT_CHECK:-0}"
PORT_CHECK_PYTHON="${PORT_CHECK_PYTHON:-/home/work/AGILAB/conda/envs/sglang/bin/python}"

RM_SCRIPT="${SCRIPT_DIR}/serve_rm_sglang_detector.sh"
DET_SCRIPT="${SCRIPT_DIR}/serve_det.sh"
SUPERVISOR_LOG="${LOG_DIR}/watchdog.log"
EXPECTED_MODEL="Qwen/Qwen3.5-35B-A3B"

mkdir -p "${LOG_DIR}"

log() {
    printf '[%(%Y-%m-%d %H:%M:%S)T] %s\n' -1 "$*" | tee -a "${SUPERVISOR_LOG}"
}

if ((${#GPUS[@]} == 0)); then
    log "ERROR: GPU_LIST is empty"
    exit 1
fi

for detector_gpu in "${DETECTOR_GPUS[@]}"; do
    found=0
    for gpu in "${GPUS[@]}"; do
        if [[ "${detector_gpu}" == "${gpu}" ]]; then
            found=1
            break
        fi
    done
    if ((found == 0)); then
        log "ERROR: detector GPU ${detector_gpu} is not included in GPU_LIST"
        exit 1
    fi
done

for required_command in curl setsid flock; do
    if ! command -v "${required_command}" >/dev/null 2>&1; then
        log "ERROR: required command not found: ${required_command}"
        exit 1
    fi
done

for required_script in "${RM_SCRIPT}" "${DET_SCRIPT}"; do
    if [[ ! -f "${required_script}" ]]; then
        log "ERROR: required script not found: ${required_script}"
        exit 1
    fi
done

exec 9>"${LOCK_FILE}"
if ! flock -n 9; then
    log "ERROR: another watchdog already holds ${LOCK_FILE}"
    exit 1
fi

declare -A RM_PID DET_PID
declare -A RM_STARTED_AT DET_STARTED_AT
declare -A RM_FAILURES DET_FAILURES
declare -A RM_RESTARTS DET_RESTARTS

port_is_in_use() {
    local port="$1"
    "${PORT_CHECK_PYTHON}" -c \
        'import socket, sys; s = socket.socket(); s.settimeout(1); sys.exit(0 if s.connect_ex(("127.0.0.1", int(sys.argv[1]))) == 0 else 1)' \
        "${port}"
}

preflight_ports() {
    local gpu rm_port det_port conflict=0

    if [[ "${SKIP_PORT_CHECK}" == "1" ]]; then
        log "WARNING: initial port conflict check is disabled"
        return 0
    fi

    for gpu in "${GPUS[@]}"; do
        rm_port=$((RM_PORT_BASE + gpu))
        if port_is_in_use "${rm_port}"; then
            log "ERROR: GPU ${gpu} RM port ${rm_port} is already in use"
            conflict=1
        fi
    done
    for gpu in "${DETECTOR_GPUS[@]}"; do
        det_port=$((DET_PORT_BASE + gpu))
        if port_is_in_use "${det_port}"; then
            log "ERROR: GPU ${gpu} detector port ${det_port} is already in use"
            conflict=1
        fi
    done

    if ((conflict)); then
        log "Stop the existing services or choose different port bases before retrying."
        return 1
    fi
}

start_service() {
    local kind="$1"
    local gpu="$2"
    local port pid log_file

    if [[ "${kind}" == "rm" ]]; then
        port=$((RM_PORT_BASE + gpu))
        log_file="${LOG_DIR}/gpu${gpu}_rm.log"
        if [[ "${gpu}" == "${HIGH_CAPACITY_GPU}" ]]; then
            setsid env GPU_ID="${gpu}" PORT="${port}" \
                MEM_FRACTION_STATIC="${HIGH_CAPACITY_MEM_FRACTION_STATIC}" \
                MAX_RUNNING_REQUESTS="${HIGH_CAPACITY_MAX_RUNNING_REQUESTS}" \
                MAX_TOTAL_TOKENS="${HIGH_CAPACITY_MAX_TOTAL_TOKENS}" \
                CHUNKED_PREFILL_SIZE="${HIGH_CAPACITY_CHUNKED_PREFILL_SIZE}" \
                MAX_PREFILL_TOKENS="${HIGH_CAPACITY_MAX_PREFILL_TOKENS}" \
                CUDA_GRAPH_MAX_BS="${HIGH_CAPACITY_CUDA_GRAPH_MAX_BS}" \
                MAX_MAMBA_CACHE_SIZE="${HIGH_CAPACITY_MAX_MAMBA_CACHE_SIZE}" \
                bash "${RM_SCRIPT}" >>"${log_file}" 2>&1 </dev/null &
        else
            setsid env GPU_ID="${gpu}" PORT="${port}" \
                bash "${RM_SCRIPT}" >>"${log_file}" 2>&1 </dev/null &
        fi
        pid=$!
        RM_PID["${gpu}"]="${pid}"
        RM_STARTED_AT["${gpu}"]="$(date +%s)"
        RM_FAILURES["${gpu}"]=0
    else
        port=$((DET_PORT_BASE + gpu))
        log_file="${LOG_DIR}/gpu${gpu}_detector.log"
        setsid env GPU_ID="${gpu}" PORT="${port}" \
            bash "${DET_SCRIPT}" >>"${log_file}" 2>&1 </dev/null &
        pid=$!
        DET_PID["${gpu}"]="${pid}"
        DET_STARTED_AT["${gpu}"]="$(date +%s)"
        DET_FAILURES["${gpu}"]=0
    fi

    log "Started ${kind} on GPU ${gpu}, port ${port}, pid ${pid}, log ${log_file}"
}

stop_service() {
    local kind="$1"
    local gpu="$2"
    local pid attempt

    if [[ "${kind}" == "rm" ]]; then
        pid="${RM_PID[${gpu}]:-}"
    else
        pid="${DET_PID[${gpu}]:-}"
    fi

    [[ -n "${pid}" ]] || return 0

    kill -TERM -- "-${pid}" 2>/dev/null || true
    for ((attempt = 0; attempt < TERM_GRACE_SECONDS; attempt++)); do
        if ! kill -0 -- "-${pid}" 2>/dev/null; then
            break
        fi
        sleep 1
    done
    if kill -0 -- "-${pid}" 2>/dev/null; then
        log "Force killing ${kind} process group ${pid} on GPU ${gpu}"
        kill -KILL -- "-${pid}" 2>/dev/null || true
    fi
    wait "${pid}" 2>/dev/null || true
}

health_ok() {
    local kind="$1"
    local gpu="$2"
    local port response

    if [[ "${kind}" == "rm" ]]; then
        port=$((RM_PORT_BASE + gpu))
        response="$(curl -fsS --max-time "${HEALTH_TIMEOUT_SECONDS}" \
            "http://127.0.0.1:${port}/v1/models" 2>/dev/null)" || return 1
        grep -Fq "${EXPECTED_MODEL}" <<< "${response}"
    else
        port=$((DET_PORT_BASE + gpu))
        response="$(curl -fsS --max-time "${HEALTH_TIMEOUT_SECONDS}" \
            "http://127.0.0.1:${port}/health" 2>/dev/null)" || return 1
        grep -Eq '"model_loaded"[[:space:]]*:[[:space:]]*true' <<< "${response}"
    fi
}

restart_service() {
    local kind="$1"
    local gpu="$2"
    local restarts delay

    stop_service "${kind}" "${gpu}"

    if [[ "${kind}" == "rm" ]]; then
        restarts=$((${RM_RESTARTS[${gpu}]:-0} + 1))
        RM_RESTARTS["${gpu}"]="${restarts}"
    else
        restarts=$((${DET_RESTARTS[${gpu}]:-0} + 1))
        DET_RESTARTS["${gpu}"]="${restarts}"
    fi

    delay=$((RESTART_DELAY_SECONDS * restarts))
    if ((delay > MAX_RESTART_DELAY_SECONDS)); then
        delay="${MAX_RESTART_DELAY_SECONDS}"
    fi
    log "Reloading ${kind} on GPU ${gpu} after ${delay}s (restart ${restarts})"
    sleep "${delay}"
    start_service "${kind}" "${gpu}"
}

check_service() {
    local kind="$1"
    local gpu="$2"
    local pid started_at now age failures threshold

    if [[ "${kind}" == "rm" ]]; then
        pid="${RM_PID[${gpu}]:-}"
        started_at="${RM_STARTED_AT[${gpu}]:-0}"
        failures="${RM_FAILURES[${gpu}]:-0}"
        threshold="${RM_HEALTH_FAILURE_THRESHOLD}"
    else
        pid="${DET_PID[${gpu}]:-}"
        started_at="${DET_STARTED_AT[${gpu}]:-0}"
        failures="${DET_FAILURES[${gpu}]:-0}"
        threshold="${DET_HEALTH_FAILURE_THRESHOLD}"
    fi

    if [[ -z "${pid}" ]] || ! kill -0 "${pid}" 2>/dev/null; then
        [[ -z "${pid}" ]] || wait "${pid}" 2>/dev/null || true
        log "Detected exited ${kind} process on GPU ${gpu}"
        restart_service "${kind}" "${gpu}"
        return
    fi

    now="$(date +%s)"
    age=$((now - started_at))
    if ((age < STARTUP_GRACE_SECONDS)); then
        return
    fi

    if health_ok "${kind}" "${gpu}"; then
        if ((failures > 0)); then
            log "${kind} on GPU ${gpu} recovered after ${failures} failed health checks"
        fi
        if [[ "${kind}" == "rm" ]]; then
            RM_FAILURES["${gpu}"]=0
            RM_RESTARTS["${gpu}"]=0
        else
            DET_FAILURES["${gpu}"]=0
            DET_RESTARTS["${gpu}"]=0
        fi
        return
    fi

    failures=$((failures + 1))
    if [[ "${kind}" == "rm" ]]; then
        RM_FAILURES["${gpu}"]="${failures}"
    else
        DET_FAILURES["${gpu}"]="${failures}"
    fi
    log "Health check failed for ${kind} on GPU ${gpu} (${failures}/${threshold})"

    if ((failures >= threshold)); then
        log "Detected unhealthy ${kind} on GPU ${gpu}"
        restart_service "${kind}" "${gpu}"
    fi
}

shutdown() {
    local exit_code=$?
    local gpu
    trap - INT TERM EXIT
    log "Stopping watchdog and all managed model servers"
    for gpu in "${GPUS[@]}"; do
        stop_service rm "${gpu}"
    done
    for gpu in "${DETECTOR_GPUS[@]}"; do
        stop_service det "${gpu}"
    done
    exit "${exit_code}"
}

trap shutdown INT TERM EXIT

if ! preflight_ports; then
    exit 1
fi

log "Starting SGLang on GPUs: ${GPUS[*]}"
log "Starting detectors on GPUs: ${DETECTOR_GPU_LIST:-none}"
log "GPU ${HIGH_CAPACITY_GPU} high-capacity SGLang profile: requests=${HIGH_CAPACITY_MAX_RUNNING_REQUESTS}, tokens=${HIGH_CAPACITY_MAX_TOTAL_TOKENS}"
for gpu in "${DETECTOR_GPUS[@]}"; do
    DET_RESTARTS["${gpu}"]=0
    start_service det "${gpu}"
done

for gpu in "${GPUS[@]}"; do
    RM_RESTARTS["${gpu}"]=0
    start_service rm "${gpu}"
    sleep "${START_STAGGER_SECONDS}"
done

log "All services launched; entering monitor loop"
while true; do
    for gpu in "${GPUS[@]}"; do
        check_service rm "${gpu}"
    done
    for gpu in "${DETECTOR_GPUS[@]}"; do
        check_service det "${gpu}"
    done
    sleep "${CHECK_INTERVAL_SECONDS}"
done
