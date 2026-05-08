#!/usr/bin/env bash
set -euo pipefail

# Knoxnet VMS Beta - Local LLM service launcher (Linux/macOS)
#
# Supports:
# - foreground mode (default): suitable for systemd or desktop-managed subprocess
# - daemon mode (--daemon): background with pidfile
# - stop/restart: best-effort pidfile-based shutdown
#
# Logs: <install_root>/logs/service_local_llm.log (override with KNOXNET_LOG_DIR)
# Ports: default 8102 (override with LLM_PORT / LLM_HOST)

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
INSTALL_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"

_default_log_dir() {
  local candidate="${INSTALL_ROOT}/logs"
  if mkdir -p "${candidate}" >/dev/null 2>&1 && [[ -w "${candidate}" ]]; then
    echo "${candidate}"
    return 0
  fi
  local state="${XDG_STATE_HOME:-${HOME}/.local/state}"
  echo "${state}/KnoxnetVMS/logs"
}

LOG_DIR="${KNOXNET_LOG_DIR:-$(_default_log_dir)}"
mkdir -p "${LOG_DIR}"

LOG_FILE="${LOG_DIR}/service_local_llm.log"
PID_FILE="${LOG_DIR}/service_local_llm.pid"

_exe_in_root() {
  if [[ -x "${INSTALL_ROOT}/KnoxnetVMS" ]]; then
    echo "${INSTALL_ROOT}/KnoxnetVMS"
    return 0
  fi
  if [[ -x "${INSTALL_ROOT}/KnoxnetVMS.bin" ]]; then
    echo "${INSTALL_ROOT}/KnoxnetVMS.bin"
    return 0
  fi
  return 1
}

_pick_python() {
  if [[ -n "${KNOXNET_SERVICE_PYTHON:-}" ]]; then
    echo "${KNOXNET_SERVICE_PYTHON}"
    return 0
  fi
  if [[ -x "${INSTALL_ROOT}/venv/bin/python" ]]; then
    echo "${INSTALL_ROOT}/venv/bin/python"
    return 0
  fi
  if command -v python3 >/dev/null 2>&1; then
    echo "python3"
    return 0
  fi
  echo "python"
}

_is_running_pid() {
  local pid="$1"
  [[ -n "${pid}" ]] && kill -0 "${pid}" >/dev/null 2>&1
}

stop_service() {
  if [[ ! -f "${PID_FILE}" ]]; then
    echo "Local LLM: not running (no pidfile at ${PID_FILE})"
    return 0
  fi
  local pid
  pid="$(cat "${PID_FILE}" 2>/dev/null || true)"
  if [[ -z "${pid}" ]]; then
    rm -f "${PID_FILE}" || true
    echo "Local LLM: stale pidfile removed"
    return 0
  fi
  if ! _is_running_pid "${pid}"; then
    rm -f "${PID_FILE}" || true
    echo "Local LLM: not running (stale pid ${pid})"
    return 0
  fi
  echo "Stopping Local LLM (pid ${pid})..."
  kill -TERM "${pid}" >/dev/null 2>&1 || true
  for _ in {1..30}; do
    if ! _is_running_pid "${pid}"; then
      rm -f "${PID_FILE}" || true
      echo "Local LLM stopped"
      return 0
    fi
    sleep 0.2
  done
  echo "Local LLM did not stop gracefully; forcing kill..."
  kill -KILL "${pid}" >/dev/null 2>&1 || true
  rm -f "${PID_FILE}" || true
  echo "Local LLM killed"
}

start_foreground() {
  exec >>"${LOG_FILE}" 2>&1

  echo ""
  echo "----------------------------------------"
  echo "Starting Local LLM: $(date -Is)"
  echo "Install root: ${INSTALL_ROOT}"
  echo "Log file: ${LOG_FILE}"

  # Consistent defaults (override via env; matches pydantic env_prefix=LLM_)
  export LLM_HOST="${LLM_HOST:-127.0.0.1}"
  export LLM_PORT="${LLM_PORT:-8102}"

  if exe="$(_exe_in_root)"; then
    echo "Runner: ${exe} --run-llm-local"
    exec "${exe}" --run-llm-local
  fi

  py="$(_pick_python)"
  echo "Runner: ${py} -m services.llm_local"
  exec "${py}" -m services.llm_local
}

start_daemon() {
  if [[ -f "${PID_FILE}" ]]; then
    local old
    old="$(cat "${PID_FILE}" 2>/dev/null || true)"
    if [[ -n "${old}" ]] && _is_running_pid "${old}"; then
      echo "Local LLM already running (pid ${old})"
      return 0
    fi
    rm -f "${PID_FILE}" || true
  fi

  (
    start_foreground
  ) &
  local pid="$!"
  echo "${pid}" >"${PID_FILE}"
  echo "Local LLM started (pid ${pid})"
}

cmd="${1:-start}"
case "${cmd}" in
  start)
    start_foreground
    ;;
  --daemon|daemon)
    start_daemon
    ;;
  stop)
    stop_service
    ;;
  restart)
    stop_service || true
    start_daemon
    ;;
  *)
    echo "Usage: $0 [start|--daemon|stop|restart]"
    exit 2
    ;;
esac


