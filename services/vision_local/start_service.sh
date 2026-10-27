#!/usr/bin/env bash
set -euo pipefail

# Knoxnet VMS Beta - Local Vision service launcher (Linux/macOS)
#
# Supports:
# - foreground mode (default): suitable for systemd or desktop-managed subprocess
# - daemon mode (--daemon): background with pidfile
# - stop/restart: best-effort pidfile-based shutdown
#
# Logs: <install_root>/logs/service_local_vision.log (override with KNOXNET_LOG_DIR)
# Ports: default 8101 (override with LOCAL_VISION_PORT)

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
INSTALL_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"

_default_log_dir() {
  # Prefer install-local logs if writable, else fall back to a per-user state dir.
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

LOG_FILE="${LOG_DIR}/service_local_vision.log"
PID_FILE="${LOG_DIR}/service_local_vision.pid"

_exe_in_root() {
  # Prefer running the frozen app executable in production installs.
  # On Linux/macOS PyInstaller onedir typically places the main binary at <root>/KnoxnetVMS.
  if [[ -x "${INSTALL_ROOT}/KnoxnetVMS" ]]; then
    echo "${INSTALL_ROOT}/KnoxnetVMS"
    return 0
  fi
  # Fallback: any other executable in root (rare)
  if [[ -x "${INSTALL_ROOT}/KnoxnetVMS.bin" ]]; then
    echo "${INSTALL_ROOT}/KnoxnetVMS.bin"
    return 0
  fi
  return 1
}

_pick_python() {
  # Allow explicit override (useful for internal venv installs)
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
    echo "Local Vision: not running (no pidfile at ${PID_FILE})"
    return 0
  fi
  local pid
  pid="$(cat "${PID_FILE}" 2>/dev/null || true)"
  if [[ -z "${pid}" ]]; then
    rm -f "${PID_FILE}" || true
    echo "Local Vision: stale pidfile removed"
    return 0
  fi
  if ! _is_running_pid "${pid}"; then
    rm -f "${PID_FILE}" || true
    echo "Local Vision: not running (stale pid ${pid})"
    return 0
  fi
  echo "Stopping Local Vision (pid ${pid})..."
  kill -TERM "${pid}" >/dev/null 2>&1 || true
  for _ in {1..30}; do
    if ! _is_running_pid "${pid}"; then
      rm -f "${PID_FILE}" || true
      echo "Local Vision stopped"
      return 0
    fi
    sleep 0.2
  done
  echo "Local Vision did not stop gracefully; forcing kill..."
  kill -KILL "${pid}" >/dev/null 2>&1 || true
  rm -f "${PID_FILE}" || true
  echo "Local Vision killed"
}

start_foreground() {
  # Redirect all output to the service log (install-local).
  exec >>"${LOG_FILE}" 2>&1

  echo ""
  echo "----------------------------------------"
  echo "Starting Local Vision: $(date -Is)"
  echo "Install root: ${INSTALL_ROOT}"
  echo "Log file: ${LOG_FILE}"

  # Consistent defaults (can be overridden via environment)
  export LOCAL_VISION_HOST="${LOCAL_VISION_HOST:-0.0.0.0}"
  export LOCAL_VISION_PORT="${LOCAL_VISION_PORT:-8101}"
  export VISION_LOCAL_MODE="${VISION_LOCAL_MODE:-production}"

  # Run via frozen executable when available; otherwise run via python module.
  if exe="$(_exe_in_root)"; then
    echo "Runner: ${exe} --run-vision-local"
    exec "${exe}" --run-vision-local
  fi

  py="$(_pick_python)"
  echo "Runner: ${py} -m services.vision_local"
  exec "${py}" -m services.vision_local
}

start_daemon() {
  # Background the process and manage a pidfile.
  if [[ -f "${PID_FILE}" ]]; then
    local old
    old="$(cat "${PID_FILE}" 2>/dev/null || true)"
    if [[ -n "${old}" ]] && _is_running_pid "${old}"; then
      echo "Local Vision already running (pid ${old})"
      return 0
    fi
    rm -f "${PID_FILE}" || true
  fi

  # Ensure the child inherits our env + logging.
  (
    start_foreground
  ) &
  local pid="$!"
  echo "${pid}" >"${PID_FILE}"
  echo "Local Vision started (pid ${pid})"
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


