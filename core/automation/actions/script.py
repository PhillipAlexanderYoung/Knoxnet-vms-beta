from __future__ import annotations

import json
import logging
import os
import shlex
import subprocess
import sys
import threading
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

from core.paths import get_data_dir, get_event_rules_scripts_dir, get_project_root

logger = logging.getLogger(__name__)

DEFAULT_SCRIPT_TIMEOUT_SEC = 30.0
MAX_SCRIPT_TIMEOUT_SEC = 600.0
MAX_OUTPUT_CHARS = 16_384

SUPPORTED_RUNNERS = frozenset({"python", "shell", "executable"})
SCRIPT_ACTION_TYPES = frozenset({"script", "run_script"})


@dataclass
class ScriptRunResult:
    success: bool
    exit_code: int
    stdout: str
    stderr: str
    timed_out: bool
    error: Optional[str] = None
    command: Optional[List[str]] = None


def _env_flag(name: str) -> bool:
    return str(os.environ.get(name, "")).strip().lower() in {"1", "true", "yes", "on"}


def allow_shell_runner() -> bool:
    return _env_flag("KNOXNET_EVENT_RULES_ALLOW_SHELL")


def allow_external_scripts() -> bool:
    return _env_flag("KNOXNET_EVENT_RULES_ALLOW_EXTERNAL_SCRIPTS")


def configured_script_roots() -> List[Path]:
    roots: List[Path] = []
    for root in (get_event_rules_scripts_dir(), get_data_dir() / "event_rules_scripts"):
        try:
            resolved = root.resolve()
        except Exception:
            continue
        if resolved not in roots:
            roots.append(resolved)
    extra = str(os.environ.get("KNOXNET_EVENT_RULES_SCRIPT_DIRS") or "").strip()
    if extra:
        for part in extra.split(os.pathsep):
            part = part.strip()
            if not part:
                continue
            try:
                p = Path(part).expanduser().resolve()
            except Exception:
                continue
            if p not in roots:
                roots.append(p)
    return roots


def ensure_script_roots() -> Path:
    primary = get_event_rules_scripts_dir()
    primary.mkdir(parents=True, exist_ok=True)
    return primary


def normalize_runner(value: Any) -> str:
    runner = str(value or "python").strip().lower()
    if runner in {"py", "python3"}:
        return "python"
    if runner in {"sh", "bash"}:
        return "shell"
    if runner in {"exe", "binary"}:
        return "executable"
    return runner if runner in SUPPORTED_RUNNERS else "python"


def normalize_args(value: Any) -> List[str]:
    if value is None:
        return []
    if isinstance(value, list):
        return [str(v) for v in value]
    text = str(value).strip()
    if not text:
        return []
    try:
        return shlex.split(text)
    except ValueError:
        return text.split()


def normalize_timeout(value: Any, *, default: float = DEFAULT_SCRIPT_TIMEOUT_SEC) -> float:
    try:
        timeout = float(value)
    except Exception:
        timeout = default
    if timeout <= 0:
        timeout = default
    return min(float(timeout), MAX_SCRIPT_TIMEOUT_SEC)


def resolve_script_path(raw_path: str, *, roots: Optional[Sequence[Path]] = None) -> Tuple[Optional[Path], Optional[str]]:
    text = str(raw_path or "").strip()
    if not text:
        return None, "missing_path"

    candidate = Path(text).expanduser()
    try:
        if candidate.is_absolute():
            resolved = candidate.resolve()
            if allow_external_scripts():
                if not resolved.is_file():
                    return None, "not_found"
                return resolved, None
            for root in roots or configured_script_roots():
                root_resolved = root.resolve()
                try:
                    resolved.relative_to(root_resolved)
                except ValueError:
                    continue
                if resolved.is_file():
                    return resolved, None
            return None, "outside_allowed_roots"

        for root in roots or configured_script_roots():
            resolved = (root / candidate).resolve()
            try:
                resolved.relative_to(root.resolve())
            except ValueError:
                continue
            if resolved.is_file():
                return resolved, None
        return None, "not_found"
    except Exception as exc:
        return None, f"invalid_path:{exc}"


def build_script_command(
    *,
    runner: str,
    script_path: Path,
    args: Optional[Sequence[str]] = None,
    python_executable: Optional[str] = None,
) -> Tuple[Optional[List[str]], Optional[str]]:
    runner_norm = normalize_runner(runner)
    extra = list(args or [])

    if runner_norm == "python":
        py = python_executable or sys.executable
        return [py, str(script_path), *extra], None

    if runner_norm == "shell":
        if not allow_shell_runner():
            return None, "shell_disabled"
        suffix = script_path.suffix.lower()
        if suffix not in {".sh", ".bash"}:
            return None, "shell_requires_sh_extension"
        shell_bin = "/bin/bash" if Path("/bin/bash").exists() else "/bin/sh"
        return [shell_bin, str(script_path), *extra], None

    if runner_norm == "executable":
        if not os.access(script_path, os.X_OK):
            return None, "not_executable"
        return [str(script_path), *extra], None

    return None, "unsupported_runner"


def build_event_context_payload(
    *,
    rule: Dict[str, Any],
    details: Dict[str, Any],
    event: Any,
    ctx: Any,
) -> Dict[str, Any]:
    payload: Dict[str, Any] = {}
    if hasattr(ctx, "payload") and isinstance(getattr(ctx, "payload"), dict):
        payload.update(ctx.payload)
    evt_payload = getattr(event, "payload", None)
    if isinstance(evt_payload, dict):
        payload.update(evt_payload)

    return {
        "camera_id": getattr(event, "camera_id", None) or rule.get("camera_id"),
        "rule_id": rule.get("id"),
        "rule_name": rule.get("name"),
        "trigger": rule.get("trigger"),
        "shape_id": rule.get("shape_id") or details.get("shape_id"),
        "event_type": details.get("event_type") or payload.get("event_type"),
        "details": details,
        "payload": payload,
    }


def run_script_subprocess(
    command: Sequence[str],
    *,
    env: Dict[str, str],
    timeout_sec: float,
    cwd: Optional[Path] = None,
    input_json: Optional[str] = None,
) -> ScriptRunResult:
    try:
        completed = subprocess.run(
            list(command),
            input=input_json,
            capture_output=True,
            text=True,
            timeout=timeout_sec,
            env=env,
            cwd=str(cwd) if cwd else None,
            shell=False,
        )
        stdout = (completed.stdout or "")[:MAX_OUTPUT_CHARS]
        stderr = (completed.stderr or "")[:MAX_OUTPUT_CHARS]
        return ScriptRunResult(
            success=completed.returncode == 0,
            exit_code=int(completed.returncode),
            stdout=stdout,
            stderr=stderr,
            timed_out=False,
            command=list(command),
        )
    except subprocess.TimeoutExpired as exc:
        stdout = (exc.stdout or "")[:MAX_OUTPUT_CHARS] if isinstance(exc.stdout, str) else ""
        stderr = (exc.stderr or "")[:MAX_OUTPUT_CHARS] if isinstance(exc.stderr, str) else ""
        stderr = (stderr + "\nScript timed out.").strip()
        return ScriptRunResult(
            success=False,
            exit_code=-1,
            stdout=stdout,
            stderr=stderr,
            timed_out=True,
            error="timeout",
            command=list(command),
        )
    except Exception as exc:
        return ScriptRunResult(
            success=False,
            exit_code=-1,
            stdout="",
            stderr=str(exc),
            timed_out=False,
            error=str(exc),
            command=list(command),
        )


class ScriptAction:
    """Server-side script runner for Event Rules."""

    def __init__(
        self,
        *,
        max_workers: int = 4,
        script_roots: Optional[Sequence[Path]] = None,
        python_executable: Optional[str] = None,
    ) -> None:
        ensure_script_roots()
        self._roots = list(script_roots) if script_roots is not None else configured_script_roots()
        self._python_executable = python_executable or sys.executable
        self._executor = ThreadPoolExecutor(
            max_workers=max(1, int(max_workers)),
            thread_name_prefix="event-rule-script",
        )
        self._lock = threading.Lock()

    def shutdown(self) -> None:
        self._executor.shutdown(wait=False, cancel_futures=True)

    def handler(self) -> Callable[..., None]:
        def _handler(*, rule: Dict[str, Any], ctx: Any, details: Dict[str, Any], action: Dict[str, Any], event: Any) -> None:
            self.submit(rule=rule, ctx=ctx, details=details, action=action, event=event)

        return _handler

    def submit(
        self,
        *,
        rule: Dict[str, Any],
        ctx: Any,
        details: Dict[str, Any],
        action: Dict[str, Any],
        event: Any,
    ) -> None:
        self._executor.submit(self.execute, rule=rule, ctx=ctx, details=details, action=action, event=event)

    def execute(
        self,
        *,
        rule: Dict[str, Any],
        ctx: Any,
        details: Dict[str, Any],
        action: Dict[str, Any],
        event: Any,
    ) -> ScriptRunResult:
        action_type = str(action.get("type") or "").strip().lower()
        if action_type not in SCRIPT_ACTION_TYPES:
            return ScriptRunResult(
                success=False,
                exit_code=-1,
                stdout="",
                stderr="unsupported_action_type",
                timed_out=False,
                error="unsupported_action_type",
            )

        raw_path = action.get("path") or action.get("command") or action.get("script")
        runner = normalize_runner(action.get("language") or action.get("runner") or "python")
        args = normalize_args(action.get("args"))
        timeout_sec = normalize_timeout(action.get("timeout_sec", action.get("timeout")))

        script_path, path_error = resolve_script_path(str(raw_path or ""), roots=self._roots)
        if path_error or script_path is None:
            logger.warning(
                "ScriptAction skipped rule=%s path=%r error=%s",
                rule.get("id"),
                raw_path,
                path_error,
            )
            return ScriptRunResult(
                success=False,
                exit_code=-1,
                stdout="",
                stderr=path_error or "invalid_path",
                timed_out=False,
                error=path_error or "invalid_path",
            )

        command, cmd_error = build_script_command(
            runner=runner,
            script_path=script_path,
            args=args,
            python_executable=self._python_executable,
        )
        if cmd_error or not command:
            logger.warning(
                "ScriptAction command build failed rule=%s runner=%s error=%s",
                rule.get("id"),
                runner,
                cmd_error,
            )
            return ScriptRunResult(
                success=False,
                exit_code=-1,
                stdout="",
                stderr=cmd_error or "command_build_failed",
                timed_out=False,
                error=cmd_error or "command_build_failed",
            )

        context_payload = build_event_context_payload(rule=rule, details=details, event=event, ctx=ctx)
        context_json = json.dumps(context_payload, ensure_ascii=False, default=str)
        env = os.environ.copy()
        env["PYTHONUNBUFFERED"] = "1"
        env["KNOXNET_EVENT_JSON"] = context_json
        env["OPEN_SENTRY_EVENT_JSON"] = context_json
        env["KNOXNET_RULE_ID"] = str(rule.get("id") or "")
        env["KNOXNET_CAMERA_ID"] = str(getattr(event, "camera_id", None) or rule.get("camera_id") or "")

        result = run_script_subprocess(
            command,
            env=env,
            timeout_sec=timeout_sec,
            cwd=script_path.parent,
            input_json=context_json,
        )
        if result.success:
            logger.info(
                "ScriptAction completed rule=%s script=%s exit=%s",
                rule.get("id"),
                script_path.name,
                result.exit_code,
            )
        else:
            logger.warning(
                "ScriptAction failed rule=%s script=%s exit=%s timed_out=%s stderr=%s",
                rule.get("id"),
                script_path.name,
                result.exit_code,
                result.timed_out,
                (result.stderr or "")[:500],
            )
        return result
