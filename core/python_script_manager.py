import json
import logging
import os
import re
import sys
import threading
import uuid
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, List, Optional, Tuple

import subprocess
import time

logger = logging.getLogger(__name__)


@dataclass
class ScriptExecutionResult:
    script_id: str
    started_at: str
    completed_at: str
    duration_ms: float
    exit_code: int
    stdout: str
    stderr: str
    success: bool
    timeout: bool
    trigger: Dict[str, Any]


DEFAULT_TEMPLATE = """#!/usr/bin/env python3
\"\"\"Knoxnet VMS Beta automation script.\"\"\"

from __future__ import annotations

import json
import os
from datetime import datetime


def main() -> int:
    # Event payload is provided through the OPEN_SENTRY_EVENT_JSON environment variable.
    raw_event = os.environ.get("OPEN_SENTRY_EVENT_JSON", "{}")
    event = json.loads(raw_event or "{}")

    camera_ref = event.get("camera_id") or os.environ.get("OPEN_SENTRY_CAMERA_ID", "unknown")
    event_type = event.get("type") or os.environ.get("OPEN_SENTRY_EVENT_TYPE", "manual")

    print(f"[{datetime.utcnow().isoformat()}] ▶ script started for camera={camera_ref} event={event_type}")
    print(f"Incoming payload keys: {sorted(event.keys())}")

    # TODO: add your automation logic here

    print(f"[{datetime.utcnow().isoformat()}] ✅ script completed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
"""


def _utcnow() -> str:
    return datetime.utcnow().replace(microsecond=0).isoformat() + "Z"


class PythonScriptManager:
    """Directory-backed manager for user automation scripts."""

    def __init__(
        self,
        base_path: Path | str,
        camera_resolver: Optional[Callable[[str], Optional[Dict[str, Any]]]] = None,
        max_workers: int = 4,
    ) -> None:
        self.base_path = Path(base_path).resolve()
        self.base_path.mkdir(parents=True, exist_ok=True)
        self.metadata_path = self.base_path / "metadata.json"
        self._lock = threading.RLock()
        self._camera_resolver = camera_resolver
        self._executor = ThreadPoolExecutor(max_workers=max(1, max_workers), thread_name_prefix="python-script")
        self._scripts: Dict[str, Dict[str, Any]] = {}
        self._load_metadata()

    # ------------------------------------------------------------------ Persistence helpers
    def _load_metadata(self) -> None:
        with self._lock:
            if not self.metadata_path.exists():
                self.metadata_path.write_text("{}", encoding="utf-8")
                self._scripts = {}
                return
            try:
                data = json.loads(self.metadata_path.read_text(encoding="utf-8") or "{}")
                if isinstance(data, dict):
                    self._scripts = data
                else:
                    logger.warning("Metadata file corrupted, resetting python script registry")
                    self._scripts = {}
            except Exception as exc:
                logger.error("Failed to load python script metadata: %s", exc)
                self._scripts = {}

    def _save_metadata(self) -> None:
        with self._lock:
            tmp_path = self.metadata_path.with_suffix(".tmp")
            tmp_path.write_text(json.dumps(self._scripts, indent=2, sort_keys=True), encoding="utf-8")
            tmp_path.replace(self.metadata_path)

    def shutdown(self) -> None:
        self._executor.shutdown(wait=False, cancel_futures=True)

    # ------------------------------------------------------------------ Public API
    def list_scripts(self) -> List[Dict[str, Any]]:
        with self._lock:
            scripts = [self._serialize_script(meta) for meta in self._scripts.values()]
        scripts.sort(key=lambda item: item["name"].lower())
        return scripts

    def get_script(self, script_id: str, include_code: bool = False) -> Optional[Dict[str, Any]]:
        with self._lock:
            meta = self._scripts.get(script_id)
            if not meta:
                return None
            return self._serialize_script(meta, include_code=include_code)

    def create_script(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        script_id = payload.get("id") or f"script-{uuid.uuid4().hex}"
        name = (payload.get("name") or f"Script {datetime.utcnow().strftime('%H:%M:%S')}").strip()
        description = (payload.get("description") or "").strip()
        tags = self._normalize_list(payload.get("tags"))
        bindings = self._normalize_bindings(payload.get("bindings") or {})
        code = payload.get("code") or DEFAULT_TEMPLATE

        code_filename = f"{script_id}.py"
        created_at = _utcnow()

        metadata = {
            "id": script_id,
            "name": name or script_id,
            "description": description,
            "tags": tags,
            "bindings": bindings,
            "code_path": code_filename,
            "created_at": created_at,
            "updated_at": created_at,
            "last_run_at": None,
            "last_run_status": None,
            "last_run_duration_ms": None,
            "last_run_error": None,
            "last_run_summary": None,
            "version": 1,
            "ai_generated": bool(payload.get("ai_generated")),
        }

        with self._lock:
            if script_id in self._scripts:
                raise ValueError(f"Script with id '{script_id}' already exists")
            (self.base_path / code_filename).write_text(code, encoding="utf-8", newline="\n")
            self._scripts[script_id] = metadata
            self._save_metadata()

        logger.info("Created python automation script %s", script_id)
        return self._serialize_script(metadata, include_code=True)

    def update_script(self, script_id: str, payload: Dict[str, Any]) -> Dict[str, Any]:
        with self._lock:
            meta = self._scripts.get(script_id)
            if not meta:
                raise KeyError(f"Script '{script_id}' not found")

            if "name" in payload:
                meta["name"] = str(payload["name"]).strip() or meta["name"]
            if "description" in payload:
                meta["description"] = str(payload["description"]).strip()
            if "tags" in payload:
                meta["tags"] = self._normalize_list(payload.get("tags"))
            if "bindings" in payload:
                meta["bindings"] = self._normalize_bindings(payload.get("bindings") or {})
            if payload.get("ai_generated") is not None:
                meta["ai_generated"] = bool(payload.get("ai_generated"))
            meta["updated_at"] = _utcnow()
            meta["version"] = int(meta.get("version", 1)) + 1

            if "code" in payload and payload["code"] is not None:
                code_path = self.base_path / meta["code_path"]
                code_path.write_text(str(payload["code"]), encoding="utf-8", newline="\n")

            self._save_metadata()
            result = self._serialize_script(meta, include_code=payload.get("include_code", False))

        logger.info("Updated python automation script %s", script_id)
        return result

    def delete_script(self, script_id: str) -> bool:
        with self._lock:
            meta = self._scripts.pop(script_id, None)
            if not meta:
                return False
            try:
                (self.base_path / meta["code_path"]).unlink(missing_ok=True)
            except Exception as exc:
                logger.warning("Failed to delete script file for %s: %s", script_id, exc)
            self._save_metadata()
        logger.info("Deleted python automation script %s", script_id)
        return True

    def run_script(
        self,
        script_id: str,
        *,
        event_type: str = "manual",
        camera_id: Optional[str] = None,
        payload: Optional[Dict[str, Any]] = None,
        timeout: float = 20.0,
        capture_output: bool = True,
    ) -> ScriptExecutionResult:
        with self._lock:
            meta = self._scripts.get(script_id)
            if not meta:
                raise KeyError(f"Script '{script_id}' not found")
            script_path = self.base_path / meta["code_path"]
            if not script_path.exists():
                raise FileNotFoundError(f"Script file missing for '{script_id}'")

        payload = payload or {}
        event_payload = dict(payload)
        event_payload.setdefault("type", event_type)
        if camera_id:
            event_payload.setdefault("camera_id", camera_id)

        env = os.environ.copy()
        env["PYTHONUNBUFFERED"] = "1"
        env["OPEN_SENTRY_SCRIPT_ID"] = script_id
        env["OPEN_SENTRY_EVENT_TYPE"] = event_type
        env["OPEN_SENTRY_EVENT_JSON"] = json.dumps(event_payload, ensure_ascii=False)
        if camera_id:
            env["OPEN_SENTRY_CAMERA_ID"] = camera_id

        trigger_info = {
            "type": event_type,
            "camera_id": camera_id,
            "payload": event_payload,
        }

        result = self._execute_script(
            script_id,
            script_path,
            env,
            trigger_info,
            timeout,
            capture_output,
            allow_dependency_install=True,
        )

        self._record_execution(meta, result)
        return result


    def _execute_script(
        self,
        script_id: str,
        script_path: Path,
        env: Dict[str, Any],
        trigger_info: Dict[str, Any],
        timeout: float,
        capture_output: bool,
        allow_dependency_install: bool,
        _retrying: bool = False,
    ) -> ScriptExecutionResult:
        started_monotonic = time.perf_counter()
        started_at = _utcnow()
        try:
            completed = subprocess.run(
                [sys.executable, str(script_path)],
                input=None,
                capture_output=capture_output,
                text=True,
                timeout=timeout,
                env=env,
                cwd=str(self.base_path),
            )
            success = completed.returncode == 0
            stdout = completed.stdout or ""
            stderr = completed.stderr or ""
            exit_code = completed.returncode
            timed_out = False
        except subprocess.TimeoutExpired as exc:
            success = False
            stdout = exc.stdout or ""
            stderr = (exc.stderr or "") + "\nScript timed out."
            exit_code = -1
            timed_out = True
            logger.warning("Python automation script %s timed out after %.2fs", script_id, timeout)

        duration_ms = (time.perf_counter() - started_monotonic) * 1000.0
        completed_at = _utcnow()

        stdout_trimmed = self._trim_output(stdout)
        stderr_trimmed = self._trim_output(stderr)

        result = ScriptExecutionResult(
            script_id=script_id,
            started_at=started_at,
            completed_at=completed_at,
            duration_ms=duration_ms,
            exit_code=exit_code,
            stdout=stdout_trimmed,
            stderr=stderr_trimmed,
            success=success,
            timeout=timed_out,
            trigger=trigger_info,
        )

        if (
            allow_dependency_install
            and not success
            and not timed_out
            and not _retrying
        ):
            missing_modules = self._extract_missing_modules(stderr_trimmed)
            if missing_modules:
                install_messages = []
                install_failed = False
                for module_name in missing_modules:
                    installed, install_stdout, install_stderr = self._install_module(module_name, env)
                    if installed:
                        message = (
                            f"[deps] Installed missing module '{module_name}' via pip."
                        )
                        if install_stdout:
                            message += f"\n{install_stdout.strip()}"
                        if install_stderr:
                            message += f"\n{install_stderr.strip()}"
                        install_messages.append(message)
                    else:
                        install_failed = True
                        failure_msg = f"[deps] Failed to install module '{module_name}'."
                        if install_stderr:
                            failure_msg += f"\n{install_stderr.strip()}"
                        install_messages.append(failure_msg)
                if install_messages and not install_failed:
                    rerun = self._execute_script(
                        script_id,
                        script_path,
                        env,
                        trigger_info,
                        timeout,
                        capture_output,
                        allow_dependency_install=False,
                        _retrying=True,
                    )
                    merged_stdout = "\n\n".join(install_messages)
                    if rerun.stdout:
                        merged_stdout = f"{merged_stdout}\n\n{rerun.stdout}"
                    rerun = ScriptExecutionResult(
                        script_id=rerun.script_id,
                        started_at=result.started_at,
                        completed_at=rerun.completed_at,
                        duration_ms=result.duration_ms + rerun.duration_ms,
                        exit_code=rerun.exit_code,
                        stdout=self._trim_output(merged_stdout),
                        stderr=rerun.stderr,
                        success=rerun.success,
                        timeout=rerun.timeout,
                        trigger=rerun.trigger,
                    )
                    return rerun
                elif install_messages:
                    combined = "\n\n".join(install_messages)
                    combined_stdout = f"{combined}\n\n{result.stdout}".strip()
                    result = ScriptExecutionResult(
                        script_id=result.script_id,
                        started_at=result.started_at,
                        completed_at=result.completed_at,
                        duration_ms=result.duration_ms,
                        exit_code=result.exit_code,
                        stdout=self._trim_output(combined_stdout),
                        stderr=result.stderr,
                        success=result.success,
                        timeout=result.timeout,
                        trigger=result.trigger,
                    )

        return result

    @staticmethod
    def _extract_missing_modules(stderr: str) -> List[str]:
        if not stderr:
            return []
        modules: List[str] = []
        patterns = [
            r"ModuleNotFoundError: No module named ['\"]([^'\"\n]+)['\"]",
            r"No module named ['\"]([^'\"\n]+)['\"]",
        ]
        for pattern in patterns:
            for match in re.finditer(pattern, stderr):
                module = match.group(1).strip()
                if module and module not in modules:
                    modules.append(module)
        return modules

    def _install_module(
        self,
        module: str,
        env: Dict[str, Any],
    ) -> Tuple[bool, str, str]:
        try:
            logger.info("Attempting to install missing module '%s' for python scripts", module)
            completed = subprocess.run(
                [sys.executable, "-m", "pip", "install", module],
                capture_output=True,
                text=True,
                timeout=120,
                env=env,
                cwd=str(self.base_path),
            )
            success = completed.returncode == 0
            if success:
                logger.info("Module '%s' installed successfully", module)
            else:
                logger.error("Failed to install module '%s': %s", module, completed.stderr.strip())
            return success, completed.stdout or "", completed.stderr or ""
        except subprocess.TimeoutExpired:
            logger.error("Timed out installing module '%s'", module)
            return False, "", "pip install timed out"
    # ------------------------------------------------------------------ Event integration
    def handle_event(
        self,
        event_type: str,
        camera_id: Optional[str],
        payload: Optional[Dict[str, Any]] = None,
    ) -> None:
        payload = payload or {}
        with self._lock:
            scripts = [meta for meta in self._scripts.values() if self._should_trigger(meta, event_type, camera_id, payload)]

        if not scripts:
            return

        logger.debug(
            "Dispatching %d python scripts for event=%s camera=%s",
            len(scripts),
            event_type,
            camera_id,
        )

        for meta in scripts:
            script_id = meta["id"]

            def _task(mid: str = script_id, m_payload: Dict[str, Any] = payload) -> None:
                try:
                    self.run_script(
                        mid,
                        event_type=event_type,
                        camera_id=camera_id,
                        payload=m_payload,
                        timeout=float(meta.get("bindings", {}).get("timeout_seconds", 20.0)),
                    )
                except Exception as exc:
                    logger.error("Python automation script %s failed for event '%s': %s", mid, event_type, exc)

            self._executor.submit(_task)

    # ------------------------------------------------------------------ Helpers
    def _serialize_script(self, meta: Dict[str, Any], *, include_code: bool = False) -> Dict[str, Any]:
        result = dict(meta)
        if include_code:
            try:
                result["code"] = (self.base_path / meta["code_path"]).read_text(encoding="utf-8")
            except Exception as exc:
                logger.error("Failed to read code for script %s: %s", meta["id"], exc)
                result["code"] = ""
        else:
            try:
                code_text = (self.base_path / meta["code_path"]).read_text(encoding="utf-8")
                result["code_preview"] = code_text.splitlines()[:20]
            except Exception:
                result["code_preview"] = []
        return result

    def _record_execution(self, meta: Dict[str, Any], result: ScriptExecutionResult) -> None:
        with self._lock:
            meta["last_run_at"] = result.completed_at
            meta["last_run_status"] = "success" if result.success else "error"
            meta["last_run_duration_ms"] = result.duration_ms
            meta["last_run_error"] = result.stderr[:2000] if not result.success else None
            meta["last_run_summary"] = result.stdout[:2000]
            self._save_metadata()

    def _normalize_list(self, value: Any) -> List[str]:
        if not value:
            return []
        if isinstance(value, str):
            return [value.strip()] if value.strip() else []
        result: List[str] = []
        for item in value if isinstance(value, Iterable) else [value]:
            text = str(item).strip()
            if text and text not in result:
                result.append(text)
        return result

    def _normalize_bindings(self, bindings: Dict[str, Any]) -> Dict[str, Any]:
        normalized = {
            "auto_run": bool(bindings.get("auto_run")),
            "any_camera": bool(bindings.get("any_camera")),
            "events": self._normalize_list(bindings.get("events")),
            "cameras": self._normalize_list(bindings.get("cameras")),
            "camera_names": self._normalize_list(bindings.get("camera_names")),
            "rules": self._normalize_list(bindings.get("rules")),
            "timeout_seconds": float(bindings.get("timeout_seconds", 20.0)),
        }
        return normalized

    def _should_trigger(
        self,
        meta: Dict[str, Any],
        event_type: str,
        camera_id: Optional[str],
        payload: Dict[str, Any],
    ) -> bool:
        bindings = meta.get("bindings") or {}
        if not bindings.get("auto_run"):
            return False

        if not self._event_matches(bindings.get("events"), event_type, payload):
            return False

        if bindings.get("any_camera"):
            return True

        cameras = bindings.get("cameras") or []
        camera_names = [name.lower() for name in bindings.get("camera_names") or []]

        if not cameras and not camera_names:
            return True  # No camera constraint

        if camera_id and camera_id in cameras:
            return True

        if camera_names:
            resolved = self._resolve_camera(camera_id)
            if resolved:
                name = str(resolved.get("name", "")).lower()
                if name and name in camera_names:
                    return True

        return False

    def _event_matches(self, configured_events: Optional[List[str]], event_type: str, payload: Dict[str, Any]) -> bool:
        if not configured_events:
            return True

        event_type = (event_type or "").lower()
        configured = [evt.lower() for evt in configured_events]

        if event_type in configured or "*" in configured or "any" in configured:
            return True

        if event_type == "detection":
            detections = payload.get("detections") or []
            detected_classes = {str(det.get("class", "")).lower() for det in detections if det}
            for evt in configured:
                if not evt.startswith("detection"):
                    continue
                parts = evt.split(":", 1)
                if len(parts) == 1:
                    return True
                desired = parts[1]
                if desired in ("*", "any"):
                    return True
                if desired in detected_classes:
                    return True
        return False

    def _resolve_camera(self, camera_id: Optional[str]) -> Optional[Dict[str, Any]]:
        if not camera_id:
            return None
        if callable(self._camera_resolver):
            try:
                return self._camera_resolver(camera_id)
            except Exception as exc:
                logger.debug("Camera resolver failed for %s: %s", camera_id, exc)
        return None

    @staticmethod
    def _trim_output(text: str, limit: int = 16000) -> str:
        if len(text) <= limit:
            return text
        head = limit - 200
        return text[:head] + "\n... [output truncated] ...\n" + text[-200:]


