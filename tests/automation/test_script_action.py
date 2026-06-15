"""Tests for Event Rule script actions."""

from __future__ import annotations

import json
import os
import stat
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import MagicMock

from core.automation.actions.script import (
    ScriptAction,
    build_event_context_payload,
    build_script_command,
    normalize_runner,
    resolve_script_path,
    run_script_subprocess,
)
from desktop.utils.shape_trigger_helpers import (
    EXPLICIT_SHAPE_TRIGGERS,
    build_rule_actions,
    effective_trigger_from_mode,
    script_action_from_settings,
    trigger_mode_options_for_kind,
)


class ScriptCommandTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)
        self.script = self.root / "hello.py"
        self.script.write_text(
            "import json, os, sys\n"
            "payload = json.loads(os.environ.get('KNOXNET_EVENT_JSON', '{}'))\n"
            "print(json.dumps(payload))\n"
            "sys.exit(0)\n",
            encoding="utf-8",
        )

    def tearDown(self):
        self.tmp.cleanup()

    def test_build_python_command(self):
        cmd, err = build_script_command(runner="python", script_path=self.script, args=["--x"])
        self.assertIsNone(err)
        self.assertEqual(cmd[0], sys.executable)
        self.assertEqual(cmd[1], str(self.script))
        self.assertEqual(cmd[2:], ["--x"])

    def test_resolve_relative_path_within_root(self):
        resolved, err = resolve_script_path("hello.py", roots=[self.root])
        self.assertIsNone(err)
        self.assertEqual(resolved, self.script.resolve())

    def test_resolve_invalid_path_outside_roots(self):
        outside = Path(self.tmp.name).parent / "outside.py"
        outside.write_text("print('nope')\n", encoding="utf-8")
        resolved, err = resolve_script_path(str(outside), roots=[self.root])
        self.assertIsNone(resolved)
        self.assertEqual(err, "outside_allowed_roots")

    def test_context_payload_roundtrip(self):
        rule = {"id": "rule_1", "name": "Test", "trigger": "zone_enter", "camera_id": "cam1", "shape_id": "z1"}
        details = {"event_type": "zone_enter", "shape_id": "z1"}
        event = MagicMock(camera_id="cam1", payload={"track_id": 7})
        ctx = MagicMock(payload={"event_type": "zone_enter", "tracker_namespace": "motion_box"})
        payload = build_event_context_payload(rule=rule, details=details, event=event, ctx=ctx)
        self.assertEqual(payload["camera_id"], "cam1")
        self.assertEqual(payload["rule_id"], "rule_1")
        self.assertEqual(payload["details"]["event_type"], "zone_enter")
        self.assertEqual(payload["payload"]["track_id"], 7)

    def test_run_script_passes_json_context(self):
        env = os.environ.copy()
        context = {"camera_id": "cam1", "event_type": "zone_enter"}
        env["KNOXNET_EVENT_JSON"] = json.dumps(context)
        result = run_script_subprocess(
            [sys.executable, str(self.script)],
            env=env,
            timeout_sec=5.0,
            cwd=self.root,
            input_json=json.dumps(context),
        )
        self.assertTrue(result.success, result.stderr)
        self.assertIn("cam1", result.stdout)

    def test_timeout_returns_failure_without_crashing(self):
        from unittest.mock import patch
        import subprocess as subprocess_module

        with patch("core.automation.actions.script.subprocess.run") as mock_run:
            mock_run.side_effect = subprocess_module.TimeoutExpired(
                cmd=[sys.executable, "slow.py"],
                timeout=0.5,
                output="",
                stderr="timed out",
            )
            result = run_script_subprocess(
                [sys.executable, "slow.py"],
                env=os.environ.copy(),
                timeout_sec=0.5,
                cwd=self.root,
            )
        self.assertFalse(result.success)
        self.assertTrue(result.timed_out)
        self.assertEqual(result.error, "timeout")

    def test_invalid_path_does_not_crash_action(self):
        action = ScriptAction(script_roots=[self.root])
        result = action.execute(
            rule={"id": "rule_x", "camera_id": "cam1"},
            ctx=MagicMock(payload={}),
            details={"event_type": "zone_enter"},
            action={"type": "script", "language": "python", "path": "missing.py", "timeout_sec": 5},
            event=MagicMock(camera_id="cam1", payload={}),
        )
        self.assertFalse(result.success)
        self.assertEqual(result.stderr, "not_found")

    def test_executable_runner_requires_executable_bit(self):
        exe = self.root / "run.sh"
        exe.write_text("#!/bin/sh\nexit 0\n", encoding="utf-8")
        cmd, err = build_script_command(runner="executable", script_path=exe)
        self.assertEqual(err, "not_executable")
        self.assertIsNone(cmd)
        exe.chmod(exe.stat().st_mode | stat.S_IXUSR)
        cmd, err = build_script_command(runner="executable", script_path=exe)
        self.assertIsNone(err)
        self.assertEqual(cmd[0], str(exe))


class ScriptActionHelperTests(unittest.TestCase):
    def test_script_action_from_settings(self):
        action = script_action_from_settings(
            enabled=True,
            path="notify.py",
            runner="python",
            args_text="--verbose",
            timeout_sec=15,
        )
        self.assertIsNotNone(action)
        self.assertEqual(action["type"], "script")
        self.assertEqual(action["language"], "python")
        self.assertEqual(action["path"], "notify.py")
        self.assertEqual(action["args"], ["--verbose"])
        self.assertEqual(action["timeout_sec"], 15)

    def test_build_rule_actions_includes_script(self):
        actions = build_rule_actions(
            take_screenshot=False,
            motion_watch_settings={},
            run_script=True,
            script_path="hook.py",
            script_runner="python",
            script_args="",
            script_timeout_sec=20,
        )
        self.assertEqual(len(actions), 1)
        self.assertEqual(actions[0]["type"], "script")

    def test_normalize_runner_aliases(self):
        self.assertEqual(normalize_runner("python3"), "python")
        self.assertEqual(normalize_runner("sh"), "shell")


class TriggerMappingTests(unittest.TestCase):
    def test_zone_trigger_options_include_explicit_and_path_modes(self):
        values = [value for _, value in trigger_mode_options_for_kind("zone")]
        self.assertEqual(values[0], "auto_path")
        self.assertIn("any_interaction", values)
        self.assertIn("auto_path", values)
        self.assertIn("path_match", values)
        self.assertIn("zone_enter", values)
        self.assertIn("zone_exit", values)
        self.assertIn("dwell_met", values)

    def test_line_trigger_options(self):
        values = [value for _, value in trigger_mode_options_for_kind("line")]
        self.assertIn("line_cross", values)
        self.assertNotIn("zone_enter", values)

    def test_effective_trigger_explicit_enter(self):
        trigger = effective_trigger_from_mode(
            mode="zone_enter",
            shape_kind="zone",
            derived_trigger="zone_exit",
            has_path=True,
        )
        self.assertEqual(trigger, "zone_enter")

    def test_effective_trigger_auto_uses_derived(self):
        trigger = effective_trigger_from_mode(
            mode="auto_path",
            shape_kind="zone",
            derived_trigger="zone_enter",
            has_path=True,
        )
        self.assertEqual(trigger, "zone_enter")

    def test_explicit_shape_triggers_set(self):
        self.assertIn("zone_enter", EXPLICIT_SHAPE_TRIGGERS)
        self.assertIn("near_tag", EXPLICIT_SHAPE_TRIGGERS)


if __name__ == "__main__":
    unittest.main()
