"""
Regression tests for the Tapo PTZ "Temporary Suspension" lockout bug.

These exercise the lockout/auth logic with a FAKE pytapo `Tapo` class so
nothing ever touches a real camera or the network:

  * a failed auth does NOT trigger an immediate second login
    (single-attempt by default; the fallback chain only runs when
    discovery is explicitly enabled),
  * the lockout cooldown is parsed/armed and short-circuits further
    logins until it expires,
  * the working auth method is memoized and reused first,
  * a probe reuses a live authenticated session instead of opening a
    second login (the test-connection -> first-control-press path).
"""

from __future__ import annotations

import asyncio
import unittest
from unittest import mock

from core import ptz_credentials
from core.ptz_controllers import pytapo_controller
from core.ptz_controllers.pytapo_controller import (
    PyTapoController,
    _parse_lockout_seconds,
)
from core.ptz_manager import PTZManager


class FakeTapo:
    """Stand-in for pytapo.Tapo that records every construction (= login)."""

    instances: list = []
    # 'ok' | 'auth' | 'lockout' | 'admin_only'
    behavior = "ok"
    lockout_msg = "Temporary Suspension: Try again in 533 seconds"

    def __init__(self, host, user, password, **kwargs):
        FakeTapo.instances.append(
            {"host": host, "user": user, "password": password, "kwargs": kwargs}
        )
        if FakeTapo.behavior == "auth":
            raise Exception("Invalid authentication data")
        if FakeTapo.behavior == "lockout":
            raise Exception(FakeTapo.lockout_msg)
        if FakeTapo.behavior == "admin_only" and user != "admin":
            # Camera-account creds rejected; only the admin/cloud method works.
            raise Exception("Invalid authentication data")

    def getBasicInfo(self):
        return {"device_model": "Tapo C200"}

    def moveMotor(self, x, y):
        return {"error_code": 0}

    @classmethod
    def reset(cls, behavior="ok"):
        cls.instances = []
        cls.behavior = behavior


def _build_controller(cfg):
    with mock.patch.object(pytapo_controller, "PYTAPO_AVAILABLE", True), \
            mock.patch.object(pytapo_controller, "Tapo", FakeTapo):
        ctl = PyTapoController(cfg)
        res = ctl._connect_sync()
    return ctl, res


class ParseLockoutTests(unittest.TestCase):
    def test_parses_seconds(self):
        self.assertEqual(
            _parse_lockout_seconds("Temporary Suspension: Try again in 533 seconds"),
            533,
        )

    def test_returns_none_without_match(self):
        self.assertIsNone(_parse_lockout_seconds("Invalid authentication data"))
        self.assertIsNone(_parse_lockout_seconds(""))


class SingleAttemptAuthTests(unittest.TestCase):
    def setUp(self):
        FakeTapo.reset("auth")

    def test_failed_auth_does_not_trigger_second_login(self):
        """Default (control/move) path must try exactly ONE login."""
        ctl, res = _build_controller(
            {
                "ip_address": "10.0.0.5",
                "username": "cam",
                "password": "pw",
                "tapo_cloud_password": "cloud",
            }
        )
        self.assertFalse(res["success"])
        self.assertTrue(res["is_auth_failure"])
        self.assertEqual(
            len(FakeTapo.instances), 1,
            "auth failure must NOT auto-fire a second login attempt",
        )

    def test_reuseSession_passed_to_tapo(self):
        FakeTapo.reset("ok")
        _build_controller(
            {
                "ip_address": "10.0.0.5",
                "username": "cam",
                "password": "pw",
                "tapo_cloud_password": "cloud",
            }
        )
        self.assertTrue(FakeTapo.instances[0]["kwargs"].get("reuseSession"))


class DiscoveryFallbackTests(unittest.TestCase):
    def setUp(self):
        FakeTapo.reset("admin_only")

    def test_fallback_only_when_explicitly_allowed(self):
        # Without fallback: single attempt (camera account) fails.
        ctl, res = _build_controller(
            {
                "ip_address": "10.0.0.6",
                "username": "cam",
                "password": "pw",
                "tapo_cloud_password": "cloud",
            }
        )
        self.assertFalse(res["success"])
        self.assertEqual(len(FakeTapo.instances), 1)

        # With fallback (the explicit "Test connection" path): camera account
        # fails, then admin/cloud succeeds -> 2 spaced attempts.
        FakeTapo.reset("admin_only")
        with mock.patch.object(pytapo_controller.time, "sleep", lambda *_: None):
            ctl, res = _build_controller(
                {
                    "ip_address": "10.0.0.6",
                    "username": "cam",
                    "password": "pw",
                    "tapo_cloud_password": "cloud",
                    "allow_auth_fallback": True,
                }
            )
        self.assertTrue(res["success"])
        self.assertEqual(len(FakeTapo.instances), 2)
        self.assertEqual(ctl.working_auth["user"], "admin")

    def test_preferred_auth_tried_first_single_login(self):
        FakeTapo.reset("admin_only")
        ctl, res = _build_controller(
            {
                "ip_address": "10.0.0.6",
                "username": "cam",
                "password": "pw",
                "tapo_cloud_password": "cloud",
                "preferred_auth": {"user": "admin", "password": "cloud", "label": "remembered"},
            }
        )
        self.assertTrue(res["success"])
        self.assertEqual(len(FakeTapo.instances), 1, "remembered method => single login")
        self.assertEqual(FakeTapo.instances[0]["user"], "admin")


class LockoutParseControllerTests(unittest.TestCase):
    def test_lockout_surfaces_seconds(self):
        FakeTapo.reset("lockout")
        ctl, res = _build_controller(
            {
                "ip_address": "10.0.0.7",
                "username": "cam",
                "password": "pw",
                "tapo_cloud_password": "cloud",
            }
        )
        self.assertFalse(res["success"])
        self.assertTrue(res["is_locked"])
        self.assertEqual(res["lockout_seconds"], 533)
        self.assertEqual(len(FakeTapo.instances), 1)


class ManagerCooldownTests(unittest.TestCase):
    def setUp(self):
        self.mgr = PTZManager()

    def test_set_and_remaining(self):
        self.mgr._set_lockout("camA", 120)
        self.assertGreater(self.mgr._lockout_remaining("camA"), 100)

    def test_expired_cooldown_clears(self):
        self.mgr._lockout_until["camB"] = 1.0  # far in the past
        self.assertEqual(self.mgr._lockout_remaining("camB"), 0.0)
        self.assertNotIn("camB", self.mgr._lockout_until)

    def test_probe_short_circuits_when_locked(self):
        FakeTapo.reset("ok")
        self.mgr._set_lockout("camC", 300)
        with mock.patch.object(pytapo_controller, "PYTAPO_AVAILABLE", True), \
                mock.patch.object(pytapo_controller, "Tapo", FakeTapo):
            res = asyncio.run(
                self.mgr.probe(
                    "camC",
                    {
                        "ip_address": "10.0.0.8",
                        "brand_hint": "tapo",
                        "username": "cam",
                        "password": "pw",
                        "tapo_cloud_password": "cloud",
                    },
                )
            )
        self.assertTrue(res["is_locked"])
        self.assertEqual(
            len(FakeTapo.instances), 0,
            "a locked-out camera must never receive a login",
        )

    def test_probe_arms_cooldown_on_suspension(self):
        FakeTapo.reset("lockout")
        with mock.patch.object(pytapo_controller, "PYTAPO_AVAILABLE", True), \
                mock.patch("core.ptz_manager.PYTAPO_AVAILABLE", True), \
                mock.patch.object(pytapo_controller, "Tapo", FakeTapo):
            res = asyncio.run(
                self.mgr.probe(
                    "camD",
                    {
                        "ip_address": "10.0.0.9",
                        "brand_hint": "tapo",
                        "username": "cam",
                        "password": "pw",
                        "tapo_cloud_password": "cloud",
                    },
                )
            )
        self.assertTrue(res["is_locked"])
        self.assertGreater(self.mgr._lockout_remaining("camD"), 0)


class ManagerSessionReuseTests(unittest.TestCase):
    def setUp(self):
        self.mgr = PTZManager()

    def test_probe_reuses_live_session_no_new_login(self):
        cfg = {
            "ip_address": "10.0.0.10",
            "username": "cam",
            "password": "pw",
            "tapo_cloud_password": "cloud",
        }
        FakeTapo.reset("ok")
        ctl, res = _build_controller(cfg)
        self.assertTrue(res["success"])
        self.assertEqual(len(FakeTapo.instances), 1)

        # Warm the manager cache like a successful test-connection would.
        self.mgr.controllers["camE"] = ctl
        self.mgr.protocols["camE"] = "tapo"

        FakeTapo.reset("ok")  # zero instances => assert no new login
        probe_cfg = dict(cfg)
        probe_cfg["brand_hint"] = "tapo"
        with mock.patch("core.ptz_manager.PYTAPO_AVAILABLE", True):
            res2 = asyncio.run(self.mgr.probe("camE", probe_cfg))
        self.assertEqual(res2["protocol_resolved"], "tapo")
        self.assertTrue(res2.get("reused_session"))
        self.assertEqual(
            len(FakeTapo.instances), 0,
            "probe must reuse the cached authenticated session (no 2nd login)",
        )


class MemoizationTests(unittest.TestCase):
    def test_remember_and_get_working_auth(self):
        cam = "cam-memo-unique"
        ptz_credentials.clear(cam)
        self.assertIsNone(ptz_credentials.get_working_auth(cam))
        ptz_credentials.remember_working_auth(cam, "admin", "cloudpw", "admin / cloud password")
        wa = ptz_credentials.get_working_auth(cam)
        self.assertEqual(wa["user"], "admin")
        self.assertEqual(wa["password"], "cloudpw")
        self.assertEqual(wa["label"], "admin / cloud password")
        ptz_credentials.clear(cam)


if __name__ == "__main__":
    unittest.main()
