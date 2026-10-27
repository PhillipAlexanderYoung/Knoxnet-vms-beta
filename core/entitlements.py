from __future__ import annotations

from typing import Any, Dict

BETA_TIER = "beta_free"
DEFAULT_CAMERA_LIMIT = 4
DEFAULT_OFFLINE_GRACE_DAYS = 0


def load_entitlement() -> Dict[str, Any]:
    """Public beta entitlement: free local use for up to four cameras."""
    return {
        "tier": BETA_TIER,
        "camera_limit": DEFAULT_CAMERA_LIMIT,
        "offline_grace_days": DEFAULT_OFFLINE_GRACE_DAYS,
        "source": "public_beta",
    }


def save_entitlement(data: Dict[str, Any]) -> None:
    """No-op in the public beta; paid/cloud entitlements are not included."""
    _ = data


def _effective_entitlement(data: Dict[str, Any]) -> Dict[str, Any]:
    """Keep compatibility with callers that ask for effective entitlement state."""
    _ = data
    return load_entitlement()


def get_camera_limit() -> int:
    return DEFAULT_CAMERA_LIMIT


def entitlement_summary(current_count: int) -> Dict[str, Any]:
    count = max(0, int(current_count or 0))
    return {
        "tier": BETA_TIER,
        "camera_limit": DEFAULT_CAMERA_LIMIT,
        "camera_count": count,
        "limit_reached": count >= DEFAULT_CAMERA_LIMIT,
        "expires_at": None,
        "last_refresh_at": None,
        "offline_grace_days": DEFAULT_OFFLINE_GRACE_DAYS,
        "license": None,
        "message": "Knoxnet VMS Beta includes free local use for up to 4 cameras.",
    }


