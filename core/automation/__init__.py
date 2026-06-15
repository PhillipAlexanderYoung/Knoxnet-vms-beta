"""
Server-side automation engine (rules -> actions).

This package is intentionally lightweight and uses only safe, structured rule conditions.
"""

from .engine import AutomationEngine  # noqa: F401
from .track_state import (  # noqa: F401
    BACKEND_SORT_NAMESPACE,
    MOTION_BOX_NAMESPACE,
    TrackSceneEngine,
)


