"""
ONVIF PTZ Controller

Real ONVIF PTZ via the `onvif-zeep` package. The library is synchronous
(zeep), so all calls are wrapped in `loop.run_in_executor` to keep the
async PTZManager flow non-blocking.

Returns truthful `{success, error}` dicts on every call so the desktop
overlay can react accurately.
"""

from __future__ import annotations

import asyncio
import logging
from concurrent.futures import ThreadPoolExecutor
from typing import Any, Dict, List, Optional

try:
    from onvif import ONVIFCamera, ONVIFError  # onvif-zeep
    ONVIF_AVAILABLE = True
except Exception:  # pragma: no cover
    ONVIFCamera = None  # type: ignore
    ONVIFError = Exception  # type: ignore
    ONVIF_AVAILABLE = False


logger = logging.getLogger(__name__)
_executor = ThreadPoolExecutor(max_workers=4)


def _clamp(value: float, lo: float, hi: float) -> float:
    try:
        return max(lo, min(hi, float(value)))
    except Exception:
        return lo


class ONVIFPTZController:
    """Minimal ONVIF PTZ controller that mirrors the PyTapo surface."""

    DEFAULT_PORTS = (80, 8080, 2020, 8000)

    def __init__(self, camera_config: Dict[str, Any]):
        if not ONVIF_AVAILABLE:
            raise ImportError(
                "onvif-zeep not installed. Run: pip install onvif-zeep"
            )

        self.host = str(camera_config.get('ip_address') or camera_config.get('ip') or '').strip()
        if not self.host:
            raise ValueError("ONVIF PTZ controller requires ip_address")

        self.username = str(camera_config.get('username') or '').strip() or 'admin'
        self.password = str(camera_config.get('password') or '')
        self.port = int(camera_config.get('onvif_port') or camera_config.get('port_onvif') or 0)

        self._camera = None
        self._ptz = None
        self._media = None
        self._profile_token: Optional[str] = None
        self.connected = False
        self.current_position = {'pan': 0.0, 'tilt': 0.0, 'zoom': 1.0}

        # Cache the port we successfully connected on
        self._working_port: Optional[int] = None

    # ------------------------------------------------------------------ #
    # Connection
    # ------------------------------------------------------------------ #

    def _connect_sync(self) -> Dict[str, Any]:
        """Try ONVIF on the supplied port first, then a small fallback list."""
        ports_to_try: List[int] = []
        if self.port and self.port not in ports_to_try:
            ports_to_try.append(self.port)
        for p in self.DEFAULT_PORTS:
            if p not in ports_to_try:
                ports_to_try.append(p)

        last_error = "no ports tried"
        for port in ports_to_try:
            try:
                logger.info("ONVIF: connecting to %s:%s as %s", self.host, port, self.username)
                cam = ONVIFCamera(self.host, port, self.username, self.password)
                media = cam.create_media_service()
                profiles = media.GetProfiles() or []
                if not profiles:
                    last_error = f"no media profiles on {self.host}:{port}"
                    continue
                profile_token = profiles[0].token
                ptz = cam.create_ptz_service()
                # Best-effort capability check
                try:
                    ptz.GetConfigurations()
                except Exception:
                    pass

                self._camera = cam
                self._media = media
                self._ptz = ptz
                self._profile_token = profile_token
                self._working_port = port
                self.port = port
                self.connected = True
                logger.info(
                    "ONVIF connected on %s:%s (profile=%s)",
                    self.host, port, profile_token,
                )
                return {
                    'success': True,
                    'connected': True,
                    'message': f'Connected via ONVIF on port {port}',
                    'port': port,
                    'profile_token': profile_token,
                }
            except ONVIFError as err:  # type: ignore[misc]
                last_error = f"{type(err).__name__}: {err}"
                logger.debug("ONVIF probe failed on %s:%s -> %s", self.host, port, last_error)
                continue
            except Exception as err:
                last_error = f"{type(err).__name__}: {err}"
                logger.debug("ONVIF probe error on %s:%s -> %s", self.host, port, last_error)
                continue

        self.connected = False
        return {
            'success': False,
            'connected': False,
            'message': 'ONVIF connection failed',
            'error': last_error,
        }

    async def connect(self) -> Dict[str, Any]:
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(_executor, self._connect_sync)

    async def disconnect(self) -> Dict[str, Any]:
        self._camera = None
        self._ptz = None
        self._media = None
        self._profile_token = None
        self.connected = False
        return {'success': True}

    # ------------------------------------------------------------------ #
    # Movement
    # ------------------------------------------------------------------ #

    def _continuous_move_sync(
        self,
        pan_speed: float,
        tilt_speed: float,
        zoom_speed: float,
        duration: float,
    ) -> Dict[str, Any]:
        if not self._ptz or not self._profile_token:
            return {'success': False, 'error': 'Not connected'}
        try:
            req = self._ptz.create_type('ContinuousMove')
            req.ProfileToken = self._profile_token
            req.Velocity = {
                'PanTilt': {
                    'x': _clamp(pan_speed, -1.0, 1.0),
                    'y': _clamp(tilt_speed, -1.0, 1.0),
                },
                'Zoom': {'x': _clamp(zoom_speed, -1.0, 1.0)},
            }
            self._ptz.ContinuousMove(req)

            # Optional auto-stop after duration (cameras differ; we rely on
            # the manager calling stop() on key release for held buttons).
            if duration and duration > 0:
                # Don't block the executor for long moves; just update pos hint.
                pass

            return {
                'success': True,
                'message': 'ONVIF continuous_move sent',
                'position': self.current_position.copy(),
            }
        except Exception as err:
            logger.error("ONVIF continuous_move error: %s", err)
            return {'success': False, 'error': str(err)}

    async def continuous_move(
        self,
        pan_speed: float = 0.0,
        tilt_speed: float = 0.0,
        zoom_speed: float = 0.0,
        duration: float = 0.0,
    ) -> Dict[str, Any]:
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(
            _executor,
            self._continuous_move_sync,
            pan_speed, tilt_speed, zoom_speed, duration,
        )

    def _stop_sync(self) -> Dict[str, Any]:
        if not self._ptz or not self._profile_token:
            return {'success': False, 'error': 'Not connected'}
        try:
            self._ptz.Stop({
                'ProfileToken': self._profile_token,
                'PanTilt': True,
                'Zoom': True,
            })
            return {'success': True, 'message': 'ONVIF stop sent'}
        except Exception as err:
            logger.error("ONVIF stop error: %s", err)
            return {'success': False, 'error': str(err)}

    async def stop(self) -> Dict[str, Any]:
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(_executor, self._stop_sync)

    # ------------------------------------------------------------------ #
    # Home / presets
    # ------------------------------------------------------------------ #

    def _go_home_sync(self) -> Dict[str, Any]:
        if not self._ptz or not self._profile_token:
            return {'success': False, 'error': 'Not connected'}
        try:
            self._ptz.GotoHomePosition({'ProfileToken': self._profile_token})
            self.current_position = {'pan': 0.0, 'tilt': 0.0, 'zoom': 1.0}
            return {'success': True, 'message': 'Returned to ONVIF home'}
        except Exception as err:
            logger.error("ONVIF go_home error: %s", err)
            return {'success': False, 'error': str(err)}

    async def go_home(self) -> Dict[str, Any]:
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(_executor, self._go_home_sync)

    def _goto_preset_sync(self, preset_token: str) -> Dict[str, Any]:
        if not self._ptz or not self._profile_token:
            return {'success': False, 'error': 'Not connected'}
        try:
            self._ptz.GotoPreset({
                'ProfileToken': self._profile_token,
                'PresetToken': str(preset_token),
            })
            return {'success': True, 'message': f'Moved to preset {preset_token}'}
        except Exception as err:
            logger.error("ONVIF goto_preset error: %s", err)
            return {'success': False, 'error': str(err)}

    async def goto_preset(self, preset_token: str) -> Dict[str, Any]:
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(_executor, self._goto_preset_sync, preset_token)

    def _set_preset_sync(self, preset_token: str, preset_name: Optional[str]) -> Dict[str, Any]:
        if not self._ptz or not self._profile_token:
            return {'success': False, 'error': 'Not connected'}
        try:
            req = self._ptz.create_type('SetPreset')
            req.ProfileToken = self._profile_token
            req.PresetName = preset_name or f'Preset {preset_token}'
            req.PresetToken = str(preset_token)
            res = self._ptz.SetPreset(req)
            return {
                'success': True,
                'message': f'Saved preset {preset_token}',
                'preset_token': str(res) if res else str(preset_token),
            }
        except Exception as err:
            logger.error("ONVIF set_preset error: %s", err)
            return {'success': False, 'error': str(err)}

    async def set_preset(self, preset_token: str, preset_name: Optional[str] = None) -> Dict[str, Any]:
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(_executor, self._set_preset_sync, preset_token, preset_name)

    def _get_presets_sync(self) -> Dict[str, Any]:
        if not self._ptz or not self._profile_token:
            return {'success': False, 'error': 'Not connected', 'presets': []}
        try:
            raw = self._ptz.GetPresets({'ProfileToken': self._profile_token}) or []
            presets = []
            for p in raw:
                token = getattr(p, 'token', None) or getattr(p, 'Name', None) or ''
                name = getattr(p, 'Name', None) or token
                presets.append({'token': str(token), 'name': str(name)})
            return {'success': True, 'presets': presets}
        except Exception as err:
            logger.error("ONVIF get_presets error: %s", err)
            return {'success': False, 'error': str(err), 'presets': []}

    async def get_presets(self) -> Dict[str, Any]:
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(_executor, self._get_presets_sync)

    async def get_status(self) -> Dict[str, Any]:
        return {
            'success': True,
            'connected': self.connected,
            'position': self.current_position.copy(),
            'patrol_active': False,
            'sweep_active': False,
            'port': self._working_port,
        }
