"""
PTZ Manager

Routes PTZ commands to the appropriate controller (PyTapo, ONVIF, or
generic CGI). Auto-detects the right protocol per camera using a small
probe and caches the resolution so subsequent commands skip the probe.

Credentials (camera username/password/IP) are pulled from the existing
`CameraManager.cameras` record. Tapo cloud passwords come from the
session/disk store in `core.ptz_credentials`. The HTTP route that calls
this manager only needs `{action, params, brand_hint?}`.
"""
from __future__ import annotations

import asyncio
import logging
import math
import socket
import threading
import time
from typing import Any, Dict, Optional, Tuple

from core import ptz_credentials
from core.ptz_controllers.generic import GenericPTZController
from core.ptz_controllers.pytapo_controller import PyTapoController, PYTAPO_AVAILABLE

try:
    from core.ptz_controllers.onvif_controller import ONVIFPTZController, ONVIF_AVAILABLE
except Exception:  # pragma: no cover
    ONVIFPTZController = None  # type: ignore
    ONVIF_AVAILABLE = False


logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------- #
# Helpers
# ---------------------------------------------------------------------------- #

def _hint_is_tapo(brand_hint: str, manufacturer: str, rtsp_url: str = '') -> bool:
    pieces = " ".join(p for p in (brand_hint, manufacturer, rtsp_url) if p).lower()
    if 'tapo' in pieces or 'tp-link' in pieces or 'tplink' in pieces:
        return True
    # Tapo cameras advertise their RTSP feed at /stream1 (main) and /stream2 (sub)
    # by default. The path is a strong signature when present.
    if '/stream1' in pieces or '/stream2' in pieces:
        return True
    return False


def _tcp_open(host: str, port: int, timeout: float = 0.6) -> bool:
    try:
        with socket.create_connection((host, port), timeout=timeout):
            return True
    except Exception:
        return False


# ---------------------------------------------------------------------------- #
# Manager
# ---------------------------------------------------------------------------- #

class PTZManager:
    """Routes PTZ commands to per-camera controllers (Tapo / ONVIF / generic)."""

    RESOLUTION_TTL_SEC = 600.0
    ONVIF_PROBE_PORTS = (80, 8080, 2020, 8000)

    def __init__(self) -> None:
        self.controllers: Dict[str, Any] = {}
        self.protocols: Dict[str, str] = {}
        self.sweep_threads: Dict[str, threading.Thread] = {}
        self.sweep_stop_events: Dict[str, threading.Event] = {}
        self.sweep_state: Dict[str, Dict[str, Any]] = {}
        self._resolution_cache: Dict[str, Dict[str, Any]] = {}
        # Sticky "stop trying" state so we don't get the camera into a
        # Temporary Suspension lockout after a few bad password retries.
        # Cleared by `invalidate_resolution` (called when the user
        # updates credentials via the UI).
        self._stuck_state: Dict[str, Dict[str, Any]] = {}
        logger.info("PTZ Manager initialised")

    # ------------------------------------------------------------------ #
    # Resolution / probing
    # ------------------------------------------------------------------ #

    def _cached_resolution(self, camera_id: str) -> Optional[Dict[str, Any]]:
        entry = self._resolution_cache.get(camera_id)
        if not entry:
            return None
        if time.time() - entry.get('ts', 0.0) > self.RESOLUTION_TTL_SEC:
            self._resolution_cache.pop(camera_id, None)
            return None
        return entry

    def _store_resolution(self, camera_id: str, protocol: str, **extra: Any) -> None:
        entry = {'protocol': protocol, 'ts': time.time(), **extra}
        self._resolution_cache[camera_id] = entry

    def invalidate_resolution(self, camera_id: str) -> None:
        self._resolution_cache.pop(camera_id, None)
        self._stuck_state.pop(camera_id, None)

    def _mark_stuck(self, camera_id: str, reason: str, *, locked: bool = False,
                    needs: Optional[list] = None) -> None:
        self._stuck_state[camera_id] = {
            'reason': reason,
            'is_locked': bool(locked),
            'needs_credentials': list(needs or []),
            'ts': time.time(),
        }

    def _stuck_status(self, camera_id: str) -> Optional[Dict[str, Any]]:
        entry = self._stuck_state.get(camera_id)
        if not entry:
            return None
        # Lockout entries auto-expire after 6 minutes (Tapo's typical
        # Temporary Suspension is around 5 min); other auth failures
        # remain stuck until the user updates credentials.
        if entry.get('is_locked') and (time.time() - entry.get('ts', 0.0)) > 360.0:
            self._stuck_state.pop(camera_id, None)
            return None
        return entry

    async def probe(self, camera_id: str, config: Dict[str, Any]) -> Dict[str, Any]:
        """
        Determine which PTZ protocol this camera speaks. Returns:
            {
              "protocol_resolved": "tapo" | "onvif" | "generic" | None,
              "needs_credentials": ["tapo_cloud_password"?],
              "brand_guess": str,
              "error": str?,
            }
        """
        ip = str(config.get('ip_address') or config.get('ip') or '').strip()
        if not ip:
            return {
                'protocol_resolved': None,
                'needs_credentials': [],
                'brand_guess': '',
                'error': 'No IP address available for camera',
            }

        # If we previously got the camera into a known-bad state (auth
        # failure or Tapo lockout), short-circuit so we don't hammer it
        # again. The credentials dialog calls `invalidate_resolution`
        # to clear this when the user supplies new credentials.
        stuck = self._stuck_status(camera_id)
        if stuck:
            res = {
                'protocol_resolved': 'tapo' if stuck.get('is_locked') else None,
                'needs_credentials': list(stuck.get('needs_credentials') or []),
                'brand_guess': '',
                'error': stuck.get('reason') or 'PTZ stuck after previous failure',
                'is_locked': bool(stuck.get('is_locked')),
            }
            return res

        brand_hint = str(config.get('brand_hint') or config.get('camera_brand') or '').lower()
        manufacturer = str(config.get('manufacturer') or '').lower()
        rtsp_url = str(config.get('rtsp_url') or '').lower()
        creds = ptz_credentials.get(camera_id) or {}
        cloud_pw = str(config.get('tapo_cloud_password') or creds.get('tapo_cloud_password') or '')
        protocol_override = str(creds.get('protocol_override') or '').lower().strip()

        # Apply credential-store local-user/password overrides for the probe path.
        local_user = creds.get('tapo_local_username')
        local_pw = creds.get('tapo_local_password')
        if local_user or local_pw:
            config = dict(config)
            if local_user:
                config['username'] = local_user
            if local_pw:
                config['password'] = local_pw

        wants_tapo = (
            protocol_override == 'tapo'
            or _hint_is_tapo(brand_hint, manufacturer, rtsp_url)
        )

        # 0) Honor explicit protocol override from the credentials dialog.
        if protocol_override == 'onvif':
            return await self._probe_onvif_only(camera_id, config, brand_hint, manufacturer)
        if protocol_override == 'generic':
            self._store_resolution(camera_id, 'generic')
            return {
                'protocol_resolved': 'generic',
                'needs_credentials': [],
                'brand_guess': brand_hint or manufacturer or 'generic',
            }

        # 1) Tapo path — sticky once we have any Tapo signal. We never
        #    silently fall back to a different protocol from here, because
        #    the user (or RTSP signature) explicitly told us this is a
        #    Tapo and a CGI/ONVIF guess on a Tapo will only confuse them.
        if wants_tapo:
            if not PYTAPO_AVAILABLE:
                msg = "pytapo not installed (run: pip install 'pytapo>=3.3.0,<4')"
                logger.error("Tapo PTZ requested for %s but %s", camera_id, msg)
                return {
                    'protocol_resolved': 'tapo',
                    'needs_credentials': [],
                    'brand_guess': 'tapo',
                    'error': msg,
                }
            if not cloud_pw:
                return {
                    'protocol_resolved': 'tapo',
                    'needs_credentials': ['tapo_cloud_password'],
                    'brand_guess': 'tapo',
                }
            ok, err, info, controller = await self._probe_tapo(config, cloud_pw)
            if ok:
                self._store_resolution(camera_id, 'tapo')
                self._stuck_state.pop(camera_id, None)
                # Promote the probe's connected controller into the live
                # controller cache so the next command reuses the existing
                # Tapo session and DOES NOT trigger another login. This is
                # critical: Tapo locks the camera after a handful of
                # logins per hour.
                if controller is not None:
                    old = self.controllers.get(camera_id)
                    if old is not None and old is not controller:
                        # Best-effort cleanup of any prior tapo session.
                        try:
                            disc = getattr(old, 'disconnect', None)
                            if disc:
                                if asyncio.iscoroutinefunction(disc):
                                    await disc()
                                else:
                                    disc()
                        except Exception:
                            pass
                    self.controllers[camera_id] = controller
                    self.protocols[camera_id] = 'tapo'
                return {
                    'protocol_resolved': 'tapo',
                    'needs_credentials': [],
                    'brand_guess': 'tapo',
                }
            # Stay on the Tapo branch and force a re-prompt with the real error.
            # Mark the camera as STUCK so we stop hammering it (this is what
            # gets cameras into Tapo's Temporary Suspension lockout). The
            # stuck state clears the moment the user submits new creds.
            logger.warning("Tapo probe failed for %s: %s", camera_id, err)
            self._mark_stuck(
                camera_id,
                err or 'Tapo connection failed',
                locked=bool(info.get('is_locked')),
                needs=['tapo_cloud_password'],
            )
            return {
                'protocol_resolved': 'tapo',
                'needs_credentials': ['tapo_cloud_password'],
                'brand_guess': 'tapo',
                'error': err or 'Tapo connection failed',
                'is_locked': bool(info.get('is_locked')),
                'is_auth_failure': bool(info.get('is_auth_failure')),
            }

        # 2) ONVIF
        if ONVIF_AVAILABLE:
            ok, port, err = await self._probe_onvif(config)
            if ok:
                self._store_resolution(camera_id, 'onvif', onvif_port=port)
                return {
                    'protocol_resolved': 'onvif',
                    'needs_credentials': [],
                    'brand_guess': brand_hint or manufacturer or 'onvif',
                    'onvif_port': port,
                }

        # 3) Tapo as a heuristic *without* explicit hint: only suggest it,
        #    don't try blindly (would lock cameras out on bad creds).
        if not cloud_pw and self._looks_like_tapo_port(ip):
            return {
                'protocol_resolved': 'tapo',
                'needs_credentials': ['tapo_cloud_password'],
                'brand_guess': 'tapo',
            }

        # 4) Generic CGI fallback
        self._store_resolution(camera_id, 'generic')
        return {
            'protocol_resolved': 'generic',
            'needs_credentials': [],
            'brand_guess': brand_hint or manufacturer or 'generic',
        }

    async def _probe_onvif_only(self, camera_id: str, config: Dict[str, Any],
                                brand_hint: str, manufacturer: str) -> Dict[str, Any]:
        if not ONVIF_AVAILABLE:
            return {
                'protocol_resolved': 'onvif',
                'needs_credentials': [],
                'brand_guess': brand_hint or manufacturer or 'onvif',
                'error': "onvif-zeep not installed (run: pip install 'onvif-zeep>=0.2.12')",
            }
        ok, port, err = await self._probe_onvif(config)
        if ok:
            self._store_resolution(camera_id, 'onvif', onvif_port=port)
            return {
                'protocol_resolved': 'onvif',
                'needs_credentials': [],
                'brand_guess': brand_hint or manufacturer or 'onvif',
                'onvif_port': port,
            }
        return {
            'protocol_resolved': 'onvif',
            'needs_credentials': [],
            'brand_guess': brand_hint or manufacturer or 'onvif',
            'error': err or 'ONVIF probe failed',
        }

    def _looks_like_tapo_port(self, ip: str) -> bool:
        # Tapo cameras expose 443 by default and rarely answer on 80
        return _tcp_open(ip, 443, 0.5) and not _tcp_open(ip, 80, 0.4)

    async def _probe_tapo(self, config: Dict[str, Any], cloud_pw: str
                          ) -> Tuple[bool, Optional[str], Dict[str, Any], Optional[Any]]:
        """
        Returns (ok, message, info, controller).

        On success, the connected `PyTapoController` is returned so the
        caller can stash it as the camera's live controller — this avoids
        a second redundant login (every Tapo login counts toward the
        5-attempts-per-hour lockout window).
        """
        loop = asyncio.get_event_loop()

        def attempt() -> Tuple[bool, Optional[str], Dict[str, Any], Optional[Any]]:
            try:
                ctl = PyTapoController({**config, 'tapo_cloud_password': cloud_pw})
                res = ctl._connect_sync()  # type: ignore[attr-defined]
                if res.get('success'):
                    return True, None, {}, ctl
                info = {
                    'is_locked': bool(res.get('is_locked')),
                    'is_auth_failure': bool(res.get('is_auth_failure')),
                }
                return False, res.get('message') or res.get('error') or 'tapo connect failed', info, None
            except Exception as err:
                return False, str(err), {}, None

        return await loop.run_in_executor(None, attempt)

    async def _probe_onvif(self, config: Dict[str, Any]) -> Tuple[bool, Optional[int], Optional[str]]:
        ip = str(config.get('ip_address') or config.get('ip') or '').strip()
        if not ip or not ONVIF_AVAILABLE:
            return False, None, 'onvif unavailable'

        # Cheap TCP probe first to skip dead ports
        candidate_ports = [p for p in self.ONVIF_PROBE_PORTS if _tcp_open(ip, p, 0.3)]
        if not candidate_ports:
            return False, None, 'no onvif port reachable'

        loop = asyncio.get_event_loop()

        def attempt() -> Tuple[bool, Optional[int], Optional[str]]:
            for port in candidate_ports:
                try:
                    ctl = ONVIFPTZController({**config, 'onvif_port': port})
                    res = ctl._connect_sync()  # type: ignore[attr-defined]
                    if res.get('success'):
                        return True, port, None
                except Exception as err:
                    logger.debug("ONVIF probe %s:%s -> %s", ip, port, err)
                    continue
            return False, None, 'onvif handshake failed'

        return await loop.run_in_executor(None, attempt)

    # ------------------------------------------------------------------ #
    # Controller lifecycle
    # ------------------------------------------------------------------ #

    async def get_or_create_controller(self, camera_id: str, config: Dict[str, Any]) -> Optional[Any]:
        """Return a connected controller for this camera, creating one if needed."""

        # Hydrate Tapo extras from the credential store: cloud password + any
        # advanced local-username/password override (some Tapo cameras require
        # `admin` regardless of the RTSP user / a different local password).
        creds = ptz_credentials.get(camera_id) or {}
        cloud_pw = config.get('tapo_cloud_password') or creds.get('tapo_cloud_password')
        if cloud_pw:
            config = {**config, 'tapo_cloud_password': cloud_pw}
        local_user = creds.get('tapo_local_username')
        local_pw = creds.get('tapo_local_password')
        if local_user or local_pw:
            config = dict(config)
            if local_user:
                config['username'] = local_user
            if local_pw:
                config['password'] = local_pw

        # Determine target protocol
        cached = self._cached_resolution(camera_id)
        if cached:
            target_protocol = cached['protocol']
            if target_protocol == 'onvif' and 'onvif_port' in cached:
                config = {**config, 'onvif_port': cached['onvif_port']}
        else:
            probe = await self.probe(camera_id, config)
            target_protocol = probe.get('protocol_resolved') or 'generic'
            if probe.get('needs_credentials'):
                logger.warning(
                    "PTZ %s requires credentials %s before a controller can be created",
                    camera_id, probe['needs_credentials'],
                )
                return None
            if probe.get('error'):
                logger.error(
                    "PTZ %s probe error (%s): %s",
                    camera_id, target_protocol, probe['error'],
                )
                return None
            if target_protocol == 'onvif' and probe.get('onvif_port'):
                config = {**config, 'onvif_port': probe['onvif_port']}

        # Reuse existing controller if protocol matches.
        # CRITICAL: the probe already promoted its connected controller
        # into self.controllers, so this check normally returns the
        # already-logged-in instance — avoiding a redundant Tapo login
        # (which would burn a slot in Tapo's 5/hour lockout window).
        existing = self.controllers.get(camera_id)
        if existing and self.protocols.get(camera_id) == target_protocol:
            return existing
        if existing:
            await self.remove_controller(camera_id)

        # Build the new controller
        try:
            controller = await self._build_controller(target_protocol, config)
        except Exception as err:
            logger.error("Failed to build %s controller for %s: %s", target_protocol, camera_id, err)
            return None

        if controller is None:
            return None

        # Connect and verify (ONVIF/PyTapo perform real handshake; generic is no-op)
        try:
            connect_method = getattr(controller, 'connect', None)
            if asyncio.iscoroutinefunction(connect_method):
                connect_result = await connect_method()
            elif callable(connect_method):
                connect_result = connect_method()
                if isinstance(connect_result, bool):
                    connect_result = {'success': connect_result}
            else:
                connect_result = {'success': True}
        except Exception as err:
            logger.error("Connect failure for %s: %s", camera_id, err)
            return None

        if isinstance(connect_result, dict) and not connect_result.get('success', False):
            logger.error(
                "Controller for %s (%s) failed to connect: %s",
                camera_id, target_protocol, connect_result.get('message') or connect_result.get('error'),
            )
            self.invalidate_resolution(camera_id)
            return None

        self.controllers[camera_id] = controller
        self.protocols[camera_id] = target_protocol
        logger.info("Created %s PTZ controller for camera %s", target_protocol, camera_id)
        return controller

    async def _build_controller(self, protocol: str, config: Dict[str, Any]) -> Optional[Any]:
        if protocol == 'tapo':
            cloud_pw = (
                config.get('tapo_cloud_password')
                or config.get('cloud_password')
                or config.get('tapoCloudPassword')
                or ''
            )
            if not (config.get('username') and config.get('password') and cloud_pw):
                logger.error(
                    "Tapo controller missing credentials (username=%s password=%s cloud_password=%s)",
                    bool(config.get('username')), bool(config.get('password')), bool(cloud_pw),
                )
                return None
            return PyTapoController(config)

        if protocol == 'onvif':
            if not ONVIF_AVAILABLE:
                logger.error("ONVIF requested but onvif-zeep is not installed")
                return None
            return ONVIFPTZController(config)

        return GenericPTZController(config)

    async def remove_controller(self, camera_id: str) -> None:
        controller = self.controllers.get(camera_id)
        if not controller:
            return
        await self._stop_sweep(camera_id, controller, silent=True)
        try:
            disconnect = getattr(controller, 'disconnect', None)
            if disconnect:
                if asyncio.iscoroutinefunction(disconnect):
                    await disconnect()
                else:
                    disconnect()
        except Exception as err:
            logger.error("Error disconnecting controller for %s: %s", camera_id, err)
        self.controllers.pop(camera_id, None)
        self.protocols.pop(camera_id, None)
        logger.info("Removed PTZ controller for camera %s", camera_id)

    async def cleanup(self) -> None:
        logger.info("Cleaning up PTZ controllers...")
        for camera_id in list(self.controllers.keys()):
            await self.remove_controller(camera_id)
        logger.info("PTZ Manager cleanup complete")

    # ------------------------------------------------------------------ #
    # Command dispatch
    # ------------------------------------------------------------------ #

    async def execute_command(self, camera_id: str, config: Dict[str, Any],
                              command: str, params: Dict[str, Any]) -> Dict[str, Any]:
        try:
            controller = await self.get_or_create_controller(camera_id, config)
            if not controller:
                # Re-run a probe so the failure response carries actionable
                # `needs_credentials` info and the underlying error; the
                # desktop client uses this to pop the credential prompt
                # on the first failed movement attempt with the real
                # cause shown to the user.
                try:
                    probe_result = await self.probe(camera_id, config)
                except Exception as probe_exc:
                    probe_result = {'error': f'probe exception: {probe_exc}'}
                needs = probe_result.get('needs_credentials') or []
                proto = probe_result.get('protocol_resolved')
                err = probe_result.get('error')
                is_locked = bool(probe_result.get('is_locked'))
                if is_locked:
                    msg = err or 'Camera is temporarily locked out by Tapo (Temporary Suspension).'
                elif 'tapo_cloud_password' in needs:
                    if err:
                        msg = f'Tapo PTZ login failed: {err}'
                    else:
                        msg = 'Tapo PTZ needs credentials'
                elif err:
                    msg = f'PTZ unavailable: {err}'
                else:
                    msg = 'PTZ controller unavailable (probe failed or missing credentials)'
                return {
                    'success': False,
                    'error': msg,
                    'message': msg,
                    'needs_credentials': needs,
                    'protocol_resolved': proto,
                    'brand_guess': probe_result.get('brand_guess'),
                    'is_locked': is_locked,
                }

            if command == 'start_sweep':
                result = await self._start_sweep(camera_id, controller, params)
                return self._augment_with_sweep_status(camera_id, result)
            if command == 'stop_sweep':
                result = await self._stop_sweep(camera_id, controller)
                return self._augment_with_sweep_status(camera_id, result)

            protocol = self.protocols.get(camera_id, 'generic')
            if protocol == 'tapo':
                result = await self._execute_pytapo_command(controller, command, params)
            elif protocol == 'onvif':
                result = await self._execute_onvif_command(controller, command, params)
            else:
                result = await self._execute_generic_command(controller, command, params)

            return self._augment_with_sweep_status(camera_id, result)

        except Exception as err:
            logger.error("PTZ command %s failed for %s: %s", command, camera_id, err)
            return {'success': False, 'error': str(err)}

    # ------------------------------------------------------------------ #
    # Sweep
    # ------------------------------------------------------------------ #

    def _is_sweep_running(self, camera_id: str) -> bool:
        thread = self.sweep_threads.get(camera_id)
        return bool(thread and thread.is_alive())

    def _augment_with_sweep_status(self, camera_id: str, result: Dict[str, Any]) -> Dict[str, Any]:
        if not isinstance(result, dict):
            return result
        sweep_active = self._is_sweep_running(camera_id)
        result['sweep_active'] = sweep_active
        if sweep_active:
            state = self.sweep_state.get(camera_id, {})
            result['sweep_settings'] = {k: v for k, v in state.items() if k != 'stop_event'}
        else:
            result.setdefault('sweep_settings', None)
        return result

    async def _start_sweep(self, camera_id: str, controller: Any, params: Dict[str, Any]) -> Dict[str, Any]:
        await self._stop_sweep(camera_id, controller, silent=True)

        speed = float(params.get('speed', 0.3) or 0.3)
        start_pan = float(params.get('start_pan', -1.0) or -1.0)
        end_pan = float(params.get('end_pan', 1.0) or 1.0)
        tilt = float(params.get('tilt', 0.0) or 0.0)
        seconds_per_side = max(1.5, min(240.0, float(
            params.get('seconds_per_side') or params.get('secondsPerSide') or params.get('duration') or 6.0
        )))
        edge_pause = max(0.0, min(20.0, float(
            params.get('edge_pause_seconds') or params.get('edgePauseSeconds') or params.get('dwell_time') or 0.0
        )))
        sweep_angle = max(10.0, min(180.0, float(
            params.get('sweep_angle') or (abs(end_pan - start_pan) * 90.0) or 180.0
        )))
        smooth_ratio = max(0.0, min(0.9, float(
            params.get('smooth_ratio') or params.get('smoothRatio') or params.get('ease') or 0.0
        )))

        raw_direction = params.get('start_direction', params.get('startDirection', 1))
        if isinstance(raw_direction, str):
            direction = -1 if raw_direction.lower().startswith('l') else 1
        else:
            direction = -1 if float(raw_direction or 1) < 0 else 1

        stop_event = threading.Event()
        self.sweep_stop_events[camera_id] = stop_event
        base_speed = max(0.002, min(1.0, speed))
        self.sweep_state[camera_id] = {
            'speed': base_speed,
            'base_speed': base_speed,
            'start_pan': max(-1.0, min(1.0, start_pan)),
            'end_pan': max(-1.0, min(1.0, end_pan)),
            'tilt': max(-1.0, min(1.0, tilt)),
            'dwell_time': edge_pause,
            'seconds_per_side': seconds_per_side,
            'sweep_angle': sweep_angle,
            'smooth_ratio': smooth_ratio,
            'direction': direction,
            'edge_pause': edge_pause,
            'active': True,
        }

        thread = threading.Thread(
            target=self._run_sweep_worker,
            name=f"PTZ-Sweep-{camera_id}",
            args=(camera_id, controller, stop_event),
            daemon=True,
        )
        self.sweep_threads[camera_id] = thread
        thread.start()

        logger.info(
            "Started sweep for %s (speed=%.3f span=%.2f->%.2f sps=%.1fs pause=%.1fs smooth=%.0f%% dir=%s)",
            camera_id, base_speed, start_pan, end_pan, seconds_per_side,
            edge_pause, smooth_ratio * 100, 'left' if direction < 0 else 'right',
        )
        return {'success': True, 'message': 'Sweep started', 'sweep_active': True}

    async def _stop_sweep(self, camera_id: str, controller: Any, silent: bool = False) -> Dict[str, Any]:
        stop_event = self.sweep_stop_events.pop(camera_id, None)
        thread = self.sweep_threads.pop(camera_id, None)
        state = self.sweep_state.get(camera_id)

        if stop_event:
            stop_event.set()
        if thread and thread.is_alive():
            thread.join(timeout=3)

        if state:
            state['active'] = False
            self.sweep_state[camera_id] = state
        else:
            self.sweep_state.pop(camera_id, None)

        if controller:
            try:
                stop_method = getattr(controller, 'stop', None)
                if asyncio.iscoroutinefunction(stop_method):
                    await stop_method()
                elif callable(stop_method):
                    stop_method()
            except Exception as err:
                if not silent:
                    logger.error("Failed to send stop for sweep on %s: %s", camera_id, err)

        if silent:
            return {'success': True, 'sweep_active': False}

        logger.info("Stopped sweep for %s", camera_id)
        return {'success': True, 'message': 'Sweep stopped', 'sweep_active': False}

    def _run_sweep_worker(self, camera_id: str, controller: Any, stop_event: threading.Event) -> None:
        settings = self.sweep_state.get(camera_id, {}).copy()
        if not settings:
            return

        base_speed = max(0.002, min(1.0, settings.get('base_speed', settings.get('speed', 0.3))))
        smooth_ratio = max(0.0, min(0.9, settings.get('smooth_ratio', 0.0)))
        start_pan = settings.get('start_pan', -1.0)
        end_pan = settings.get('end_pan', 1.0)
        dwell_time = max(0.0, settings.get('dwell_time', 0.4))
        seconds_per_side = max(1.5, settings.get('seconds_per_side', 6.0))
        sweep_angle = settings.get('sweep_angle', abs(end_pan - start_pan) * 90.0)

        approx_degrees = max(10.0, min(180.0, sweep_angle or (abs(end_pan - start_pan) * 90.0)))
        move_duration = max(0.5, seconds_per_side)
        steps = max(4, int(math.ceil(move_duration / 0.25)))
        step_duration = max(0.12, min(0.75, move_duration / steps))
        steps = max(4, int(math.ceil(move_duration / step_duration)))
        step_duration = move_duration / steps

        direction = -1 if settings.get('direction', 1) < 0 else 1
        base_speed_int = max(1, min(10, int(round(base_speed * 10))))

        is_pytapo = isinstance(controller, PyTapoController)
        is_onvif = ONVIFPTZController is not None and isinstance(controller, ONVIFPTZController)

        logger.info(
            "Sweep worker for %s (deg~%.1f sps~%.1fs steps=%d type=%s)",
            camera_id, approx_degrees, move_duration, steps,
            'tapo' if is_pytapo else ('onvif' if is_onvif else 'generic'),
        )

        try:
            while not stop_event.is_set():
                state_ref = self.sweep_state.get(camera_id)
                if state_ref is not None:
                    state_ref['direction'] = direction

                if is_pytapo:
                    self._sweep_step_pytapo(controller, stop_event, steps, step_duration,
                                             base_speed, smooth_ratio, direction)
                elif is_onvif:
                    self._sweep_step_onvif(controller, stop_event, base_speed, direction, move_duration)
                else:
                    self._sweep_step_generic(controller, stop_event, steps, step_duration,
                                              base_speed_int, smooth_ratio, direction)

                if stop_event.wait(dwell_time):
                    break
                direction *= -1
                state_ref = self.sweep_state.get(camera_id)
                if state_ref is not None:
                    state_ref['direction'] = direction
        finally:
            self.sweep_threads.pop(camera_id, None)
            self.sweep_stop_events.pop(camera_id, None)
            state = self.sweep_state.get(camera_id)
            if state:
                state['active'] = False
                self.sweep_state[camera_id] = state

    def _sweep_step_pytapo(self, controller, stop_event, steps, step_duration,
                            base_speed, smooth_ratio, direction):
        try:
            for step in range(steps):
                if stop_event.is_set():
                    return
                progress = step / max(1, steps - 1)
                if smooth_ratio > 0:
                    eased = 0.5 - 0.5 * math.cos(progress * math.pi)
                    multiplier = (1 - smooth_ratio) + smooth_ratio * eased
                else:
                    multiplier = 1.0
                effective = max(0.0015, min(1.0, base_speed * multiplier))
                asyncio.run(controller.continuous_move(
                    pan_speed=effective * direction,
                    tilt_speed=0.0,
                    duration=step_duration,
                ))
                if stop_event.wait(step_duration):
                    return
            asyncio.run(controller.stop())
        except Exception as err:
            logger.error("Sweep PyTapo step failed: %s", err)

    def _sweep_step_onvif(self, controller, stop_event, base_speed, direction, move_duration):
        try:
            asyncio.run(controller.continuous_move(
                pan_speed=base_speed * direction,
                tilt_speed=0.0,
                zoom_speed=0.0,
                duration=move_duration,
            ))
            if stop_event.wait(move_duration):
                pass
            asyncio.run(controller.stop())
        except Exception as err:
            logger.error("Sweep ONVIF step failed: %s", err)

    def _sweep_step_generic(self, controller, stop_event, steps, step_duration,
                             base_speed_int, smooth_ratio, direction):
        try:
            direction_name = 'right' if direction > 0 else 'left'
            for step in range(steps):
                if stop_event.is_set():
                    return
                progress = step / max(1, steps - 1)
                if smooth_ratio > 0:
                    eased = 0.5 - 0.5 * math.cos(progress * math.pi)
                    multiplier = (1 - smooth_ratio) + smooth_ratio * eased
                else:
                    multiplier = 1.0
                speed_int = max(1, min(10, int(round(base_speed_int * multiplier))))
                controller.execute_command('pan', {'direction': direction_name, 'speed': speed_int})
                if step < steps - 1 and stop_event.wait(step_duration):
                    return
            controller.stop()
        except Exception as err:
            logger.error("Sweep generic step failed: %s", err)

    # ------------------------------------------------------------------ #
    # Per-protocol command dispatchers
    # ------------------------------------------------------------------ #

    async def _execute_pytapo_command(self, controller: PyTapoController,
                                      command: str, params: Dict[str, Any]) -> Dict[str, Any]:
        if command == 'connect':
            res = await controller.connect()
            res.setdefault('controller', 'pytapo')
            res.setdefault('connected', controller.connected)
            res.setdefault('message',
                           'Connected to Tapo camera' if res.get('success') else 'Connection failed')
            return res

        if command == 'continuous_move':
            return await controller.continuous_move(
                pan_speed=params.get('pan_speed', 0.0),
                tilt_speed=params.get('tilt_speed', 0.0),
                zoom_speed=params.get('zoom_speed', 0.0),
                duration=params.get('duration', 0.5),
            )
        if command == 'stop':
            return await controller.stop()
        if command == 'go_home':
            return await controller.go_home()
        if command == 'goto_preset':
            return await controller.goto_preset(params.get('preset_token', '1'))
        if command == 'set_preset':
            return await controller.set_preset(
                preset_token=params.get('preset_token', '1'),
                preset_name=params.get('preset_name'),
            )
        if command == 'get_presets':
            return await controller.get_presets()
        if command == 'status':
            return await controller.get_status()

        return {'success': False, 'error': f'Command {command} not supported by PyTapo controller'}

    async def _execute_onvif_command(self, controller: Any, command: str,
                                     params: Dict[str, Any]) -> Dict[str, Any]:
        if command == 'connect':
            res = await controller.connect()
            res.setdefault('controller', 'onvif')
            res.setdefault('connected', getattr(controller, 'connected', False))
            return res
        if command == 'continuous_move':
            return await controller.continuous_move(
                pan_speed=params.get('pan_speed', 0.0),
                tilt_speed=params.get('tilt_speed', 0.0),
                zoom_speed=params.get('zoom_speed', 0.0),
                duration=params.get('duration', 0.0),
            )
        if command == 'stop':
            return await controller.stop()
        if command == 'go_home':
            return await controller.go_home()
        if command == 'goto_preset':
            return await controller.goto_preset(params.get('preset_token', '1'))
        if command == 'set_preset':
            return await controller.set_preset(
                preset_token=params.get('preset_token', '1'),
                preset_name=params.get('preset_name'),
            )
        if command == 'get_presets':
            return await controller.get_presets()
        if command == 'status':
            return await controller.get_status()

        return {'success': False, 'error': f'Command {command} not supported by ONVIF controller'}

    async def _execute_generic_command(self, controller: GenericPTZController,
                                       command: str, params: Dict[str, Any]) -> Dict[str, Any]:
        if command == 'connect':
            try:
                connected = controller.connect()
                return {
                    'success': bool(connected),
                    'message': 'Generic controller ready' if connected else 'Connection failed',
                    'position': controller.current_position,
                }
            except Exception as err:
                logger.error("Generic connect error: %s", err)
                return {'success': False, 'error': str(err)}

        if command == 'continuous_move':
            try:
                pan_speed = float(params.get('pan_speed', 0.0))
                tilt_speed = float(params.get('tilt_speed', 0.0))
                zoom_speed = float(params.get('zoom_speed', 0.0))

                if pan_speed < 0:
                    return controller.execute_command('pan', {'direction': 'left', 'speed': int(abs(pan_speed) * 10)})
                if pan_speed > 0:
                    return controller.execute_command('pan', {'direction': 'right', 'speed': int(pan_speed * 10)})
                if tilt_speed < 0:
                    return controller.execute_command('tilt', {'direction': 'down', 'speed': int(abs(tilt_speed) * 10)})
                if tilt_speed > 0:
                    return controller.execute_command('tilt', {'direction': 'up', 'speed': int(tilt_speed * 10)})
                if zoom_speed:
                    direction = 'in' if zoom_speed > 0 else 'out'
                    return controller.execute_command('zoom', {'direction': direction, 'speed': int(abs(zoom_speed) * 10)})

                return {'success': True, 'message': 'No movement', 'position': controller.current_position}
            except Exception as err:
                logger.error("Generic continuous_move error: %s", err)
                return {'success': False, 'error': str(err)}

        if command in ('goto_preset', 'set_preset'):
            try:
                action = 'goto' if command == 'goto_preset' else 'set'
                return controller.execute_command('preset', {
                    'action': action,
                    'presetId': params.get('preset_token', params.get('presetId', '')),
                })
            except Exception as err:
                logger.error("Generic preset error: %s", err)
                return {'success': False, 'error': str(err)}

        if command == 'go_home':
            try:
                return controller.execute_command('home', {})
            except Exception as err:
                return {'success': False, 'error': str(err)}

        if command == 'stop':
            try:
                return controller.execute_command('stop', {})
            except Exception as err:
                return {'success': False, 'error': str(err)}

        if command == 'get_presets':
            try:
                return controller.get_presets()
            except Exception as err:
                return {'success': False, 'error': str(err)}

        if command == 'status':
            try:
                return controller.get_position()
            except Exception as err:
                return {'success': False, 'error': str(err)}

        return {'success': False, 'error': f'Unknown command: {command}'}


# ---------------------------------------------------------------------------- #
# Singleton accessor
# ---------------------------------------------------------------------------- #

_ptz_manager: Optional[PTZManager] = None


def get_ptz_manager() -> PTZManager:
    global _ptz_manager
    if _ptz_manager is None:
        _ptz_manager = PTZManager()
    return _ptz_manager
