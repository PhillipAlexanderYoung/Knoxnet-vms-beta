"""
PyTapo PTZ Controller

Native TP-Link Tapo camera control via the `pytapo` library.

Per the pytapo project's official authentication guidance there are
two credential patterns supported by Tapo cameras:

    1. **Camera Account** (set up in the Tapo phone app under
       Settings -> Advanced Settings -> Camera Account):
           Tapo(host, camera_user, camera_password, cloudPassword=cloud_pw)
    2. **Admin fallback** (used when the Camera Account isn't created
       or rejected by newer firmware):
           Tapo(host, "admin", cloud_pw, cloudPassword=cloud_pw)

Lockout safety (critical):
  Tapo cameras enter a "Temporary Suspension" lockout after only a
  handful of logins / failed auth attempts in quick succession. To
  avoid tripping it:
    * The Tapo session is created ONCE with ``reuseSession=True`` and
      reused for every probe / move (pytapo caches ``stok`` on the
      instance, so reusing the controller performs NO new login).
    * By default a single ``connect()`` performs exactly ONE login
      attempt. The rapid camera-account -> admin/cloud fallback chain
      is only attempted when ``allow_auth_fallback`` is explicitly set
      (the user-initiated "Test connection" path) and the attempts are
      spaced out. Control / move actions never rapid-fire a fallback.
    * The credential set that worked is remembered (``working_auth``)
      so the manager can persist it and skip re-discovery next time.
    * "Temporary Suspension: try again in N seconds" is parsed into
      ``lockout_seconds`` so the manager can refuse to touch the camera
      until the cooldown elapses.

Pre-requisites the user must satisfy in the Tapo phone app:
  * Me -> Third-Party Services -> Third-Party Compatibility = ON.
  * (Recommended) Settings -> Advanced Settings -> Camera Account
    created with a strong password.
"""
import logging
import asyncio
import re
import time
from concurrent.futures import ThreadPoolExecutor
from typing import Dict, Any, List, Optional, Tuple

try:
    from pytapo import Tapo
    PYTAPO_AVAILABLE = True
except ImportError:
    PYTAPO_AVAILABLE = False
    Tapo = None

logger = logging.getLogger(__name__)

DEFAULT_ADMIN_USERNAME = "admin"

# Thread pool for running pytapo (which uses its own asyncio internally)
_executor = ThreadPoolExecutor(max_workers=4)


def _looks_like_auth_failure(err_msg: str) -> bool:
    msg = (err_msg or "").lower()
    return (
        "invalid authentication data" in msg
        or "invalid auth" in msg
        or "401" in msg
        or "unauthor" in msg
    )


def _looks_like_lockout(err_msg: str) -> bool:
    msg = (err_msg or "").lower()
    return (
        "temporary suspension" in msg
        or "locked out" in msg
        or "-40404" in msg
        or "-40411" in msg
    )


def _parse_lockout_seconds(err_msg: str) -> Optional[int]:
    """Extract N from "Temporary Suspension: Try again in N seconds"."""
    if not err_msg:
        return None
    match = re.search(r"try again in\s+(\d+)\s*second", err_msg.lower())
    if match:
        try:
            return int(match.group(1))
        except (TypeError, ValueError):
            return None
    return None


class PyTapoController:
    """Simple PTZ controller for TP-Link Tapo cameras using pytapo."""

    def __init__(self, camera_config: Dict[str, Any]):
        if not PYTAPO_AVAILABLE:
            raise ImportError("pytapo not installed. Run: pip install pytapo")

        self.ip_address = str(camera_config.get('ip_address') or '').strip()
        if not self.ip_address:
            raise ValueError("Tapo PTZ controller requires ip_address.")

        self.username = (camera_config.get('username') or DEFAULT_ADMIN_USERNAME).strip() or DEFAULT_ADMIN_USERNAME
        self.password = str(camera_config.get('password') or '').strip()
        self.tapo_cloud_password = str(
            camera_config.get('tapo_cloud_password')
            or camera_config.get('cloud_password')
            or camera_config.get('tapoCloudPassword')
            or ''
        ).strip()

        if not self.password:
            raise ValueError("Tapo PTZ controller requires camera password in configuration.")
        if not self.tapo_cloud_password:
            raise ValueError("Tapo PTZ controller requires tapo_cloud_password in configuration.")

        # Remember the ORIGINAL requested credentials so the manager can tell
        # whether a later config still matches this live session (and thus
        # reuse it instead of opening a new login). self.username/password get
        # overwritten with the *working* values after a successful connect.
        self._req_username = self.username
        self._req_password = self.password
        self._req_cloud_password = self.tapo_cloud_password

        # Optional control port override (pytapo defaults to 443).
        self.control_port = (
            camera_config.get('control_port')
            or camera_config.get('controlPort')
            or None
        )

        # Single-attempt auth by default. Only the explicit, user-initiated
        # "Test connection" path passes allow_auth_fallback=True so the
        # camera-account -> admin/cloud discovery chain can run (spaced out).
        self.allow_fallback = bool(camera_config.get('allow_auth_fallback', False))

        # A remembered/working credential set to try FIRST (and, when
        # fallback is disabled, the ONLY set tried). Shape: {user,password,label}.
        pref = camera_config.get('preferred_auth')
        self.preferred_auth: Optional[Dict[str, str]] = pref if isinstance(pref, dict) else None

        # Populated after a successful connect so the manager can persist the
        # winning method and avoid re-discovering it next session.
        self.working_auth: Optional[Dict[str, str]] = None
        # Set when Tapo reports a Temporary Suspension; seconds remaining.
        self.lockout_seconds: Optional[int] = None

        self.tapo = None
        self.connected = False
        self.current_position = {'pan': 0.0, 'tilt': 0.0, 'zoom': 1.0}
        self.last_connection_result: Dict[str, Any] = {
            'success': False,
            'connected': False,
            'message': 'Not connected',
            'ip_address': self.ip_address,
        }

        logger.info("PyTapo controller created for %s", self.ip_address)

    def matches_credentials(self, config: Dict[str, Any],
                            cloud_pw: Optional[str] = None) -> bool:
        """
        True if `config` carries the same credentials this controller was
        constructed with. Used by the manager to decide whether a live
        (already-authenticated) session can be reused for a probe / move
        without opening a brand-new login.
        """
        cfg_user = (config.get('username') or DEFAULT_ADMIN_USERNAME).strip() or DEFAULT_ADMIN_USERNAME
        cfg_pw = str(config.get('password') or '').strip()
        cfg_cloud = str(
            cloud_pw if cloud_pw is not None else (
                config.get('tapo_cloud_password')
                or config.get('cloud_password')
                or config.get('tapoCloudPassword')
                or ''
            )
        ).strip()
        return (
            cfg_user == self._req_username
            and cfg_pw == self._req_password
            and cfg_cloud == self._req_cloud_password
        )

    # ------------------------------------------------------------------ #
    # Connection
    # ------------------------------------------------------------------ #

    def _candidate_attempts(self) -> List[Tuple[str, str, str]]:
        """
        Build the (label, user, password) attempt list.

        Order:
          0) The remembered/preferred working method, if known. When a
             method has already been proven for this camera we use ONLY
             it (deterministic, single login).
          1) Camera Account credentials (as configured in the Tapo app).
          2) Admin fallback: ("admin", cloud_password). Required for some
             firmware where the Camera Account hash isn't loaded.

        IMPORTANT: unless ``self.allow_fallback`` is True, the list is
        truncated to a SINGLE attempt. Rapid camera-account -> admin/cloud
        retries are the #1 trigger for Tapo's Temporary Suspension lockout,
        so control/move paths must never auto-fire the chain.
        """
        attempts: List[Tuple[str, str, str]] = []
        seen = set()

        def _add(label: str, user: str, password: str) -> None:
            if not user or not password:
                return
            key = (user.lower(), password)
            if key in seen:
                return
            seen.add(key)
            attempts.append((label, user, password))

        # 0) Remembered/working method first (deterministic re-auth).
        if self.preferred_auth:
            _add(
                self.preferred_auth.get('label') or 'remembered method',
                self.preferred_auth.get('user') or '',
                self.preferred_auth.get('password') or '',
            )

        # 1) Camera-account method (user-supplied local creds)
        _add("camera account", self.username, self.password)

        # 2) Admin / cloud-password fallback (the pytapo README's official
        #    second attempt). Only meaningful when we have a cloud password.
        _add("admin / cloud password", DEFAULT_ADMIN_USERNAME, self.tapo_cloud_password)

        # Single login unless an explicit discovery (test-connection) asked
        # for the full chain. Never rapid-fire on control/move paths.
        if not self.allow_fallback and attempts:
            attempts = attempts[:1]
        return attempts

    def _connect_sync(self) -> Dict[str, Any]:
        result: Dict[str, Any] = {
            'success': False,
            'connected': False,
            'message': 'Connection not established',
            'ip_address': self.ip_address,
        }

        if self.connected and self.tapo:
            message = f"Already connected to Tapo camera at {self.ip_address}"
            logger.info(message)
            result.update({'success': True, 'connected': True, 'message': message})
            self.last_connection_result = dict(result)
            return result

        attempts = self._candidate_attempts()
        if not attempts:
            message = "Tapo PTZ requires a camera password (and a TP-Link cloud password for the admin fallback)."
            logger.error(message)
            result.update({
                'success': False,
                'message': message,
                'error': message,
                'is_locked': False,
                'is_auth_failure': True,
            })
            self.last_connection_result = dict(result)
            return result

        # pytapo kwargs shared by every attempt. reuseSession=True keeps a
        # single requests.Session (and the cached stok) alive on the instance
        # so reusing this controller performs NO additional login.
        tapo_kwargs: Dict[str, Any] = {'cloudPassword': self.tapo_cloud_password, 'reuseSession': True}
        if self.control_port:
            try:
                tapo_kwargs['controlPort'] = int(self.control_port)
            except (TypeError, ValueError):
                pass

        last_error: str = ""
        last_label: str = ""
        for idx, (label, user, password) in enumerate(attempts):
            # Small delay before any attempt after the first to avoid
            # looking like a brute-force flood to Tapo's rate limiter.
            # (Only reachable in explicit discovery mode; single-attempt
            # connects never hit this.)
            if idx > 0:
                time.sleep(1.5)
            logger.info(
                "Connecting to Tapo camera %s using '%s' (user=%s)",
                self.ip_address, label, user,
            )
            try:
                self.tapo = Tapo(self.ip_address, user, password, **tapo_kwargs)
                info = self.tapo.getBasicInfo() or {}
                model = (
                    info.get('device_model')
                    or info.get('model')
                    or info.get('device_model_name')
                    or 'Tapo camera'
                )
                self.connected = True
                self.lockout_seconds = None
                # Cache the working credentials on the instance so
                # subsequent reconnects skip the failing attempt, and expose
                # them so the manager can persist the winning method.
                self.username = user
                self.password = password
                self.working_auth = {'user': user, 'password': password, 'label': label}
                result = {
                    'success': True,
                    'connected': True,
                    'message': f'Connected to {model} via {label}',
                    'model': model,
                    'ip_address': self.ip_address,
                    'auth_method': label,
                    'working_auth': dict(self.working_auth),
                }
                self.last_connection_result = dict(result)
                return result
            except Exception as err:
                last_error = str(err)
                last_label = label
                logger.warning(
                    "PyTapo '%s' attempt failed for %s: %s",
                    label, self.ip_address, last_error[:200],
                )
                # On a hard lockout, do NOT try the next pattern (would
                # extend the lockout). Surface immediately with the parsed
                # cooldown so the manager can refuse further logins.
                if _looks_like_lockout(last_error):
                    self.connected = False
                    self.tapo = None
                    secs = _parse_lockout_seconds(last_error)
                    self.lockout_seconds = secs
                    if secs:
                        message = (
                            "Camera is temporarily locked out by Tapo "
                            f"(Temporary Suspension). Try again in {secs} seconds."
                        )
                    else:
                        message = (
                            "Camera is temporarily locked out by Tapo "
                            "(Temporary Suspension). Reboot the camera or "
                            "wait ~30 minutes before retrying."
                        )
                    result.update({
                        'success': False,
                        'message': message,
                        'error': last_error,
                        'is_locked': True,
                        'is_auth_failure': False,
                        'auth_method': label,
                        'lockout_seconds': secs,
                    })
                    self.last_connection_result = dict(result)
                    return result
                # Otherwise continue to the next attempt (discovery only).

        # All attempts exhausted.
        self.connected = False
        self.tapo = None
        is_auth = _looks_like_auth_failure(last_error)
        if is_auth:
            message = (
                "Tapo authentication failed. Check that the Camera Account "
                "in the Tapo app matches the username/password stored here, "
                "AND that the TP-Link cloud password is the one you use to "
                "log in to the Tapo phone app. Make sure Me -> Third-Party "
                "Services -> Third-Party Compatibility is enabled in the app. "
                "(No second login was attempted automatically — that would "
                "risk a Temporary Suspension lockout; use 'Test connection' "
                "to try the admin/cloud method.)"
            )
        else:
            message = f"PyTapo connection failed via {last_label}: {last_error[:200]}"
        result.update({
            'success': False,
            'connected': False,
            'message': message,
            'error': last_error,
            'is_locked': False,
            'is_auth_failure': is_auth,
            'auth_method': last_label,
        })
        self.last_connection_result = dict(result)
        return result

    async def connect(self) -> Dict[str, Any]:
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(_executor, self._connect_sync)

    # ------------------------------------------------------------------ #
    # Movement
    # ------------------------------------------------------------------ #

    def _move_sync(self, pan_degrees: float, tilt_degrees: float) -> Dict[str, Any]:
        try:
            if abs(pan_degrees) > 0.1 or abs(tilt_degrees) > 0.1:
                logger.info("Moving Tapo camera: pan=%.1f tilt=%.1f", pan_degrees, tilt_degrees)
                self.tapo.moveMotor(pan_degrees, tilt_degrees)
                self.current_position['pan'] += pan_degrees
                self.current_position['tilt'] += tilt_degrees
                return {
                    'success': True,
                    'message': f'Moved pan={pan_degrees:.1f} tilt={tilt_degrees:.1f}',
                    'position': self.current_position.copy(),
                }
            return {
                'success': True,
                'message': 'No movement',
                'position': self.current_position.copy(),
            }
        except Exception as err:
            logger.error("Move error: %s", err)
            err_msg = str(err)
            out = {'success': False, 'error': err_msg}
            # A move that triggers an internal stok refresh can surface a
            # lockout; bubble it up so the manager sets the cooldown.
            if _looks_like_lockout(err_msg):
                secs = _parse_lockout_seconds(err_msg)
                self.connected = False
                self.lockout_seconds = secs
                out['is_locked'] = True
                out['lockout_seconds'] = secs
            return out

    async def continuous_move(self, pan_speed: float = 0.0, tilt_speed: float = 0.0,
                              zoom_speed: float = 0.0, duration: float = 0.5) -> Dict[str, Any]:
        if not self.tapo or not self.connected:
            return {'success': False, 'error': 'Not connected'}

        duration = max(0.1, float(duration or 0.1))
        # Speed (-1..1) -> degrees, scaled by duration (~40 deg/s at full speed).
        degrees_per_second = 40.0
        pan_degrees = pan_speed * degrees_per_second * duration
        tilt_degrees = tilt_speed * degrees_per_second * duration

        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(_executor, self._move_sync, pan_degrees, tilt_degrees)

    async def stop(self) -> Dict[str, Any]:
        # Tapo's moveMotor is stepwise; there is no continuous-stop semantic.
        return {
            'success': True,
            'message': 'Tapo movement stops automatically',
            'position': self.current_position.copy(),
        }

    # ------------------------------------------------------------------ #
    # Calibration / presets
    # ------------------------------------------------------------------ #

    def _calibrate_sync(self) -> Dict[str, Any]:
        try:
            logger.info("Calibrating Tapo to home position")
            self.tapo.calibrateMotor()
            self.current_position = {'pan': 0.0, 'tilt': 0.0, 'zoom': 1.0}
            return {
                'success': True,
                'message': 'Calibrated to home',
                'position': self.current_position.copy(),
            }
        except Exception as err:
            logger.error("Calibrate error: %s", err)
            return {'success': False, 'error': str(err)}

    async def go_home(self) -> Dict[str, Any]:
        if not self.tapo or not self.connected:
            return {'success': False, 'error': 'Not connected'}
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(_executor, self._calibrate_sync)

    async def goto_preset(self, preset_token: str) -> Dict[str, Any]:
        try:
            if not self.tapo or not self.connected:
                return {'success': False, 'error': 'Not connected'}
            logger.info("Going to preset: %s", preset_token)
            self.tapo.setPreset(preset_token)
            return {'success': True, 'message': f'Moved to preset {preset_token}'}
        except Exception as err:
            logger.error("Preset error: %s", err)
            return {'success': False, 'error': str(err)}

    async def set_preset(self, preset_token: str, preset_name: Optional[str] = None) -> Dict[str, Any]:
        try:
            if not self.tapo or not self.connected:
                return {'success': False, 'error': 'Not connected'}
            name = preset_name or preset_token
            logger.info("Saving preset: %s", name)
            self.tapo.savePreset(name)
            return {'success': True, 'message': f'Preset {name} saved'}
        except Exception as err:
            logger.error("Save preset error: %s", err)
            return {'success': False, 'error': str(err)}

    async def get_presets(self) -> Dict[str, Any]:
        try:
            if not self.tapo or not self.connected:
                return {'success': False, 'error': 'Not connected', 'presets': []}
            raw = []
            for name in ('getPresets', 'getPresetList', 'getPreset'):
                method = getattr(self.tapo, name, None)
                if method:
                    try:
                        raw = method() or []
                        break
                    except Exception:
                        continue
            presets = []
            if isinstance(raw, list):
                for item in raw:
                    if isinstance(item, dict):
                        token = str(item.get('id') or item.get('token') or item.get('name') or '')
                        name = str(item.get('name') or token)
                    else:
                        token = name = str(item)
                    if token:
                        presets.append({'token': token, 'name': name})
            return {'success': True, 'presets': presets}
        except Exception as err:
            logger.error("Get presets error: %s", err)
            return {'success': False, 'error': str(err), 'presets': []}

    async def get_status(self) -> Dict[str, Any]:
        return {
            'success': True,
            'connected': self.connected,
            'position': self.current_position.copy(),
            'patrol_active': False,
            'sweep_active': False,
        }

    async def disconnect(self) -> Dict[str, Any]:
        self.connected = False
        self.tapo = None
        logger.info("Disconnected from Tapo camera")
        return {'success': True}
