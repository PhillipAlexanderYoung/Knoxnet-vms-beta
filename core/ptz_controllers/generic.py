"""
Generic / CGI PTZ controller

Last-resort PTZ controller for cameras that expose a vendor CGI URL or
a custom URL template. Tries the well-known patterns from common
brands (Hikvision, Dahua, Axis, Amcrest) and reports truthful
success/failure based on the underlying HTTP result.
"""
import logging
import time
from typing import Any, Dict, List, Optional

import requests

logger = logging.getLogger(__name__)


class GenericPTZController:
    HTTP_TIMEOUT = 1.0
    HTTP_TIMEOUT_FAST = 0.5

    def __init__(self, camera_config: Dict[str, Any]):
        self.config = camera_config

        ip_address = camera_config.get('ip_address') or camera_config.get('ip') or ''
        if ip_address and not str(ip_address).startswith('http'):
            self.base_url = f"http://{ip_address}"
        else:
            self.base_url = str(camera_config.get('http_url') or ip_address)

        self.username = camera_config.get('username') or 'admin'
        self.password = camera_config.get('password') or ''

        self.ptz_config = camera_config.get('ptz_config') or {}
        self.pan_range = self.ptz_config.get('pan_range', [-180, 180])
        self.tilt_range = self.ptz_config.get('tilt_range', [-90, 90])
        self.zoom_range = self.ptz_config.get('zoom_range', [1, 10])
        self.presets = dict(self.ptz_config.get('presets') or {})

        self.current_position = {'pan': 0, 'tilt': 0, 'zoom': 1}

        # Cache the working URL pattern *on the instance* (do not mutate
        # the shared config dict that other code passes around).
        self._working_pattern_index: Optional[int] = None
        self._last_http_error: Optional[str] = None

        logger.info("Generic PTZ Controller initialised for %s", self.base_url)

    # ------------------------------------------------------------------ #
    # HTTP / probing
    # ------------------------------------------------------------------ #

    def _build_url_patterns(self, code: str, speed: int, action: str) -> List[str]:
        return [
            f"{self.base_url}/cgi-bin/ptz.cgi?action={action}&channel=0&code={code}&arg1={speed}&arg2=0",
            f"{self.base_url}/cgi-bin/ptz.cgi?action={action}&code={code}&channel=0&arg1=0&arg2={speed}&arg3=0",
            f"{self.base_url}/axis-cgi/com/ptz.cgi?{code.lower()}={speed}",
            f"{self.base_url}/cgi-bin/ptz.cgi?action={action}&code={code}&arg1={speed}&arg2=0&arg3=0",
            f"{self.base_url}/onvif/ptz?action={code.lower()}&speed={speed}",
        ]

    def _attempt_http(self, url: str, auth) -> Optional[bool]:
        """Return True on 200, False on auth/4xx, None on retryable errors."""
        try:
            response = requests.get(url, auth=auth, timeout=self.HTTP_TIMEOUT_FAST)
        except requests.exceptions.Timeout:
            self._last_http_error = "timeout"
            return None
        except requests.exceptions.ConnectionError as err:
            self._last_http_error = f"connection_error: {err}"
            return None
        except Exception as err:
            self._last_http_error = f"{type(err).__name__}: {err}"
            return None

        status = response.status_code
        if status == 200:
            self._last_http_error = None
            return True
        if status in (401, 403):
            self._last_http_error = f"http_{status} (auth failure)"
            return False
        # 404/501/etc. -> caller should try next pattern
        self._last_http_error = f"http_{status}"
        return None

    def _send_ptz_command(self, code: str, speed: int = 5, action: str = 'start') -> bool:
        custom_url = self.config.get('custom_ptz_url') or ''
        auth = (self.username, self.password) if self.username and self.password else None

        # Custom URL takes precedence and is authoritative.
        if custom_url:
            url = custom_url.replace('{CMD}', code).replace('{SPEED}', str(speed))
            if not url.startswith('http'):
                url = self.base_url + url
            logger.info("Using custom PTZ URL: %s", url)
            result = self._attempt_http(url, auth)
            return bool(result)

        patterns = self._build_url_patterns(code, speed, action)

        # Cached working pattern first.
        if self._working_pattern_index is not None and 0 <= self._working_pattern_index < len(patterns):
            cached_url = patterns[self._working_pattern_index]
            logger.debug("Trying cached pattern %d: %s", self._working_pattern_index + 1, cached_url)
            result = self._attempt_http(cached_url, auth)
            if result is True:
                return True
            if result is False:
                # Auth failure -> no point trying other patterns
                return False
            # Otherwise fall through and probe the rest.

        for i, url in enumerate(patterns):
            if i == self._working_pattern_index:
                continue
            logger.debug("[%d/%d] Trying: %s", i + 1, len(patterns), url)
            result = self._attempt_http(url, auth)
            if result is True:
                self._working_pattern_index = i
                logger.info("PTZ pattern %d works: %s", i + 1, url)
                return True
            if result is False:
                # Auth failed, no use trying others
                return False

        logger.warning(
            "No working PTZ URL pattern found for %s (last_error=%s)",
            self.base_url, self._last_http_error,
        )
        return False

    # ------------------------------------------------------------------ #
    # Public API
    # ------------------------------------------------------------------ #

    def connect(self) -> bool:
        logger.info("Generic PTZ controller ready for %s", self.base_url)
        return True

    def execute_command(self, command: str, params: Dict[str, Any]) -> Dict[str, Any]:
        try:
            if command == 'pan':
                return self.pan(params.get('direction'), params.get('speed', 5))
            if command == 'tilt':
                return self.tilt(params.get('direction'), params.get('speed', 5))
            if command == 'zoom':
                return self.zoom(params.get('direction'), params.get('speed', 5))
            if command == 'stop':
                return self.stop()
            if command == 'home':
                return self.go_home()
            if command == 'preset':
                action = params.get('action')
                preset_id = params.get('presetId')
                if action == 'goto':
                    return self.goto_preset(preset_id)
                if action == 'set':
                    return self.set_preset(preset_id)
            return {'success': False, 'error': f'Unknown command: {command}'}
        except Exception as err:
            logger.error("Error executing PTZ command %s: %s", command, err)
            return {'success': False, 'error': str(err)}

    def _movement_response(self, success: bool, label: str) -> Dict[str, Any]:
        return {
            'success': bool(success),
            'message': f"{label} {'sent' if success else 'failed'}",
            'error': None if success else self._last_http_error,
            'position': self.current_position.copy(),
        }

    def pan(self, direction: str, speed: int = 5) -> Dict[str, Any]:
        if direction not in ('left', 'right'):
            return {'success': False, 'error': 'Invalid pan direction'}
        logger.info("Pan %s at speed %s", direction, speed)
        code = 'Left' if direction == 'left' else 'Right'
        success = self._send_ptz_command(code, speed)
        if success:
            delta = speed * 2
            if direction == 'left':
                self.current_position['pan'] = max(self.pan_range[0], self.current_position['pan'] - delta)
            else:
                self.current_position['pan'] = min(self.pan_range[1], self.current_position['pan'] + delta)
        return self._movement_response(success, f'Pan {direction}')

    def tilt(self, direction: str, speed: int = 5) -> Dict[str, Any]:
        if direction not in ('up', 'down'):
            return {'success': False, 'error': 'Invalid tilt direction'}
        logger.info("Tilt %s at speed %s", direction, speed)
        code = 'Up' if direction == 'up' else 'Down'
        success = self._send_ptz_command(code, speed)
        if success:
            delta = speed * 2
            if direction == 'up':
                self.current_position['tilt'] = min(self.tilt_range[1], self.current_position['tilt'] + delta)
            else:
                self.current_position['tilt'] = max(self.tilt_range[0], self.current_position['tilt'] - delta)
        return self._movement_response(success, f'Tilt {direction}')

    def zoom(self, direction: str, speed: int = 5) -> Dict[str, Any]:
        if direction not in ('in', 'out'):
            return {'success': False, 'error': 'Invalid zoom direction'}
        logger.info("Zoom %s at speed %s", direction, speed)
        code = 'ZoomTele' if direction == 'in' else 'ZoomWide'
        success = self._send_ptz_command(code, speed)
        if success:
            delta = speed * 0.2
            if direction == 'in':
                self.current_position['zoom'] = min(self.zoom_range[1], self.current_position['zoom'] + delta)
            else:
                self.current_position['zoom'] = max(self.zoom_range[0], self.current_position['zoom'] - delta)
        return self._movement_response(success, f'Zoom {direction}')

    def stop(self) -> Dict[str, Any]:
        logger.info("PTZ stop")
        success = self._send_ptz_command('Stop', action='stop')
        return {
            'success': bool(success),
            'message': 'PTZ stop ' + ('sent' if success else 'failed'),
            'error': None if success else self._last_http_error,
            'position': self.current_position.copy(),
        }

    def go_home(self) -> Dict[str, Any]:
        logger.info("PTZ go home")
        success = self._send_ptz_command('GotoHome', action='start')
        if success:
            home_position = self.presets.get('home', {'pan': 0, 'tilt': 0, 'zoom': 1})
            time.sleep(0.5)
            self.current_position = dict(home_position)
        return {
            'success': bool(success),
            'message': 'Home ' + ('sent' if success else 'failed'),
            'error': None if success else self._last_http_error,
            'position': self.current_position.copy(),
        }

    def goto_preset(self, preset_id: str) -> Dict[str, Any]:
        try:
            preset_num = int(preset_id)
        except Exception:
            preset_num = None

        # Locally stored named preset
        if preset_num is None and preset_id in self.presets:
            self.current_position = dict(self.presets[preset_id])
            return {
                'success': True,
                'message': f'Moved to preset {preset_id}',
                'position': self.current_position.copy(),
            }

        success = self._send_ptz_command('GotoPreset', speed=preset_num or 1)
        return {
            'success': bool(success),
            'message': f"Goto preset {preset_id} " + ('sent' if success else 'failed'),
            'error': None if success else self._last_http_error,
            'position': self.current_position.copy(),
        }

    def set_preset(self, preset_id: str) -> Dict[str, Any]:
        self.presets[preset_id] = self.current_position.copy()
        logger.info("Set preset %s to position: %s", preset_id, self.current_position)

        try:
            preset_num = int(preset_id)
            success = self._send_ptz_command('SetPreset', speed=preset_num)
        except Exception:
            success = True  # named preset, only stored locally

        return {
            'success': bool(success),
            'message': f'Preset {preset_id} ' + ('saved' if success else 'failed'),
            'error': None if success else self._last_http_error,
            'position': self.current_position.copy(),
        }

    def get_position(self) -> Dict[str, Any]:
        return {'success': True, 'position': self.current_position.copy()}

    def get_presets(self) -> Dict[str, Any]:
        return {'success': True, 'presets': self.presets.copy()}
