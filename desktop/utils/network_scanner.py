"""Fast network camera scanner with encoder/multi-channel awareness.

Strategy (completes in seconds, not minutes):

1. ARP-first discovery: Read the OS ARP cache, then do a single broadcast
   ping to populate it. Finds live hosts in ~1 second.

2. RTSP-port-first filtering: Only probe port 554 (and 8554/7447) with a
   very short TCP connect timeout. Hosts without an RTSP port open are
   almost certainly not cameras.

3. Parallel everything: Port checks, vendor lookups, and RTSP URL probes
   all run concurrently with a large thread pool.

4. Vendor-aware RTSP: When a vendor is identified via MAC OUI, auto-fill
   RTSP URL template, default credentials, and port.

5. Encoder / multi-channel detection: When port 80/8080 is open alongside
   RTSP, or a known encoder vendor is detected, probe for multiple channels.
"""

from __future__ import annotations

import logging
import re
import socket
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Dict, List, Optional, Tuple

import psutil
from PySide6.QtCore import QThread, Signal

logger = logging.getLogger(__name__)

# ═══════════════════════════════════════════════════════════════════════════
#  Vendor OUI database  (100+ prefixes)
# ═══════════════════════════════════════════════════════════════════════════

CAMERA_OUI_PREFIXES: Dict[str, str] = {
    # ── Hikvision ────────────────────────────────────────────────────────
    "44:2C:05": "Hikvision", "BC:51:FE": "Hikvision", "9C:8E:CD": "Hikvision",
    "C0:56:E3": "Hikvision", "18:68:CB": "Hikvision", "54:C4:15": "Hikvision",
    "A4:14:37": "Hikvision", "28:57:BE": "Hikvision", "C4:2F:90": "Hikvision",
    "80:1F:12": "Hikvision", "E0:50:8B": "Hikvision", "4C:BD:8F": "Hikvision",
    "C0:51:7E": "Hikvision", "EC:F2:2C": "Hikvision", "40:6A:8E": "Hikvision",
    "D4:43:A8": "Hikvision", "A0:E4:CB": "Hikvision", "68:C6:3A": "Hikvision",
    # ── Ezviz (Hikvision consumer) ───────────────────────────────────────
    "50:3A:A0": "Ezviz", "80:76:55": "Ezviz", "CC:D2:71": "Ezviz",
    # ── Dahua ────────────────────────────────────────────────────────────
    "38:AF:29": "Dahua", "94:9F:3F": "Dahua", "3C:EF:8C": "Dahua",
    "A0:BD:1D": "Dahua", "40:2C:76": "Dahua", "E0:24:7F": "Dahua",
    "B0:C5:CA": "Dahua", "D0:48:02": "Dahua", "DC:BF:E9": "Dahua",
    "78:A7:14": "Dahua", "14:A7:8B": "Dahua", "20:17:42": "Dahua",
    # ── Lorex (Dahua OEM) ────────────────────────────────────────────────
    "F4:C7:14": "Lorex", "A8:4E:3F": "Lorex",
    # ── Amcrest (Dahua OEM) ──────────────────────────────────────────────
    "FC:AD:0F": "Amcrest", "9C:8E:CD": "Amcrest",
    # ── Swann (Dahua OEM) ────────────────────────────────────────────────
    "EC:71:DB": "Swann",
    # ── Annke / ZOSI / SANNCE (HiSilicon/Dahua OEM) ─────────────────────
    "00:42:52": "Annke",
    # ── Axis ─────────────────────────────────────────────────────────────
    "64:16:7F": "Axis", "00:40:8C": "Axis", "AC:CC:8E": "Axis",
    "B8:A4:4F": "Axis", "00:1A:07": "Axis",
    # ── Ubiquiti / UniFi Protect ─────────────────────────────────────────
    "24:A4:3C": "Ubiquiti", "44:D9:E7": "Ubiquiti", "FC:EC:DA": "Ubiquiti",
    "78:8A:20": "Ubiquiti", "74:AC:B9": "Ubiquiti", "68:72:51": "Ubiquiti",
    "18:E8:29": "Ubiquiti", "B4:FB:E4": "Ubiquiti", "24:5A:4C": "Ubiquiti",
    "F0:9F:C2": "Ubiquiti", "80:2A:A8": "Ubiquiti",
    # ── Reolink ──────────────────────────────────────────────────────────
    "54:62:66": "Reolink", "EC:71:DB": "Reolink", "DC:54:75": "Reolink",
    "C8:D7:B0": "Reolink",
    # ── Wyze ─────────────────────────────────────────────────────────────
    "2C:AA:8E": "Wyze", "7C:78:B2": "Wyze", "D0:3F:27": "Wyze",
    # ── Hanwha / Samsung Wisenet ─────────────────────────────────────────
    "00:09:18": "Hanwha", "00:16:6C": "Hanwha", "00:68:EB": "Hanwha",
    "F4:E1:1E": "Hanwha", "C8:B2:1E": "Hanwha",
    # ── Vivotek ──────────────────────────────────────────────────────────
    "00:02:D1": "Vivotek", "00:22:F7": "Vivotek",
    # ── Bosch ────────────────────────────────────────────────────────────
    "00:07:5F": "Bosch", "00:04:13": "Bosch",
    # ── Pelco ────────────────────────────────────────────────────────────
    "00:07:83": "Pelco",
    # ── GeoVision ────────────────────────────────────────────────────────
    "00:13:E2": "GeoVision",
    # ── TP-Link / Tapo ───────────────────────────────────────────────────
    "98:25:4A": "TP-Link", "5C:A6:E6": "TP-Link", "60:A4:B7": "TP-Link",
    "B0:A7:B9": "TP-Link", "14:EB:B6": "TP-Link",
    # ── FLIR ─────────────────────────────────────────────────────────────
    "00:40:7F": "FLIR", "70:71:BC": "FLIR",
    # ── Honeywell ────────────────────────────────────────────────────────
    "00:1F:C6": "Honeywell", "00:D0:23": "Honeywell",
    # ── Avigilon ─────────────────────────────────────────────────────────
    "00:18:85": "Avigilon",
    # ── Mobotix ──────────────────────────────────────────────────────────
    "00:17:FC": "Mobotix",
    # ── Sony ─────────────────────────────────────────────────────────────
    "00:04:1F": "Sony", "00:80:92": "Sony", "A8:E3:EE": "Sony",
    # ── Panasonic / i-PRO ────────────────────────────────────────────────
    "00:80:F0": "Panasonic", "00:B0:C7": "Panasonic", "80:ED:2C": "Panasonic",
    "08:60:6E": "Panasonic",
    # ── Arecont Vision ───────────────────────────────────────────────────
    "00:26:74": "ArecontVision",
    # ── Grandstream ──────────────────────────────────────────────────────
    "00:0B:82": "Grandstream",
    # ── Uniview ──────────────────────────────────────────────────────────
    "24:24:05": "Uniview", "24:16:9D": "Uniview", "74:DA:88": "Uniview",
    # ── Tiandy ───────────────────────────────────────────────────────────
    "00:12:E8": "Tiandy",
    # ── ACTi ─────────────────────────────────────────────────────────────
    "00:0E:53": "ACTi",
    # ── March Networks ───────────────────────────────────────────────────
    "00:13:5F": "MarchNetworks",
    # ── Digital Watchdog ─────────────────────────────────────────────────
    "00:1C:27": "DigitalWatchdog",
    # ── Milesight ────────────────────────────────────────────────────────
    "10:12:FB": "Milesight",
    # ── Verkada ──────────────────────────────────────────────────────────
    "B4:A3:82": "Verkada",
    # ── Nest / Google ────────────────────────────────────────────────────
    "18:B4:30": "Google", "64:16:66": "Google",
    # ── Ring / Amazon ────────────────────────────────────────────────────
    "34:D2:70": "Ring", "9C:53:22": "Ring",
    # ── Eufy / Anker ─────────────────────────────────────────────────────
    "98:83:89": "Eufy",
    # ── Arlo / Netgear ───────────────────────────────────────────────────
    "9C:B7:0D": "Arlo",
}

RTSP_PORTS = [554, 8554, 7447]

SECONDARY_PORTS = [80, 443, 8000, 8080, 1935, 37777, 34567]

# ═══════════════════════════════════════════════════════════════════════════
#  RTSP templates with {user}, {password}, {ip}, {port}, {channel}
# ═══════════════════════════════════════════════════════════════════════════

VENDOR_RTSP_TEMPLATES: Dict[str, List[str]] = {
    "Hikvision": [
        "rtsp://{user}:{password}@{ip}:{port}/Streaming/Channels/{channel}01",
        "rtsp://{user}:{password}@{ip}:{port}/h264/ch{channel}/main/av_stream",
        "rtsp://{user}:{password}@{ip}:{port}/ISAPI/streaming/channels/{channel}01",
    ],
    "Ezviz": [
        "rtsp://{user}:{password}@{ip}:{port}/Streaming/Channels/{channel}01",
        "rtsp://{user}:{password}@{ip}:{port}/h264/ch{channel}/main/av_stream",
    ],
    "Dahua": [
        "rtsp://{user}:{password}@{ip}:{port}/cam/realmonitor?channel={channel}&subtype=0",
        "rtsp://{user}:{password}@{ip}:{port}/cam/realmonitor?channel={channel}&subtype=1",
    ],
    "Amcrest": [
        "rtsp://{user}:{password}@{ip}:{port}/cam/realmonitor?channel={channel}&subtype=0",
    ],
    "Lorex": [
        "rtsp://{user}:{password}@{ip}:{port}/cam/realmonitor?channel={channel}&subtype=0",
    ],
    "Swann": [
        "rtsp://{user}:{password}@{ip}:{port}/cam/realmonitor?channel={channel}&subtype=0",
    ],
    "Annke": [
        "rtsp://{user}:{password}@{ip}:{port}/Streaming/Channels/{channel}01",
    ],
    "Reolink": [
        "rtsp://{user}:{password}@{ip}:{port}/h264Preview_{channel:02d}_main",
        "rtsp://{user}:{password}@{ip}:{port}/Preview_{channel:02d}_main",
    ],
    "Axis": [
        "rtsp://{user}:{password}@{ip}:{port}/axis-media/media.amp?camera={channel}",
        "rtsp://{user}:{password}@{ip}:{port}/axis-media/media.amp?videocodec=h264&camera={channel}",
    ],
    "Ubiquiti": [
        "rtsps://{user}:{password}@{ip}:7441/{channel_hex}",
        "rtsp://{user}:{password}@{ip}:7447/{channel_hex}",
    ],
    "Hanwha": [
        "rtsp://{user}:{password}@{ip}:{port}/profile{channel}/media.smp",
        "rtsp://{user}:{password}@{ip}:{port}/stw-cgi/video.cgi?msubmenu=stream&action=view&channel={channel}",
    ],
    "Vivotek": [
        "rtsp://{user}:{password}@{ip}:{port}/live.sdp",
        "rtsp://{user}:{password}@{ip}:{port}/live{channel}.sdp",
    ],
    "Bosch": [
        "rtsp://{user}:{password}@{ip}:{port}/rtsp_tunnel",
        "rtsp://{user}:{password}@{ip}:{port}/video{channel}",
    ],
    "TP-Link": [
        "rtsp://{user}:{password}@{ip}:{port}/stream1",
        "rtsp://{user}:{password}@{ip}:{port}/stream2",
    ],
    "FLIR": [
        "rtsp://{user}:{password}@{ip}:{port}/avc",
        "rtsp://{user}:{password}@{ip}:{port}/ch0{channel}",
    ],
    "Honeywell": [
        "rtsp://{user}:{password}@{ip}:{port}/VideoInput/{channel}/h264/1",
    ],
    "Avigilon": [
        "rtsp://{user}:{password}@{ip}:{port}/defaultPrimary?streamType=u",
    ],
    "Mobotix": [
        "rtsp://{user}:{password}@{ip}:{port}/mobotix.h264",
    ],
    "Sony": [
        "rtsp://{user}:{password}@{ip}:{port}/media/video{channel}",
        "rtsp://{user}:{password}@{ip}:{port}/image{channel}",
    ],
    "Panasonic": [
        "rtsp://{user}:{password}@{ip}:{port}/MediaInput/h264/stream_{channel}",
        "rtsp://{user}:{password}@{ip}:{port}/nphMotionJpeg?Resolution=640x480",
    ],
    "ArecontVision": [
        "rtsp://{user}:{password}@{ip}:{port}/h264.sdp",
    ],
    "Grandstream": [
        "rtsp://{user}:{password}@{ip}:{port}/0",
        "rtsp://{user}:{password}@{ip}:{port}/{channel}",
    ],
    "Uniview": [
        "rtsp://{user}:{password}@{ip}:{port}/unicast/c{channel}/s0/live",
        "rtsp://{user}:{password}@{ip}:{port}/media/video{channel}",
    ],
    "Tiandy": [
        "rtsp://{user}:{password}@{ip}:{port}/Streaming/Channels/{channel}01",
    ],
    "ACTi": [
        "rtsp://{user}:{password}@{ip}:{port}/ISAPI/streaming/channels/{channel}01",
    ],
    "MarchNetworks": [
        "rtsp://{user}:{password}@{ip}:{port}/live/ch{channel:02d}_0",
    ],
    "DigitalWatchdog": [
        "rtsp://{user}:{password}@{ip}:{port}/ch{channel}/0",
    ],
    "Milesight": [
        "rtsp://{user}:{password}@{ip}:{port}/main",
    ],
    "Wyze": [
        "rtsp://{user}:{password}@{ip}:{port}/live",
    ],
    "default": [
        "rtsp://{user}:{password}@{ip}:{port}/stream1",
        "rtsp://{user}:{password}@{ip}:{port}/live",
        "rtsp://{user}:{password}@{ip}:{port}/ch{channel}/0",
        "rtsp://{user}:{password}@{ip}:{port}/Streaming/Channels/{channel}01",
        "rtsp://{user}:{password}@{ip}:{port}",
    ],
}

VENDOR_DEFAULT_CREDENTIALS: Dict[str, List[Tuple[str, str]]] = {
    "Hikvision": [("admin", "admin12345"), ("admin", "12345"), ("admin", "")],
    "Ezviz": [("admin", ""), ("admin", "admin")],
    "Dahua": [("admin", "admin"), ("admin", "admin123"), ("admin", "")],
    "Amcrest": [("admin", "admin"), ("admin", "")],
    "Lorex": [("admin", "admin"), ("admin", "")],
    "Swann": [("admin", "admin"), ("admin", "12345")],
    "Annke": [("admin", "admin"), ("admin", "")],
    "Reolink": [("admin", ""), ("admin", "admin")],
    "Axis": [("root", "pass"), ("root", "root"), ("root", "")],
    "Ubiquiti": [("ubnt", "ubnt"), ("admin", "admin")],
    "Hanwha": [("admin", "4321"), ("admin", "admin")],
    "Vivotek": [("root", ""), ("root", "root")],
    "Bosch": [("service", "service"), ("admin", "admin")],
    "TP-Link": [("admin", "admin"), ("admin", "")],
    "FLIR": [("admin", "admin"), ("admin", "")],
    "Honeywell": [("admin", "1234"), ("admin", "admin")],
    "Avigilon": [("admin", "admin"), ("administrator", "")],
    "Mobotix": [("admin", "meinsm"), ("admin", "")],
    "Sony": [("admin", "admin"), ("admin", "")],
    "Panasonic": [("admin", "12345"), ("admin", "admin")],
    "ArecontVision": [("admin", ""), ("admin", "admin")],
    "Grandstream": [("admin", "admin"), ("admin", "")],
    "Uniview": [("admin", "123456"), ("admin", "admin")],
    "Tiandy": [("admin", "admin123"), ("admin", "admin")],
    "ACTi": [("Admin", "123456"), ("admin", "admin")],
    "MarchNetworks": [("admin", "admin")],
    "DigitalWatchdog": [("admin", "admin"), ("admin", "")],
    "Milesight": [("admin", "ms1234")],
    "Wyze": [("admin", "")],
    "default": [("admin", "admin"), ("admin", "12345"), ("admin", ""), ("root", ""), ("root", "root")],
}

VENDOR_DEFAULT_PORTS: Dict[str, int] = {
    "Ubiquiti": 7447,
    "default": 554,
}

VENDOR_ENCODER_HINTS: Dict[str, dict] = {
    "Hikvision": {"max_probe": 32, "likely_multi": True},
    "Ezviz":     {"max_probe": 8,  "likely_multi": False},
    "Dahua":     {"max_probe": 64, "likely_multi": True},
    "Amcrest":   {"max_probe": 16, "likely_multi": True},
    "Lorex":     {"max_probe": 16, "likely_multi": True},
    "Swann":     {"max_probe": 16, "likely_multi": True},
    "Annke":     {"max_probe": 16, "likely_multi": True},
    "Axis":      {"max_probe": 16, "likely_multi": True},
    "Reolink":   {"max_probe": 16, "likely_multi": True},
    "Uniview":   {"max_probe": 32, "likely_multi": True},
    "Hanwha":    {"max_probe": 16, "likely_multi": True},
}

# ═══════════════════════════════════════════════════════════════════════════
#  Helpers
# ═══════════════════════════════════════════════════════════════════════════

_NO_WINDOW = subprocess.CREATE_NO_WINDOW if sys.platform == "win32" else 0


def _normalize_mac(mac: Optional[str]) -> str:
    if not mac:
        return ""
    return mac.upper().replace("-", ":")


def _lookup_vendor(mac: str) -> Optional[str]:
    mac_norm = _normalize_mac(mac)
    if len(mac_norm) < 8:
        return None
    return CAMERA_OUI_PREFIXES.get(mac_norm[:8])


def _local_subnets() -> List[str]:
    """Return /24 network prefixes from host interfaces."""
    prefixes: List[str] = []
    try:
        for _name, addrs in psutil.net_if_addrs().items():
            for a in addrs:
                if a.family != socket.AF_INET:
                    continue
                ip = a.address
                if ip.startswith("127.") or ip.startswith("169.254."):
                    continue
                octets = ip.split(".")
                if len(octets) == 4:
                    prefix = ".".join(octets[:3])
                    if prefix not in prefixes:
                        prefixes.append(prefix)
    except Exception as exc:
        logger.warning("Failed to enumerate interfaces: %s", exc)
    return prefixes


def _populate_arp_cache(subnets: List[str]) -> None:
    """Broadcast-ping each subnet to fill the OS ARP table quickly."""
    for prefix in subnets:
        broadcast = f"{prefix}.255"
        try:
            subprocess.run(
                ["ping", "-n" if sys.platform == "win32" else "-c", "1",
                 "-w" if sys.platform == "win32" else "-W",
                 "200" if sys.platform == "win32" else "1",
                 broadcast],
                stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
                timeout=2, creationflags=_NO_WINDOW,
            )
        except Exception:
            pass


def _read_arp_table() -> Dict[str, str]:
    """Parse ``arp -a`` into {ip: mac}."""
    ip_mac: Dict[str, str] = {}
    try:
        out = subprocess.check_output(
            ["arp", "-a"], timeout=10, text=True,
            stderr=subprocess.DEVNULL, creationflags=_NO_WINDOW,
        )
        ip_re = re.compile(r"(\d+\.\d+\.\d+\.\d+)")
        mac_re = re.compile(r"([0-9a-fA-F]{2}[:\-]){5}[0-9a-fA-F]{2}")
        for line in out.splitlines():
            ip_m = ip_re.search(line)
            mac_m = mac_re.search(line)
            if ip_m and mac_m:
                mac_str = _normalize_mac(mac_m.group())
                if mac_str != "FF:FF:FF:FF:FF:FF":
                    ip_mac[ip_m.group()] = mac_str
    except Exception as exc:
        logger.debug("ARP table read failed: %s", exc)
    return ip_mac


def _tcp_connect(ip: str, port: int, timeout: float = 0.3) -> bool:
    """Fast non-blocking TCP connect test."""
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    sock.settimeout(timeout)
    try:
        return sock.connect_ex((ip, port)) == 0
    except Exception:
        return False
    finally:
        try:
            sock.close()
        except Exception:
            pass


def _check_rtsp_ports(ip: str, timeout: float = 0.3) -> List[int]:
    return [p for p in RTSP_PORTS if _tcp_connect(ip, p, timeout)]


def _check_secondary_ports(ip: str, timeout: float = 0.3) -> List[int]:
    return [p for p in SECONDARY_PORTS if _tcp_connect(ip, p, timeout)]


def _score_device(open_ports: List[int], vendor: Optional[str]) -> Tuple[int, bool]:
    has_rtsp = bool(set(open_ports) & {554, 8554, 7447})
    vendor_bonus = 3 if vendor else 0
    port_score = len(open_ports) * 2
    score = port_score + vendor_bonus
    likely_camera = has_rtsp or (vendor is not None and score >= 3)
    return score, likely_camera


# ═══════════════════════════════════════════════════════════════════════════
#  RTSP URL building
# ═══════════════════════════════════════════════════════════════════════════

def build_rtsp_url(
    vendor: str,
    ip: str,
    port: int = 554,
    channel: int = 1,
    user: str = "",
    password: str = "",
    template_index: int = 0,
) -> str:
    """Build a complete RTSP URL from vendor template + parameters."""
    templates = VENDOR_RTSP_TEMPLATES.get(vendor, [])
    if not templates:
        templates = VENDOR_RTSP_TEMPLATES["default"]
    idx = min(template_index, len(templates) - 1)
    tmpl = templates[idx]

    cred_prefix = ""
    if user:
        cred_prefix = f"{user}:{password}@" if password else f"{user}@"

    try:
        url = tmpl.format(
            ip=ip, port=port, channel=channel, channel_hex=f"{channel:x}",
            user=user, password=password,
        )
    except (KeyError, IndexError, ValueError):
        url = f"rtsp://{cred_prefix}{ip}:{port}"

    if not user and "@" in url:
        url = re.sub(r"rtsp://[^@]*@", "rtsp://", url)

    return url


def get_vendor_defaults(vendor: str) -> dict:
    """Return default port, first credential pair, and first RTSP template for a vendor."""
    creds = VENDOR_DEFAULT_CREDENTIALS.get(vendor, VENDOR_DEFAULT_CREDENTIALS["default"])
    user, pw = creds[0] if creds else ("admin", "")
    port = VENDOR_DEFAULT_PORTS.get(vendor, VENDOR_DEFAULT_PORTS["default"])
    return {"user": user, "password": pw, "port": port}


# ═══════════════════════════════════════════════════════════════════════════
#  Connection testing
# ═══════════════════════════════════════════════════════════════════════════

def _probe_rtsp(url: str, timeout_ms: int = 2000) -> bool:
    """Try to open an RTSP stream; returns True on success."""
    try:
        import cv2
        cap = cv2.VideoCapture(url, cv2.CAP_FFMPEG)
        cap.set(cv2.CAP_PROP_OPEN_TIMEOUT_MSEC, timeout_ms)
        cap.set(cv2.CAP_PROP_READ_TIMEOUT_MSEC, timeout_ms)
        ok = cap.isOpened()
        cap.release()
        return ok
    except Exception:
        return False


def _probe_rtsp_with_creds(ip: str, port: int, path: str, user: str, pw: str, timeout: float = 2.0) -> bool:
    """Probe an RTSP stream with credentials embedded in the URL."""
    cred = f"{user}:{pw}@" if user else ""
    url = f"rtsp://{cred}{ip}:{port}{path}"
    return _probe_rtsp(url, timeout_ms=int(timeout * 1000))


def test_rtsp_connection(url: str, timeout_ms: int = 3000) -> dict:
    """Test an RTSP URL and return rich metadata.

    Returns dict with keys:
        ok (bool), width (int), height (int), fps (float), error (str)
    """
    result = {"ok": False, "width": 0, "height": 0, "fps": 0.0, "error": ""}
    try:
        import cv2
        cap = cv2.VideoCapture(url, cv2.CAP_FFMPEG)
        cap.set(cv2.CAP_PROP_OPEN_TIMEOUT_MSEC, timeout_ms)
        cap.set(cv2.CAP_PROP_READ_TIMEOUT_MSEC, timeout_ms)

        if not cap.isOpened():
            result["error"] = "Connection refused or timeout"
            return result

        w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
        h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)
        fps = cap.get(cv2.CAP_PROP_FPS) or 0.0

        if w == 0 or h == 0:
            ret, frame = cap.read()
            if ret and frame is not None:
                h, w = frame.shape[:2]
                result["ok"] = True
            else:
                result["error"] = "Stream opened but no frames"
        else:
            result["ok"] = True

        result["width"] = w
        result["height"] = h
        result["fps"] = round(fps, 1)
        cap.release()
    except Exception as exc:
        result["error"] = str(exc)[:80]
    return result


def enumerate_channels(
    ip: str,
    port: int,
    vendor: str,
    user: str = "",
    password: str = "",
    max_probe: int = 16,
    timeout_ms: int = 2000,
) -> List[int]:
    """Probe channels 1..N on an encoder/NVR and return list of valid channel numbers."""
    found: List[int] = []
    consecutive_failures = 0

    for ch in range(1, max_probe + 1):
        url = build_rtsp_url(vendor, ip, port, ch, user, password)
        if _probe_rtsp(url, timeout_ms=timeout_ms):
            found.append(ch)
            consecutive_failures = 0
        else:
            consecutive_failures += 1
            if consecutive_failures >= 2 and ch > 2:
                break

    return found


# ═══════════════════════════════════════════════════════════════════════════
#  QThread scanner
# ═══════════════════════════════════════════════════════════════════════════

class NetworkCameraScanner(QThread):
    """Fast camera scanner using ARP + RTSP-port-first strategy.

    Typical scan time: 3-8 seconds for a /24 subnet.
    """

    progress = Signal(str)
    device_found = Signal(dict)
    finished_all = Signal(list)
    error = Signal(str)

    def __init__(self, port_timeout: float = 0.3, rtsp_probe: bool = True,
                 max_workers: int = 80, parent=None):
        super().__init__(parent)
        self.port_timeout = max(0.1, min(port_timeout, 1.0))
        self.rtsp_probe = rtsp_probe
        self.max_workers = max_workers
        self._stop_requested = False

    def request_stop(self):
        self._stop_requested = True

    def run(self):
        try:
            self._stop_requested = False
            results: List[dict] = []

            self.progress.emit("Detecting network interfaces\u2026")
            subnets = _local_subnets()
            if not subnets:
                self.progress.emit("No usable network interfaces found.")
                self.finished_all.emit([])
                return

            self.progress.emit(
                f"Scanning {', '.join(s + '.0/24' for s in subnets)}"
            )

            _populate_arp_cache(subnets)
            arp_table = _read_arp_table()

            arp_ips = set()
            for ip in arp_table:
                for prefix in subnets:
                    if ip.startswith(prefix + "."):
                        arp_ips.add(ip)

            all_ips: List[str] = []
            for prefix in subnets:
                all_ips.extend(f"{prefix}.{i}" for i in range(1, 255))

            self.progress.emit(f"Probing {len(all_ips)} addresses for RTSP ports\u2026")

            arp_first = sorted(arp_ips)
            remaining = [ip for ip in all_ips if ip not in arp_ips]
            ordered_ips = arp_first + remaining

            candidates: List[dict] = []
            scanned = 0
            total = len(ordered_ips)

            with ThreadPoolExecutor(max_workers=self.max_workers) as pool:
                future_to_ip = {
                    pool.submit(_check_rtsp_ports, ip, self.port_timeout): ip
                    for ip in ordered_ips
                }

                for future in as_completed(future_to_ip):
                    if self._stop_requested:
                        return
                    scanned += 1
                    ip = future_to_ip[future]
                    try:
                        rtsp_ports = future.result()
                    except Exception:
                        rtsp_ports = []

                    if scanned % 40 == 0 or rtsp_ports:
                        self.progress.emit(
                            f"Scanning\u2026 {scanned}/{total}  "
                            f"({len(candidates)} camera{'s' if len(candidates) != 1 else ''} found)"
                        )

                    if not rtsp_ports:
                        mac = arp_table.get(ip, "")
                        vendor = _lookup_vendor(mac)
                        if not vendor:
                            continue
                        candidates.append({
                            "ip": ip, "mac": mac, "vendor": vendor,
                            "rtsp_ports": [], "from_arp": ip in arp_ips,
                        })
                    else:
                        mac = arp_table.get(ip, "")
                        vendor = _lookup_vendor(mac)
                        candidates.append({
                            "ip": ip, "mac": mac, "vendor": vendor,
                            "rtsp_ports": rtsp_ports, "from_arp": ip in arp_ips,
                        })

            if self._stop_requested:
                return

            self.progress.emit(
                f"Found {len(candidates)} candidate(s). Identifying\u2026"
            )

            for idx, cand in enumerate(candidates, 1):
                if self._stop_requested:
                    return

                ip = cand["ip"]
                mac = cand["mac"]
                vendor = cand["vendor"]
                rtsp_ports = cand["rtsp_ports"]

                secondary = []
                if rtsp_ports:
                    secondary = _check_secondary_ports(ip, self.port_timeout)

                all_ports = sorted(set(rtsp_ports + secondary))
                score, likely = _score_device(all_ports, vendor)

                has_http = bool(set(all_ports) & {80, 8080, 443})
                encoder_hint = VENDOR_ENCODER_HINTS.get(vendor or "", {})
                likely_encoder = (
                    encoder_hint.get("likely_multi", False)
                    and (has_http or bool(rtsp_ports))
                )

                defaults = get_vendor_defaults(vendor or "default")
                best_port = rtsp_ports[0] if rtsp_ports else defaults["port"]

                max_ch = encoder_hint.get("max_probe", 1)

                # ── Auto-probe channels for likely encoders ──────
                channels_found: List[int] = [1]
                if likely_encoder and likely and best_port:
                    self.progress.emit(
                        f"Probing encoder channels on {ip} "
                        f"({vendor or 'unknown'})\u2026"
                    )
                    try:
                        ch_list = enumerate_channels(
                            ip, best_port, vendor or "default",
                            defaults["user"], defaults["password"],
                            max_probe=min(max_ch, 16),
                            timeout_ms=2000,
                        )
                        if ch_list:
                            channels_found = ch_list
                    except Exception as exc:
                        logger.debug("Channel probe failed for %s: %s", ip, exc)

                is_encoder = len(channels_found) > 1

                for ch in channels_found:
                    rtsp_url = build_rtsp_url(
                        vendor or "default", ip, best_port, ch,
                        defaults["user"], defaults["password"],
                    )
                    device = {
                        "ip": ip,
                        "mac": mac,
                        "manufacturer": vendor or "",
                        "open_ports": all_ports,
                        "rtsp_ports": rtsp_ports,
                        "score": score,
                        "likely_camera": likely,
                        "likely_encoder": is_encoder,
                        "is_encoder": is_encoder,
                        "encoder_channels": len(channels_found),
                        "rtsp_url": rtsp_url,
                        "rtsp_port": best_port,
                        "default_user": defaults["user"],
                        "default_password": defaults["password"],
                        "channel": ch,
                        "max_channels": max_ch,
                    }
                    results.append(device)
                    self.device_found.emit(device)

            results.sort(key=lambda d: (
                -int(d.get("likely_camera", False)),
                -d.get("score", 0),
                d.get("ip", ""),
                d.get("channel", 1),
            ))
            self.finished_all.emit(results)

        except Exception as exc:
            logger.exception("Network scan failed")
            self.error.emit(str(exc))
