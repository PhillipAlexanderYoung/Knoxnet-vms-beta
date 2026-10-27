import base64
import cv2
import numpy as np
from typing import List, Dict, Any, Optional, Tuple


def draw_overlays(frame: np.ndarray,
                  zones: List[List[Dict[str, float]]],
                  lines: List[Dict[str, Dict[str, float]]],
                  tracks: List[Dict[str, Any]],
                  detections: List[Dict[str, Any]]) -> np.ndarray:
    img = frame.copy()
    h, w = img.shape[:2]

    # Zones: polygons
    for poly in zones or []:
        pts = np.array([[int(p['x']), int(p['y'])] for p in poly], np.int32)
        if len(pts) >= 3:
            cv2.polylines(img, [pts], isClosed=True, color=(36, 255, 255), thickness=2)

    # Lines
    for ln in lines or []:
        p1 = ln.get('p1', {})
        p2 = ln.get('p2', {})
        cv2.line(img, (int(p1.get('x', 0)), int(p1.get('y', 0))), (int(p2.get('x', 0)), int(p2.get('y', 0))), (0, 165, 255), 2)

    # Tracks: boxes only (no trajectory/history lines to avoid heavy overlays)
    for t in tracks or []:
        b = t.get('bbox', {})
        x, y, ww, hh = int(b.get('x', 0)), int(b.get('y', 0)), int(b.get('w', 0)), int(b.get('h', 0))
        cv2.rectangle(img, (x, y), (x+ww, y+hh), (0, 215, 255), 2)
        cv2.putText(img, f"ID {t.get('id')}", (x, max(0, y-6)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,0), 2)
        cv2.putText(img, f"ID {t.get('id')}", (x, max(0, y-6)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,215,255), 1)

    # Detections
    for d in detections or []:
        b = d.get('bbox', {})
        x, y, ww, hh = int(b.get('x', 0)), int(b.get('y', 0)), int(b.get('w', 0)), int(b.get('h', 0))
        cv2.rectangle(img, (x, y), (x+ww, y+hh), (50, 205, 50), 2)
        label = f"{d.get('class_name','obj')} {int((d.get('confidence',0))*100)}%"
        cv2.putText(img, label, (x, max(0, y-8)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,0), 2)
        cv2.putText(img, label, (x, max(0, y-8)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (50,205,50), 1)

    return img


def encode_image_base64(img: np.ndarray, fmt: str = '.jpg', quality: int = 85) -> str:
    params = []
    if fmt == '.jpg':
        params = [int(cv2.IMWRITE_JPEG_QUALITY), int(quality)]
    _, buf = cv2.imencode(fmt, img, params)
    return 'data:image/jpeg;base64,' + base64.b64encode(buf.tobytes()).decode('utf-8')


# Composite utilities

def build_track_composite(
    frame: np.ndarray,
    track_history: List[Tuple[float, float]],
    current_bbox: Tuple[int, int, int, int],
    tiles: int = 4,
    crop_margin: int = 8
) -> np.ndarray:
    """
    Build a small composite image for a track using current bbox and a few historical positions.

    The composite is a grid of tiles (2x2 by default) showing crops around the object's
    historical positions for temporal context. This helps detectors stabilize classification.
    """
    img_h, img_w = frame.shape[:2]
    num_tiles = max(1, int(tiles))
    grid = int(num_tiles ** 0.5)
    if grid * grid < num_tiles:
        grid += 1

    # Determine crop boxes: current + up to (num_tiles-1) previous centers
    crops: List[np.ndarray] = []
    x, y, w, h = current_bbox
    def clamp_rect(xx, yy, ww, hh):
        xx = max(0, min(xx, img_w - 1))
        yy = max(0, min(yy, img_h - 1))
        ww = max(1, min(ww, img_w - xx))
        hh = max(1, min(hh, img_h - yy))
        return xx, yy, ww, hh

    # Current crop with margin
    mx = max(0, x - crop_margin)
    my = max(0, y - crop_margin)
    mw = min(img_w - mx, w + crop_margin * 2)
    mh = min(img_h - my, h + crop_margin * 2)
    mx, my, mw, mh = clamp_rect(mx, my, mw, mh)
    crops.append(frame[my:my+mh, mx:mx+mw])

    # Historical crops around past centers (reverse order, latest first)
    for cx, cy in list(track_history)[-num_tiles:][::-1]:
        hx = int(cx - w / 2)
        hy = int(cy - h / 2)
        hx = max(0, hx - crop_margin)
        hy = max(0, hy - crop_margin)
        hw = min(img_w - hx, w + crop_margin * 2)
        hh = min(img_h - hy, h + crop_margin * 2)
        hx, hy, hw, hh = clamp_rect(hx, hy, hw, hh)
        crop = frame[hy:hy+hh, hx:hx+hw]
        if crop.size > 0:
            crops.append(crop)
        if len(crops) >= num_tiles:
            break

    # Normalize crop sizes and build grid
    tile_w = 128
    tile_h = 128
    canvas = np.zeros((tile_h * grid, tile_w * grid, 3), dtype=frame.dtype)
    i = 0
    for r in range(grid):
        for c in range(grid):
            if i >= len(crops):
                break
            tile = cv2.resize(crops[i], (tile_w, tile_h), interpolation=cv2.INTER_LINEAR)
            y0 = r * tile_h
            x0 = c * tile_w
            canvas[y0:y0+tile_h, x0:x0+tile_w] = tile
            i += 1

    return canvas

