"""Which physical cameras macOS has right now (light module: no ML imports)."""
from __future__ import annotations

import json
import subprocess
from typing import Optional

# The door camera is the Mac's built-in webcam, found by name. Never the iPhone.
DOOR_CAMERA_NAME = ("FaceTime", "MacBook")


def camera_names() -> list[str]:
    """Names of the cameras macOS currently sees."""
    try:
        out = subprocess.run(["system_profiler", "SPCameraDataType", "-json"],
                             capture_output=True, text=True, timeout=10).stdout
        return [c.get("_name", "") for c in json.loads(out).get("SPCameraDataType", [])]
    except Exception:
        return []


def door_camera_index() -> Optional[int]:
    """OpenCV index of the Mac's built-in webcam, or None if there isn't one.

    OpenCV lists Continuity/external cameras (the iPhone) first and the built-in camera after
    them, so the built-in camera's index is the number of non-built-in cameras."""
    names = camera_names()
    door = [n for n in names
            if "iphone" not in n.lower() and any(k.lower() in n.lower() for k in DOOR_CAMERA_NAME)]
    if not door:
        return None
    return len(names) - len(door)
