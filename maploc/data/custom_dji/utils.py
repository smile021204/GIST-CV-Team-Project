"""poses.txt loader for the custom DJI dataset.

poses.txt format (whitespace separated, one header line skipped if it starts
with '<'):

    <filename> <lat> <lon> <alt> <roll_deg> <pitch_deg> <yaw_deg>

Yaw follows DJI convention: clockwise from North, degrees.
"""

import json
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np


def parse_poses_file(path: Path) -> Dict[str, dict]:
    out = {}
    for line in path.read_text().strip().splitlines():
        if not line or line.startswith("#") or line.startswith("<"):
            continue
        parts = line.split()
        name = parts[0]
        lat, lon, alt = map(float, parts[1:4])
        roll, pitch, yaw = map(float, parts[4:7])
        out[name] = dict(
            latlon=np.array([lat, lon], dtype=np.float64),
            alt=alt,
            roll=roll,
            pitch=pitch,
            yaw=yaw,
        )
    return out


def load_intrinsics(path: Path) -> dict:
    cfg = json.loads(path.read_text())
    assert cfg["model"] == "PINHOLE", cfg["model"]
    return dict(
        model="PINHOLE",
        width=int(cfg["width"]),
        height=int(cfg["height"]),
        params=np.array([cfg["fx"], cfg["fy"], cfg["cx"], cfg["cy"]], dtype=np.float64),
    )


def load_splits(path: Path) -> Dict[str, List[str]]:
    return json.loads(path.read_text())
