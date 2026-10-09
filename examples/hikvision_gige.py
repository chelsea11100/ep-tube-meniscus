"""海康 GigE 取图插件：config 设 backend: plugin + plugin_module: examples.hikvision_gige"""

from __future__ import annotations

from typing import Any

import numpy as np

from src.capture_frame import CaptureSettings
from src.mvs_hikvision import grab_one_frame_bgr_mvs


def grab_frame(settings: CaptureSettings) -> tuple[np.ndarray, dict[str, Any]]:
    return grab_one_frame_bgr_mvs(settings)
