"""
取图插件模板（backend: plugin）。

在 Linux + 海康 / 机器人 SDK 取到一帧后，解码成 OpenCV BGR 数组返回即可。
config.yaml 示例::

  server:
    capture:
      backend: plugin
      plugin_module: examples.camera_plugin_stub
      plugin_function: grab_frame
"""

from __future__ import annotations

from typing import Any

import numpy as np

# from your_sdk import capture_one_frame_bgr  # 接入厂商 SDK 后在这里 import


def grab_frame(settings) -> tuple[np.ndarray, dict[str, Any]]:
    """
    参数 settings 为 CaptureSettings（宽高等来自 config.yaml）。
    返回 (BGR 图像, meta)，meta 建议含 camera_name 等说明字段。
    """
    # 示例：接入 SDK 后替换下面两行
    # frame = capture_one_frame_bgr(width=settings.width, height=settings.height)
    # return frame, {"source": "plugin", "camera_name": "Hikvision-GigE-0"}
    raise NotImplementedError(
        "请在此函数内调用海康/机器人 SDK 取一帧 BGR；"
        "开发阶段可改用 backend: usb 或 POST /v1/level 上传图片"
    )
