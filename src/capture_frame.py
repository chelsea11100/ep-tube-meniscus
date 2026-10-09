"""取图一层：从 USB 或 RTSP 抓一帧，供 HTTP 一键识别。"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import cv2
import numpy as np

from .camera_util import (
    CameraDevice,
    list_camera_devices,
    open_video_capture,
    prefer_usb_camera,
)


@dataclass
class CaptureSettings:
    backend: str = "usb"
    camera_index: int | None = None
    prefer_usb: bool = True
    fallback_index: int = 1
    width: int = 1280
    height: int = 720
    rtsp_url: str = ""
    jpeg_quality: int = 85
    plugin_module: str = ""
    plugin_function: str = "grab_frame"
    mvs_development_dir: str = ""
    mvs_runtime_dir: str = ""
    mvs_device_index: int = 0
    mvs_feature_file: str = ""
    mvs_warmup_frames: int = 45
    roi_percent_x: float = 100.0
    roi_percent_y: float = 100.0


def apply_roi_center_percent(
    bgr: np.ndarray, pct_x: float, pct_y: float
) -> tuple[np.ndarray, tuple[int, int]]:
    """取画面中心 pct（0–100）区域，减少手指/背景干扰。"""
    px = max(25.0, min(100.0, float(pct_x))) / 100.0
    py = max(25.0, min(100.0, float(pct_y))) / 100.0
    h, w = bgr.shape[:2]
    nw, nh = max(32, int(w * px)), max(32, int(h * py))
    x0 = (w - nw) // 2
    y0 = (h - nh) // 2
    return bgr[y0 : y0 + nh, x0 : x0 + nw].copy(), (x0, y0)


def capture_settings_from_config(config: dict[str, Any]) -> CaptureSettings:
    srv = config.get("server") or {}
    cap = srv.get("capture") or {}
    idx = cap.get("camera_index", 1)
    return CaptureSettings(
        backend=str(cap.get("backend", "usb")).lower(),
        camera_index=None if idx is None or idx == "" else int(idx),
        prefer_usb=bool(cap.get("prefer_usb", True)),
        fallback_index=int(cap.get("fallback_index", 1 if idx in (None, "") else idx)),
        width=int(cap.get("width", 1280)),
        height=int(cap.get("height", 720)),
        rtsp_url=str(cap.get("rtsp_url", "") or "").strip(),
        jpeg_quality=int(cap.get("jpeg_quality", 85)),
        plugin_module=str(cap.get("plugin_module", "") or "").strip(),
        plugin_function=str(cap.get("plugin_function", "grab_frame") or "grab_frame"),
        mvs_development_dir=str(cap.get("mvs_development_dir", "") or "").strip(),
        mvs_runtime_dir=str(cap.get("mvs_runtime_dir", "") or "").strip(),
        mvs_device_index=int(cap.get("mvs_device_index", 0)),
        mvs_feature_file=str(cap.get("mvs_feature_file", "") or "").strip(),
        mvs_warmup_frames=int(cap.get("mvs_warmup_frames", 45)),
        roi_percent_x=float(cap.get("roi_percent_x", 100)),
        roi_percent_y=float(cap.get("roi_percent_y", 100)),
    )


def resolve_usb_device(settings: CaptureSettings) -> CameraDevice:
    devices = list_camera_devices(8)
    if not devices:
        raise RuntimeError("未找到可用 USB 摄像头，请检查连接后 GET /v1/cameras")
    if settings.camera_index is not None:
        for d in devices:
            if d.index == settings.camera_index:
                return d
        raise RuntimeError(
            f"camera_index={settings.camera_index} 不存在，请先 GET /v1/cameras"
        )
    pick = prefer_usb_camera(
        devices,
        prefer_usb=settings.prefer_usb,
        fallback_index=settings.fallback_index,
    )
    if pick is None:
        raise RuntimeError("无法选择 USB 摄像头")
    return pick


def grab_one_frame_bgr(settings: CaptureSettings) -> tuple[np.ndarray, dict[str, Any]]:
    """抓一帧 BGR；返回 (图像, 来源信息 meta)。"""
    backend = settings.backend.lower()
    if backend == "plugin":
        from .camera_plugin import grab_via_plugin

        return grab_via_plugin(settings)

    if backend == "v4l2":
        return _grab_usb_v4l2(settings, force_v4l2=True)

    if backend == "rtsp":
        if not settings.rtsp_url:
            raise RuntimeError("server.capture.rtsp_url 未配置")
        cap = cv2.VideoCapture(settings.rtsp_url)
        if not cap.isOpened():
            raise RuntimeError(f"无法打开 RTSP: {settings.rtsp_url}")
        try:
            for _ in range(8):
                ok, frame = cap.read()
                if ok and frame is not None and frame.size > 0:
                    meta = {"source": "rtsp", "rtsp_url": settings.rtsp_url}
                    return frame, meta
            raise RuntimeError("RTSP 读帧失败")
        finally:
            cap.release()

    return _grab_usb_v4l2(settings, force_v4l2=False)


def _grab_usb_v4l2(
    settings: CaptureSettings, *, force_v4l2: bool
) -> tuple[np.ndarray, dict[str, Any]]:
    import sys

    dev = resolve_usb_device(settings)
    cap = open_video_capture(dev)
    if cap is None and (force_v4l2 or sys.platform.startswith("linux")):
        idx = dev.index if settings.camera_index is not None else settings.fallback_index
        cap = cv2.VideoCapture(idx, cv2.CAP_V4L2)
    if cap is None:
        raise RuntimeError(f"无法打开 {dev.name}")
    try:
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, settings.width)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, settings.height)
        for _ in range(4):
            cap.read()
        ok, frame = cap.read()
    finally:
        cap.release()
    if not ok or frame is None or frame.size == 0:
        raise RuntimeError("USB 摄像头读帧失败")
    meta = {
        "source": "v4l2" if force_v4l2 or __import__("sys").platform.startswith("linux") else "usb",
        "camera_index": dev.index,
        "camera_name": dev.name,
    }
    return frame, meta
