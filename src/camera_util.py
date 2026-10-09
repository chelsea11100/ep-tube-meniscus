"""USB / 本机摄像头：Windows 下列设备名并优先外接 USB（DirectShow 索引）。"""

from __future__ import annotations

import re
import sys
from dataclasses import dataclass

import cv2


@dataclass(frozen=True)
class CameraDevice:
    index: int
    name: str
    backend: int = cv2.CAP_DSHOW


_BUILTIN_RE = re.compile(
    r"integrated|facetime|virtual|infrared|\bir\b|内置|built-?in|lenovo.*camera|hp.*camera",
    re.I,
)
_USB_RE = re.compile(
    r"usb|uvc|webcam|hd camera|camera \d|工业|显微镜|endoscope|logitech|hikvision|daheng",
    re.I,
)


def _dshow_device_names() -> list[str]:
    if sys.platform != "win32":
        return []
    try:
        from pygrabber.dshow_graph import FilterGraph

        return list(FilterGraph().get_input_devices())
    except Exception:
        return []


def device_label(dev: CameraDevice) -> str:
    return f"[{dev.index}] {dev.name}"


def list_camera_devices_from_names(names: list[str]) -> list[CameraDevice]:
    """仅根据设备名列表构建条目（不打开相机，避免扫描卡顿/闪退）。"""
    return [
        CameraDevice(index=idx, name=(name.strip() or f"Camera {idx}"))
        for idx, name in enumerate(names)
    ]


def list_camera_devices_opencv_probe(max_probe: int = 4) -> list[CameraDevice]:
    """无 pygrabber 时慢速探测；仅在后台线程调用，勿与 GUI 扫描并发。"""
    devices: list[CameraDevice] = []
    backends = [cv2.CAP_DSHOW] if sys.platform == "win32" else [cv2.CAP_V4L2, cv2.CAP_ANY]
    for idx in range(max_probe + 1):
        for backend in backends:
            cap = None
            try:
                cap = cv2.VideoCapture(idx, backend)
                if not cap.isOpened():
                    continue
                cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
                ok, frame = cap.read()
                if ok and frame is not None and frame.size > 0:
                    devices.append(
                        CameraDevice(
                            index=idx,
                            name=f"Camera {idx}",
                            backend=backend,
                        )
                    )
                    break
            except Exception:
                continue
            finally:
                if cap is not None:
                    cap.release()
    return devices


def list_camera_devices(max_probe: int = 8) -> list[CameraDevice]:
    """列出摄像头。Windows 优先 pygrabber 设备名（不逐路 open）。"""
    names = _dshow_device_names()
    if names:
        return list_camera_devices_from_names(names)
    return list_camera_devices_opencv_probe(max_probe)


def prefer_usb_camera(
    devices: list[CameraDevice],
    *,
    prefer_usb: bool = True,
    fallback_index: int = 1,
) -> CameraDevice | None:
    if not devices:
        return None
    if not prefer_usb:
        for d in devices:
            if d.index == fallback_index:
                return d
        return devices[0]

    scored: list[tuple[float, CameraDevice]] = []
    for d in devices:
        score = float(d.index)
        if _BUILTIN_RE.search(d.name):
            score -= 200.0
        if _USB_RE.search(d.name):
            score += 80.0
        if d.index == 0 and len(devices) > 1:
            score -= 30.0
        scored.append((score, d))
    scored.sort(key=lambda x: x[0], reverse=True)
    best = scored[0][1]
    if len(devices) > 1 and best.index == 0 and not _USB_RE.search(best.name):
        for _, d in scored:
            if d.index != 0:
                return d
    return best


def open_video_capture(dev: CameraDevice | int) -> cv2.VideoCapture | None:
    """打开摄像头；Windows 外接 USB 请用 list 返回的 CameraDevice（固定 DSHOW 索引）。"""
    if isinstance(dev, int):
        index = dev
        backend = cv2.CAP_DSHOW if sys.platform == "win32" else cv2.CAP_ANY
    else:
        index = dev.index
        backend = dev.backend

    cap = cv2.VideoCapture(index, backend)
    if cap.isOpened():
        ok, frame = cap.read()
        if ok and frame is not None and frame.size > 0:
            return cap
        cap.release()

    if sys.platform == "win32" and backend != cv2.CAP_MSMF:
        cap = cv2.VideoCapture(index, cv2.CAP_MSMF)
        if cap.isOpened():
            ok, frame = cap.read()
            if ok and frame is not None and frame.size > 0:
                return cap
            cap.release()
    return None
