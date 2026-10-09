"""EP 管液面识别统一入口（类封装，便于机器人/工位集成）。"""

from __future__ import annotations

import base64
import json
import threading
from dataclasses import replace
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from .camera_util import device_label, list_camera_devices, prefer_usb_camera
from .capture_frame import CaptureSettings, capture_settings_from_config, grab_one_frame_bgr
from .common import load_config
from .detect_tube import TubeDetector, imread_chinese
from .judge_liquid import LiquidJudgement, create_liquid_judge

TUBE_NOMINAL_ML = 1.5


class EpTubeInspector:
    """
    本地一体化识别：检测离心管 + 液面 → 结构化 dict（与 HTTP JSON 字段一致）。

    示例::

        inspector = EpTubeInspector()
        result = inspector.snapshot()          # USB 取一帧并识别
        result = inspector.inspect("a.jpg")    # 从文件识别
    """

    def __init__(
        self,
        config_path: str | Path | None = None,
        *,
        verbose: bool = False,
    ) -> None:
        self.config = load_config(config_path)
        if not verbose:
            self.config["verbose"] = False
        self.min_fill = int(self.config.get("min_fill_percent", 10))
        self._capture = capture_settings_from_config(self.config)
        self._detector = TubeDetector(self.config, verbose=verbose)
        self._judge = create_liquid_judge(self.config, verbose=verbose)
        self._lock = threading.Lock()

    def list_cameras(self) -> dict[str, Any]:
        devices = list_camera_devices(8)
        preferred = prefer_usb_camera(
            devices,
            prefer_usb=self._capture.prefer_usb,
            fallback_index=self._capture.fallback_index,
        )
        return {
            "count": len(devices),
            "preferred_index": preferred.index if preferred else None,
            "cameras": [
                {"index": d.index, "name": d.name, "label": device_label(d)} for d in devices
            ],
        }

    def inspect(
        self,
        image: str | Path | np.ndarray,
        *,
        filename: str | None = None,
        include_image: bool = True,
    ) -> dict[str, Any]:
        """识别一张图（路径或 BGR ndarray）。"""
        frame = self._load_image(image)
        name = filename or (str(image) if isinstance(image, (str, Path)) else "frame")
        with self._lock:
            return self._analyze_frame(frame, filename=name, include_image=include_image)

    def inspect_bytes(
        self,
        data: bytes,
        *,
        filename: str = "upload.jpg",
        include_image: bool = True,
    ) -> dict[str, Any]:
        if not data:
            return self._bad_result(filename, "空文件", include_image=include_image)
        frame = cv2.imdecode(np.frombuffer(data, np.uint8), cv2.IMREAD_COLOR)
        if frame is None:
            return self._bad_result(
                filename, "空文件或无法解码，请传 jpg/png", include_image=include_image
            )
        with self._lock:
            return self._analyze_frame(frame, filename=filename, include_image=include_image)

    def snapshot(
        self,
        *,
        camera_index: int | None = None,
        include_image: bool = True,
    ) -> dict[str, Any]:
        """取摄像机一帧 → 识别 → 返回 JSON 结构 dict。"""
        settings = self._capture_settings(camera_index)
        try:
            frame, meta = grab_one_frame_bgr(settings)
        except RuntimeError as exc:
            return self._bad_result("snapshot", str(exc), include_image=include_image)
        label = meta.get("camera_name") or meta.get("rtsp_url") or "snapshot"
        with self._lock:
            out = self._analyze_frame(frame, filename=str(label), include_image=include_image)
        out.update(meta)
        if meta.get("source") == "usb":
            out["camera_index"] = meta.get("camera_index")
            out["camera_name"] = meta.get("camera_name")
        return out

    def capture_and_inspect(
        self,
        *,
        camera_index: int | None = None,
        include_image: bool = True,
    ) -> dict[str, Any]:
        """一次调用：取图 → 识别 → JSON（与 snapshot 相同，便于工位集成命名）。"""
        return self.snapshot(camera_index=camera_index, include_image=include_image)

    def _capture_settings(self, camera_index: int | None) -> CaptureSettings:
        base = self._capture
        if camera_index is None:
            return base
        return replace(base, camera_index=camera_index)

    @staticmethod
    def _load_image(image: str | Path | np.ndarray) -> np.ndarray:
        if isinstance(image, np.ndarray):
            return image
        path = Path(image)
        frame = imread_chinese(path)
        if frame is None:
            raise FileNotFoundError(f"无法读取图片: {path}")
        return frame

    def _analyze_frame(
        self,
        frame: np.ndarray,
        *,
        filename: str | None,
        include_image: bool,
    ) -> dict[str, Any]:
        detection = self._detector.detect(frame)
        judgement: LiquidJudgement = self._judge.judge(detection.crop)
        geometry = self._geometry(judgement.raw_text)
        vis = self._compose(frame, detection.bbox, detection.annotated, judgement.level_image)
        ok_tube = not detection.used_fallback
        fill = int(judgement.fill_percent)
        allow = (not judgement.too_little) and ok_tube and judgement.meniscus_y is not None
        error = None
        if not ok_tube:
            error = "未检出离心管，结果按整图估计，不要直接夹取"
        elif judgement.meniscus_y is None:
            error = "检出了管，但没有可靠液面"
        out: dict[str, Any] = {
            "ok": ok_tube and judgement.meniscus_y is not None,
            "fill_percent": fill,
            "volume_ml": round(fill / 100.0 * TUBE_NOMINAL_ML, 3),
            "allow_grasp": allow,
            "min_fill_percent": self.min_fill,
            "meniscus_y": judgement.meniscus_y,
            "tube_top": geometry.get("tube_top"),
            "tube_bottom": geometry.get("tube_bottom"),
            "bbox": [int(v) for v in detection.bbox],
            "confidence": round(float(judgement.confidence), 4),
            "evidence": judgement.evidence,
            "class_name": detection.class_name,
            "error": error,
            "image_base64": self._jpeg_b64(vis) if include_image else "",
        }
        if filename:
            out["filename"] = filename
        return out

    def _bad_result(self, filename: str, error: str, *, include_image: bool) -> dict[str, Any]:
        return {
            "filename": filename,
            "ok": False,
            "fill_percent": 0,
            "volume_ml": 0.0,
            "allow_grasp": False,
            "min_fill_percent": self.min_fill,
            "meniscus_y": None,
            "tube_top": None,
            "tube_bottom": None,
            "bbox": None,
            "confidence": 0.0,
            "evidence": "",
            "class_name": None,
            "error": error,
            "image_base64": "" if not include_image else "",
        }

    @staticmethod
    def _geometry(raw_text: str) -> dict:
        try:
            data = json.loads(raw_text)
        except (json.JSONDecodeError, TypeError):
            return {}
        return data if isinstance(data, dict) else {}

    @staticmethod
    def _compose(
        frame: np.ndarray,
        bbox: tuple[int, int, int, int],
        annotated: np.ndarray,
        level: np.ndarray | None,
    ) -> np.ndarray:
        x1, y1, x2, y2 = bbox
        if level is None:
            return annotated
        vis = frame.copy()
        ch, cw = level.shape[:2]
        if (y2 - y1) == ch and (x2 - x1) == cw:
            vis[y1:y2, x1:x2] = level
            cv2.rectangle(vis, (x1, y1), (x2, y2), (0, 200, 0), 2)
            return vis
        if level.shape[:2] == frame.shape[:2]:
            return level
        return annotated

    @staticmethod
    def _jpeg_b64(image: np.ndarray) -> str:
        ok, buf = cv2.imencode(".jpg", image, [int(cv2.IMWRITE_JPEG_QUALITY), 85])
        if not ok:
            return ""
        return base64.b64encode(buf.tobytes()).decode("ascii")


class EpTubeHttpClient:
    """远程 HTTP 服务客户端（类封装，对接 run_server.py）。"""

    def __init__(
        self,
        base_url: str = "http://127.0.0.1:8000",
        *,
        timeout: float = 120.0,
    ) -> None:
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout

    def list_cameras(self) -> dict[str, Any]:
        import requests

        r = requests.get(f"{self.base_url}/v1/cameras", timeout=self.timeout)
        r.raise_for_status()
        return r.json()

    def inspect_file(self, path: str | Path, *, include_image: bool = True) -> dict[str, Any]:
        import requests

        path = Path(path)
        with path.open("rb") as f:
            r = requests.post(
                f"{self.base_url}/v1/level",
                files={"image": (path.name, f, "image/jpeg")},
                timeout=self.timeout,
            )
        r.raise_for_status()
        data = r.json()
        if not include_image:
            data["image_base64"] = ""
        return data

    def snapshot(
        self,
        *,
        camera_index: int | None = None,
        include_image: bool = True,
    ) -> dict[str, Any]:
        import requests

        params: dict[str, Any] = {"include_image": include_image}
        if camera_index is not None:
            params["camera_index"] = camera_index
        r = requests.post(
            f"{self.base_url}/v1/inspect",
            params=params,
            timeout=self.timeout,
        )
        r.raise_for_status()
        return r.json()

    def capture_and_inspect(
        self,
        *,
        camera_index: int | None = None,
        include_image: bool = True,
    ) -> dict[str, Any]:
        return self.snapshot(camera_index=camera_index, include_image=include_image)
