#!/usr/bin/env python3
"""
工位一键：摄像头拍照 → 液面识别 → JSON + JPG（原图 / 标注图）。

用法::

  .venv/bin/python run_inspect.py -o output/inspect_result.json
"""

from __future__ import annotations

import argparse
import base64
import json
import sys
from dataclasses import replace
from pathlib import Path

import cv2
import numpy as np
import yaml

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.capture_frame import (
    CAPTURE_SOURCES,
    apply_roi_center_percent,
    capture_settings_from_config,
    grab_one_frame_bgr,
)
from src.ep_tube_inspector import EpTubeInspector


def _write_jpg_bgr(path: Path, bgr: np.ndarray) -> None:
    ok, enc = cv2.imencode(".jpg", bgr, [int(cv2.IMWRITE_JPEG_QUALITY), 92])
    if not ok:
        raise RuntimeError(f"JPEG 编码失败: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(enc.tobytes())


def _write_jpg_b64(path: Path, b64: str) -> None:
    raw = base64.b64decode(b64)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(raw)


def _resolve_out(path: Path | None, default: Path) -> Path:
    if path is None:
        return default
    return path if path.is_absolute() else ROOT / path


def main() -> int:
    parser = argparse.ArgumentParser(description="一键拍照 + 识别 → JSON + JPG")
    parser.add_argument(
        "-o",
        "--output",
        type=Path,
        default=None,
        help="JSON 路径，默认 output/inspect_result.json",
    )
    parser.add_argument(
        "--capture-jpg",
        type=Path,
        default=None,
        help="相机原图 JPG，默认 output/inspect_capture.jpg",
    )
    parser.add_argument(
        "--annotated-jpg",
        type=Path,
        default=None,
        help="标注结果 JPG，默认 output/inspect_annotated.jpg",
    )
    parser.add_argument(
        "--no-image",
        action="store_true",
        help="JSON 中不包含 image_base64",
    )
    parser.add_argument(
        "--source",
        choices=CAPTURE_SOURCES,
        default=None,
        help="取图方式：usb / v4l2 / rtsp / plugin；省略则按 config.yaml",
    )
    parser.add_argument(
        "--camera",
        type=int,
        default=None,
        help="USB 设备索引，省略则自动优先外接 USB",
    )
    args = parser.parse_args()

    cfg_path = ROOT / "config.yaml"
    with cfg_path.open(encoding="utf-8") as f:
        config = yaml.safe_load(f) or {}

    settings = capture_settings_from_config(config)
    if args.source:
        settings = replace(settings, backend=args.source)
    if args.camera is not None:
        settings = replace(settings, camera_index=args.camera)
    bgr, meta = grab_one_frame_bgr(settings)
    roi_off = (0, 0)
    if settings.roi_percent_x < 99.5 or settings.roi_percent_y < 99.5:
        bgr, roi_off = apply_roi_center_percent(
            bgr, settings.roi_percent_x, settings.roi_percent_y
        )
        meta = dict(meta)
        meta["roi_offset"] = list(roi_off)
        meta["roi_percent"] = [settings.roi_percent_x, settings.roi_percent_y]

    json_out = _resolve_out(args.output, ROOT / "output" / "inspect_result.json")
    capture_jpg = _resolve_out(args.capture_jpg, ROOT / "output" / "inspect_capture.jpg")
    annotated_jpg = _resolve_out(
        args.annotated_jpg, ROOT / "output" / "inspect_annotated.jpg"
    )

    _write_jpg_bgr(capture_jpg, bgr)
    print(f"原图: {capture_jpg.resolve()}", file=sys.stderr)

    inspector = EpTubeInspector(verbose=False)
    result = inspector.inspect(bgr, filename="snapshot", include_image=not args.no_image)
    result.update(meta)

    b64 = result.get("image_base64") or ""
    if b64 and not args.no_image:
        _write_jpg_b64(annotated_jpg, b64)
        print(f"标注图: {annotated_jpg.resolve()}", file=sys.stderr)

    json_out.parent.mkdir(parents=True, exist_ok=True)
    json_out.write_text(
        json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(f"JSON: {json_out.resolve()}", file=sys.stderr)

    display = dict(result)
    if display.pop("image_base64", ""):
        display["_image_base64_note"] = f"见 {annotated_jpg.name} 或 JSON 文件"
    display["capture_jpg"] = str(capture_jpg.resolve())
    if b64 and not args.no_image:
        display["annotated_jpg"] = str(annotated_jpg.resolve())
    print(json.dumps(display, ensure_ascii=False, indent=2))

    return 0 if result.get("ok") else 1


if __name__ == "__main__":
    raise SystemExit(main())
