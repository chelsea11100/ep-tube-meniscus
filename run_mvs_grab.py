#!/usr/bin/env python3
"""海康相机取一帧测试图（会 FeatureLoad + 预热，输出 output/mvs_test.jpg）。"""

from __future__ import annotations

import sys
from pathlib import Path

import cv2
import yaml

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.capture_frame import capture_settings_from_config
from src.mvs_hikvision import grab_one_frame_bgr_mvs


def main() -> None:
    cfg_path = ROOT / "config.yaml"
    with cfg_path.open(encoding="utf-8") as f:
        config = yaml.safe_load(f) or {}
    settings = capture_settings_from_config(config)
    bgr, meta = grab_one_frame_bgr_mvs(settings)
    out_dir = ROOT / "output"
    out_dir.mkdir(parents=True, exist_ok=True)
    out = (out_dir / "mvs_test.jpg").resolve()
    ok, enc = cv2.imencode(".jpg", bgr, [int(cv2.IMWRITE_JPEG_QUALITY), 92])
    if not ok:
        raise RuntimeError("JPEG 编码失败")
    out.write_bytes(enc.tobytes())
    if not out.is_file():
        raise RuntimeError(f"写入失败: {out}")
    print(f"已保存: {out}  ({out.stat().st_size} bytes)  shape={bgr.shape}  meta={meta}")


if __name__ == "__main__":
    main()
