#!/usr/bin/env python3
"""从海康相机导出当前参数到 config/mv_cu120_features.mfs（先 MVS 调亮并关闭 Client）。"""

from __future__ import annotations

import sys
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.capture_frame import capture_settings_from_config
from src.mvs_hikvision import export_camera_features


def main() -> None:
    cfg_path = ROOT / "config.yaml"
    with cfg_path.open(encoding="utf-8") as f:
        config = yaml.safe_load(f) or {}
    settings = capture_settings_from_config(config)
    out = export_camera_features(settings)
    print(f"已导出相机参数: {out}")
    print("请在 config.yaml 的 server.capture.mvs_feature_file 中指向该文件。")


if __name__ == "__main__":
    main()
