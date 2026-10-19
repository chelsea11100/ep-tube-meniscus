"""启动 Level Meter 图形界面（可拖动液面线）。"""

from pathlib import Path

from src.level_meter_gui import main

if __name__ == "__main__":
    import argparse

    p = argparse.ArgumentParser(description="EP 管液面 Level Meter GUI")
    p.add_argument("--config", type=Path, default=None, help="config.yaml 路径")
    p.add_argument(
        "--camera",
        type=int,
        default=None,
        help="USB 摄像头索引（省略则用 config.yaml 的 gui.camera_index）",
    )
    p.add_argument(
        "--open-camera",
        action="store_true",
        help="启动后立刻打开摄像头",
    )
    args = p.parse_args()
    main(args.config, camera_index=args.camera, open_camera=args.open_camera)
