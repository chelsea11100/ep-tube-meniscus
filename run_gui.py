"""启动 Level Meter 图形界面（可拖动液面线）。"""

from pathlib import Path

from src.level_meter_gui import main

if __name__ == "__main__":
    import argparse

    p = argparse.ArgumentParser(description="EP 管液面 Level Meter GUI")
    p.add_argument("--config", type=Path, default=None, help="config.yaml 路径")
    args = p.parse_args()
    main(args.config)
