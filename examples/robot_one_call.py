"""工位/机器人一键调用：取图 + 识别（相机需接在跑 run_server.py 的本机，或配置 RTSP）。"""

from __future__ import annotations

import argparse
import base64
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.ep_tube_inspector import EpTubeHttpClient, EpTubeInspector


def main() -> int:
    p = argparse.ArgumentParser(description="类封装：一键取图识别")
    p.add_argument("--local", action="store_true", help="不经过 HTTP，直接用 EpTubeInspector")
    p.add_argument("--host", default="127.0.0.1", help="run_server.py 所在机器 IP")
    p.add_argument("--port", type=int, default=8000)
    p.add_argument("--camera", type=int, default=None, help="USB 索引，省略则用 config")
    p.add_argument("--no-image", action="store_true", help="不要 JSON 里的标注图")
    p.add_argument("--save", type=Path, default=None, help="保存标注图到本地")
    p.add_argument("-o", "--json", type=Path, default=None, help="保存完整 JSON")
    args = p.parse_args()

    if args.local:
        data = EpTubeInspector().capture_and_inspect(
            camera_index=args.camera,
            include_image=not args.no_image,
        )
    else:
        client = EpTubeHttpClient(f"http://{args.host}:{args.port}")
        data = client.snapshot(
            camera_index=args.camera,
            include_image=not args.no_image,
        )
    import json

    if args.json:
        args.json.parent.mkdir(parents=True, exist_ok=True)
        args.json.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
        print("json:", args.json.resolve())
    else:
        print(json.dumps(data, ensure_ascii=False, indent=2))

    b64 = data.get("image_base64") or ""
    if args.save and b64:
        args.save.parent.mkdir(parents=True, exist_ok=True)
        args.save.write_bytes(base64.b64decode(b64))
        print("saved:", args.save)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
