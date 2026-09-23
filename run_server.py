"""启动液面 HTTP 服务。在项目目录执行: python run_server.py"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def main() -> int:
    parser = argparse.ArgumentParser(description="EP 管液面识别 HTTP 服务")
    parser.add_argument("--host", default="0.0.0.0", help="监听地址，默认所有网卡")
    parser.add_argument("--port", type=int, default=8000)
    args = parser.parse_args()

    import uvicorn

    shown = "127.0.0.1" if args.host in ("0.0.0.0", "::") else args.host
    print(f"网页测试: http://{shown}:{args.port}/ui")
    print(f"单张 API: http://{shown}:{args.port}/v1/level  (字段 image)")
    print(f"批量: http://{shown}:{args.port}/v1/level/batch  (字段 images，可多文件)")
    print("返回 JSON，图在每条结果的 image_base64。")
    print(f"健康检查: http://{shown}:{args.port}/health")
    uvicorn.run("src.server:app", host=args.host, port=args.port, app_dir=str(ROOT))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
