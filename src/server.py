"""HTTP 服务：上传一张图，返回画好线的 JPEG（base64）和液面 JSON。"""

from __future__ import annotations

import base64
import json
import threading
from contextlib import asynccontextmanager

import cv2
import numpy as np
from fastapi import FastAPI, File, HTTPException, UploadFile

from .common import load_config
from .detect_tube import TubeDetector
from .judge_liquid import LiquidJudgement, create_liquid_judge

TUBE_NOMINAL_ML = 1.5

_lock = threading.Lock()
_detector: TubeDetector | None = None
_judge = None
_min_fill = 10


def _startup() -> None:
    global _detector, _judge, _min_fill
    config = load_config()
    config["verbose"] = False
    _min_fill = int(config.get("min_fill_percent", 10))
    _detector = TubeDetector(config, verbose=False)
    _judge = create_liquid_judge(config, verbose=False)


@asynccontextmanager
async def _lifespan(_app: FastAPI):
    _startup()
    yield


app = FastAPI(title="EP 管液面服务", lifespan=_lifespan)


@app.get("/health")
def health() -> dict:
    return {"ok": _detector is not None, "service": "ep-tube-level"}


@app.post("/v1/level")
async def level(image: UploadFile = File(...)) -> dict:
    raw = await image.read()
    if not raw:
        raise HTTPException(status_code=400, detail="空文件")
    frame = cv2.imdecode(np.frombuffer(raw, np.uint8), cv2.IMREAD_COLOR)
    if frame is None:
        raise HTTPException(status_code=400, detail="无法解码图片，请传 jpg 或 png")
    if _detector is None or _judge is None:
        raise HTTPException(status_code=503, detail="检测器尚未就绪")
    with _lock:
        return analyze_frame(frame)


def analyze_frame(frame: np.ndarray) -> dict:
    assert _detector is not None and _judge is not None
    detection = _detector.detect(frame)
    judgement: LiquidJudgement = _judge.judge(detection.crop)
    geometry = _geometry(judgement.raw_text)
    vis = _compose(frame, detection.bbox, detection.annotated, judgement.level_image)
    ok_tube = not detection.used_fallback
    fill = int(judgement.fill_percent)
    allow = (not judgement.too_little) and ok_tube and judgement.meniscus_y is not None
    error = None
    if not ok_tube:
        error = "未检出离心管，结果按整图估计，不要直接夹取"
    elif judgement.meniscus_y is None:
        error = "检出了管，但没有可靠液面"
    return {
        "ok": ok_tube and judgement.meniscus_y is not None,
        "fill_percent": fill,
        "volume_ml": round(fill / 100.0 * TUBE_NOMINAL_ML, 3),
        "allow_grasp": allow,
        "min_fill_percent": _min_fill,
        "meniscus_y": judgement.meniscus_y,
        "tube_top": geometry.get("tube_top"),
        "tube_bottom": geometry.get("tube_bottom"),
        "bbox": [int(v) for v in detection.bbox],
        "confidence": round(float(judgement.confidence), 4),
        "evidence": judgement.evidence,
        "class_name": detection.class_name,
        "error": error,
        "image_base64": _jpeg_b64(vis),
    }


def _geometry(raw_text: str) -> dict:
    try:
        data = json.loads(raw_text)
    except (json.JSONDecodeError, TypeError):
        return {}
    if not isinstance(data, dict):
        return {}
    return data


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


def _jpeg_b64(image: np.ndarray) -> str:
    ok, buf = cv2.imencode(".jpg", image, [int(cv2.IMWRITE_JPEG_QUALITY), 85])
    if not ok:
        return ""
    return base64.b64encode(buf.tobytes()).decode("ascii")
