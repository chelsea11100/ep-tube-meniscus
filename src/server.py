"""HTTP 服务：上传一张图，返回画好线的 JPEG（base64）和液面 JSON。"""

from __future__ import annotations

import base64
import json
import threading
from contextlib import asynccontextmanager
from typing import Annotated

import cv2
import numpy as np
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.responses import HTMLResponse

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


@app.get("/")
def root() -> dict:
    return {
        "service": "ep-tube-level",
        "ui": "/ui",
        "docs": "/docs",
        "health": "/health",
        "single": "POST /v1/level  (form field: image)",
        "batch": "POST /v1/level/batch  (form field: images)",
    }


@app.get("/health")
def health() -> dict:
    return {"ok": _detector is not None, "service": "ep-tube-level"}


@app.get("/ui", response_class=HTMLResponse)
def ui_page() -> str:
    return """<!DOCTYPE html>
<html lang="zh-CN"><head><meta charset="utf-8"><title>EP 管液面</title>
<style>
body{font-family:sans-serif;max-width:960px;margin:24px auto;padding:0 16px}
section{margin:20px 0;padding:16px;border:1px solid #ddd;border-radius:8px}
button{margin-top:8px;padding:8px 16px}
img{max-width:100%;margin-top:12px;border:1px solid #ccc}
pre{background:#f5f5f5;padding:12px;overflow:auto;font-size:13px}
</style></head><body>
<h1>EP 管液面识别</h1>
<p>在浏览器里选图片测试。对接程序请直接调 API，不要用这个页。</p>
<section>
<h2>单张</h2>
<input type="file" id="one" accept="image/*">
<button onclick="runOne()">识别</button>
<div id="oneOut"></div>
</section>
<section>
<h2>批量</h2>
<p>可多选图片，或选一个文件夹（自动取里面全部 jpg/png）。</p>
<input type="file" id="many" accept="image/jpeg,image/png,image/webp,image/bmp" multiple hidden>
<input type="file" id="folder" webkitdirectory directory multiple hidden>
<button type="button" onclick="document.getElementById('many').click()">选择多张</button>
<button type="button" onclick="document.getElementById('folder').click()">选择文件夹</button>
<button type="button" onclick="runBatch()">批量识别</button>
<p id="pickLabel" style="color:#555;margin-top:8px"></p>
<div id="manyOut"></div>
</section>
<script>
let batchFiles = [];
const IMG_RE = /\.(jpe?g|png|webp|bmp)$/i;
function pickBatch(files) {
  batchFiles = [...files].filter(f => IMG_RE.test(f.name));
  const el = document.getElementById('pickLabel');
  el.textContent = batchFiles.length
    ? ('已选 ' + batchFiles.length + ' 张：' + batchFiles.map(f => f.name).slice(0, 5).join('、') + (batchFiles.length > 5 ? '…' : ''))
    : '文件夹里没有 jpg/png';
}
document.getElementById('many').onchange = e => pickBatch(e.target.files);
document.getElementById('folder').onchange = e => pickBatch(e.target.files);
function showResult(el, data, imgB64) {
  let html = '<pre>' + JSON.stringify(data, null, 2) + '</pre>';
  if (imgB64) html += '<img src="data:image/jpeg;base64,' + imgB64 + '">';
  el.innerHTML = html;
}
async function runOne() {
  const f = document.getElementById('one').files[0];
  const out = document.getElementById('oneOut');
  if (!f) { out.textContent = '请先选一张图'; return; }
  out.textContent = '识别中…';
  const fd = new FormData(); fd.append('image', f);
  const r = await fetch('/v1/level', { method: 'POST', body: fd });
  const data = await r.json();
  showResult(out, data, data.image_base64);
}
async function runBatch() {
  const out = document.getElementById('manyOut');
  if (!batchFiles.length) { out.textContent = '请先「选择多张」或「选择文件夹」'; return; }
  out.textContent = '识别中 ' + batchFiles.length + ' 张…';
  const fd = new FormData();
  for (const f of batchFiles) fd.append('images', f, f.name.split(/[/\\\\]/).pop());
  const r = await fetch('/v1/level/batch', { method: 'POST', body: fd });
  const data = await r.json();
  let html = '<pre>' + JSON.stringify(data, null, 2) + '</pre>';
  for (const item of data.results || []) {
    if (item.image_base64) {
      html += '<p>' + item.filename + ' — fill ' + item.fill_percent + '%</p>';
      html += '<img src="data:image/jpeg;base64,' + item.image_base64 + '">';
    }
  }
  out.innerHTML = html;
}
</script></body></html>"""


@app.post("/v1/level")
async def level(
    image: Annotated[UploadFile, File(description="选一张 jpg 或 png")],
) -> dict:
    if _detector is None or _judge is None:
        raise HTTPException(status_code=503, detail="检测器尚未就绪")
    raw = await image.read()
    frame = _decode_image(raw)
    if frame is None:
        raise HTTPException(status_code=400, detail="空文件或无法解码，请传 jpg/png")
    with _lock:
        return analyze_frame(frame, filename=image.filename)


@app.post("/v1/level/batch")
async def level_batch(
    images: Annotated[list[UploadFile], File(description="选多张 jpg/png")],
) -> dict:
    """一次上传多张图，返回与单张接口相同结构的列表。"""
    if _detector is None or _judge is None:
        raise HTTPException(status_code=503, detail="检测器尚未就绪")
    if not images:
        raise HTTPException(status_code=400, detail="请至少上传一张图片，字段名 images")
    results: list[dict] = []
    with _lock:
        for upload in images:
            raw = await upload.read()
            frame = _decode_image(raw)
            name = upload.filename or "unknown"
            if frame is None:
                results.append(_bad_file_result(name, "空文件或无法解码"))
                continue
            item = analyze_frame(frame, filename=name)
            item["filename"] = name
            results.append(item)
    ok_count = sum(1 for r in results if r.get("ok"))
    return {
        "count": len(results),
        "ok_count": ok_count,
        "results": results,
    }


def _decode_image(raw: bytes) -> np.ndarray | None:
    if not raw:
        return None
    frame = cv2.imdecode(np.frombuffer(raw, np.uint8), cv2.IMREAD_COLOR)
    return frame


def _bad_file_result(filename: str, error: str) -> dict:
    return {
        "filename": filename,
        "ok": False,
        "fill_percent": 0,
        "volume_ml": 0.0,
        "allow_grasp": False,
        "min_fill_percent": _min_fill,
        "meniscus_y": None,
        "tube_top": None,
        "tube_bottom": None,
        "bbox": None,
        "confidence": 0.0,
        "evidence": "",
        "class_name": None,
        "error": error,
        "image_base64": "",
    }


def analyze_frame(frame: np.ndarray, filename: str | None = None) -> dict:
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
    out = {
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
    if filename:
        out["filename"] = filename
    return out


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
