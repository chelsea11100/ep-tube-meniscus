"""HTTP 服务：上传一张图，返回画好线的 JPEG（base64）和液面 JSON。"""

from __future__ import annotations

from contextlib import asynccontextmanager
from typing import Annotated

from fastapi import FastAPI, File, HTTPException, Query, UploadFile

from .capture_frame import CAPTURE_SOURCES
from .ep_tube_inspector import EpTubeInspector

_inspector: EpTubeInspector | None = None


def _startup() -> None:
    global _inspector
    _inspector = EpTubeInspector(verbose=False)


@asynccontextmanager
async def _lifespan(_app: FastAPI):
    _startup()
    yield


app = FastAPI(title="EP 管液面服务", lifespan=_lifespan)


@app.get("/")
def root() -> dict:
    return {
        "service": "ep-tube-level",
        "docs": "/docs",
        "health": "/health",
        "single": "POST /v1/level  (form field: image)",
        "batch": "POST /v1/level/batch  (form field: images)",
        "cameras": "GET /v1/cameras",
        "inspect_once": "POST /v1/inspect  (取图+识别一体，与 /v1/snapshot 相同；?source=usb|v4l2|rtsp)",
        "snapshot": "POST /v1/snapshot",
        "capture": "POST /v1/level/capture  (兼容旧名)",
    }


@app.get("/health")
def health() -> dict:
    return {"ok": _inspector is not None, "service": "ep-tube-level"}


@app.get("/v1/cameras")
def cameras_list() -> dict:
    if _inspector is None:
        raise HTTPException(status_code=503, detail="检测器尚未就绪")
    return _inspector.list_cameras()


def _snapshot_once(
    camera_index: int | None,
    include_image: bool,
    source: str | None = None,
) -> dict:
    if _inspector is None:
        raise HTTPException(status_code=503, detail="检测器尚未就绪")
    if source and source.lower() not in CAPTURE_SOURCES:
        raise HTTPException(
            status_code=400,
            detail=f"source 只能是：{', '.join(CAPTURE_SOURCES)}",
        )
    out = _inspector.snapshot(
        source=source, camera_index=camera_index, include_image=include_image
    )
    if out.get("error") and not out.get("ok") and out.get("filename") == "snapshot":
        msg = str(out.get("error") or "取图失败")
        code = 400 if "不存在" in msg or "camera_index" in msg else 503
        raise HTTPException(status_code=code, detail=msg)
    return out


_SOURCE_QUERY = Query(
    None,
    description="usb / v4l2 / rtsp / plugin；省略则按 config.yaml",
)
_CAMERA_INDEX_QUERY = Query(
    None,
    description="USB 设备索引；省略则自动优先外接 USB",
)
_INCLUDE_IMAGE_QUERY = Query(True, description="是否返回 image_base64 标注图")


@app.post("/v1/inspect")
@app.post("/v1/snapshot")
def snapshot(
    source: str | None = _SOURCE_QUERY,
    camera_index: int | None = _CAMERA_INDEX_QUERY,
    include_image: bool = _INCLUDE_IMAGE_QUERY,
) -> dict:
    """取摄像机一帧 JPG → 液面识别 → 返回 JSON（机器人/工位推荐入口）。"""
    return _snapshot_once(camera_index, include_image, source)


@app.post("/v1/level/capture")
def level_capture(
    source: str | None = _SOURCE_QUERY,
    camera_index: int | None = Query(None),
    include_image: bool = Query(True),
) -> dict:
    """与 POST /v1/snapshot 相同（兼容旧路径）。"""
    return _snapshot_once(camera_index, include_image, source)


@app.post("/v1/level")
async def level(
    image: Annotated[UploadFile, File(description="选一张 jpg 或 png")],
) -> dict:
    if _inspector is None:
        raise HTTPException(status_code=503, detail="检测器尚未就绪")
    raw = await image.read()
    if not raw:
        raise HTTPException(status_code=400, detail="空文件或无法解码，请传 jpg/png")
    out = _inspector.inspect_bytes(raw, filename=image.filename or "upload.jpg")
    err = out.get("error") or ""
    if "无法解码" in err or err == "空文件":
        raise HTTPException(status_code=400, detail=err)
    return out


@app.post("/v1/level/batch")
async def level_batch(
    images: Annotated[list[UploadFile], File(description="选多张 jpg/png")],
) -> dict:
    """一次上传多张图，返回与单张接口相同结构的列表。"""
    if _inspector is None:
        raise HTTPException(status_code=503, detail="检测器尚未就绪")
    if not images:
        raise HTTPException(status_code=400, detail="请至少上传一张图片，字段名 images")
    results: list[dict] = []
    for upload in images:
        raw = await upload.read()
        name = upload.filename or "unknown"
        item = _inspector.inspect_bytes(raw, filename=name)
        item["filename"] = name
        results.append(item)
    ok_count = sum(1 for r in results if r.get("ok"))
    return {
        "count": len(results),
        "ok_count": ok_count,
        "results": results,
    }
