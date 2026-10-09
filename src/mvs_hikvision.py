"""海康 MVS GigE/USB 取图与 FeatureSave/Load（Windows + 已安装 MVS Runtime）。"""

from __future__ import annotations

import os
import shutil
import sys
import tempfile
from ctypes import byref, cast, c_ubyte, memset, sizeof
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from .capture_frame import CaptureSettings

_DEFAULT_DEV = Path(r"D:\桌面\明德实习资料\目前使用的机器人的资料\mvs\MVS\Development")
_DEFAULT_RUNTIME = Path(r"C:\Program Files (x86)\Common Files\MVS\Runtime\Win64_x64")
_MV_OK = 0
_sdk_refs = 0


def _project_root() -> Path:
    return Path(__file__).resolve().parent.parent


def resolve_mvs_paths(settings: CaptureSettings) -> tuple[Path, Path]:
    dev = (settings.mvs_development_dir or os.getenv("MVCAM_COMMON_RUNENV") or "").strip()
    rt = (settings.mvs_runtime_dir or "").strip()
    dev_path = Path(dev) if dev else _DEFAULT_DEV
    rt_path = Path(rt) if rt else _DEFAULT_RUNTIME
    return dev_path, rt_path


def _sdk_file_path(path: Path) -> str:
    """MVS FeatureSave/Load 仅接受 ASCII 路径；中文路径先走系统 Temp。"""
    path = path.resolve()
    text = str(path)
    if text.isascii():
        return text
    tmp = Path(tempfile.gettempdir()) / path.name
    return str(tmp)


def _feature_load(cam, path: Path) -> None:
    src = path.resolve()
    sdk_path = Path(_sdk_file_path(src))
    if sdk_path != src:
        if not src.is_file():
            raise RuntimeError(f"参数文件不存在: {src}")
        shutil.copy2(src, sdk_path)
    ret = cam.MV_CC_FeatureLoad(str(sdk_path))
    if ret != _MV_OK:
        raise RuntimeError(f"FeatureLoad 失败: {src} ret=0x{ret:x}")


def _feature_save(cam, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    sdk_path = Path(_sdk_file_path(path))
    ret = cam.MV_CC_FeatureSave(str(sdk_path))
    if ret != _MV_OK:
        raise RuntimeError(f"FeatureSave 失败: 0x{ret:x}")
    if sdk_path.resolve() != path.resolve():
        shutil.copy2(sdk_path, path)


def _resolve_feature_path(settings: CaptureSettings) -> Path | None:
    raw = (settings.mvs_feature_file or "").strip()
    if not raw:
        return None
    p = Path(raw)
    if not p.is_absolute():
        p = _project_root() / p
    return p


def prepare_mvs_import(development_dir: Path, runtime_dir: Path) -> None:
    if sys.platform != "win32":
        raise RuntimeError("海康 MVS Python 接口仅支持 Windows")
    if not development_dir.is_dir():
        raise RuntimeError(f"MVS Development 目录不存在: {development_dir}")
    mv_import = development_dir / "Samples" / "Python" / "MvImport"
    if not mv_import.is_dir():
        raise RuntimeError(f"未找到 MvImport: {mv_import}")
    if runtime_dir.is_dir():
        os.environ["PATH"] = str(runtime_dir) + os.pathsep + os.environ.get("PATH", "")
    os.environ.setdefault("MVCAM_COMMON_RUNENV", str(development_dir))
    imp = str(mv_import)
    if imp not in sys.path:
        sys.path.append(imp)


def _sdk_init() -> None:
    global _sdk_refs
    from MvCameraControl_class import MvCamera

    if _sdk_refs == 0:
        ret = MvCamera.MV_CC_Initialize()
        if ret != 0:
            raise RuntimeError(f"MVS SDK 初始化失败: 0x{ret:x}")
    _sdk_refs += 1


def _sdk_finalize() -> None:
    global _sdk_refs
    from MvCameraControl_class import MvCamera

    if _sdk_refs <= 0:
        return
    _sdk_refs -= 1
    if _sdk_refs == 0:
        MvCamera.MV_CC_Finalize()


def _is_hb(en_pixel: int, pt) -> bool:
    hb = (
        pt.PixelType_Gvsp_HB_Mono8,
        pt.PixelType_Gvsp_HB_BGR8_Packed,
        pt.PixelType_Gvsp_HB_RGB8_Packed,
    )
    return en_pixel in hb


def _is_mono(en_pixel: int, pt) -> bool:
    mono = (
        pt.PixelType_Gvsp_Mono8,
        pt.PixelType_Gvsp_Mono10,
        pt.PixelType_Gvsp_Mono12,
        pt.PixelType_Gvsp_Mono16,
    )
    return en_pixel in mono


def _frame_to_bgr(cam, st_out_frame, pt, mv_structs) -> np.ndarray:
    MV_CC_HB_DECODE_PARAM = mv_structs["MV_CC_HB_DECODE_PARAM"]
    MV_CC_PIXEL_CONVERT_PARAM_EX = mv_structs["MV_CC_PIXEL_CONVERT_PARAM_EX"]
    PixelType_Gvsp_Mono8 = pt.PixelType_Gvsp_Mono8
    PixelType_Gvsp_RGB8_Packed = pt.PixelType_Gvsp_RGB8_Packed

    st_convert = MV_CC_PIXEL_CONVERT_PARAM_EX()
    memset(byref(st_convert), 0, sizeof(st_convert))
    info = st_out_frame.stFrameInfo

    if _is_hb(info.enPixelType, pt):
        decode_len = info.nWidth * info.nHeight * 3
        decode_buf = (c_ubyte * decode_len)()
        st_decode = MV_CC_HB_DECODE_PARAM()
        st_decode.pSrcBuf = st_out_frame.pBufAddr
        st_decode.nSrcLen = info.nFrameLen
        st_decode.pDstBuf = decode_buf
        st_decode.nDstBufSize = decode_len
        ret = cam.MV_CC_HBDecode(st_decode)
        if ret != 0:
            raise RuntimeError(f"HB 解码失败: 0x{ret:x}")
        st_convert.pSrcData = st_decode.pDstBuf
        st_convert.nSrcDataLen = st_decode.nDstBufLen
        st_convert.enSrcPixelType = st_decode.enDstPixelType
    else:
        st_convert.pSrcData = st_out_frame.pBufAddr
        st_convert.nSrcDataLen = info.nFrameLen
        st_convert.enSrcPixelType = info.enPixelType

    if _is_mono(st_convert.enSrcPixelType, pt):
        dst_type = PixelType_Gvsp_Mono8
        channels = 1
    else:
        dst_type = PixelType_Gvsp_RGB8_Packed
        channels = 3

    convert_len = channels * info.nWidth * info.nHeight
    dst_buf = (c_ubyte * convert_len)()
    st_convert.nWidth = info.nWidth
    st_convert.nHeight = info.nHeight
    st_convert.enDstPixelType = dst_type
    st_convert.pDstBuffer = dst_buf
    st_convert.nDstBufferSize = convert_len
    ret = cam.MV_CC_ConvertPixelTypeEx(st_convert)
    if ret != 0:
        raise RuntimeError(f"像素格式转换失败: 0x{ret:x}")

    arr = np.frombuffer(dst_buf, dtype=np.uint8, count=convert_len)
    if channels == 1:
        gray = arr.reshape(info.nHeight, info.nWidth)
        return cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
    rgb = arr.reshape(info.nHeight, info.nWidth, 3)
    return cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)


def _pixel_types():
    import PixelType_header as pt

    return pt


def _mv_structs():
    from CameraParams_header import MV_CC_HB_DECODE_PARAM, MV_CC_PIXEL_CONVERT_PARAM_EX

    return {
        "MV_CC_HB_DECODE_PARAM": MV_CC_HB_DECODE_PARAM,
        "MV_CC_PIXEL_CONVERT_PARAM_EX": MV_CC_PIXEL_CONVERT_PARAM_EX,
    }


def _connect_camera(settings: CaptureSettings, *, load_feature: bool):
    from MvCameraControl_class import (
        MV_ACCESS_Exclusive,
        MV_CC_DEVICE_INFO,
        MV_CC_DEVICE_INFO_LIST,
        MV_GENTL_GIGE_DEVICE,
        MV_GIGE_DEVICE,
        MV_GENTL_CAMERALINK_DEVICE,
        MV_GENTL_CXP_DEVICE,
        MV_GENTL_XOF_DEVICE,
        MV_USB_DEVICE,
        MvCamera,
        POINTER,
    )

    device_list = MV_CC_DEVICE_INFO_LIST()
    tlayer = (
        MV_GIGE_DEVICE
        | MV_USB_DEVICE
        | MV_GENTL_CAMERALINK_DEVICE
        | MV_GENTL_CXP_DEVICE
        | MV_GENTL_XOF_DEVICE
    )
    ret = MvCamera.MV_CC_EnumDevices(tlayer, device_list)
    if ret != 0:
        raise RuntimeError(f"枚举相机失败: 0x{ret:x}")
    if device_list.nDeviceNum == 0:
        raise RuntimeError("未发现相机 — 请检查网线和 MVS Client 是否已关闭")

    idx = int(settings.mvs_device_index)
    if idx < 0 or idx >= device_list.nDeviceNum:
        raise RuntimeError(
            f"mvs_device_index={idx} 无效，当前共 {device_list.nDeviceNum} 台设备"
        )

    st_dev = cast(device_list.pDeviceInfo[idx], POINTER(MV_CC_DEVICE_INFO)).contents
    cam = MvCamera()
    ret = cam.MV_CC_CreateHandle(st_dev)
    if ret != 0:
        raise RuntimeError(f"CreateHandle 失败: 0x{ret:x}")
    ret = cam.MV_CC_OpenDevice(MV_ACCESS_Exclusive, 0)
    if ret != 0:
        cam.MV_CC_DestroyHandle()
        raise RuntimeError(f"打开相机失败(0x{ret:x}) — MVS 是否仍占用设备?")

    if st_dev.nTLayerType in (MV_GIGE_DEVICE, MV_GENTL_GIGE_DEVICE):
        pkt = cam.MV_CC_GetOptimalPacketSize()
        if int(pkt) > 0:
            cam.MV_CC_SetIntValue("GevSCPSPacketSize", pkt)

    if load_feature:
        feature = _resolve_feature_path(settings)
        if feature is not None and feature.is_file():
            _feature_load(cam, feature)

    from MvCameraControl_class import MV_TRIGGER_MODE_OFF

    cam.MV_CC_SetEnumValue("TriggerMode", MV_TRIGGER_MODE_OFF)
    return cam


def _close_camera(cam, *, was_grabbing: bool) -> None:
    if was_grabbing:
        cam.MV_CC_StopGrabbing()
    cam.MV_CC_CloseDevice()
    cam.MV_CC_DestroyHandle()


def export_camera_features(settings: CaptureSettings, dest: Path | None = None) -> Path:
    """从相机导出当前参数到 .mfs（请先 MVS 调亮并关闭 Client）。"""
    dev_dir, rt_dir = resolve_mvs_paths(settings)
    prepare_mvs_import(dev_dir, rt_dir)
    out = dest or (_project_root() / "config" / "mv_cu120_features.mfs")
    out.parent.mkdir(parents=True, exist_ok=True)

    _sdk_init()
    cam = None
    try:
        cam = _connect_camera(settings, load_feature=False)
        _feature_save(cam, out)
        return out
    finally:
        if cam is not None:
            _close_camera(cam, was_grabbing=False)
        _sdk_finalize()


def grab_one_frame_bgr_mvs(settings: CaptureSettings) -> tuple[np.ndarray, dict[str, Any]]:
    dev_dir, rt_dir = resolve_mvs_paths(settings)
    prepare_mvs_import(dev_dir, rt_dir)
    warmup = max(1, int(settings.mvs_warmup_frames))

    _sdk_init()
    cam = None
    try:
        cam = _connect_camera(settings, load_feature=True)
        pixel_types = _pixel_types()
        mv_structs = _mv_structs()
        ret = cam.MV_CC_StartGrabbing()
        if ret != 0:
            raise RuntimeError(f"StartGrabbing 失败: 0x{ret:x}")

        from MvCameraControl_class import MV_FRAME_OUT

        st_out = MV_FRAME_OUT()
        bgr: np.ndarray | None = None
        for _ in range(warmup):
            memset(byref(st_out), 0, sizeof(st_out))
            ret = cam.MV_CC_GetImageBuffer(st_out, 2000)
            if ret != 0 or st_out.pBufAddr is None:
                continue
            try:
                bgr = _frame_to_bgr(cam, st_out, pixel_types, mv_structs)
            finally:
                cam.MV_CC_FreeImageBuffer(st_out)

        if bgr is None:
            raise RuntimeError("预热后仍无法取到有效帧")

        w, h = int(settings.width), int(settings.height)
        if w > 0 and h > 0 and (bgr.shape[1] != w or bgr.shape[0] != h):
            bgr = cv2.resize(bgr, (w, h), interpolation=cv2.INTER_AREA)

        meta: dict[str, Any] = {
            "source": "mvs_gige",
            "mvs_device_index": settings.mvs_device_index,
            "feature_file": str(_resolve_feature_path(settings) or ""),
            "shape": list(bgr.shape),
        }
        return bgr, meta
    finally:
        if cam is not None:
            _close_camera(cam, was_grabbing=True)
        _sdk_finalize()
