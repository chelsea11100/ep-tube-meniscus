"""自定义取图插件：海康 / 机器人 SDK 等由你在独立模块里实现。"""

from __future__ import annotations

import importlib
from typing import Any, Callable

import numpy as np

from .capture_frame import CaptureSettings

GrabFn = Callable[[CaptureSettings], tuple[np.ndarray, dict[str, Any]]]

_loaded: GrabFn | None = None
_loaded_key: str | None = None


def load_grab_function(module: str, function: str = "grab_frame") -> GrabFn:
    mod = importlib.import_module(module)
    fn = getattr(mod, function, None)
    if fn is None or not callable(fn):
        raise RuntimeError(f"取图插件 {module}.{function} 不存在或不可调用")
    return fn


def grab_via_plugin(settings: CaptureSettings) -> tuple[np.ndarray, dict[str, Any]]:
    global _loaded, _loaded_key
    if not settings.plugin_module:
        raise RuntimeError("backend=plugin 但未配置 plugin_module")
    key = f"{settings.plugin_module}:{settings.plugin_function}"
    if _loaded is None or _loaded_key != key:
        _loaded = load_grab_function(settings.plugin_module, settings.plugin_function)
        _loaded_key = key
    frame, meta = _loaded(settings)
    if not isinstance(frame, np.ndarray) or frame.size == 0:
        raise RuntimeError("插件 grab_frame 必须返回非空 BGR numpy 图像")
    meta = dict(meta or {})
    meta.setdefault("source", "plugin")
    meta.setdefault("plugin_module", settings.plugin_module)
    return frame, meta
