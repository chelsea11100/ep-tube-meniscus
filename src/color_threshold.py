"""CanMV 风格颜色阈值：把选中的像素做成白 mask，再用白区域上沿当液面。"""

from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np

from .judge_meniscus import MeniscusResult, fill_percent_from_meniscus


SPACE_LIMITS: dict[str, list[tuple[str, int, int]]] = {
    "LAB": [("L", 0, 100), ("A", -128, 127), ("B", -128, 127)],
    "HSV": [("H", 0, 179), ("S", 0, 255), ("V", 0, 255)],
    "RGB": [("R", 0, 255), ("G", 0, 255), ("B", 0, 255)],
}

# 先对准浅粉色液柱：背景偏白（A≈0）和蓝色夹爪（A 为负）默认不进范围。
SPACE_DEFAULTS: dict[str, list[tuple[int, int]]] = {
    "LAB": [(18, 96), (6, 50), (-35, 35)],
    "HSV": [(0, 22), (18, 255), (50, 255)],
    "RGB": [(150, 255), (90, 220), (100, 230)],
}


@dataclass
class ColorSpec:
    space: str
    bounds: tuple[tuple[int, int], tuple[int, int], tuple[int, int]]
    invert: bool = False

    def text(self) -> str:
        nums = ", ".join(str(int(v)) for pair in self.bounds for v in pair)
        return f"{self.space}阈值 ({nums})"


def _in_range(ch: np.ndarray, lo: int, hi: int) -> np.ndarray:
    """最小值大于最大值时按 CanMV 环绕：通道 >= lo 或 <= hi。"""
    if lo <= hi:
        return (ch >= lo) & (ch <= hi)
    return (ch >= lo) | (ch <= hi)


def _channels(bgr: np.ndarray, space: str) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    space = space.upper()
    if space == "LAB":
        lab = cv2.cvtColor(bgr, cv2.COLOR_BGR2LAB).astype(np.int16)
        lightness = lab[:, :, 0] * 100 // 255
        a_star = lab[:, :, 1] - 128
        b_star = lab[:, :, 2] - 128
        return lightness, a_star, b_star
    if space == "HSV":
        hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
        return hsv[:, :, 0], hsv[:, :, 1], hsv[:, :, 2]
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    return rgb[:, :, 0], rgb[:, :, 1], rgb[:, :, 2]


def color_mask(bgr: np.ndarray, spec: ColorSpec) -> np.ndarray:
    channels = _channels(bgr, spec.space)
    keep = np.ones(bgr.shape[:2], dtype=bool)
    for channel, (lo, hi) in zip(channels, spec.bounds):
        keep &= _in_range(channel, int(lo), int(hi))
    out = keep.astype(np.uint8) * 255
    if spec.invert:
        out = cv2.bitwise_not(out)
    return out


def meniscus_y_from_mask(
    mask: np.ndarray,
    tube_top: int,
    tube_bottom: int,
    x_left: int,
    x_right: int,
) -> tuple[int | None, float, int, int]:
    """白色像素里，贴着管底的那一段上沿就是液面。"""
    h, w = mask.shape[:2]
    x0 = int(np.clip(x_left, 0, max(0, w - 2)))
    x1 = int(np.clip(x_right, x0 + 2, w))
    y0 = int(np.clip(tube_top, 0, max(0, h - 2)))
    y1 = int(np.clip(tube_bottom, y0 + 2, h))
    roi = mask[y0:y1, x0:x1] > 0
    if roi.size == 0:
        return None, 0.0, x0, x1
    col = roi.sum(axis=0)
    if int(col.max()) < 4:
        return None, 0.0, x0, x1
    xs = np.where(col >= max(4, int(col.max() * 0.45)))[0]
    if xs.size < 1:
        return None, 0.0, x0, x1
    blob_x0 = x0 + int(xs[0])
    blob_x1 = x0 + int(xs[-1]) + 1
    band = roi[:, int(xs[0]) : int(xs[-1]) + 1]
    on = band.mean(axis=1) >= 0.28
    ys = np.where(on)[0]
    if ys.size < 4:
        return None, 0.0, blob_x0, blob_x1
    runs: list[np.ndarray] = []
    start = 0
    for i in range(1, int(ys.size) + 1):
        if i == ys.size or int(ys[i]) > int(ys[i - 1]) + 4:
            runs.append(ys[start:i])
            start = i
    span = max(1, y1 - y0)
    best: np.ndarray | None = None
    for run in runs:
        if run.size < 4:
            continue
        if int(run[-1]) < int(span * 0.55):
            continue
        if best is None or int(run[-1]) > int(best[-1]):
            best = run
    if best is None:
        return None, 0.0, blob_x0, blob_x1
    y = y0 + int(best[0])
    conf = min(0.95, 0.55 + float(best.size) / max(10.0, float(span)))
    return y, float(conf), blob_x0, blob_x1


def meniscus_with_color(crop_bgr: np.ndarray, ms: MeniscusResult, spec: ColorSpec) -> MeniscusResult:
    mask = color_mask(crop_bgr, spec)
    width = crop_bgr.shape[1]
    y, conf, x0, x1 = meniscus_y_from_mask(mask, ms.tube_top, ms.tube_bottom, 0, width)
    if y is None:
        return MeniscusResult(None, 0.0, conf, ms.tube_top, ms.tube_bottom, ms.x_left, ms.x_right)
    fill = fill_percent_from_meniscus(y, ms.tube_top, ms.tube_bottom)
    return MeniscusResult(y, fill, conf, ms.tube_top, ms.tube_bottom, x0, min(width, x1))
