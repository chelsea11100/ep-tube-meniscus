from __future__ import annotations

import json
from dataclasses import dataclass

import cv2
import numpy as np

from .common import load_config, log_verbose
from .detect_tube import put_chinese
from .judge_liquid import LiquidJudgement, clamp_int, clamp_float


@dataclass
class MeniscusResult:
    meniscus_y: int | None
    fill_percent: float
    peak_score: float
    tube_top: int
    tube_bottom: int
    x_left: int
    x_right: int


class LocalLiquidJudge:
    """背光 EP 管 crop 上检测弯月面横线，按管身高换算 fill_percent。"""

    def __init__(self, config: dict | None = None, verbose: bool | None = None):
        self.config = config or load_config()
        self.verbose = bool(self.config.get("verbose", True)) if verbose is None else verbose
        self.min_fill_percent = int(self.config.get("min_fill_percent", 10))
        liq = self.config.get("liquid") or {}

        self.tube_top_frac = float(liq.get("tube_top_frac", 0.12))
        self.tube_bottom_frac = float(liq.get("tube_bottom_frac", 0.05))
        self.center_x_left = float(liq.get("center_x_left", 0.28))
        self.center_x_right = float(liq.get("center_x_right", 0.72))
        self.meniscus_min_peak = float(liq.get("meniscus_min_peak", 0.40))
        self.empty_peak_max = float(liq.get("empty_peak_max", 0.32))

    def judge(self, image: np.ndarray) -> LiquidJudgement:
        ms = detect_meniscus(
            image,
            self.tube_top_frac,
            self.tube_bottom_frac,
            self.center_x_left,
            self.center_x_right,
            self.meniscus_min_peak,
            self.empty_peak_max,
        )
        level_vis = draw_level_overlay(image, ms)
        fill = clamp_int(ms.fill_percent, 0, 100)
        liquid_present = ms.meniscus_y is not None and ms.fill_percent > 1.0
        if ms.meniscus_y is None:
            liquid_present = False
            fill = 0
        confidence = clamp_float(ms.peak_score if liquid_present else 1.0 - ms.peak_score, 0.0, 1.0)
        if liquid_present:
            confidence = clamp_float(0.55 + ms.peak_score * 0.45, 0.0, 1.0)
        too_little = ms.meniscus_y is None or ms.fill_percent < self.min_fill_percent
        if ms.meniscus_y is None:
            evidence = "未检测到明显弯月面横线，判定为空或极低液量"
        else:
            evidence = (
                f"本地弯月面 y={ms.meniscus_y}px，相对管身高度约 {fill}%，"
                f"峰强度 {ms.peak_score:.2f}"
            )
        raw = json.dumps(
            {
                "method": "meniscus_local",
                "meniscus_y": ms.meniscus_y,
                "fill_percent": fill,
                "peak_score": round(ms.peak_score, 4),
                "tube_top": ms.tube_top,
                "tube_bottom": ms.tube_bottom,
                "x_left": ms.x_left,
                "x_right": ms.x_right,
            },
            ensure_ascii=False,
        )
        log_verbose(
            self.verbose,
            f"[液面] meniscus_y={ms.meniscus_y} fill={fill}% peak={ms.peak_score:.3f} "
            f"too_little={too_little}",
        )
        return LiquidJudgement(
            fill_percent=fill,
            liquid_present=liquid_present,
            too_little=too_little,
            confidence=confidence,
            evidence=evidence,
            raw_text=raw,
            level_image=level_vis,
            meniscus_y=ms.meniscus_y,
        )

 
def _row_tube_width(row: np.ndarray, tube_thr: float, w: int) -> int:
    x_lo, x_hi = int(w * 0.12), int(w * 0.88)
    xs = np.where(row[x_lo:x_hi] < tube_thr)[0]
    if xs.size < 4:
        return 0
    xs = xs + x_lo
    return int(xs[-1] - xs[0] + 1)


def _edge_width_profile(gray: np.ndarray) -> np.ndarray:
    """竖向管壁边缘跨度。透光锥尖往往不够暗，暗像素宽度会在液柱处断掉。"""
    h, w = gray.shape[:2]
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    sx = np.abs(cv2.Sobel(blur, cv2.CV_32F, 1, 0, ksize=3))
    sx = cv2.GaussianBlur(sx, (5, 9), 0)
    x_lo, x_hi = int(w * 0.08), int(w * 0.92)
    cx = w / 2.0
    widths = np.zeros(h, dtype=np.float32)
    band = sx[:, x_lo:x_hi]
    if band.size == 0:
        return widths
    for y in range(h):
        row = band[y]
        thr = max(float(row.mean()) * 2.4, float(np.percentile(row, 90)))
        xs = np.where(row >= thr)[0]
        if xs.size < 2:
            continue
        left = int(xs[0] + x_lo)
        right = int(xs[-1] + x_lo)
        span = right - left
        if span < 6 or span > w * 0.62:
            continue
        if abs((left + right) / 2.0 - cx) > w * 0.30:
            continue
        widths[y] = float(span)
    return np.convolve(widths, np.ones(7, dtype=np.float32) / 7.0, mode="same")


def _y_bottom_from_last_segment(
    widths: np.ndarray, y_top: int, h: int, max_w: float
) -> int | None:
    """手指遮挡中间时宽度为 0；取最下方「够宽」管段里的锥尖（非第一段假窄行）。"""
    min_thr = max(8.0, max_w * 0.11)
    peak_thr = max_w * 0.35
    y_min = y_top + max(12, int((h - y_top) * 0.22))
    segments: list[tuple[int, int]] = []
    in_seg = False
    seg_start = 0
    for y in range(y_min, h):
        ok = float(widths[y]) >= min_thr
        if ok and not in_seg:
            seg_start = y
            in_seg = True
        elif not ok and in_seg:
            segments.append((seg_start, y - 1))
            in_seg = False
    if in_seg:
        segments.append((seg_start, h - 1))
    good = [(a, b) for a, b in segments if float(widths[a : b + 1].max()) >= peak_thr]
    if not good:
        return None
    s0, s1 = good[-1]
    seg = widths[s0 : s1 + 1]
    tail_start = int(len(seg) * 0.32)
    tail = seg[tail_start:]
    if tail.size == 0:
        return s1
    rel = int(np.argmin(tail))
    return s0 + tail_start + rel


def _detect_tube_bounds(
    gray: np.ndarray,
    tube_top_frac: float,
    tube_bottom_frac: float,
    center_x_left: float,
    center_x_right: float,
) -> tuple[int, int, int, int]:
    """背光图：管口=管宽足够大；锥尖=下半段管宽最窄行（避免中心液柱误判）。"""
    h, w = gray.shape[:2]
    x0_def = int(w * center_x_left)
    x1_def = int(w * center_x_right)
    cx = w // 2

    bg = float(np.percentile(gray, 94))
    tube_thr = bg - 12.0

    widths = np.array([_row_tube_width(gray[y], tube_thr, w) for y in range(h)], dtype=np.float32)
    if float(widths.max()) < 5:
        return int(h * tube_top_frac), int(h * (1.0 - tube_bottom_frac)), x0_def, x1_def

    widths = np.convolve(widths, np.ones(5, dtype=np.float32) / 5.0, mode="same")
    max_w = float(widths.max())

    y_top = 0
    for y in range(h):
        if widths[y] >= max_w * 0.30:
            y_top = y
            break

    y_start = y_top + max(8, int((h - y_top) * 0.42))
    cone = widths[y_start:h].copy()
    cone[cone < 4] = np.inf
    if not np.isfinite(cone).any():
        y_bottom = min(h - 1, int(h * (1.0 - tube_bottom_frac)))
    else:
        y_bottom = y_start + int(np.argmin(cone))

    y_seg = _y_bottom_from_last_segment(widths, y_top, h, max_w)
    if y_seg is not None and y_seg > y_bottom:
        y_bottom = y_seg
    y_cone = y_bottom

    tube_w_px = max(8, x1_def - x0_def, int(max_w + 4))
    max_h = int(tube_w_px * 8.6)
    if y_bottom - y_top > max_h:
        y_bottom = y_top + max_h

    tip_w = float(widths[y_bottom])
    narrow = tip_w * 1.18 + 2.0
    for y in range(y_bottom, min(h, y_bottom + 12)):
        if 4.0 <= float(widths[y]) <= narrow:
            y_bottom = y

    # 只有画面底边仍是很宽的管身时，才说明锥尖被裁出画面，往下补一截。
    if y_bottom >= h - 8 and float(widths[min(h - 2, y_bottom)]) > max_w * 0.55:
        y_bottom = min(h - 1, y_bottom + int(max_w * 0.35))

    # 暗像素宽度会在透光锥尖/液柱处断开。沿边缘继续向下，直到跨度突然变宽（管外背景）。
    edge_w = _edge_width_profile(gray)
    start_w = float(edge_w[min(y_bottom, h - 1)])
    if start_w < 6.0:
        start_w = float(widths[min(y_bottom, h - 1)])
    width_cap = max(start_w * 1.22, start_w + 14.0, 10.0)
    gap = 0
    limit = min(h - 1, y_top + int(max(8.0, max_w) * 8.6))
    x_c0, x_c1 = int(w * 0.36), int(w * 0.64)
    bright_thr = bg - 10.0
    bright_run = 0
    last_tube_row = y_bottom
    max_extend = max(14, int(max_w * 0.45))
    for y2 in range(y_bottom + 1, min(limit + 1, y_bottom + max_extend + 1)):
        cent = float(gray[y2, x_c0:x_c1].mean())
        if cent >= bright_thr:
            bright_run += 1
            if bright_run >= 2:
                y_bottom = last_tube_row
                break
        else:
            bright_run = 0
        ew = float(edge_w[y2])
        if ew < 4.0 or ew > width_cap:
            gap += 1
            if gap > 4:
                break
            continue
        y_bottom = y2
        last_tube_row = y2
        gap = 0

    if y_bottom >= h - 10:
        pull = y_cone
        if y_seg is not None:
            pull = min(pull, y_seg)
        y_bottom = min(y_bottom, pull + max(6, int(max_w * 0.2)))
    # 手拿/整图时管身宽度常撑满画面底，勿把底缘贴在最后一行
    if y_bottom >= h - 12 and max_w > w * 0.38:
        y_bottom = min(y_bottom, max(y_top + 20, y_cone, int(h * 0.88)))

    y_top = max(0, y_top)
    y_bottom = min(h - 1, max(y_top + 12, y_bottom))

    x0, x1 = _horizontal_edges_at_row(gray, y_bottom, tube_thr, cx)
    x0t, x1t = _horizontal_edges_at_row(gray, y_top, tube_thr, cx)
    x0 = min(x0, x0t)
    x1 = max(x1, x1t)
    if x1 - x0 < w * 0.12:
        x0, x1 = x0_def, x1_def
    return y_top, y_bottom, x0, x1


def _horizontal_edges_at_row(gray: np.ndarray, y: int, tube_thr: float, cx: int) -> tuple[int, int]:
    h, w = gray.shape[:2]
    y = max(0, min(h - 1, y))
    row = gray[y, :]
    x_left = cx
    for x in range(cx, 0, -1):
        if row[x] < tube_thr:
            x_left = x
            break
    x_right = cx
    for x in range(cx, w):
        if row[x] < tube_thr:
            x_right = x
            break
    pad = max(2, w // 40)
    return max(0, x_left - pad), min(w - 1, x_right + pad)


def _meniscus_from_saturated_liquid(roi_bgr: np.ndarray) -> tuple[int | None, float]:
    """粉/红等有色液体：最大饱和连通域的上缘为弯月面（忽略管口零星染色）。"""
    h, w = roi_bgr.shape[:2]
    if h < 40 or w < 12:
        return None, 0.0
    hsv = cv2.cvtColor(roi_bgr, cv2.COLOR_BGR2HSV)
    sat = hsv[:, :, 1]
    val = hsv[:, :, 2]
    mask = ((sat > 24) & (val > 38)).astype(np.uint8) * 255
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=1)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=2)
    mask[: max(1, int(h * 0.22)), :] = 0
    nz = int(cv2.countNonZero(mask))
    if nz < 60:
        return None, 0.0

    n_labels, labels, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
    best_label = -1
    best_score = -1.0
    for lab in range(1, n_labels):
        area = int(stats[lab, cv2.CC_STAT_AREA])
        if area < 40:
            continue
        y0 = int(stats[lab, cv2.CC_STAT_TOP])
        y1 = y0 + int(stats[lab, cv2.CC_STAT_HEIGHT])
        cy = 0.5 * (y0 + y1)
        # 真液柱多在下半段；管口小噪点面积小且靠上
        bottom_bias = 0.35 + 0.65 * (cy / max(1, h - 1))
        score = area * bottom_bias
        if score > best_score:
            best_score = score
            best_label = lab
    if best_label < 0:
        return None, 0.0

    blob = labels == best_label
    ys = np.where(blob)[0]
    if ys.size == 0:
        return None, 0.0
    y_top_liq = int(np.min(ys))
    y_bot_liq = int(np.max(ys))
    blob_area = int(np.count_nonzero(blob))

    sat_top = float(sat[int(h * 0.22) : int(h * 0.42), :].mean())
    sat_bot = float(sat[int(h * 0.72) :, :].mean())
    # 持管时管口/手指常高饱和；真液柱应贴底或明显长于管口噪点
    reaches_bottom = y_bot_liq >= int(h * 0.86)
    tall_enough = (y_bot_liq - y_top_liq) >= int(h * 0.22)
    finger_stain = (
        y_top_liq < int(h * 0.38)
        and sat_top > sat_bot + 18.0
        and not reaches_bottom
    )
    if finger_stain or (not reaches_bottom and not tall_enough):
        return None, 0.0

    mean_sat = float(sat[blob].mean()) if blob.any() else 0.0
    if mean_sat < 38.0:
        return None, 0.0

    conf = min(0.92, 0.45 + (blob_area / max(1, mask.size)) * 12.0)
    return y_top_liq, float(conf)


def _meniscus_from_center_pink(
    roi_bgr: np.ndarray,
) -> tuple[int | None, float, int | None, int, int]:
    """找贴着这段画面底部的粉/红液柱上沿。背光白底的红绿差很小，不算液体。"""
    h, w = roi_bgr.shape[:2]
    if h < 48 or w < 10:
        return None, 0.0, None, 0, 0
    _blue, green, red = cv2.split(roi_bgr)
    rg = red.astype(np.float32) - green.astype(np.float32)
    hsv = cv2.cvtColor(roi_bgr, cv2.COLOR_BGR2HSV)
    hue, sat, val = hsv[:, :, 0], hsv[:, :, 1], hsv[:, :, 2]
    pink = (
        ((hue < 20) | (hue > 150))
        & (sat > 12)
        & (val > 70)
        & (rg > 8)
    ).astype(np.uint8)
    if int(pink.sum()) < 40:
        return None, 0.0, None, 0, 0
    n, _labels, stats, _cent = cv2.connectedComponentsWithStats(pink, 8)
    best: tuple[int, int, int, int, int] | None = None
    for i in range(1, n):
        x, y, ww, hh, area = (int(v) for v in stats[i])
        if area < 40 or hh < 8 or ww < 4:
            continue
        bottom = y + hh - 1
        if h - 1 - bottom > int(h * 0.28):
            continue
        if y < int(h * 0.12):
            continue
        if best is None or area > best[0]:
            best = (area, y, bottom, x, x + ww)
    if best is None:
        return None, 0.0, None, 0, 0
    _area, y_top_liq, y_bot_liq, x0, x1 = best
    block = rg[y_top_liq : y_bot_liq + 1, x0:x1]
    mask = pink[y_top_liq : y_bot_liq + 1, x0:x1] > 0
    liquid = block[mask]
    if liquid.size < 20:
        return None, 0.0, None, 0, 0
    above = rg[max(0, y_top_liq - 36) : y_top_liq, x0:x1]
    base = float(np.median(above)) if above.size else 0.0
    delta = float(liquid.mean()) - base
    if delta < 4.0:
        return None, 0.0, None, 0, 0
    conf = min(0.93, 0.62 + delta / 40.0)
    return int(y_top_liq), float(conf), int(y_bot_liq), int(x0), int(x1)


def _meniscus_from_bottom_tint(
    roi_bgr: np.ndarray, *, grip: bool = False
) -> tuple[int | None, float]:
    """淡粉/红：从管底向上找液柱上缘（饱和度很低时也适用）。"""
    h, w = roi_bgr.shape[:2]
    if h < 50 or w < 12:
        return None, 0.0
    y0 = int(h * (0.68 if grip else 0.50))
    x0, x1 = int(w * 0.22), int(w * 0.78)
    if x1 <= x0 + 4:
        return None, 0.0
    band = roi_bgr[y0:, x0:x1]
    bh = band.shape[0]
    if bh < 16:
        return None, 0.0
    hsv = cv2.cvtColor(band, cv2.COLOR_BGR2HSV)
    b, g, r = cv2.split(band)
    sat = hsv[:, :, 1].astype(np.float32)
    rg = r.astype(np.float32) - g.astype(np.float32)
    rb = r.astype(np.float32) - b.astype(np.float32)
    row_sig = np.maximum.reduce(
        [sat.mean(axis=1), rg.mean(axis=1) * 1.05, rb.mean(axis=1) * 0.85]
    )
    row_sig = np.convolve(row_sig, np.ones(5, dtype=np.float32) / 5.0, mode="same")
    base = float(np.median(row_sig[: max(4, bh // 6)]))
    peak = float(row_sig.max())
    min_delta = 3.8 if grip else 5.5
    if peak < base + min_delta:
        return None, 0.0
    thr = base + max(3.2 if grip else 4.5, (peak - base) * (0.22 if grip else 0.28))
    liquid_rows: list[int] = []
    for i in range(bh - 1, -1, -1):
        if row_sig[i] >= thr:
            liquid_rows.append(i)
        elif len(liquid_rows) >= 4:
            break
        elif liquid_rows and i < bh - 3:
            break
    if len(liquid_rows) < 2:
        return None, 0.0
    y_top_liq = min(liquid_rows)
    span = max(liquid_rows) - y_top_liq + 1
    rel_y = y0 + y_top_liq
    if rel_y < int(h * (0.62 if grip else 0.55)):
        return None, 0.0
    conf = min(0.88, 0.40 + span / max(1, bh) * 2.2 + (peak - base) / 40.0)
    return int(rel_y), float(conf)


def detect_meniscus(
    crop_bgr: np.ndarray,
    tube_top_frac: float,
    tube_bottom_frac: float,
    center_x_left: float,
    center_x_right: float,
    meniscus_min_peak: float,
    empty_peak_max: float,
    black_max: int = 100,
    ignore_top_frac: float = 0.0,
) -> MeniscusResult:
    h, w = crop_bgr.shape[:2]
    gray_full = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2GRAY)
    y_top, y_bot, x0, x1 = _detect_tube_bounds(
        gray_full, tube_top_frac, tube_bottom_frac, center_x_left, center_x_right
    )
    if y_bot <= y_top + 8 or x1 <= x0 + 8:
        return MeniscusResult(None, 0.0, 0.0, y_top, y_bot, x0, x1)

    ignore_frac = max(0.0, min(0.8, float(ignore_top_frac)))
    ignore_y = y_top + int((y_bot - y_top) * ignore_frac)
    roi = crop_bgr[y_top:y_bot, x0:x1]
    gray_early = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    grip_early = (
        float((gray_early[: max(1, int(roi.shape[0] * 0.30))] < 92).mean())
        if roi.shape[0] > 16
        else 0.0
    )
    grip_mode = grip_early > 0.32
    pink_band = crop_bgr[y_top:y_bot, :]
    pink_y, pink_conf, pink_bot, pink_x0, pink_x1 = _meniscus_from_center_pink(pink_band)
    if pink_bot is not None:
        pink_bot_abs = y_top + int(pink_bot)
        if pink_y is not None:
            pink_y = y_top + int(pink_y)
        # 底线贴在画面最后一行、粉色液体还在上面：锥尖就在液柱下面一点。
        if y_bot >= h - 8 and pink_bot_abs + 6 < y_bot:
            if pink_y is not None:
                pad = max(8, int((pink_bot_abs - pink_y) * 0.15))
            else:
                pad = max(8, min(24, int((pink_bot_abs - y_top) * 0.06)))
            y_bot = min(h - 1, pink_bot_abs + pad)
            if pink_x1 - pink_x0 >= 6:
                x0, x1 = int(pink_x0), min(w - 1, int(pink_x1))
            ignore_y = y_top + int((y_bot - y_top) * ignore_frac)
    if (
        pink_y is not None
        and pink_conf >= 0.55
        and ignore_y + 2 < pink_y < y_bot - 2
    ):
        cy = int(pink_y)
        tube_h = max(1, y_bot - y_top)
        fill = 100.0 * (y_bot - cy) / tube_h
        if fill < 45:
            fill = fill * 0.35
        fill = max(0.0, min(100.0, fill))
        return MeniscusResult(cy, fill, pink_conf, y_top, y_bot, x0, x1)

    color_rel, color_conf = _meniscus_from_saturated_liquid(roi)
    bottom_rel, bottom_conf = _meniscus_from_bottom_tint(roi, grip=grip_mode)

    def _result_from_color() -> MeniscusResult | None:
        if color_rel is None or color_conf < 0.48:
            return None
        cy = y_top + int(color_rel)
        tube_h = max(1, y_bot - y_top)
        fill = 100.0 * (y_bot - cy) / tube_h
        if fill < 45:
            fill = fill * 0.35
        fill = max(0.0, min(100.0, fill))
        return MeniscusResult(cy, fill, color_conf, y_top, y_bot, x0, x1)

    def _result_from_bottom_tint(force: bool = False) -> MeniscusResult | None:
        rel, conf = bottom_rel, bottom_conf
        if force or rel is None:
            rel, conf = _meniscus_from_bottom_tint(roi, grip=True)
        min_conf = 0.38 if grip_mode else 0.46
        if rel is None or conf < min_conf:
            return None
        cy = y_top + int(rel)
        tube_h = max(1, y_bot - y_top)
        fill = 100.0 * (y_bot - cy) / tube_h
        if fill < 45:
            fill = fill * 0.35
        fill = max(0.0, min(100.0, fill))
        return MeniscusResult(cy, fill, conf, y_top, y_bot, x0, x1)

    if grip_mode:
        grip_first = _result_from_bottom_tint(force=True)
        if grip_first is not None:
            return grip_first

    roi_edge = roi
    if ignore_frac > 0.01:
        roi_edge = roi.copy()
        cut = int(roi_edge.shape[0] * ignore_frac)
        if cut > 2:
            gray_cut = cv2.cvtColor(roi_edge[:cut], cv2.COLOR_BGR2GRAY)
            roi_edge[:cut][gray_cut < int(black_max)] = (248, 248, 248)
    gray = cv2.cvtColor(roi_edge, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (3, 3), 0)
    sobel_y = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
    row_edge = np.mean(np.abs(sobel_y), axis=1)
    row_dark = 255.0 - np.mean(gray.astype(np.float32), axis=1)
    gw = gray.shape[1]
    cx0, cx1 = int(gw * 0.36), int(gw * 0.64)
    sobel_c = cv2.Sobel(gray[:, cx0:cx1], cv2.CV_32F, 0, 1, ksize=3)
    row_edge_c = np.mean(np.abs(sobel_c), axis=1)

    def _norm(arr: np.ndarray) -> np.ndarray:
        lo, hi = float(arr.min()), float(arr.max())
        return (arr - lo) / (hi - lo + 1e-6)

    score = 0.55 * _norm(row_edge) + 0.45 * _norm(row_dark)
    score_c = 0.70 * _norm(row_edge_c) + 0.30 * _norm(row_dark)
    kernel = np.ones(9, dtype=np.float32) / 9.0
    score_sm = np.convolve(score, kernel, mode="same")
    score_sm_c = np.convolve(score_c, kernel, mode="same")
    # 夹爪是成块的黑，液面在亮的管壁上。黑块以及它下面几行的强边缘都不当液面。
    min_run = max(16, int(gray.shape[1] * 0.05))
    solid = cv2.erode((gray < 88).astype(np.uint8), cv2.getStructuringElement(cv2.MORPH_RECT, (min_run, 1)))
    dark_rows = solid.any(axis=1)
    suppress = np.zeros(len(dark_rows), dtype=bool)
    rad = max(12, int(len(dark_rows) * 0.08))
    for i in np.where(dark_rows)[0]:
        if i > len(dark_rows) * 0.55:
            continue
        suppress[i : min(len(dark_rows), int(i) + rad)] = True
    score_sm = score_sm.copy()
    score_sm_c = score_sm_c.copy()
    score_sm[suppress] = 0.0
    score_sm_c[suppress] = 0.0

    tube_thr = float(np.percentile(gray_full, 94)) - 12.0
    widths_full = np.array(
        [_row_tube_width(gray_full[y], tube_thr, w) for y in range(h)], dtype=np.float32
    )
    widths_full = np.convolve(widths_full, np.ones(5, dtype=np.float32) / 5.0, mode="same")
    tube_span = widths_full[y_top : y_bot + 1]
    max_w_full = float(tube_span.max()) if tube_span.size else float(widths_full.max())

    rh = len(score_sm)
    grip_top = float((gray[: max(1, int(rh * 0.30))] < 92).mean()) if rh > 16 else 0.0
    if grip_top > 0.45:
        search_lo = int(rh * 0.58)
    elif grip_top > 0.32:
        search_lo = int(rh * 0.42)
    else:
        search_lo = int(rh * 0.28)
    search_lo = max(search_lo, int(rh * ignore_frac))
    search_hi = int(rh * 0.94)
    edge_skip = max(12, int(rh * 0.05))
    lo = min(search_hi - 4, search_lo + edge_skip)
    if search_hi <= lo + 4:
        return MeniscusResult(None, 0.0, 0.0, y_top, y_bot, x0, x1)

    segment = score_sm[lo:search_hi]
    rel_idx = int(np.argmax(segment))
    best_val = float(segment[rel_idx])
    meniscus_idx = lo + rel_idx

    # 管口/搜索窗上沿常出现假峰；若明显高于管身中段，改取中段最强峰
    mid_lo, mid_hi = int(rh * 0.30), int(rh * 0.78)
    if mid_hi > mid_lo + 8:
        mid_seg = score_sm[mid_lo:mid_hi]
        mid_best = float(mid_seg.max())
        mid_idx = mid_lo + int(np.argmax(mid_seg))
        if meniscus_idx < mid_lo and best_val > mid_best * 1.05 and mid_best >= meniscus_min_peak * 0.85:
            meniscus_idx = mid_idx
            best_val = mid_best

    if meniscus_idx < int(rh * 0.40) and grip_top > 0.32:
        if mid_hi > mid_lo + 8 and mid_best >= meniscus_min_peak * 0.72:
            meniscus_idx = mid_idx
            best_val = mid_best
        else:
            lo2 = int(rh * (0.40 if grip_top > 0.45 else 0.38))
            if search_hi > lo2 + 8:
                lower = score_sm[lo2:search_hi]
                li = lo2 + int(np.argmax(lower))
                if grip_top > 0.45 or float(score_sm[li]) >= best_val * 0.42:
                    meniscus_idx = li
                    best_val = float(score_sm[li])

    meniscus_y_try = y_top + int(meniscus_idx)
    rel_y = (meniscus_y_try - y_top) / max(1, y_bot - y_top)
    taper_row = (
        max_w_full > 5
        and 0 <= meniscus_y_try < h
        and float(widths_full[meniscus_y_try]) < max_w_full * 0.72
        and rel_y > 0.52
    )
    used_taper_refine = False
    if taper_row:
        orig_idx, orig_val = meniscus_idx, best_val
        taper_lo = int(rh * 0.38)
        taper_hi = search_hi
        if taper_hi > taper_lo + 10:
            seg_c = score_sm_c[taper_lo:taper_hi]
            peaks: list[tuple[int, float]] = []
            for i in range(1, len(seg_c) - 1):
                if seg_c[i] >= seg_c[i - 1] and seg_c[i] > seg_c[i + 1]:
                    if seg_c[i] >= meniscus_min_peak * 0.42:
                        peaks.append((taper_lo + i, float(seg_c[i])))
            if peaks:
                below = [
                    (i, v)
                    for i, v in peaks
                    if i > orig_idx and v >= meniscus_min_peak * 0.52
                ]
                if below:
                    new_idx, new_val = max(below, key=lambda t: t[1])
                    meniscus_idx, best_val = new_idx, new_val
                    used_taper_refine = True
            else:
                rel_c = int(np.argmax(seg_c))
                new_val = float(seg_c[rel_c])
                new_idx = taper_lo + rel_c
                if new_idx > orig_idx and new_val >= meniscus_min_peak * 0.52:
                    meniscus_idx, best_val = new_idx, new_val
                    used_taper_refine = True

    if best_val < empty_peak_max:
        for resolver in (_result_from_bottom_tint, _result_from_color):
            alt = resolver()
            if alt is not None:
                return alt
        return MeniscusResult(None, 0.0, best_val, y_top, y_bot, x0, x1)

    min_peak = meniscus_min_peak * (0.55 if used_taper_refine else 1.0)
    prominence = best_val - float(np.median(segment))
    grip_relaxed = grip_top > 0.45 and meniscus_idx > int(rh * 0.38)
    if (best_val < min_peak or prominence < 0.06) and not grip_relaxed:
        for resolver in (_result_from_bottom_tint, _result_from_color):
            alt = resolver()
            if alt is not None:
                return alt
        return MeniscusResult(None, 0.0, best_val, y_top, y_bot, x0, x1)

    meniscus_y = y_top + int(meniscus_idx)
    tube_h = max(1, y_bot - y_top)
    fill = 100.0 * (y_bot - meniscus_y) / tube_h
    rel_m = (meniscus_y - y_top) / tube_h

    # 夹爪/管口边缘常在上半段；粉液上缘更低时以颜色为准
    tint_res = _result_from_bottom_tint(force=True)
    if (
        tint_res is not None
        and tint_res.meniscus_y is not None
        and rel_m < 0.48
        and tint_res.meniscus_y > meniscus_y + int(tube_h * 0.10)
    ):
        return tint_res

    # 持管时边缘常跟到管口/手指；液面应在下半段
    if grip_top > 0.32 and rel_m < 0.50:
        for resolver in (
            lambda: _result_from_bottom_tint(force=True),
            _result_from_color,
        ):
            alt = resolver()
            if alt is not None:
                return alt
        return MeniscusResult(None, 0.0, best_val, y_top, y_bot, x0, x1)

    if color_rel is not None:
        color_y = y_top + int(color_rel)
        color_fill = 100.0 * (y_bot - color_y) / tube_h
        rel_edge = (meniscus_y - y_top) / tube_h
        rel_color = (color_y - y_top) / tube_h
        use_color = fill > 48 and color_fill < fill - 12
        use_color = use_color or (rel_edge < 0.45 and color_y > meniscus_y + 8)
        use_color = use_color or (best_val < meniscus_min_peak and color_conf >= 0.5)
        # 边缘误跟手指/管口（高 fill），有色液块在下方
        use_color = use_color or (
            fill > 55
            and rel_color > 0.52
            and color_fill < fill - 18
            and color_conf >= 0.55
        )
        if use_color:
            meniscus_y = color_y
            fill = color_fill
            best_val = color_conf

    if meniscus_y < ignore_y:
        tint_below = _result_from_bottom_tint(force=True)
        if tint_below is not None and tint_below.meniscus_y is not None and tint_below.meniscus_y >= ignore_y:
            return tint_below
        return MeniscusResult(None, 0.0, best_val, y_top, y_bot, x0, x1)

    if fill < 45:
        fill = fill * 0.35
    fill = max(0.0, min(100.0, fill))
    return MeniscusResult(meniscus_y, fill, best_val, y_top, y_bot, x0, x1)


def fill_percent_from_meniscus(meniscus_y: int, tube_top: int, tube_bottom: int) -> float:
    tube_h = max(1, tube_bottom - tube_top)
    fill = 100.0 * (tube_bottom - meniscus_y) / tube_h
    if fill < 45:
        fill = fill * 0.35
    return max(0.0, min(100.0, fill))


def draw_level_overlay(crop_bgr: np.ndarray, ms: MeniscusResult) -> np.ndarray:
    vis = crop_bgr.copy()
    h, w = vis.shape[:2]
    cx = (ms.x_left + ms.x_right) // 2
    red = (0, 0, 255)
    x0, x1 = ms.x_left, ms.x_right
    cv2.line(vis, (cx, ms.tube_top), (cx, ms.tube_bottom), red, 1, cv2.LINE_AA)
    cv2.line(vis, (x0, ms.tube_top), (x1, ms.tube_top), red, 1, cv2.LINE_AA)
    cv2.line(vis, (x0, ms.tube_bottom), (x1, ms.tube_bottom), red, 2, cv2.LINE_AA)

    if ms.meniscus_y is not None:
        cv2.line(vis, (x0, ms.meniscus_y), (x1, ms.meniscus_y), red, 2, cv2.LINE_AA)
        peak_pct = ms.peak_score * 100.0
        label_en = f"peak {peak_pct:.1f}%  Y={ms.meniscus_y}px"
        cv2.putText(
            vis,
            label_en,
            (ms.x_left, max(ms.tube_top + 24, ms.meniscus_y - 12)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.45,
            red,
            1,
            cv2.LINE_AA,
        )
        fill_show = fill_percent_from_meniscus(
            ms.meniscus_y, ms.tube_top, ms.tube_bottom
        )
        fill_label = f"fill {fill_show:.1f}%"
        vis = put_chinese(vis, fill_label, (ms.x_left, min(h - 40, ms.meniscus_y + 28)))
    else:
        vis = put_chinese(vis, "未检测到弯月面", (ms.x_left, ms.tube_top + 8))
    if ms.tube_bottom >= h - 4:
        vis = put_chinese(vis, "底缘近裁剪边", (ms.x_left, min(h - 28, ms.tube_bottom - 6)))

    return vis
