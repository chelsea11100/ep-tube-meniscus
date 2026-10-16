from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

from .common import ROOT, load_config, log_verbose


@dataclass
class TubeDetection:
    bbox: tuple[int, int, int, int]
    confidence: float
    class_name: str
    crop: np.ndarray
    annotated: np.ndarray
    used_fallback: bool = False
    rbox: list[list[int]] | None = None
    angle_deg: float = 0.0


class TubeDetector:
    def __init__(self, config: dict | None = None, verbose: bool | None = None):
        self.config = config or load_config()
        self.verbose = bool(self.config.get("verbose", True)) if verbose is None else verbose
        self.padding = float(self.config.get("crop_padding", 0.10))
        self.padding_bottom = float(
            self.config.get("crop_padding_bottom", self.padding + 0.12)
        )
        yolo_cfg = self.config.get("yolo") or {}
        self.backend = str(yolo_cfg.get("backend") or "opencv").lower()
        self.fallback_yolo = bool(yolo_cfg.get("fallback_yolo", True))
        self._yolo = None
        if self.backend == "yolo" or (
            self.backend == "opencv_first" and self.fallback_yolo
        ):
            self._yolo = self._load_yolo(yolo_cfg)

    def _load_yolo(self, yolo_cfg: dict):
        import os

        os.environ.setdefault("YOLO_OFFLINE", "1")
        from ultralytics import YOLO
        from ultralytics.utils import SETTINGS, checks

        SETTINGS.update({"offline": True, "sync": False})

        original = checks.check_requirements

        def _skip_clip(req=(), *args, **kwargs):
            text = str(req)
            if "CLIP" in text:
                return True
            return original(req, *args, **kwargs)

        checks.check_requirements = _skip_clip
        weights = yolo_cfg.get("weights", "yolov8s-worldv2.pt")
        weights_path = Path(weights)
        if not weights_path.is_absolute():
            local = ROOT / weights
            weights = str(local) if local.exists() else weights
        model = YOLO(weights)
        classes = list(yolo_cfg.get("classes") or ["centrifuge tube"])
        if hasattr(model, "set_classes"):
            model.set_classes(classes)
        log_verbose(
            self.verbose,
            f"[YOLO] 已加载权重: {weights}\n[YOLO] 开放类名: {classes}",
        )
        return model

    def detect(self, image: np.ndarray) -> TubeDetection:
        h, w = image.shape[:2]
        log_verbose(self.verbose, f"[检测] 输入尺寸 {w}x{h}，backend={self.backend}")

        if self.backend == "opencv_first":
            found = detect_tube_opencv(
                image,
                self.padding,
                verbose=self.verbose,
                padding_bottom=self.padding_bottom,
            )
            if found is not None:
                log_verbose(
                    self.verbose,
                    f"[OpenCV] 选中 {found.class_name} conf={found.confidence:.2f} "
                    f"bbox={found.bbox} angle={found.angle_deg:.1f}°",
                )
                return found
            log_verbose(self.verbose, "[OpenCV] 未检测到管，尝试 YOLO 兜底")
            if self._yolo is not None:
                found = self._detect_yolo(image)
                if found is not None:
                    return found
            log_verbose(self.verbose, "[检测] 改用整图")
            return self._fallback_full_image(image)

        if self.backend == "opencv":
            found = detect_tube_opencv(
                image,
                self.padding,
                verbose=self.verbose,
                padding_bottom=self.padding_bottom,
            )
            if found is None:
                log_verbose(self.verbose, "[OpenCV] 未检测到管，改用整图")
                return self._fallback_full_image(image)
            log_verbose(
                self.verbose,
                f"[OpenCV] 选中 {found.class_name} conf={found.confidence:.2f} "
                f"bbox={found.bbox} angle={found.angle_deg:.1f}°",
            )
            return found

        if self._yolo is not None:
            found = self._detect_yolo(image)
            if found is not None:
                return found
            log_verbose(self.verbose, "[YOLO] 无可用框，改用 OpenCV")
        found = detect_tube_opencv(
            image,
            self.padding,
            verbose=self.verbose,
            padding_bottom=self.padding_bottom,
        )
        if found is None:
            log_verbose(self.verbose, "[OpenCV] 未检测到管，改用整图")
            return self._fallback_full_image(image)
        log_verbose(
            self.verbose,
            f"[OpenCV] 选中 {found.class_name} conf={found.confidence:.2f} "
            f"bbox={found.bbox} angle={found.angle_deg:.1f}°",
        )
        return found

    def detect_path(self, image_path: str | Path) -> TubeDetection:
        image = imread_chinese(image_path)
        if image is None:
            raise FileNotFoundError(f"无法读取图片: {image_path}")
        return self.detect(image)

    def _detect_yolo(self, image: np.ndarray) -> TubeDetection | None:
        yolo_cfg = self.config.get("yolo") or {}
        conf_thr = float(yolo_cfg.get("conf", 0.01))
        iou = float(yolo_cfg.get("iou", 0.50))
        imgsz = int(yolo_cfg.get("imgsz", 1280))
        log_verbose(
            self.verbose,
            f"[YOLO] predict conf={conf_thr} iou={iou} imgsz={imgsz}",
        )
        results = self._yolo.predict(
            image,
            conf=conf_thr,
            iou=iou,
            imgsz=imgsz,
            verbose=self.verbose,
        )
        best = None
        best_score = -1.0
        img_h, img_w = image.shape[:2]
        names = getattr(self._yolo, "names", {}) or {}
        classes = list(yolo_cfg.get("classes") or ["tube"])
        candidates: list[tuple] = []
        for result in results:
            boxes = getattr(result, "boxes", None)
            if boxes is None or boxes.xyxy is None or len(boxes) == 0:
                continue
            xyxy = boxes.xyxy.cpu().numpy()
            confs = boxes.conf.cpu().numpy()
            clss = boxes.cls.cpu().numpy() if boxes.cls is not None else [0] * len(xyxy)
            for box, conf, cls_id in zip(xyxy, confs, clss):
                x1, y1, x2, y2 = [int(round(v)) for v in box.tolist()]
                conf_f = float(conf)
                score = _score_yolo_tube_box(x1, y1, x2, y2, conf_f, img_h, img_w)
                name = names.get(int(cls_id), classes[0])
                candidates.append((score, name, conf_f, x1, y1, x2, y2))
                if score <= best_score:
                    continue
                best = (x1, y1, x2, y2, conf_f, str(name))
                best_score = score
        if self.verbose:
            if not candidates:
                log_verbose(self.verbose, "[YOLO] 候选框: （无）")
            else:
                print("[YOLO] 候选框（按综合分 score 降序）:", flush=True)
                for score, name, conf_f, x1, y1, x2, y2 in sorted(
                    candidates, key=lambda c: c[0], reverse=True
                ):
                    flag = " ← 选用" if best and (x1, y1, x2, y2) == best[:4] else ""
                    print(
                        f"  score={score:.3f}  {name}  conf={conf_f:.4f}  "
                        f"bbox=({x1},{y1},{x2},{y2}){flag}",
                        flush=True,
                    )
        if best is None:
            return None
        x1, y1, x2, y2, conf, name = best
        log_verbose(
            self.verbose,
            f"[YOLO] 最终: {name} conf={conf:.4f} bbox=({x1},{y1},{x2},{y2})",
        )
        crop_box = pad_box((x1, y1, x2, y2), image.shape, self.padding)
        return TubeDetection(
            bbox=(x1, y1, x2, y2),
            confidence=conf,
            class_name=name,
            crop=crop_image(image, crop_box),
            annotated=draw_box(image, (x1, y1, x2, y2), f"{name} {conf:.2f}"),
            used_fallback=False,
        )

    def _fallback_full_image(self, image: np.ndarray) -> TubeDetection:
        h, w = image.shape[:2]
        annotated = image.copy()
        annotated = put_chinese(annotated, "未框到离心管，改用整张图", (20, 20))
        return TubeDetection(
            bbox=(0, 0, w, h),
            confidence=0.0,
            class_name="full_image_fallback",
            crop=image.copy(),
            annotated=annotated,
            used_fallback=True,
            rbox=[[0, 0], [w, 0], [w, h], [0, h]],
            angle_deg=0.0,
        )


def _score_yolo_tube_box(
    x1: int, y1: int, x2: int, y2: int, conf: float, img_h: int, img_w: int
) -> float:
    bw = max(1, x2 - x1)
    bh = max(1, y2 - y1)
    if bh < img_h * 0.18:
        return -1.0
    aspect = bh / bw
    if aspect < 1.25:
        return -1.0
    if y2 < img_h * 0.40:
        return -1.0
    cx = (x1 + x2) / 2.0
    center = 1.0 - min(1.0, abs(cx - img_w / 2.0) / (img_w / 2.0))
    return conf * 2.0 + aspect * 0.35 + center * 0.8 + (bh / img_h) * 1.2


def pad_box_tube(
    box: tuple[int, int, int, int],
    shape: tuple[int, ...],
    padding: float,
    padding_bottom: float | None = None,
) -> tuple[int, int, int, int]:
    x1, y1, x2, y2 = box
    h, w = shape[:2]
    bw = max(1, x2 - x1)
    bh = max(1, y2 - y1)
    px = int(round(bw * padding))
    py_top = int(round(bh * padding))
    py_bot = int(round(bh * (padding_bottom if padding_bottom is not None else padding)))
    return (
        max(0, x1 - px),
        max(0, y1 - py_top),
        min(w, x2 + px),
        min(h, y2 + py_bot),
    )


def _grip_exclude_mask(bgr: np.ndarray, skin: np.ndarray) -> np.ndarray:
    """背光图握持区：仅图像**上半段**合并肤色 + 无透光暗块，避免削掉整根管壁边缘。"""
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    h, w = gray.shape
    exclude = skin.copy()
    upper_h = max(32, int(h * 0.42))
    gray_u = gray[:upper_h, :]
    skin_u = skin[:upper_h, :]
    x0, x1 = w // 4, (3 * w) // 4
    band = gray_u[:, x0:x1].astype(np.float32)
    col_bright = cv2.GaussianBlur(band, (7, 17), 0)
    col_full = np.zeros_like(gray_u, dtype=np.float32)
    col_full[:, x0:x1] = col_bright
    solid = (gray_u.astype(np.float32) < 95) & (col_full < 118)
    solid = solid.astype(np.uint8) * 255
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
    solid = cv2.morphologyEx(solid, cv2.MORPH_CLOSE, kernel, iterations=1)
    top_exclude = cv2.bitwise_or(skin_u, solid)
    exclude[:upper_h, :] = top_exclude
    return exclude


def _backlit_center_bright_ratio(bgr: np.ndarray, box: tuple[int, int, int, int]) -> float:
    return _tube_center_signal(bgr, box)


def _tube_center_signal(bgr: np.ndarray, box: tuple[int, int, int, int]) -> float:
    """管心透光或有色液体（粉/蓝等）均视为有效管身信号。"""
    x1, y1, x2, y2 = box
    patch = bgr[y1:y2, x1:x2]
    if patch.size == 0:
        return 0.0
    h, w = patch.shape[:2]
    c0, c1 = int(w * 0.32), int(w * 0.68)
    if c1 <= c0:
        return 0.0
    center = patch[:, c0:c1]
    gray = cv2.cvtColor(center, cv2.COLOR_BGR2GRAY)
    bright = float((gray > 165).mean())
    hsv = cv2.cvtColor(center, cv2.COLOR_BGR2HSV)
    colored = float((hsv[:, :, 1] > 22).mean())
    return max(bright, colored * 0.9)


def _dark_grip_ratio(bgr: np.ndarray, box: tuple[int, int, int, int]) -> float:
    x1, y1, x2, y2 = box
    patch = bgr[y1:y2, x1:x2]
    if patch.size == 0:
        return 1.0
    h = patch.shape[0]
    top = patch[: max(1, int(h * 0.24)), :]
    gray = cv2.cvtColor(top, cv2.COLOR_BGR2GRAY)
    return float((gray < 95).mean())


def _is_solid_black_block(bgr: np.ndarray, box: tuple[int, int, int, int]) -> bool:
    """背光图：实心黑块（手指剪影等）不要；EP 管应有透光亮带或明显有色液柱。"""
    x1, y1, x2, y2 = box
    patch = bgr[y1:y2, x1:x2]
    if patch.size == 0:
        return True
    h, w = patch.shape[:2]
    if h < 12 or w < 8:
        return True
    gray_all = cv2.cvtColor(patch, cv2.COLOR_BGR2GRAY)
    dark_frac = float((gray_all < 100).mean())
    c0, c1 = int(w * 0.30), int(w * 0.70)
    if c1 <= c0:
        return dark_frac > 0.55
    center = patch[:, c0:c1]
    gray = cv2.cvtColor(center, cv2.COLOR_BGR2GRAY)
    hsv = cv2.cvtColor(center, cv2.COLOR_BGR2HSV)
    val = hsv[:, :, 2]
    sat = hsv[:, :, 1]
    mean_gray = float(gray.mean())
    bright_frac = float((gray > 145).mean())
    mean_val = float(val.mean())
    mean_sat = float(sat.mean())
    colored_frac = float(((sat > 28) & (val > 55)).mean())
    # 整块偏黑 + 管心几乎不亮 → 不要
    if dark_frac > 0.62 and bright_frac < 0.26 and colored_frac < 0.35:
        return True
    if dark_frac > 0.50 and mean_gray < 105:
        return True
    if mean_gray < 112 and bright_frac < 0.22 and mean_sat < 48:
        return True
    if mean_val < 95 and bright_frac < 0.18:
        return True
    return False


def _is_finger_slit_occlusion(bgr: np.ndarray, box: tuple[int, int, int, int]) -> bool:
    """窄缝/竖条里手指：两侧暗、中间亮；背光 EP 管则整幅都偏亮。"""
    x1, y1, x2, y2 = box
    patch = bgr[y1:y2, x1:x2]
    h, w = patch.shape[:2]
    if h < 24 or w < 10:
        return True
    band = patch[int(h * 0.15) : int(h * 0.88)]
    gray = cv2.cvtColor(band, cv2.COLOR_BGR2GRAY)
    core = float(gray[:, int(w * 0.38) : int(w * 0.62)].mean())
    lw = float(gray[:, : max(1, int(w * 0.18))].mean())
    rw = float(gray[:, max(0, int(w * 0.82)) :].mean())
    wing = min(lw, rw)
    if wing < 95 and core > 148 and (core - wing) > 28:
        return True
    return wing < 88 and core > 155 and (core - wing) > 35


def _is_uniform_bright_slab(
    bgr: np.ndarray, box: tuple[int, int, int, int], img_h: int
) -> bool:
    """整段灰亮、上下差不多：常见于窄条里的手指，不是透光 EP 管。"""
    x1, y1, x2, y2 = box
    bh = y2 - y1
    if bh < img_h * 0.32:
        return False
    patch = bgr[y1:y2, x1:x2]
    gray = cv2.cvtColor(patch, cv2.COLOR_BGR2GRAY)
    h = gray.shape[0]
    top = float(gray[: max(1, int(h * 0.25))].mean())
    bot = float(gray[int(h * 0.70) :].mean())
    diff = bot - top
    if top > 135 and bot > 168 and abs(diff) < 18:
        return True
    return top > 128 and bot < 172 and diff < 26


def _score_tube_box(
    bgr: np.ndarray, box: tuple[int, int, int, int], img_h: int, img_w: int
) -> float:
    x1, y1, x2, y2 = box
    bw, bh = max(1, x2 - x1), max(1, y2 - y1)
    min_bw = max(40, int(img_w * 0.12))
    if bh < img_h * 0.14 or bw < min_bw:
        return -1.0
    aspect = bh / bw
    if aspect < 1.35 or aspect > 14.0:
        return -1.0
    cx = (x1 + x2) / 2.0
    if abs(cx - img_w / 2) > img_w * 0.22:
        return -1.0
    if _is_solid_black_block(bgr, box):
        return -1.0
    if _is_finger_slit_occlusion(bgr, box):
        return -1.0
    if _is_uniform_bright_slab(bgr, box, img_h):
        return -1.0
    signal = _tube_center_signal(bgr, box)
    grip = _dark_grip_ratio(bgr, box)
    if signal < 0.10:
        return -1.0
    if signal < 0.24 and grip > 0.65:
        return -1.0
    if bw / img_w > 0.42:
        return -1.0
    if y2 < img_h * 0.42:
        return -1.0
    center = 1.0 - min(1.0, abs(cx - img_w / 2) / (img_w / 2))
    return (
        aspect * 0.22
        + (bh / img_h) * 1.15
        + signal * 2.8
        + center * 0.75
        - grip * 0.55
    )


def _build_tube_edges(gray: np.ndarray, exclude: np.ndarray) -> np.ndarray:
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    mean = float(np.mean(blur))
    if mean > 125:
        grad = cv2.morphologyEx(blur, cv2.MORPH_GRADIENT, np.ones((3, 3), np.uint8))
        e_lo = cv2.Canny(blur, 12, 48)
        _, e_grad = cv2.threshold(grad, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        edges = cv2.bitwise_or(e_lo, e_grad)
    else:
        edges = cv2.Canny(blur, 30, 100)
    edges[exclude > 0] = 0
    return edges


def _box_from_vertical_walls(
    gray: np.ndarray, exclude: np.ndarray, w: int, h: int
) -> tuple[int, int, int, int] | None:
    """淡管缘 / 过曝图：用竖向 Sobel 找左右壁。"""
    y0, y1 = int(h * 0.06), int(h * 0.94)
    x_lo, x_hi = int(w * 0.10), int(w * 0.90)
    if x_hi - x_lo < 40 or y1 - y0 < int(h * 0.2):
        return None
    band = gray[y0:y1, x_lo:x_hi].astype(np.float32)
    ex = exclude[y0:y1, x_lo:x_hi]
    band[ex > 0] = 255.0
    sx = np.abs(cv2.Sobel(band, cv2.CV_32F, 1, 0, ksize=5))
    col = sx.sum(axis=0)
    if col.size < 16:
        return None
    col = np.convolve(col, np.ones(11, dtype=np.float32) / 11.0, mode="same")
    mid = len(col) // 2
    if mid < 8:
        return None
    xl = x_lo + int(np.argmax(col[:mid]))
    xr = x_lo + mid + int(np.argmax(col[mid:]))
    bw = xr - xl
    if bw < max(20, int(w * 0.07)) or bw > int(w * 0.40):
        return None
    cx = (xl + xr) / 2.0
    if abs(cx - w / 2.0) > w * 0.22:
        return None
    return (xl, y0, xr, y1)


def _iter_nonskin_blob_boxes(
    gray: np.ndarray, exclude: np.ndarray, w: int, h: int
) -> list[tuple[int, int, int, int]]:
    bright = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]
    fg = cv2.bitwise_not(bright)
    fg[exclude > 0] = 0
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    fg = cv2.morphologyEx(fg, cv2.MORPH_OPEN, kernel, iterations=1)
    fg = cv2.morphologyEx(fg, cv2.MORPH_CLOSE, kernel, iterations=2)
    contours, _ = cv2.findContours(fg, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    boxes: list[tuple[int, int, int, int]] = []
    for contour in contours:
        x, y, bw, bh = cv2.boundingRect(contour)
        if bh < h * 0.16 or bw < 10:
            continue
        if bh / max(bw, 1) < 1.45:
            continue
        cx = x + bw / 2
        if abs(cx - w / 2) > w * 0.32:
            continue
        if y < h * 0.06 and bh < h * 0.38:
            continue
        boxes.append((x, y, x + bw, y + bh))
    return boxes


def _pick_best_tube_box(
    bgr: np.ndarray,
    candidates: list[tuple[int, int, int, int]],
    img_h: int,
    img_w: int,
    min_score: float = 1.05,
) -> tuple[tuple[int, int, int, int], float] | None:
    scored: list[tuple[tuple[int, int, int, int], float]] = []
    for box in candidates:
        s = _score_tube_box(bgr, box, img_h, img_w)
        if s >= min_score:
            scored.append((box, s))
    if not scored:
        fallback: list[tuple[tuple[int, int, int, int], float]] = []
        for box in candidates:
            s = _score_tube_box(bgr, box, img_h, img_w)
            if s > 0:
                fallback.append((box, s))
        if fallback:
            best_box, best_score = max(fallback, key=lambda t: t[1])
            if best_score >= 0.72:
                conf = float(min(0.88, max(0.42, best_score / 3.2)))
                return best_box, conf
        return None
    max_s = max(s for _, s in scored)
    # 分数接近的候选里取更高的框，避免只框到管身中段（手指过滤后常见）
    near = [(box, s) for box, s in scored if s >= max_s * 0.88]
    best_box, best_score = max(near, key=lambda t: (t[1], t[0][3] - t[0][1]))
    conf = float(min(0.92, max(0.45, best_score / 3.2)))
    return best_box, conf


def _padded_rect_aabb(
    rect: tuple[tuple[float, float], tuple[float, float], float],
    img_h: int,
    img_w: int,
    y_max: int,
    padding: float,
) -> tuple[tuple[int, int, int, int], np.ndarray, float]:
    center, size, angle = rect
    size, angle = _make_height_major(size, angle)
    pad_w = 1.0 + padding + 0.22
    raw_h = size[1]
    raw_ratio = raw_h / max(1, img_h)
    if raw_ratio < 0.28:
        pad_h = 1.0 + padding + 0.48
        cy_shift = raw_h * 0.14
    elif raw_ratio < 0.36:
        pad_h = 1.0 + padding + 0.34
        cy_shift = raw_h * 0.10
    else:
        pad_h = 1.0 + padding + 0.28
        cy_shift = raw_h * 0.08
    size = (max(12.0, size[0] * pad_w), max(24.0, size[1] * pad_h))
    center = (center[0], center[1] + cy_shift)
    corners = cv2.boxPoints((center, size, angle)).astype(np.float32)
    aabb = _aabb_from_points(corners, img_w, y_max)
    return aabb, corners, float(angle)


def detect_tube_opencv(
    image: np.ndarray,
    padding: float = 0.10,
    verbose: bool = False,
    padding_bottom: float | None = None,
) -> TubeDetection | None:
    h, w = image.shape[:2]
    log_verbose(verbose, "[OpenCV] 边缘/旋转矩形检测…")
    y_bottom = int(h * 0.92)
    roi = image[:y_bottom]
    gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    skin = _skin_mask(roi)
    exclude = _grip_exclude_mask(roi, skin)
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    edges = cv2.Canny(blur, 30, 100)
    edges[exclude > 0] = 0
    rect = _rotated_rect_from_edges(edges, w, y_bottom)
    if rect is None:
        edges = _build_tube_edges(gray, exclude)
        rect = _rotated_rect_from_edges(edges, w, y_bottom)

    candidates: list[tuple[int, int, int, int]] = []
    corners: np.ndarray | None = None
    angle_deg = 0.0
    if rect is not None:
        aabb_rect, corners, angle_deg = _padded_rect_aabb(rect, h, w, h, padding)
        candidates.append(aabb_rect)
        center, size, angle = rect
        size, angle = _make_height_major(size, angle)
        raw_corners = cv2.boxPoints((center, size, angle)).astype(np.float32)
        candidates.append(_aabb_from_points(raw_corners, w, y_bottom))

    blob_boxes = _iter_nonskin_blob_boxes(gray, exclude, w, y_bottom)
    candidates.extend(blob_boxes)
    if rect is None and not blob_boxes:
        wall = _box_from_vertical_walls(gray, exclude, w, y_bottom)
        if wall is not None:
            candidates.append(wall)
            log_verbose(verbose, f"[OpenCV] 淡缘竖壁兜底框 {wall}")

    picked = _pick_best_tube_box(image, candidates, h, w)
    if picked is None:
        log_verbose(verbose, "[OpenCV] 无合格管框（含实心黑块/手指剪影已丢弃）")
        return None
    aabb, conf = picked
    if (
        _is_solid_black_block(image, aabb)
        or _is_finger_slit_occlusion(image, aabb)
        or _is_uniform_bright_slab(image, aabb, h)
    ):
        log_verbose(verbose, "[OpenCV] 最终框疑为手指/黑块，不当作离心管")
        return None
    if rect is not None:
        padded_aabb, corners, angle_deg = _padded_rect_aabb(rect, h, w, h, padding)
        pad_score = _score_tube_box(image, padded_aabb, h, w)
        if pad_score >= 1.05:
            aabb = padded_aabb
            conf = float(min(0.92, max(conf, pad_score / 3.2)))
    log_verbose(verbose, f"[OpenCV] 选中框 {aabb} tube_score={conf:.2f}")

    crop_box = pad_box_tube(aabb, image.shape, padding, padding_bottom)
    crop = crop_image(image, crop_box)
    annotated = draw_box(image, crop_box, "tube")
    rbox = (
        [[int(p[0]), int(p[1])] for p in corners]
        if corners is not None
        else [
            [crop_box[0], crop_box[1]],
            [crop_box[2], crop_box[1]],
            [crop_box[2], crop_box[3]],
            [crop_box[0], crop_box[3]],
        ]
    )
    return TubeDetection(
        bbox=crop_box,
        confidence=conf,
        class_name="tube_opencv",
        crop=crop,
        annotated=annotated,
        used_fallback=False,
        rbox=rbox,
        angle_deg=angle_deg,
    )


def _skin_mask(bgr: np.ndarray) -> np.ndarray:
    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
    m1 = cv2.inRange(hsv, np.array([0, 25, 40]), np.array([30, 220, 255]))
    m2 = cv2.inRange(hsv, np.array([155, 25, 40]), np.array([180, 220, 255]))
    skin = cv2.bitwise_or(m1, m2)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (13, 13))
    skin = cv2.morphologyEx(skin, cv2.MORPH_CLOSE, kernel, iterations=1)
    skin = cv2.dilate(skin, kernel, iterations=1)
    return skin


def _rotated_rect_from_edges(
    edges: np.ndarray, w: int, h: int
) -> tuple[tuple[float, float], tuple[float, float], float] | None:
    vkernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 17))
    closed = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, vkernel)
    lines = cv2.HoughLinesP(
        closed,
        1,
        np.pi / 180,
        threshold=24,
        minLineLength=int(h * 0.14),
        maxLineGap=32,
    )
    if lines is None:
        return None
    pts = []
    for x1, y1, x2, y2 in lines[:, 0]:
        dx, dy = int(x2 - x1), int(y2 - y1)
        length = max(1.0, float((dx * dx + dy * dy) ** 0.5))
        if abs(dy) / length < 0.70:
            continue
        cx = (x1 + x2) / 2.0
        if abs(cx - w / 2) > w * 0.38:
            continue
        n = max(4, int(length / 8))
        for t in np.linspace(0.0, 1.0, n):
            pts.append((x1 + t * (x2 - x1), y1 + t * (y2 - y1)))
    if len(pts) < 8:
        return None
    arr = np.array(pts, dtype=np.float32)
    rect = cv2.minAreaRect(arr)
    (cx, cy), (rw, rh), angle = rect
    long_side = max(rw, rh)
    short_side = min(rw, rh)
    if long_side < h * 0.16:
        return None
    if short_side < 8:
        short_side = max(12.0, long_side * 0.28)
        if rw >= rh:
            rw, rh = long_side, short_side
        else:
            rw, rh = short_side, long_side
    if short_side / max(long_side, 1) < 0.18:
        short_side = long_side * 0.30
        if rw >= rh:
            rh = short_side
        else:
            rw = short_side
    return (float(cx), float(cy)), (float(rw), float(rh)), float(angle)


def _make_height_major(
    size: tuple[float, float], angle: float
) -> tuple[tuple[float, float], float]:
    width, height = size
    if width > height:
        return (height, width), angle + 90.0
    return (width, height), angle


def crop_along_axis(
    image: np.ndarray,
    center: tuple[float, float],
    size: tuple[float, float],
    angle: float,
) -> np.ndarray:
    width, height = int(max(12, round(size[0]))), int(max(24, round(size[1])))
    matrix = cv2.getRotationMatrix2D(center, angle, 1.0)
    matrix[0, 2] += width / 2.0 - center[0]
    matrix[1, 2] += height / 2.0 - center[1]
    crop = cv2.warpAffine(
        image,
        matrix,
        (width, height),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=(245, 245, 245),
    )
    cap_y = _map_point(matrix, (center[0], min(center[1], center[1] - 20)))
    if cap_y > height / 2:
        crop = cv2.rotate(crop, cv2.ROTATE_180)
    return crop


def _map_point(matrix: np.ndarray, point: tuple[float, float]) -> float:
    x, y = point
    return float(matrix[1, 0] * x + matrix[1, 1] * y + matrix[1, 2])


def _aabb_from_points(
    corners: np.ndarray, w: int, h: int
) -> tuple[int, int, int, int]:
    xs = corners[:, 0]
    ys = corners[:, 1]
    x1 = max(0, int(np.floor(xs.min())))
    y1 = max(0, int(np.floor(ys.min())))
    x2 = min(w, int(np.ceil(xs.max())))
    y2 = min(h, int(np.ceil(ys.max())))
    return x1, y1, x2, y2


_CJK_FONTS = (
    "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc",
    "/usr/share/fonts/noto-cjk/NotoSansCJK-Regular.ttc",
    "/usr/share/fonts/google-noto-cjk/NotoSansCJK-Regular.ttc",
    "/usr/share/fonts/truetype/wqy/wqy-microhei.ttc",
    "/usr/share/fonts/truetype/wqy/wqy-zenhei.ttc",
    r"C:\Windows\Fonts\msyh.ttc",
    r"C:\Windows\Fonts\simhei.ttf",
)


def put_chinese(image: np.ndarray, text: str, org: tuple[int, int]) -> np.ndarray:
    from PIL import Image, ImageDraw, ImageFont

    rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    pil = Image.fromarray(rgb)
    draw = ImageDraw.Draw(pil)
    font_path = next((Path(p) for p in _CJK_FONTS if Path(p).exists()), None)
    font = ImageFont.truetype(str(font_path), 36) if font_path else ImageFont.load_default()
    draw.text(org, text, fill=(220, 0, 0), font=font)
    return cv2.cvtColor(np.array(pil), cv2.COLOR_RGB2BGR)


def pad_box(
    box: tuple[int, int, int, int],
    shape: tuple[int, ...],
    padding: float,
) -> tuple[int, int, int, int]:
    x1, y1, x2, y2 = box
    h, w = shape[:2]
    bw = max(1, x2 - x1)
    bh = max(1, y2 - y1)
    px = int(round(bw * padding))
    py = int(round(bh * padding))
    return max(0, x1 - px), max(0, y1 - py), min(w, x2 + px), min(h, y2 + py)


def crop_image(image: np.ndarray, box: tuple[int, int, int, int]) -> np.ndarray:
    x1, y1, x2, y2 = box
    return image[y1:y2, x1:x2].copy()


def draw_box(image: np.ndarray, box: tuple[int, int, int, int], label: str) -> np.ndarray:
    vis = image.copy()
    x1, y1, x2, y2 = box
    cv2.rectangle(vis, (x1, y1), (x2, y2), (0, 180, 0), 3)
    return _draw_label(vis, x1, y1, label)


def draw_rotated_box(image: np.ndarray, corners: np.ndarray, label: str) -> np.ndarray:
    vis = image.copy()
    pts = corners.astype(np.int32).reshape((-1, 1, 2))
    cv2.polylines(vis, [pts], isClosed=True, color=(0, 180, 0), thickness=3)
    x1 = int(corners[:, 0].min())
    y1 = int(corners[:, 1].min())
    return _draw_label(vis, x1, y1, label)


def _draw_label(vis: np.ndarray, x1: int, y1: int, label: str) -> np.ndarray:
    font = cv2.FONT_HERSHEY_SIMPLEX
    scale = 0.7
    thickness = 2
    (tw, th), _ = cv2.getTextSize(label, font, scale, thickness)
    ty = max(0, y1 - 8)
    cv2.rectangle(vis, (x1, max(0, ty - th - 6)), (x1 + tw + 8, ty + 4), (0, 180, 0), -1)
    cv2.putText(vis, label, (x1 + 4, ty), font, scale, (255, 255, 255), thickness, cv2.LINE_AA)
    return vis


def imread_chinese(path: str | Path) -> np.ndarray | None:
    data = np.fromfile(str(path), dtype=np.uint8)
    if data.size == 0:
        return None
    return cv2.imdecode(data, cv2.IMREAD_COLOR)


def imwrite_chinese(path: str | Path, image: np.ndarray) -> None:
    dest = Path(path)
    dest.parent.mkdir(parents=True, exist_ok=True)
    suffix = dest.suffix or ".jpg"
    ok, buf = cv2.imencode(suffix, image)
    if not ok:
        raise RuntimeError(f"无法编码图片: {dest}")
    buf.tofile(str(dest))
