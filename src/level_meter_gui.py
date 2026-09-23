"""Level Meter 风格界面：预览 → 手动分析；ROI；双液面；摄像头实时跟线。"""

from __future__ import annotations

import threading
import time
import tkinter as tk
from pathlib import Path
from tkinter import filedialog, ttk

import cv2
import numpy as np
from PIL import Image, ImageTk

from .common import ROOT, load_config
from .detect_tube import TubeDetector, imread_chinese, put_chinese
from .color_threshold import ColorSpec, meniscus_with_color
from .judge_meniscus import MeniscusResult, detect_meniscus, draw_level_overlay, fill_percent_from_meniscus
from .threshold_editor import ThresholdEditor


TUBE_NOMINAL_ML = 1.5


def apply_roi_percent(bgr: np.ndarray, pct_x: float, pct_y: float) -> tuple[np.ndarray, tuple[int, int]]:
    """取画面中心 pct_x/y % 区域再送去检测。"""
    pct_x = max(0.25, min(1.0, pct_x / 100.0))
    pct_y = max(0.25, min(1.0, pct_y / 100.0))
    h, w = bgr.shape[:2]
    nw, nh = max(32, int(w * pct_x)), max(32, int(h * pct_y))
    x0 = (w - nw) // 2
    y0 = (h - nh) // 2
    return bgr[y0 : y0 + nh, x0 : x0 + nw].copy(), (x0, y0)


def ml_from_y(y: int | None, tube_top: int, tube_bottom: int) -> str:
    if y is None or tube_bottom <= tube_top:
        return "None"
    fill = fill_percent_from_meniscus(y, tube_top, tube_bottom)
    return f"{fill / 100.0 * TUBE_NOMINAL_ML:.2f}"


class LevelMeterApp:
    def __init__(self, config_path: Path | None = None) -> None:
        self.config = load_config(config_path)
        self.min_fill = int(self.config.get("min_fill_percent", 10))
        liq = self.config.get("liquid") or {}
        self.tube_top_frac = float(liq.get("tube_top_frac", 0.18))
        self.tube_bottom_frac = float(liq.get("tube_bottom_frac", 0.05))
        self.center_x_left = float(liq.get("center_x_left", 0.28))
        self.center_x_right = float(liq.get("center_x_right", 0.72))
        self.meniscus_min_peak = float(liq.get("meniscus_min_peak", 0.30))
        self.empty_peak_max = float(liq.get("empty_peak_max", 0.26))

        self.detector = TubeDetector(self.config, verbose=False)

        self.root = tk.Tk()
        self.root.title("EP 管液面 Level Meter")
        self.root.minsize(1000, 680)
        self.root.geometry("1180x760")

        self.source_bgr: np.ndarray | None = None
        self.source_label = ""
        self.roi_offset = (0, 0)
        self.crop_bgr: np.ndarray | None = None
        self.crop_box: tuple[int, int, int, int] | None = None
        self.ms: MeniscusResult | None = None
        self.meniscus_y: int | None = None
        self.meniscus_y2: int | None = None
        self.peak_score = 0.0
        self.analyzed = False
        self.show_crop_view = False
        self.color_spec: ColorSpec | None = None
        self._thr_editor: ThresholdEditor | None = None

        self._photo: ImageTk.PhotoImage | None = None
        self._scale = 1.0
        self._offset_x = 0
        self._offset_y = 0
        self._display_h = 0
        self._display_w = 0
        self._dragging: str | None = None
        self._drag_offset = 0.0

        self.cap: cv2.VideoCapture | None = None
        self._camera_job: str | None = None
        self._live_mode = False
        self._last_frame: np.ndarray | None = None
        self._brightness = 120
        self._camera_index = 0
        self._last_analyze_t = 0.0
        self._analyze_busy = False
        self._fps_display = 0.0

        self._build_ui()

    def _build_ui(self) -> None:
        main = ttk.Panedwindow(self.root, orient=tk.HORIZONTAL)
        main.pack(fill=tk.BOTH, expand=True, padx=6, pady=6)

        left = ttk.Frame(main)
        main.add(left, weight=3)
        self.canvas = tk.Canvas(left, bg="#1a1a1a", highlightthickness=0)
        self.canvas.pack(fill=tk.BOTH, expand=True)
        self.canvas.bind("<Configure>", lambda _e: self._redraw())
        self.canvas.bind("<ButtonPress-1>", self._on_press)
        self.canvas.bind("<B1-Motion>", self._on_drag)
        self.canvas.bind("<ButtonRelease-1>", self._on_release)

        right_wrap = ttk.Frame(main)
        main.add(right_wrap, weight=1)
        canvas_r = tk.Canvas(right_wrap, width=340, highlightthickness=0)
        scroll = ttk.Scrollbar(right_wrap, orient=tk.VERTICAL, command=canvas_r.yview)
        right = ttk.Frame(canvas_r)
        right.bind("<Configure>", lambda e: canvas_r.configure(scrollregion=canvas_r.bbox("all")))
        canvas_r.create_window((0, 0), window=right, anchor=tk.NW)
        canvas_r.configure(yscrollcommand=scroll.set)
        canvas_r.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        scroll.pack(side=tk.RIGHT, fill=tk.Y)

        cam = ttk.LabelFrame(right, text="Camera / 输入", padding=8)
        cam.pack(fill=tk.X, pady=(0, 8))
        ttk.Button(cam, text="打开图片…（仅预览）", command=self._open_image).pack(fill=tk.X, pady=2)
        row_cam = ttk.Frame(cam)
        row_cam.pack(fill=tk.X, pady=2)
        ttk.Button(row_cam, text="打开本机摄像头", command=self._start_camera).pack(side=tk.LEFT, fill=tk.X, expand=True)
        self.cam_idx_var = tk.IntVar(value=0)
        ttk.Spinbox(row_cam, from_=0, to=4, width=3, textvariable=self.cam_idx_var).pack(side=tk.LEFT, padx=4)
        ttk.Button(cam, text="分析当前画面", command=self._analyze_current).pack(fill=tk.X, pady=2)
        ttk.Button(cam, text="停止摄像头", command=self._stop_camera).pack(fill=tk.X, pady=2)
        ttk.Label(cam, text="亮度").pack(anchor=tk.W, pady=(6, 0))
        self.bright_var = tk.IntVar(value=self._brightness)
        ttk.Scale(cam, from_=0, to=255, variable=self.bright_var, command=self._on_brightness).pack(fill=tk.X)
        ttk.Label(cam, text="分辨率（只读）").pack(anchor=tk.W, pady=(6, 0))
        self.res_var = tk.StringVar(value="—")
        ttk.Label(cam, textvariable=self.res_var).pack(anchor=tk.W)

        roi = ttk.LabelFrame(right, text="ROI Selection", padding=8)
        roi.pack(fill=tk.X, pady=(0, 8))
        ttk.Label(roi, text="Percent X").pack(anchor=tk.W)
        self.roi_x_var = tk.IntVar(value=100)
        ttk.Scale(
            roi,
            from_=40,
            to=100,
            variable=self.roi_x_var,
            command=lambda _v: self._on_roi_changed(),
        ).pack(fill=tk.X)
        ttk.Label(roi, text="Percent Y").pack(anchor=tk.W, pady=(6, 0))
        self.roi_y_var = tk.IntVar(value=100)
        ttk.Scale(
            roi,
            from_=40,
            to=100,
            variable=self.roi_y_var,
            command=lambda _v: self._on_roi_changed(),
        ).pack(fill=tk.X)

        track = ttk.LabelFrame(right, text="实时跟线（视频）", padding=8)
        track.pack(fill=tk.X, pady=(0, 8))
        self.auto_track_var = tk.BooleanVar(value=True)
        ttk.Checkbutton(
            track,
            text="摄像头开启时自动分析跟线",
            variable=self.auto_track_var,
        ).pack(anchor=tk.W)
        ttk.Label(track, text="分析帧率 (Hz，越高越吃 CPU)").pack(anchor=tk.W, pady=(6, 0))
        self.track_hz_var = tk.DoubleVar(value=3.0)
        ttk.Scale(track, from_=0.5, to=8.0, variable=self.track_hz_var).pack(fill=tk.X)
        self.fps_var = tk.StringVar(value="预览 FPS: — | 分析: — Hz")
        ttk.Label(track, textvariable=self.fps_var, wraplength=300).pack(anchor=tk.W, pady=(4, 0))
        self.view_crop_var = tk.BooleanVar(value=False)
        ttk.Checkbutton(
            track,
            text="显示管裁剪图（否则在原画面上叠线）",
            variable=self.view_crop_var,
            command=self._redraw,
        ).pack(anchor=tk.W, pady=(6, 0))

        color = ttk.LabelFrame(right, text="颜色阈值", padding=8)
        color.pack(fill=tk.X, pady=(0, 8))
        ttk.Label(
            color,
            text="左边原图，右边白色是要跟踪的液体。调到粉液变白、夹爪变黑。",
            wraplength=300,
        ).pack(anchor=tk.W)
        ttk.Button(color, text="打开阈值编辑器", command=self._open_threshold_editor).pack(fill=tk.X, pady=(6, 0))
        self.thr_status_var = tk.StringVar(value="未锁定阈值")
        ttk.Label(color, textvariable=self.thr_status_var, wraplength=300).pack(anchor=tk.W, pady=(4, 0))

        tube = ttk.LabelFrame(right, text="Tube Volumes / 标定", padding=8)
        tube.pack(fill=tk.X, pady=(0, 8))
        ttk.Label(tube, text="Min Mark (ml) / (px)").pack(anchor=tk.W)
        row1 = ttk.Frame(tube)
        row1.pack(fill=tk.X)
        self.min_ml_var = tk.StringVar(value="0")
        self.min_px_var = tk.StringVar(value="—")
        ttk.Entry(row1, width=6, textvariable=self.min_ml_var, state="readonly").pack(side=tk.LEFT)
        ttk.Entry(row1, width=8, textvariable=self.min_px_var, state="readonly").pack(side=tk.LEFT, padx=4)
        ttk.Label(tube, text="Max Mark (ml) / (px)").pack(anchor=tk.W, pady=(6, 0))
        row2 = ttk.Frame(tube)
        row2.pack(fill=tk.X)
        self.max_ml_var = tk.StringVar(value=str(TUBE_NOMINAL_ML))
        self.max_px_var = tk.StringVar(value="—")
        ttk.Entry(row2, width=6, textvariable=self.max_ml_var, state="readonly").pack(side=tk.LEFT)
        ttk.Entry(row2, width=8, textvariable=self.max_px_var, state="readonly").pack(side=tk.LEFT, padx=4)

        vol = ttk.LabelFrame(right, text="Volume Reading / 读数", padding=8)
        vol.pack(fill=tk.X, pady=(0, 8))
        ttk.Label(vol, text="Interface1 (ml)").pack(anchor=tk.W)
        self.iface1_var = tk.StringVar(value="None")
        ttk.Label(vol, textvariable=self.iface1_var, font=("Segoe UI", 13, "bold")).pack(anchor=tk.W)
        ttk.Label(vol, text="Interface2 (ml)").pack(anchor=tk.W, pady=(8, 0))
        self.iface2_var = tk.StringVar(value="None")
        ttk.Label(vol, textvariable=self.iface2_var, font=("Segoe UI", 13, "bold")).pack(anchor=tk.W)
        ttk.Label(vol, text="fill_percent (%)").pack(anchor=tk.W, pady=(8, 0))
        self.fill_var = tk.StringVar(value="—")
        ttk.Label(vol, textvariable=self.fill_var).pack(anchor=tk.W)
        ttk.Label(vol, text="allow_grasp").pack(anchor=tk.W, pady=(6, 0))
        self.grasp_var = tk.StringVar(value="—")
        ttk.Label(vol, textvariable=self.grasp_var).pack(anchor=tk.W)
        ttk.Label(vol, text=f"夹取判定（阈值 fill ≥ {self.min_fill}%）").pack(anchor=tk.W, pady=(8, 0))
        self.grasp_mode_var = tk.StringVar(value="auto")
        grasp_row = ttk.Frame(vol)
        grasp_row.pack(fill=tk.X, pady=2)
        for val, label in (
            ("auto", "自动"),
            ("allow", "可夹"),
            ("deny", "不夹"),
        ):
            ttk.Radiobutton(
                grasp_row,
                text=label,
                value=val,
                variable=self.grasp_mode_var,
                command=self._update_readings,
            ).pack(side=tk.LEFT, padx=(0, 8))

        act = ttk.LabelFrame(right, text="液面线", padding=8)
        act.pack(fill=tk.X)
        ttk.Label(
            act,
            text="Interface1=主液面（红，可拖）。手动放置后 fill 随拖动变；\n「夹取判定」可选自动/可夹/不夹。",
            wraplength=300,
        ).pack(anchor=tk.W)
        ttk.Button(act, text="自动重新检测弯月面", command=self._auto_meniscus).pack(fill=tk.X, pady=(8, 2))
        ttk.Button(act, text="手动放置 Interface1（仅自动失败时用）", command=self._manual_iface1).pack(
            fill=tk.X, pady=2
        )
        ttk.Button(act, text="添加 / 清除 Interface2", command=self._toggle_iface2).pack(fill=tk.X, pady=2)

        self.status_var = tk.StringVar(value="请打开图片（仅预览）或本机摄像头")
        ttk.Label(right, textvariable=self.status_var, wraplength=300).pack(pady=8)

    def _open_threshold_editor(self) -> None:
        editor = self._thr_editor
        if editor is not None and editor.alive():
            editor.win.lift()
        else:
            self._thr_editor = ThresholdEditor(self.root, self._on_color_threshold)
        if self.source_bgr is not None and self._thr_editor is not None:
            self._thr_editor.push_frame(self.source_bgr)

    def _on_color_threshold(self, spec: ColorSpec, locked: bool) -> None:
        self.color_spec = spec if locked else None
        self.thr_status_var.set(("已锁定  " if locked else "未锁定  ") + spec.text())
        if not locked:
            return
        if self.crop_bgr is None:
            self._analyze_current()
        else:
            self._auto_meniscus()

    def _push_editor_frame(self, bgr: np.ndarray) -> None:
        editor = self._thr_editor
        if editor is not None and editor.alive():
            editor.push_frame(bgr)

    def _with_color(self, image: np.ndarray, ms: MeniscusResult) -> MeniscusResult:
        if self.color_spec is None:
            return ms
        return meniscus_with_color(image, ms, self.color_spec)

    def _on_roi_changed(self) -> None:
        if self.source_bgr is not None and not self.analyzed:
            self._redraw()

    def _open_image(self) -> None:
        self._stop_camera()
        path = filedialog.askopenfilename(
            title="选择图片",
            initialdir=str(ROOT),
            filetypes=[("Images", "*.jpg *.jpeg *.png *.bmp"), ("All", "*.*")],
        )
        if not path:
            return
        bgr = imread_chinese(path)
        if bgr is None:
            self.status_var.set("无法读取图片")
            return
        self._reset_analysis()
        self.source_bgr = bgr
        self.source_label = Path(path).name
        self.res_var.set(f"{bgr.shape[1]}x{bgr.shape[0]}")
        self.status_var.set(f"已加载 {self.source_label} — 点击「分析当前画面」")
        self._push_editor_frame(bgr)
        self._redraw()

    def _start_camera(self) -> None:
        self._stop_camera()
        self._reset_analysis()
        idx = int(self.cam_idx_var.get())
        self.cap = cv2.VideoCapture(idx, cv2.CAP_DSHOW)
        if not self.cap.isOpened():
            self.cap = cv2.VideoCapture(idx)
        if not self.cap.isOpened():
            self.status_var.set(f"无法打开摄像头 {idx}，可换索引 0–4 再试")
            self.cap = None
            return
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
        self._live_mode = True
        self.source_label = f"camera:{idx}"
        self._preview_times: list[float] = []
        self.status_var.set("本机摄像头已打开 — 对准 EP 管即可（可勾选自动跟线）")
        self._camera_loop()

    def _stop_camera(self) -> None:
        self._live_mode = False
        if self._camera_job is not None:
            self.root.after_cancel(self._camera_job)
            self._camera_job = None
        if self.cap is not None:
            self.cap.release()
            self.cap = None

    def _reset_analysis(self) -> None:
        self.analyzed = False
        self.crop_bgr = None
        self.crop_box = None
        self.ms = None
        self.meniscus_y = None
        self.meniscus_y2 = None
        self.min_px_var.set("—")
        self.max_px_var.set("—")
        self.grasp_mode_var.set("auto")
        self._update_readings()

    def _grasp_label(self, fill: float) -> str:
        mode = self.grasp_mode_var.get()
        if mode == "allow":
            return "是，可夹（手动指定）"
        if mode == "deny":
            return "否，不夹（手动指定）"
        if fill < self.min_fill:
            return f"否（fill < {self.min_fill}%）"
        return "是，可夹（自动）"

    def allow_grasp_effective(self, fill: float) -> bool:
        mode = self.grasp_mode_var.get()
        if mode == "allow":
            return True
        if mode == "deny":
            return False
        return fill >= self.min_fill

    def _on_brightness(self, _val: str) -> None:
        self._brightness = int(float(_val))

    def _apply_brightness(self, bgr: np.ndarray) -> np.ndarray:
        beta = self._brightness - 128
        return cv2.convertScaleAbs(bgr, alpha=1.0, beta=beta)

    def _track_interval_sec(self) -> float:
        hz = max(0.5, float(self.track_hz_var.get()))
        return 1.0 / hz

    def _camera_loop(self) -> None:
        if not self._live_mode or self.cap is None:
            return
        t0 = time.perf_counter()
        ok, frame = self.cap.read()
        if ok:
            frame = self._apply_brightness(frame)
            self._last_frame = frame
            self.source_bgr = frame
            self.res_var.set(f"{frame.shape[1]}x{frame.shape[0]}")
            now = time.perf_counter()
            self._preview_times.append(now)
            self._preview_times = [t for t in self._preview_times if now - t < 1.0]
            prev_fps = len(self._preview_times)
            if (
                self.auto_track_var.get()
                and not self._analyze_busy
                and now - self._last_analyze_t >= self._track_interval_sec()
            ):
                self._run_analysis_async(frame, live=True)
            self._fps_display = prev_fps
            self._refresh_fps_label()
            self._push_editor_frame(frame)
            self._redraw()
        delay = max(1, int(1000 / 30 - (time.perf_counter() - t0) * 1000))
        self._camera_job = self.root.after(delay, self._camera_loop)

    def _refresh_fps_label(self) -> None:
        hz = float(self.track_hz_var.get()) if self.auto_track_var.get() else 0.0
        self.fps_var.set(f"预览 FPS: ~{self._fps_display} | 分析: ~{hz:.1f} Hz")

    def _analyze_current(self) -> None:
        if self._last_frame is not None:
            bgr = self._last_frame
        elif self.source_bgr is not None:
            bgr = self.source_bgr
        else:
            self.status_var.set("请先打开图片或摄像头")
            return
        self._run_analysis(bgr, live=False)

    def _run_analysis_async(self, bgr: np.ndarray, live: bool) -> None:
        self._analyze_busy = True
        self._last_analyze_t = time.perf_counter()
        snap = bgr.copy()
        roi_x, roi_y = self.roi_x_var.get(), self.roi_y_var.get()

        def work() -> None:
            try:
                payload = self._analyze_to_payload(snap, roi_x, roi_y)
            except Exception:
                payload = None

            def apply() -> None:
                self._analyze_busy = False
                if payload is not None:
                    self._apply_analysis_payload(payload, live=live, quiet=True)
                elif live and not quiet:
                    self.status_var.set("本帧未检出管 — 保持上一帧结果")

            self.root.after(0, apply)

        threading.Thread(target=work, daemon=True).start()

    def _analyze_to_payload(
        self, bgr: np.ndarray, roi_x: int, roi_y: int
    ) -> dict | None:
        roi_img, off = apply_roi_percent(bgr, roi_x, roi_y)
        det = self.detector.detect(roi_img)
        if det is None or det.crop is None or det.crop.size == 0:
            return None
        x1, y1, x2, y2 = det.bbox
        gx1, gy1 = x1 + off[0], y1 + off[1]
        gx2, gy2 = x2 + off[0], y2 + off[1]
        ms = detect_meniscus(
            det.crop,
            self.tube_top_frac,
            self.tube_bottom_frac,
            self.center_x_left,
            self.center_x_right,
            self.meniscus_min_peak,
            self.empty_peak_max,
        )
        ms = self._with_color(det.crop, ms)
        return {
            "roi_offset": off,
            "crop_box": (gx1, gy1, gx2, gy2),
            "crop_bgr": det.crop.copy(),
            "ms": ms,
        }

    def _apply_analysis_payload(self, payload: dict, live: bool = False, quiet: bool = False) -> None:
        self.roi_offset = payload["roi_offset"]
        self.crop_box = payload["crop_box"]
        self.crop_bgr = payload["crop_bgr"]
        self.ms = payload["ms"]
        self.peak_score = self.ms.peak_score
        self.meniscus_y = self.ms.meniscus_y
        self.min_px_var.set(str(self.ms.tube_top))
        self.max_px_var.set(str(self.ms.tube_bottom))
        self.analyzed = True
        self._update_readings()
        self._redraw()
        if not quiet:
            tag = "实时" if live else self.source_label
            self.status_var.set(f"已分析: {tag}")

    def _run_analysis(self, bgr: np.ndarray, live: bool = False, quiet: bool = False) -> None:
        payload = self._analyze_to_payload(bgr, self.roi_x_var.get(), self.roi_y_var.get())
        if payload is None:
            if not quiet:
                self.status_var.set(
                    "未检测到离心管（黑块/手指剪影已忽略，请 ROI 100% 或移开手）"
                )
            self.analyzed = False
            return
        self._apply_analysis_payload(payload, live=live, quiet=quiet)

    def _auto_meniscus(self) -> None:
        if self.crop_bgr is None:
            return
        ms = detect_meniscus(
            self.crop_bgr,
            self.tube_top_frac,
            self.tube_bottom_frac,
            self.center_x_left,
            self.center_x_right,
            self.meniscus_min_peak,
            self.empty_peak_max,
        )
        ms = self._with_color(self.crop_bgr, ms)
        self.ms = ms
        self.peak_score = ms.peak_score
        self.meniscus_y = ms.meniscus_y
        self.min_px_var.set(str(ms.tube_top))
        self.max_px_var.set(str(ms.tube_bottom))
        self._update_readings()
        self._redraw()

    def _manual_iface1(self) -> None:
        if self.ms is None:
            return
        self.meniscus_y = (self.ms.tube_top + self.ms.tube_bottom) // 2
        self._update_readings()
        self._redraw()
        self.status_var.set("已手动放置 Interface1 — 可拖动红线，并在「夹取判定」选自动/可夹/不夹")

    def _toggle_iface2(self) -> None:
        if self.ms is None:
            return
        if self.meniscus_y2 is None:
            mid = (self.ms.tube_top + self.ms.tube_bottom) // 2
            self.meniscus_y2 = min(self.ms.tube_bottom - 4, mid + 40)
        else:
            self.meniscus_y2 = None
        self._update_readings()
        self._redraw()

    def _update_readings(self) -> None:
        if self.ms is None:
            self.iface1_var.set("None")
            self.iface2_var.set("None")
            self.fill_var.set("—")
            self.grasp_var.set("—")
            return
        self.iface1_var.set(ml_from_y(self.meniscus_y, self.ms.tube_top, self.ms.tube_bottom))
        self.iface2_var.set(ml_from_y(self.meniscus_y2, self.ms.tube_top, self.ms.tube_bottom))
        if self.meniscus_y is None:
            self.iface1_var.set("None")
            self.fill_var.set("0")
            self.grasp_var.set("否（太少 / 未检出液面）")
            return
        fill = fill_percent_from_meniscus(self.meniscus_y, self.ms.tube_top, self.ms.tube_bottom)
        self.fill_var.set(f"{fill:.1f}")
        self.grasp_var.set(self._grasp_label(fill))

    def _compose_crop_overlay(self) -> np.ndarray | None:
        if self.crop_bgr is None or self.ms is None:
            return None
        fill = (
            fill_percent_from_meniscus(self.meniscus_y, self.ms.tube_top, self.ms.tube_bottom)
            if self.meniscus_y is not None
            else 0.0
        )
        ms = MeniscusResult(
            meniscus_y=self.meniscus_y,
            fill_percent=fill,
            peak_score=self.peak_score,
            tube_top=self.ms.tube_top,
            tube_bottom=self.ms.tube_bottom,
            x_left=self.ms.x_left,
            x_right=self.ms.x_right,
        )
        vis = draw_level_overlay(self.crop_bgr, ms)
        if self.meniscus_y2 is not None:
            x0, x1 = self.ms.x_left, self.ms.x_right
            cv2.line(vis, (x0, self.meniscus_y2), (x1, self.meniscus_y2), (0, 220, 255), 2, cv2.LINE_AA)
            vis = put_chinese(vis, "IF2", (x1 + 4, self.meniscus_y2 - 4))
        return vis

    def _compose_full_overlay(self) -> np.ndarray | None:
        if self.source_bgr is None:
            return None
        vis = self.source_bgr.copy()
        h = vis.shape[0]
        rx, ry = self.roi_x_var.get(), self.roi_y_var.get()
        roi_img, off = apply_roi_percent(self.source_bgr, rx, ry)
        rh, rw = roi_img.shape[:2]
        cv2.rectangle(vis, (off[0], off[1]), (off[0] + rw, off[1] + rh), (255, 180, 0), 2)
        if self.crop_box is not None:
            gx1, gy1, gx2, gy2 = self.crop_box
            cv2.rectangle(vis, (gx1, gy1), (gx2, gy2), (0, 200, 0), 2)
        if self.ms is not None and self.crop_box is not None and self.analyzed:
            gx1, gy1, _, _ = self.crop_box
            x0 = gx1 + self.ms.x_left
            x1 = gx1 + self.ms.x_right
            cx = (x0 + x1) // 2
            y_top_g = gy1 + self.ms.tube_top
            y_bot_g = gy1 + self.ms.tube_bottom
            cv2.line(vis, (x0, y_top_g), (x1, y_top_g), (0, 0, 255), 1, cv2.LINE_AA)
            cv2.line(vis, (x0, y_bot_g), (x1, y_bot_g), (0, 0, 255), 2, cv2.LINE_AA)
            cv2.line(vis, (cx, y_top_g), (cx, y_bot_g), (0, 0, 255), 1, cv2.LINE_AA)
            if self.meniscus_y is not None:
                y = gy1 + self.meniscus_y
                cv2.line(vis, (x0, y), (x1, y), (0, 0, 255), 2, cv2.LINE_AA)
                fill = fill_percent_from_meniscus(
                    self.meniscus_y, self.ms.tube_top, self.ms.tube_bottom
                )
                peak_pct = self.peak_score * 100.0
                cv2.putText(
                    vis,
                    f"peak {peak_pct:.1f}%  Y={self.meniscus_y}px",
                    (x0, max(y_top_g + 20, y - 10)),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.45,
                    (0, 0, 255),
                    1,
                    cv2.LINE_AA,
                )
                vis = put_chinese(vis, f"fill {fill:.1f}%", (x0, min(h - 36, y + 26)))
            else:
                vis = put_chinese(vis, "未检出液面 — 可点「手动放置 Interface1」", (x0, y_top_g + 24))
            if self.meniscus_y2 is not None:
                y2 = gy1 + self.meniscus_y2
                cv2.line(vis, (x0, y2), (x1, y2), (0, 220, 255), 2, cv2.LINE_AA)
        elif not self.analyzed:
            vis = put_chinese(vis, "未分析 — 点击「分析当前画面」", (12, 28))
        return vis

    def _redraw(self) -> None:
        if self.view_crop_var.get() and self.analyzed and self.crop_bgr is not None:
            vis = self._compose_crop_overlay()
        else:
            vis = self._compose_full_overlay()
        if vis is None:
            return
        rgb = cv2.cvtColor(vis, cv2.COLOR_BGR2RGB)
        self._display_rgb(rgb)

    def _display_rgb(self, rgb: np.ndarray) -> None:
        cw = max(self.canvas.winfo_width(), 1)
        ch = max(self.canvas.winfo_height(), 1)
        h, w = rgb.shape[:2]
        self._scale = min(cw / w, ch / h)
        nw, nh = max(1, int(w * self._scale)), max(1, int(h * self._scale))
        self._offset_x = (cw - nw) // 2
        self._offset_y = (ch - nh) // 2
        self._display_w, self._display_h = nw, nh
        img = Image.fromarray(rgb).resize((nw, nh), Image.Resampling.LANCZOS)
        self._photo = ImageTk.PhotoImage(img)
        self.canvas.delete("all")
        self.canvas.create_image(self._offset_x, self._offset_y, anchor=tk.NW, image=self._photo)

    def _image_y_from_event(self, event: tk.Event) -> int | None:
        if not self.view_crop_var.get():
            if self.crop_box is None or self.ms is None:
                return None
            gy1 = self.crop_box[1]
            y_full = (event.y - self._offset_y) / max(self._scale, 1e-6)
            y_crop = int(round(y_full - gy1))
            return y_crop
        y = (event.y - self._offset_y) / max(self._scale, 1e-6)
        if self.crop_bgr is None:
            return None
        return int(round(y))

    def _line_canvas_y(self, y_crop: int) -> float:
        if not self.view_crop_var.get() and self.crop_box is not None:
            return self._offset_y + (self.crop_box[1] + y_crop) * self._scale
        return self._offset_y + y_crop * self._scale

    def _on_press(self, event: tk.Event) -> None:
        if self.ms is None:
            return
        y_crop = self._image_y_from_event(event)
        if y_crop is None:
            return
        hits: list[tuple[str, int]] = []
        if self.meniscus_y is not None:
            if abs(event.y - self._line_canvas_y(self.meniscus_y)) <= 14:
                hits.append(("if1", self.meniscus_y))
        if self.meniscus_y2 is not None:
            if abs(event.y - self._line_canvas_y(self.meniscus_y2)) <= 14:
                hits.append(("if2", self.meniscus_y2))
        if not hits:
            return
        hits.sort(key=lambda t: abs(t[1] - y_crop))
        self._dragging = hits[0][0]
        self._drag_offset = event.y - self._line_canvas_y(hits[0][1])

    def _on_drag(self, event: tk.Event) -> None:
        if not self._dragging or self.ms is None:
            return
        y_crop = self._image_y_from_event(event)
        if y_crop is None:
            y_full = (event.y - self._drag_offset - self._offset_y) / max(self._scale, 1e-6)
            if not self.view_crop_var.get() and self.crop_box is not None:
                y_crop = int(round(y_full - self.crop_box[1]))
            else:
                y_crop = int(round(y_full))
        y_crop = max(self.ms.tube_top + 2, min(self.ms.tube_bottom - 2, y_crop))
        if self._dragging == "if1":
            self.meniscus_y = y_crop
        else:
            self.meniscus_y2 = y_crop
        self._update_readings()
        self._redraw()

    def _on_release(self, _event: tk.Event) -> None:
        self._dragging = None

    def run(self) -> None:
        self.root.protocol("WM_DELETE_WINDOW", self._on_close)
        self.root.mainloop()

    def _on_close(self) -> None:
        self._stop_camera()
        self.root.destroy()


def main(config_path: Path | None = None) -> None:
    LevelMeterApp(config_path).run()


if __name__ == "__main__":
    main()
