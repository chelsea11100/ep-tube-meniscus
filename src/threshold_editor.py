"""CanMV 风格阈值编辑器：左原图，右二值图，六条最小/最大滑块。"""

from __future__ import annotations

import tkinter as tk
from tkinter import ttk

import cv2
import numpy as np
from PIL import Image, ImageTk

from .color_threshold import SPACE_DEFAULTS, SPACE_LIMITS, ColorSpec, color_mask, meniscus_y_from_mask
from .detect_tube import put_chinese

BG = "#1e1e1e"
FG = "#f0f0f0"
MUTED = "#bdbdbd"
PANEL = "#2a2a2a"


class ThresholdEditor:
    def __init__(self, master: tk.Misc, on_apply) -> None:
        self.on_apply = on_apply
        self.frame_bgr: np.ndarray | None = None
        self.locked = False
        self.threshold_var = tk.StringVar(value="")
        self._job: str | None = None
        self._apply_job: str | None = None
        self._src_photo: ImageTk.PhotoImage | None = None
        self._bin_photo: ImageTk.PhotoImage | None = None
        self._alive = True

        self.win = tk.Toplevel(master)
        self.win.title("阈值编辑器")
        self.win.configure(bg=BG)
        self.win.geometry("1080x760")
        self.win.minsize(860, 640)
        self.win.protocol("WM_DELETE_WINDOW", self._close)

        panes = tk.Frame(self.win, bg=BG)
        panes.pack(fill=tk.BOTH, expand=True, padx=10, pady=(10, 4))
        self.src_title, self.src_label = self._pane(panes, "源图像")
        self.bin_title, self.bin_label = self._pane(panes, "二进制图像（白色像素是被跟踪的像素）")

        bar = tk.Frame(self.win, bg=BG)
        bar.pack(fill=tk.X, padx=12, pady=(4, 0))
        tk.Label(bar, text="选择最佳的颜色跟踪阈值。", bg=BG, fg=FG).pack(side=tk.LEFT)
        self.space_var = tk.StringVar(value="LAB")
        space = ttk.Combobox(
            bar,
            textvariable=self.space_var,
            values=("LAB", "HSV", "RGB"),
            width=8,
            state="readonly",
        )
        space.pack(side=tk.LEFT, padx=8)
        space.bind("<<ComboboxSelected>>", lambda _e: self._rebuild_sliders(reset=True))
        self.invert_var = tk.BooleanVar(value=False)
        tk.Checkbutton(
            bar,
            text="反转",
            variable=self.invert_var,
            command=self._on_slider,
            bg=BG,
            fg=FG,
            selectcolor="#3a3a3a",
            activebackground=BG,
            activeforeground=FG,
        ).pack(side=tk.LEFT, padx=8)
        tk.Button(bar, text="重置滑块", command=lambda: self._rebuild_sliders(reset=True)).pack(side=tk.RIGHT)

        self.slider_host = tk.Frame(self.win, bg=BG)
        self.slider_host.pack(fill=tk.X, padx=12, pady=6)
        self._lo_vars: list[tk.IntVar] = []
        self._hi_vars: list[tk.IntVar] = []
        self._lo_text: list[tk.StringVar] = []
        self._hi_text: list[tk.StringVar] = []
        self._rebuild_sliders(reset=True)

        foot = tk.Frame(self.win, bg=BG)
        foot.pack(fill=tk.X, padx=12, pady=(0, 10))
        tk.Label(foot, textvariable=self.threshold_var, bg=BG, fg="#7dcea0", anchor=tk.W).pack(fill=tk.X)
        tk.Label(
            foot,
            text="把液体调成白色，夹爪和背景保持黑色。点「用白色区域找液面」后，主画面用白色上沿当液面。关闭时复制阈值。",
            bg=BG,
            fg=MUTED,
            anchor=tk.W,
            justify=tk.LEFT,
        ).pack(fill=tk.X, pady=(2, 6))
        row = tk.Frame(foot, bg=BG)
        row.pack(fill=tk.X)
        tk.Button(row, text="用白色区域找液面", command=self._lock).pack(side=tk.LEFT)
        tk.Button(row, text="取消锁定", command=self._unlock).pack(side=tk.LEFT, padx=8)
        tk.Button(row, text="关闭", command=self._close).pack(side=tk.RIGHT)
        self._refresh_text()

    def alive(self) -> bool:
        return self._alive and bool(self.win.winfo_exists())

    def push_frame(self, bgr: np.ndarray) -> None:
        if not self.alive():
            return
        self.frame_bgr = bgr
        self._schedule()

    def _pane(self, parent: tk.Frame, title: str) -> tuple[tk.Label, tk.Label]:
        box = tk.Frame(parent, bg=PANEL, highlightbackground="#4a4a4a", highlightthickness=1)
        box.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        title_lb = tk.Label(box, text=title, bg=PANEL, fg=FG, anchor=tk.W)
        title_lb.pack(fill=tk.X, padx=6, pady=(4, 2))
        image_lb = tk.Label(box, bg="#111111")
        image_lb.pack(fill=tk.BOTH, expand=True, padx=6, pady=(0, 6))
        return title_lb, image_lb

    def _rebuild_sliders(self, reset: bool = False) -> None:
        for child in self.slider_host.winfo_children():
            child.destroy()
        space = self.space_var.get()
        limits = SPACE_LIMITS[space]
        defaults = SPACE_DEFAULTS[space]
        self._lo_vars = []
        self._hi_vars = []
        self._lo_text = []
        self._hi_text = []
        for (name, lo, hi), (d0, d1) in zip(limits, defaults):
            row_min = self._slider_row(name, "最小值", lo, hi, d0, self._lo_vars, self._lo_text)
            row_min.pack(fill=tk.X, pady=1)
            row_max = self._slider_row(name, "最大值", lo, hi, d1, self._hi_vars, self._hi_text)
            row_max.pack(fill=tk.X, pady=1)
        if reset:
            self._on_slider()

    def _slider_row(
        self,
        name: str,
        which: str,
        lo: int,
        hi: int,
        value: int,
        bag: list[tk.IntVar],
        texts: list[tk.StringVar],
    ) -> tk.Frame:
        row = tk.Frame(self.slider_host, bg=BG)
        tk.Label(row, text=f"{name}{which}", width=10, anchor=tk.W, bg=BG, fg=FG).pack(side=tk.LEFT)
        var = tk.IntVar(value=int(value))
        text = tk.StringVar(value=str(int(value)))
        tk.Scale(
            row,
            from_=lo,
            to=hi,
            orient=tk.HORIZONTAL,
            variable=var,
            showvalue=False,
            resolution=1,
            command=self._on_slider,
            bg=BG,
            fg=FG,
            troughcolor="#3a3a3a",
            highlightthickness=0,
            sliderlength=18,
        ).pack(side=tk.LEFT, fill=tk.X, expand=True, padx=8)
        tk.Label(row, textvariable=text, width=5, anchor=tk.E, bg=BG, fg=FG).pack(side=tk.RIGHT)
        bag.append(var)
        texts.append(text)
        return row

    def _spec(self) -> ColorSpec:
        bounds = tuple(
            (int(lo.get()), int(hi.get())) for lo, hi in zip(self._lo_vars, self._hi_vars)
        )
        return ColorSpec(self.space_var.get(), bounds, bool(self.invert_var.get()))  # type: ignore[arg-type]

    def _refresh_text(self) -> None:
        for var, text in zip(self._lo_vars, self._lo_text):
            text.set(str(int(var.get())))
        for var, text in zip(self._hi_vars, self._hi_text):
            text.set(str(int(var.get())))
        spec = self._spec()
        mark = "已锁定  " if self.locked else ""
        self.threshold_var.set(mark + spec.text())

    def _on_slider(self, _v: str | None = None) -> None:
        if not self._lo_vars:
            return
        self._refresh_text()
        self._schedule()
        if self.locked:
            self._schedule_apply()

    def _schedule(self) -> None:
        if not self.alive() or self._job is not None:
            return
        self._job = self.win.after(40, self._render)

    def _schedule_apply(self) -> None:
        if not self.alive():
            return
        if self._apply_job is not None:
            self.win.after_cancel(self._apply_job)
        self._apply_job = self.win.after(120, self._emit)

    def _emit(self) -> None:
        self._apply_job = None
        if self.alive():
            self.on_apply(self._spec(), self.locked)

    def _lock(self) -> None:
        self.locked = True
        self._refresh_text()
        self._emit()

    def _unlock(self) -> None:
        self.locked = False
        self._refresh_text()
        self._emit()

    def _close(self) -> None:
        text = self._spec().text()
        try:
            self.win.clipboard_clear()
            self.win.clipboard_append(text)
        except tk.TclError:
            pass
        self._alive = False
        self.win.destroy()

    def _render(self) -> None:
        self._job = None
        if not self.alive():
            return
        src, binary = self._views()
        self._src_photo = self._to_photo(src)
        self._bin_photo = self._to_photo(binary)
        self.src_label.configure(image=self._src_photo)
        self.bin_label.configure(image=self._bin_photo)

    def _views(self) -> tuple[np.ndarray, np.ndarray]:
        if self.frame_bgr is None:
            blank = np.full((360, 480, 3), 28, np.uint8)
            blank = put_chinese(blank, "请先打开摄像头或图片", (24, 170))
            return blank, blank.copy()
        view = _fit(self.frame_bgr, 500, 380)
        spec = self._spec()
        mask = color_mask(view, spec)
        binary = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)
        h, w = mask.shape[:2]
        y, _conf, _x0, _x1 = meniscus_y_from_mask(mask, int(h * 0.02), h - 1, int(w * 0.05), w - max(1, int(w * 0.05)))
        if y is not None:
            cv2.line(view, (0, y), (w - 1, y), (0, 0, 255), 2, cv2.LINE_AA)
            cv2.line(binary, (0, y), (w - 1, y), (0, 0, 255), 2, cv2.LINE_AA)
        return view, binary

    def _to_photo(self, bgr: np.ndarray) -> ImageTk.PhotoImage:
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        return ImageTk.PhotoImage(Image.fromarray(rgb))


def _fit(bgr: np.ndarray, max_w: int, max_h: int) -> np.ndarray:
    h, w = bgr.shape[:2]
    scale = min(max_w / max(1, w), max_h / max(1, h), 1.0)
    if scale >= 0.999:
        return bgr.copy()
    return cv2.resize(bgr, (max(1, int(w * scale)), max(1, int(h * scale))), interpolation=cv2.INTER_AREA)
