from __future__ import annotations

import base64
import json
import os
import re
import time
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
from dotenv import load_dotenv
from openai import OpenAI

from .common import ROOT, load_config, log_verbose
from .detect_tube import imread_chinese

load_dotenv(ROOT / ".env")


@dataclass
class LiquidJudgement:
    fill_percent: int
    liquid_present: bool
    too_little: bool
    confidence: float
    evidence: str
    raw_text: str
    level_image: np.ndarray | None = None
    meniscus_y: int | None = None


def create_liquid_judge(config: dict | None = None, verbose: bool | None = None):
    cfg = config or load_config()
    liquid_cfg = cfg.get("liquid") or {}
    backend = str(liquid_cfg.get("backend") or "local").lower()
    if backend in ("qwen", "vlm", "api"):
        return LiquidJudge(cfg, verbose=verbose)
    from .judge_meniscus import LocalLiquidJudge

    return LocalLiquidJudge(cfg, verbose=verbose)


class LiquidJudge:
    def __init__(self, config: dict | None = None, verbose: bool | None = None):
        self.config = config or load_config()
        self.verbose = bool(self.config.get("verbose", True)) if verbose is None else verbose
        vlm = self.config["vlm"]
        self.model = vlm.get("model", "qwen-vl-max")
        self.api_base = vlm.get("api_base", "https://dashscope.aliyuncs.com/compatible-mode/v1")
        self.timeout_sec = float(vlm.get("timeout_sec", 90))
        self.max_image_side = int(vlm.get("max_image_side", 1280))
        self.min_fill_percent = int(self.config.get("min_fill_percent", 10))
        prompt_file = Path(vlm.get("prompt_file", "prompts/liquid_judge.txt"))
        if not prompt_file.is_absolute():
            prompt_file = ROOT / prompt_file
        self.prompt = prompt_file.read_text(encoding="utf-8").strip()
        api_key = (os.getenv("DASHSCOPE_API_KEY") or os.getenv("OPENAI_API_KEY") or "").strip()
        if not api_key:
            raise RuntimeError("未设置 DASHSCOPE_API_KEY（或 OPENAI_API_KEY）")
        self.client = OpenAI(api_key=api_key, base_url=self.api_base, timeout=self.timeout_sec)

    def judge(self, image: np.ndarray) -> LiquidJudgement:
        h, w = image.shape[:2]
        jpeg = encode_jpeg(image, self.max_image_side)
        jpeg_kb = len(jpeg) * 3 / 4 / 1024
        log_verbose(
            self.verbose,
            f"[千问] 模型={self.model}\n"
            f"[千问] 接口={self.api_base}\n"
            f"[千问] 裁剪图 {w}x{h} → JPEG 约 {jpeg_kb:.0f} KB（max_side={self.max_image_side}）\n"
            f"[千问] 正在请求…",
        )
        t0 = time.perf_counter()
        response = self.client.chat.completions.create(
            model=self.model,
            temperature=0.1,
            messages=[
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "image_url",
                            "image_url": {"url": f"data:image/jpeg;base64,{jpeg}"},
                        },
                        {"type": "text", "text": self.prompt},
                    ],
                }
            ],
        )
        elapsed = time.perf_counter() - t0
        raw = (response.choices[0].message.content or "").strip()
        usage = getattr(response, "usage", None)
        usage_line = ""
        if usage is not None:
            usage_line = (
                f"[千问] tokens: prompt={getattr(usage, 'prompt_tokens', '?')} "
                f"completion={getattr(usage, 'completion_tokens', '?')} "
                f"total={getattr(usage, 'total_tokens', '?')}\n"
            )
        log_verbose(
            self.verbose,
            f"[千问] 响应耗时 {elapsed:.2f}s\n{usage_line}"
            f"[千问] 原始输出:\n{raw}\n[千问] ---",
        )
        parsed = parse_json_object(raw)
        fill_percent = clamp_int(parsed.get("fill_percent", 0), 0, 100)
        liquid_present = bool(parsed.get("liquid_present", fill_percent > 0))
        confidence = clamp_float(parsed.get("confidence", 0.5), 0.0, 1.0)
        evidence = str(parsed.get("evidence", "")).strip() or "模型未给出依据"
        too_little = fill_percent < self.min_fill_percent
        log_verbose(
            self.verbose,
            f"[千问] 解析: fill_percent={fill_percent} liquid_present={liquid_present} "
            f"confidence={confidence} too_little={too_little} (阈值 {self.min_fill_percent}%)",
        )
        return LiquidJudgement(
            fill_percent=fill_percent,
            liquid_present=liquid_present,
            too_little=too_little,
            confidence=confidence,
            evidence=evidence,
            raw_text=raw,
            level_image=None,
            meniscus_y=None,
        )

    def judge_path(self, image_path: str | Path) -> LiquidJudgement:
        image = imread_chinese(image_path)
        if image is None:
            raise FileNotFoundError(f"无法读取图片: {image_path}")
        return self.judge(image)


def encode_jpeg(image: np.ndarray, max_side: int) -> str:
    h, w = image.shape[:2]
    scale = min(1.0, max_side / max(h, w))
    if scale < 1.0:
        image = cv2.resize(image, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_AREA)
    ok, buf = cv2.imencode(".jpg", image, [int(cv2.IMWRITE_JPEG_QUALITY), 90])
    if not ok:
        raise RuntimeError("JPEG 编码失败")
    return base64.b64encode(buf.tobytes()).decode("ascii")


def parse_json_object(text: str) -> dict:
    cleaned = text.strip()
    fenced = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", cleaned, re.DOTALL)
    if fenced:
        cleaned = fenced.group(1)
    else:
        start = cleaned.find("{")
        end = cleaned.rfind("}")
        if start >= 0 and end > start:
            cleaned = cleaned[start : end + 1]
    try:
        data = json.loads(cleaned)
    except json.JSONDecodeError as exc:
        raise ValueError(f"视觉模型未返回合法 JSON: {text}") from exc
    if not isinstance(data, dict):
        raise ValueError(f"视觉模型 JSON 不是对象: {text}")
    return data


def clamp_int(value, low: int, high: int) -> int:
    try:
        number = int(float(value))
    except (TypeError, ValueError):
        number = low
    return max(low, min(high, number))


def clamp_float(value, low: float, high: float) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError):
        number = low
    return max(low, min(high, number))
