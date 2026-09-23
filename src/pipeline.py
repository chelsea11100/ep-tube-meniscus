from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from .common import ROOT, load_config
from .detect_tube import TubeDetector, imwrite_chinese
from .judge_liquid import create_liquid_judge


IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def case_dir_for(output_dir: Path, image_path: Path) -> Path:
    """每张图一个文件夹，避免几十张图的 crop/det/json 堆在同一层。"""
    folder = output_dir / image_path.stem
    folder.mkdir(parents=True, exist_ok=True)
    return folder


def run_one(
    image_path: Path,
    detector: TubeDetector,
    judge,
    output_dir: Path,
) -> dict:
    detection = detector.detect_path(image_path)
    folder = case_dir_for(output_dir, image_path)
    crop_path = folder / "crop.jpg"
    det_path = folder / "det.jpg"
    imwrite_chinese(crop_path, detection.crop)
    imwrite_chinese(det_path, detection.annotated)

    judgement = judge.judge(detection.crop)
    level_path = folder / "level.jpg"
    if judgement.level_image is not None:
        imwrite_chinese(level_path, judgement.level_image)
    result = {
        "image": str(image_path),
        "detection": {
            "bbox": [int(v) for v in detection.bbox],
            "confidence": round(detection.confidence, 4),
            "class_name": detection.class_name,
            "used_fallback": detection.used_fallback,
            "rbox": detection.rbox,
            "angle_deg": round(float(detection.angle_deg), 2),
        },
        "crop_path": str(crop_path),
        "det_path": str(det_path),
        "fill_percent": judgement.fill_percent,
        "liquid_present": judgement.liquid_present,
        "too_little": judgement.too_little,
        "allow_grasp": not judgement.too_little,
        "confidence": judgement.confidence,
        "evidence": judgement.evidence,
        "judge_backend": str((detector.config.get("liquid") or {}).get("backend", "local")),
        "meniscus_y": judgement.meniscus_y,
        "level_path": str(level_path) if judgement.level_image is not None else None,
        "judge_raw": judgement.raw_text,
    }
    (folder / "result.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return result


def _detect_only(image_path: Path, detector: TubeDetector, output_dir: Path) -> dict:
    detection = detector.detect_path(image_path)
    folder = case_dir_for(output_dir, image_path)
    crop_path = folder / "crop.jpg"
    det_path = folder / "det.jpg"
    imwrite_chinese(crop_path, detection.crop)
    imwrite_chinese(det_path, detection.annotated)
    return {
        "image": str(image_path),
        "detection": {
            "bbox": [int(v) for v in detection.bbox],
            "confidence": round(detection.confidence, 4),
            "class_name": detection.class_name,
            "used_fallback": detection.used_fallback,
            "rbox": detection.rbox,
            "angle_deg": round(float(detection.angle_deg), 2),
        },
        "crop_path": str(crop_path),
        "det_path": str(det_path),
    }


def default_images() -> list[Path]:
    sample_dir = ROOT / "照片test"
    if sample_dir.is_dir():
        found = collect_images([sample_dir])
        if found:
            return found
    return sorted(p for p in ROOT.iterdir() if p.is_file() and p.suffix.lower() in IMAGE_EXTS)


def collect_images(paths: list[Path]) -> list[Path]:
    found: list[Path] = []
    for path in paths:
        if path.is_dir():
            found.extend(
                sorted(p for p in path.iterdir() if p.is_file() and p.suffix.lower() in IMAGE_EXTS)
            )
        elif path.is_file() and path.suffix.lower() in IMAGE_EXTS:
            found.append(path)
    return found


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="离心管液量识别：OpenCV/YOLO 裁管 + 本地弯月面或千问")
    parser.add_argument(
        "images",
        nargs="*",
        type=Path,
        help="图片或文件夹。给文件夹会测里面全部 jpg/png。默认测 照片test 里的全部图片",
    )
    parser.add_argument("--config", type=Path, default=None, help="配置文件路径")
    parser.add_argument("--detect-only", action="store_true", help="只跑 YOLO 检测裁剪，不调用视觉模型")
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="打印 YOLO / 千问详细过程（config 里 verbose 为 true 时默认开启）",
    )
    parser.add_argument("--quiet", action="store_true", help="不打印 YOLO / 千问过程")
    args = parser.parse_args(argv)

    config = load_config(args.config)
    if args.quiet:
        verbose = False
    elif args.verbose:
        verbose = True
    else:
        verbose = bool(config.get("verbose", True))
    config["verbose"] = verbose
    output_dir = Path(config.get("paths", {}).get("output_dir", "output"))
    if not output_dir.is_absolute():
        output_dir = ROOT / output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    given = [p.resolve() if p.is_absolute() else (Path.cwd() / p).resolve() for p in args.images]
    images = collect_images(given) if given else default_images()
    if not images:
        print("未找到待处理图片", file=sys.stderr)
        return 1

    detector = TubeDetector(config, verbose=verbose)
    judge = None if args.detect_only else create_liquid_judge(config, verbose=verbose)
    results = []
    for image_path in images:
        print(f"处理: {image_path}")
        if args.detect_only:
            result = _detect_only(image_path, detector, output_dir)
        else:
            result = run_one(image_path, detector, judge, output_dir)
            status = "太少，不夹" if result["too_little"] else "足够，可夹"
            print(f"  液量 {result['fill_percent']}%  |  {status}")
            print(f"  依据: {result['evidence']}")
        results.append(result)
        print(
            f"  检测 {result['detection']['class_name']} "
            f"{result['detection']['confidence']:.2f}  fallback={result['detection']['used_fallback']}"
        )

    summary_path = output_dir / "summary.json"
    summary_path.write_text(json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"结果已写入 {summary_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
