# EP Tube Meniscus

Vision service that locates a 1.5 mL microcentrifuge tube (EP tube), finds the liquid meniscus, and returns fill level, estimated volume, and a grasp decision for a lab robot or workstation.

面向实验室工位和机械臂：从摄像头或图片里定位 1.5 mL 离心管（EP 管），检测弯月面液位，输出液量百分比、估算体积，以及「液体是否足够、能否夹取」。

本项目来自中国科学院深圳先进技术研究院（中科院深圳先进院），与温氏集团合作开展。

**Affiliation:** Shenzhen Institutes of Advanced Technology, Chinese Academy of Sciences (SIAT, CAS), in collaboration with Wens Group.

**Repository:** https://github.com/chelsea11100/ep-tube-meniscus

---

## For reviewers

A robot that picks EP tubes should not grasp a tube that is empty or nearly empty. This project was developed at the Shenzhen Institutes of Advanced Technology, Chinese Academy of Sciences (SIAT, CAS), in collaboration with Wens Group. It turns one camera frame into that grasp decision, without requiring a person to read the tube by eye.

Given a USB / V4L2 / RTSP frame, or an uploaded JPEG/PNG, the pipeline:

1. Crops the centre of the frame to reduce fingers and background.
2. Localises the tube with classical vision (edges and a rotated rectangle). An optional YOLO-World model can be enabled as a fallback.
3. Estimates the meniscus on the tube crop from the horizontal intensity peak along the tube body, then converts that height into a fill percentage of a nominal 1.5 mL tube.
4. Returns structured JSON plus an annotated image. `allow_grasp` is true only when the tube is found, a meniscus is reliable, and the fill is at or above the configured minimum (default 10%).

The same logic is exposed as a FastAPI service (the integration point for a robot), a desktop GUI for camera preview and threshold tuning, a small Python class (`EpTubeInspector`), and command-line scripts for offline images. A cloud vision-language model (Qwen-VL) can replace the local meniscus estimator if an API key is set; the default path stays fully local.

Stack: Python, OpenCV, NumPy, FastAPI. Optional: Ultralytics YOLO, an OpenAI-compatible vision API.

---

## 这个项目解决什么问题

离心管夹取前需要知道管内还有没有足够液体。靠人眼看管壁刻度慢，也不适合接到机械臂或工位程序里。本项目在中科院深圳先进院完成，与温氏集团合作，把「拍一张图」收成一次调用：检出管子、找到液面、给出液量和是否允许夹取。

默认算法全部在本机运行，不依赖网络。检出失败或液面不可靠时，`allow_grasp` 为 false，并在 `error` 里说明原因，避免把整图估计当成可以夹取的结果。

## 处理流程

```text
摄像头 / 上传图片
        │
        ▼
  中心区域裁剪（减弱手指和背景）
        │
        ▼
  定位离心管
  默认：OpenCV 边缘 + 旋转矩形
  可选：YOLO-World 兜底
        │
        ▼
  在管身裁剪图上检测弯月面
  默认：沿管身的水平亮度峰
  可选：千问视觉模型
        │
        ▼
  JSON（液量、体积、是否可夹取）+ 标注图
```

名义容量按 1.5 mL 换算：`volume_ml = fill_percent / 100 × 1.5`。

## 一次调用会返回什么

| 字段 | 含义 |
| --- | --- |
| `ok` | 检出了管子，且液面可靠 |
| `fill_percent` | 相对满管的液量百分比，0–100 |
| `volume_ml` | 按 1.5 mL 名义容量估算的体积 |
| `allow_grasp` | 管子有效、液面可靠，且液量不低于 `min_fill_percent` |
| `meniscus_y` | 弯月面在裁剪图中的竖直位置（像素） |
| `bbox` | 管子外接框 |
| `confidence` | 本次液面判断的置信度 |
| `evidence` | 判断依据的简短说明 |
| `error` | 失败原因；成功时为 null |
| `image_base64` | 画好管框和液面线的 JPEG |

`allow_grasp` 为 false 的典型情况：没检出管子、检出了管子但没有可靠液面、液量低于配置阈值（默认 10%）。

## 环境要求

- Python 3.10 及以上
- 摄像头场景需要本机能打开 USB 相机（Linux 使用 V4L2），或可访问的 RTSP 地址
- 只对已有图片做识别时，不需要摄像头

## 快速开始

```bash
git clone https://github.com/chelsea11100/ep-tube-meniscus.git
cd ep-tube-meniscus
python3 -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

启动 HTTP 服务（工位和机械臂推荐用这个入口）：

```bash
python run_server.py
```

- 接口文档：http://127.0.0.1:8000/docs
- 健康检查：http://127.0.0.1:8000/health

查看当前摄像头：

```bash
curl http://127.0.0.1:8000/v1/cameras
```

拍一帧并识别（`source` 可省略，省略时按 `config.yaml`）：

```bash
curl -X POST "http://127.0.0.1:8000/v1/snapshot?source=usb&include_image=true"
```

上传一张已有图片：

```bash
curl -X POST "http://127.0.0.1:8000/v1/level" -F "image=@tube.jpg"
```

不启服务、直接在本机拍一张并写出 JSON 和标注图：

```bash
python run_inspect.py -o output/inspect_result.json
```

结果在 `output/`：`inspect_result.json`、相机原图、标注图。

对一个文件夹里的图片批量跑检测（不经过摄像头）：

```bash
python run.py path/to/images
```

每张图会在 `output/<文件名>/` 下留下裁剪图、检测图、液面图和 `result.json`。

## 桌面界面

现场看画面、拖动液面线、调 ROI 和阈值时用桌面界面，不必先起 HTTP 服务。

```bash
python run_gui.py
python run_gui.py --camera 0 --open-camera
```

1. 右侧点 **扫描**，在下拉框里选外接 USB，不要选笔记本内置摄像头。
2. 点 **打开 USB 摄像头**。
3. 勾选 **摄像头开启时自动分析跟线**，或点 **分析当前画面**。

默认相机写在 `config.yaml` 的 `gui.camera_index`、`prefer_usb`、`auto_open_camera`。改完代码后关掉窗口再开。

## HTTP 接口

| 方法 | 路径 | 用途 |
| --- | --- | --- |
| `GET` | `/health` | 服务是否就绪 |
| `GET` | `/v1/cameras` | 列出本机摄像头，并给出优先使用的索引 |
| `POST` | `/v1/snapshot` | 取一帧并识别。工位主入口 |
| `POST` | `/v1/level` | 上传一张 jpg/png，表单字段名 `image` |
| `POST` | `/v1/level/batch` | 一次上传多张，表单字段名 `images` |

`POST /v1/snapshot` 的查询参数：

| 参数 | 说明 |
| --- | --- |
| `source` | `usb`、`v4l2`、`rtsp`、`plugin`。省略则用配置文件 |
| `camera_index` | USB 设备索引。省略则优先外接 USB |
| `include_image` | 是否在 JSON 里带标注图，默认 true |

Python 侧可以直接用类，不必自己拼 HTTP：

```python
from src.ep_tube_inspector import EpTubeInspector, EpTubeHttpClient

# 本进程内：取图 + 识别
result = EpTubeInspector().snapshot(source="usb")

# 已有图片
result = EpTubeInspector().inspect("tube.jpg")

# 调用已经启动的 run_server.py
client = EpTubeHttpClient("http://127.0.0.1:8000")
result = client.snapshot(source="usb", include_image=True)
```

`examples/robot_one_call.py` 是给工位程序看的最小调用示例，支持走 HTTP，也支持 `--local` 在进程内直接跑。

## 配置

主配置是仓库根目录的 `config.yaml`。

| 配置 | 作用 |
| --- | --- |
| `min_fill_percent` | 低于该百分比则 `allow_grasp` 为 false，默认 10 |
| `yolo.backend` | 默认 `opencv_first`：先用 OpenCV，需要时再走 YOLO |
| `yolo.fallback_yolo` | 默认 false。打开后才加载 YOLO 权重 |
| `liquid.backend` | 默认 `local`（弯月面）。改为 `qwen` 时改走视觉模型 |
| `server.capture` | 取图方式、分辨率、相机索引、RTSP、中心裁剪比例 |
| `gui` | 桌面界面的默认相机、分辨率，以及启动时是否自动打开摄像头 |

取图相关：

- `backend: usb`：OpenCV 打开摄像头；Linux 上自动使用 V4L2
- `backend: rtsp`：填写 `rtsp_url`
- `backend: plugin`：用自己的相机 SDK。实现 `grab_frame`，模块路径写在 `plugin_module`。参考 `examples/camera_plugin_stub.py`
- `roi_percent_x` / `roi_percent_y`：识别前只保留画面中心区域，减少手指和背景干扰

## 可选能力

这两项都不是默认路径，不装也不影响上面的快速开始。

**YOLO-World 兜底。** 在 `config.yaml` 里把 `yolo.fallback_yolo` 设为 true，并准备权重 `yolov8s-worldv2.pt`，然后安装：

```bash
pip install 'ultralytics>=8.3.0,<8.4'
```

开放类别名写在 `yolo.classes`（离心管、EP 管等）。OpenCV 没检出管子时才会调用它。

**千问视觉模型。** 把 `liquid.backend` 改为 `qwen`，并设置环境变量，不要把密钥写进 `config.yaml`：

```bash
export DASHSCOPE_API_KEY=your_key
```

模型、接口和提示词在 `config.yaml` 的 `vlm` 段，提示词文件是 `prompts/liquid_judge.txt`。视觉模型只估计液量百分比，不返回本地算法里的弯月面像素坐标。

## 目录

```text
ep-tube-meniscus/
├── run_server.py          # 启动 HTTP 服务
├── run_gui.py             # 桌面界面：摄像头、液面线、阈值
├── run_inspect.py         # 本机拍一张：JSON + 原图 + 标注图
├── run.py                 # 对图片或文件夹批量识别
├── config.yaml
├── requirements.txt
├── prompts/liquid_judge.txt
├── examples/
│   ├── robot_one_call.py      # 工位一键调用
│   └── camera_plugin_stub.py  # 自定义相机插件示例
└── src/
    ├── server.py              # FastAPI 路由
    ├── ep_tube_inspector.py   # 统一入口：EpTubeInspector / EpTubeHttpClient
    ├── detect_tube.py         # 管子定位（OpenCV，可选 YOLO）
    ├── judge_meniscus.py      # 本地弯月面与液量
    ├── judge_liquid.py        # 视觉模型液量判断
    ├── capture_frame.py       # USB / RTSP / 插件取图
    ├── pipeline.py            # 批量图片流水线
    ├── level_meter_gui.py     # 桌面界面
    ├── threshold_editor.py    # 阈值编辑
    └── color_threshold.py     # 颜色阈值
```
