# ep-tube-level

1.5 mL EP 离心管液面识别。输入一张背光照片，检出管子和液面，给出液量百分比、约多少毫升、能不能夹，并返回画好线的图。

默认液面用本地算法。`config.yaml` 里 `liquid.backend` 改成 `qwen` 时，改走千问视觉模型，需要环境变量 `DASHSCOPE_API_KEY`。

可夹阈值是 `min_fill_percent`（默认 10%）。标称容量按 1.5 mL 换算。

以 **Linux** 为主要运行环境；Windows 也能跑，命令里的 `.venv/bin/python` 换成 `.venv\Scripts\python.exe` 即可。

**新人本地试效果**：见 **[本地测试](#本地测试)**；**机器人/产线对接**：见 **[调用方式](#调用方式机器人--工位集成)**。

## 从 GitHub 拉下来怎么用

```bash
git clone https://github.com/chelsea11100/ep-tube-meniscus.git
cd ep-tube-meniscus
# 然后按下面「环境」建 venv，再按「本地测试」跑通
```

**两种常见角色**

| 角色 | 要做什么 | 看 README 哪一节 |
| --- | --- | --- |
| 工位/视觉同事 | clone → 装依赖 → 接摄像头 → `run_inspect.py` 或 `run_server.py` | [环境](#环境)、[本地测试](#本地测试) |
| 机器人/上位机同事 | **不必**管相机；工位机已跑 `run_server.py` 时，只连 HTTP 或用 `EpTubeHttpClient` | [调用方式](#调用方式机器人--工位集成)、[HTTP 服务](#http-服务) |

机器人侧最小对接：对工位 IP 发 **`POST http://<工位IP>:8000/v1/snapshot?include_image=false`**（或 `/v1/level` 上传 JPEG），读 JSON 里的 **`allow_grasp`**。

**Git 里不包含**（拉代码后需自行准备）：

| 不包含 | 原因 |
| --- | --- |
| `.venv/` | 每人本机建 venv 并 `pip install -r requirements.txt` |
| `output/` | 运行结果，本地生成 |
| `.env`、API Key | 安全；用千问时才设 `DASHSCOPE_API_KEY` |
| `*.pt` YOLO 权重 | 默认 OpenCV 检管不需要；开 YOLO 时自行下载 |

## 环境

Python 3.10 或 3.11。

```bash
python3 -m venv .venv
.venv/bin/pip install -r requirements.txt
export PYTHONUTF8=1
```

说明：

- 始终用 **`.venv/bin/python`** 跑脚本，不用激活 venv。系统自带的 `python3` 通常没装这些依赖。
- **没有显示器的服务器**上，`import cv2` 若报 `libGL.so.1` 找不到：`sudo apt install libgl1`。
- **标注图上的中文**需要系统有中文字体，否则会显示成方框（只影响图上的文字，不影响识别和 JSON）：`sudo apt install fonts-noto-cjk`。
- **摄像头权限**：当前用户要能读 `/dev/video*`，不行就 `sudo usermod -aG video $USER` 后重新登录。
- 默认用 OpenCV 检管，**不需要** YOLO 权重。只有 `config.yaml` 里打开 `yolo.fallback_yolo: true` 或 `backend: yolo` 时，才需在根目录放置 `yolov8s-worldv2.pt`。`requirements.txt` 里的 `ultralytics` 会连带装 PyTorch，体积较大。

## 本地测试

按下面顺序测即可；**集成与机器人对接**见后面「调用方式」。

### 1. 没有相机：用已有照片

适合先确认算法与依赖是否正常。把 jpg/png 放进 `照片test/`（或任意路径）。

```bash
.venv/bin/python run.py 照片test
.venv/bin/python run.py 照片test/某张.jpg   # 单张
```

看 `output/某图名/result.json` 里的 **`fill_percent`**、**`allow_grasp`**，以及 `level.jpg` 液面线是否合理。

### 2. USB 摄像头：拍一帧 → 识别

普通免驱（UVC）USB 摄像头直接插上即可。先确认系统认到了：

```bash
ls /dev/video*          # 例如 /dev/video0、/dev/video2
```

拍照 + 识别（`--camera` 对应 `/dev/videoN` 的 N；省略则自动选外接 USB）：

```bash
.venv/bin/python run_inspect.py --camera 0 -o output/inspect_result.json
```

**输出文件**

| 文件 | 内容 |
| --- | --- |
| `output/inspect_result.json` | 完整结果；**前几行**看 **`allow_grasp`**、**`ok`**、**`fill_percent`**、**`error`** |
| `output/inspect_capture.jpg` | 送入识别的图（含 ROI 裁剪后） |
| `output/inspect_annotated.jpg` | 标注液面/管子的图 |

终端会打印一份 **不含超长 base64** 的 JSON 摘要；完整字段以 **`inspect_result.json`** 为准。

**怎么算测通**

- `inspect_result.json` 里 **`ok: true`**，`allow_grasp` 按液量给出 true/false。
- `class_name` 为 `tube_opencv` 等（不是整图 fallback 时更稳）。
- 打开 `inspect_annotated.jpg`，液面线与肉眼大致一致。

若 **`ok: false`** 或 **`allow_grasp: false`**：先看 `error` 和 `inspect_capture.jpg`，再调背光、摄像头位置、`config.yaml` 里的 `roi_percent_x` / `roi_percent_y`（裁中心区域，减少手指和背景）。

**拍摄条件**：算法是按「背光 + 透明 EP 管竖直放在画面中间」调的。要摆好背光板、固定好摄像头，结果才稳定。认不出时 `allow_grasp` 会是 `false`，不会误夹。

### 3. 本机 HTTP 服务（模拟机器人连网）

终端 1，启动服务：

```bash
.venv/bin/python run_server.py
```

终端 2，健康检查与一键识别：

```bash
curl http://127.0.0.1:8000/health
curl -X POST "http://127.0.0.1:8000/v1/snapshot?include_image=false"

# 或用示例脚本
.venv/bin/python examples/robot_one_call.py --host 127.0.0.1 --no-image -o output/api_result.json
```

没有摄像头、只有照片时，测上传识别：

```bash
curl -F "image=@照片test/某张.jpg" "http://127.0.0.1:8000/v1/level"
```

浏览器可看 Swagger：`http://127.0.0.1:8000/docs`。

### 4. 常见问题

| 现象 | 处理 |
| --- | --- |
| 找不到 `cv2` / 脚本无输出 | 用 `.venv/bin/python`，不要用系统 `python3` |
| `libGL.so.1: cannot open shared object file` | `sudo apt install libgl1` |
| 未找到可用 USB 摄像头 | `ls /dev/video*` 看有没有设备；检查 video 组权限；换个 USB 口 |
| 一个摄像头出现两个 `/dev/video*` | 正常，通常编号小的那个能出图；用 `--camera` 逐个试 |
| JSON 里找不到能不能夹 | 打开 **`inspect_result.json` 文件开头**，字段名是 **`allow_grasp`** |
| 服务端口被占用 | 换端口：`run_server.py --port 8001`，客户端 URL 一并修改 |

## 调用方式（机器人 / 工位集成）

对外统一返回 **dict / JSON**，字段与 HTTP 一致。机器人决策看 **`allow_grasp`**（建议再同时看 **`ok`**）；为 false 时读 **`error`**，不要直接夹取。

### 先选一种模式

| 场景 | 推荐做法 |
| --- | --- |
| 摄像头和识别在 **同一台机器、同一个 Python 程序** | **`EpTubeInspector`** |
| 识别服务 **常驻工位机**，机器人所在机 **只连网** | 工位上 **`run_server.py`**，调用方用 **`EpTubeHttpClient`** 或裸 HTTP |
| 相机在 **机器人侧**，识别在 **远端** | 机器人 **`POST /v1/level`** 上传 JPEG，或 **`EpTubeHttpClient.inspect_file`** |
| 不写代码、命令行联调 | **`run_inspect.py`** 或 **`examples/robot_one_call.py`** |

### 取图方式

在 `config.yaml` 的 `server.capture.backend` 里设置，每次调用也能临时覆盖（HTTP 加 `?source=...`，类方法传 `source=...`，命令行加 `--source ...`）。

| `backend` / `source` | 说明 |
| --- | --- |
| `usb`（默认） | 普通 USB 摄像头，OpenCV 打开；Linux 上自动用 V4L2 |
| `v4l2` | 强制用 V4L2 打开 USB 摄像头（仅 Linux） |
| `rtsp` | 网络摄像头，需配 `rtsp_url` |
| `plugin` | 工业相机等厂商 SDK，自己写取图模块，模板见 `examples/camera_plugin_stub.py` |

返回 JSON 里的 `source`、`camera_index`、`camera_name` 表示实际用了哪个摄像头。

### 模式 A：本机类 `EpTubeInspector`

识别逻辑与 `run_inspect.py` / HTTP 相同，**不经过网络**。

```python
from src.ep_tube_inspector import EpTubeInspector

inspector = EpTubeInspector()  # 默认读项目根 config.yaml
# inspector = EpTubeInspector("/path/to/config.yaml", verbose=False)

# 拍照 → 识别 → dict（字段同 inspect_result.json）
result = inspector.capture_and_inspect(include_image=False)

if result.get("allow_grasp"):
    # 允许夹取：fill_percent / volume_ml 可用
    pass
else:
    # result["ok"] 可能为 false；看 result["error"]
    pass

# 其它入口
result = inspector.capture_and_inspect(camera_index=2)            # 指定 /dev/video2
result = inspector.inspect("照片test/某张.jpg")
result = inspector.inspect_bytes(jpeg_bytes, filename="frame.jpg")
cameras = inspector.list_cameras()
```

| 方法 | 作用 |
| --- | --- |
| `capture_and_inspect()` / `snapshot()` | 按 config 取 **1 帧** → 识别 |
| `inspect(path)` / `inspect(ndarray)` | 已有图片路径或 BGR 数组 |
| `inspect_bytes(bytes)` | 内存里的 jpg/png |
| `list_cameras()` | 列出本机摄像头 |

注意：`config.yaml` 里的 ROI 裁剪（`roi_percent_x/y`）目前只有 `run_inspect.py` 会用，类方法和 HTTP 拍照是整帧识别。

### 模式 B：远程类 `EpTubeHttpClient`

先在 **接摄像头的那台机** 启动服务（默认监听 `0.0.0.0:8000`，局域网可访问）：

```bash
.venv/bin/python run_server.py
# 可选：.venv/bin/python run_server.py --host 0.0.0.0 --port 8000
```

调用方：

```python
from src.ep_tube_inspector import EpTubeHttpClient

client = EpTubeHttpClient("http://192.168.1.10:8000", timeout=120.0)

# 服务端拍照 + 识别
result = client.capture_and_inspect(include_image=False)
result = client.capture_and_inspect(camera_index=2, include_image=False)

# 调用方自己拍照，只让服务端识别
result = client.inspect_file("frame.jpg", include_image=False)

devices = client.list_cameras()
```

### 模式 C：命令行

```bash
# 拍照 + 识别，输出 JSON 和两张 jpg（见「本地测试」）
.venv/bin/python run_inspect.py -o output/inspect_result.json

# 本机 EpTubeInspector
.venv/bin/python examples/robot_one_call.py --local --no-image -o output/result.json

# 远程 HTTP（改 --host 为 run_server 所在 IP）
.venv/bin/python examples/robot_one_call.py --host 192.168.1.10 --no-image
```

### 模式 D：裸 HTTP（任意语言）

| 方法 | URL | 说明 |
| --- | --- | --- |
| GET | `/health` | 服务是否存活 |
| GET | `/v1/cameras` | 本机摄像头列表 |
| POST | `/v1/inspect` 或 `/v1/snapshot` | 服务端取 1 帧并识别；查询参数 `source`、`camera_index`、`include_image=false` |
| POST | `/v1/level` | 表单字段 **`image`**，上传一张 jpg/png |
| POST | `/v1/level/batch` | 表单字段 **`images`**（同名多文件） |

### 返回字段（类与 HTTP 相同）

| 字段 | 含义 |
| --- | --- |
| `allow_grasp` | **是否建议夹取**（液量够 + 管检出 + 液面可靠） |
| `ok` | 管与液面均可靠时为 true |
| `fill_percent` | 液量百分比 |
| `volume_ml` | 按 1.5 mL 标称换算 |
| `min_fill_percent` | 配置阈值（默认 10） |
| `error` | 不可信或不可夹时的说明（字符串或 null） |
| `evidence` | 本地算法简要依据 |
| `bbox` / `meniscus_y` / `tube_top` / `tube_bottom` | 几何信息 |
| `source` / `camera_index` / `camera_name` | 实际用的摄像头（拍照接口才有） |
| `image_base64` | 标注图 JPEG（`include_image=false` 时为空） |

批量 HTTP 额外：`count`、`ok_count`、`results`（数组，每项结构同上）。

### 接到自己的 Python 工程

项目未打包成 pip 包时，任选其一：

1. 在机器人程序里把 **本项目根目录** 加入 `sys.path`（见 `examples/robot_one_call.py`）。
2. 设置环境变量 **`PYTHONPATH=项目根目录`** 后 `from src.ep_tube_inspector import EpTubeInspector`。
3. 不嵌代码：只调 **`run_server.py`**，用 HTTP 或 **`EpTubeHttpClient`**。

同一时间服务只处理 **一个** 识别请求（单张或批量）；高并发需自行排队。

## 批量测试

照片放进 `照片test/`。不带参数会测这个文件夹里的全部 jpg/png。

```bash
.venv/bin/python run.py
.venv/bin/python run.py 照片test
.venv/bin/python run.py a.jpg b.png
```

结果在 `output/`：

```text
output/
  summary.json
  某张图名/
    det.jpg
    crop.jpg
    level.jpg
    result.json
```

`summary.json` 是这一批的总表。每张图一个文件夹，互不覆盖别的图。

## HTTP 服务

给外面的系统调用。启动一次，检测器常驻。集成步骤见上文「调用方式」，本节是服务地址与补充示例。

```bash
.venv/bin/python run_server.py
```

默认监听 `0.0.0.0:8000`。改代码后要先关掉旧进程再启动。

| 地址 | 用途 |
| --- | --- |
| http://127.0.0.1:8000/docs | 接口文档（Swagger，可人工试 POST） |
| http://127.0.0.1:8000/health | 健康检查 |
| http://127.0.0.1:8000/ | 接口说明（JSON） |

根路径以外的识别接口都是 **POST**，浏览器直接打开 `/v1/level` 会显示 Method Not Allowed，这是正常的。兼容旧名：`POST /v1/level/capture`（行为与 `/v1/snapshot` 相同）。

机器人相机若接在 **机器人控制器** 上、不在本机，用 `POST /v1/level` 上传 JPEG；若相机 RTSP 网口对本机可达，可设 `server.capture.backend: rtsp` 后仍调 `/v1/snapshot`。

### requests 示例（上传图片）

```python
import base64
import requests

with open("照片test/少.jpg", "rb") as f:
    resp = requests.post("http://127.0.0.1:8000/v1/level", files={"image": f})
data = resp.json()
open("result.jpg", "wb").write(base64.b64decode(data["image_base64"]))
print(data["fill_percent"], data["volume_ml"], data["allow_grasp"])
```

批量调用示例（字段名都是 `images`）：

```python
import base64
import requests
from pathlib import Path

files = [
    ("images", (p.name, p.read_bytes(), "image/jpeg"))
    for p in Path("照片test").glob("*.jpg")
]
batch = requests.post("http://127.0.0.1:8000/v1/level/batch", files=files).json()
print(batch["count"], batch["ok_count"])
for item in batch["results"]:
    print(item["filename"], item["fill_percent"], item["allow_grasp"])
    if item["image_base64"]:
        out = Path("output") / f"{Path(item['filename']).stem}_api.jpg"
        out.parent.mkdir(exist_ok=True)
        out.write_bytes(base64.b64decode(item["image_base64"]))
```

`allow_grasp` 为 false 或 `ok` 为 false 时，调用方应停下来，不要只看有没有返回图片。
