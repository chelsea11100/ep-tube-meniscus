# ep-tube-level

1.5 mL EP 离心管液面识别。输入一张背光照片，检出管子和液面，给出液量百分比、约多少毫升、能不能夹，并返回画好线的图。

默认液面用本地算法。`config.yaml` 里 `liquid.backend` 改成 `qwen` 时，改走千问视觉模型，需要环境变量 `DASHSCOPE_API_KEY`。

可夹阈值是 `min_fill_percent`（默认 10%）。标称容量按 1.5 mL 换算。

**新人本地试效果**：见 **[本地测试](#本地测试给他人上手)**；**机器人/产线对接**：见 **[调用方式](#调用方式机器人--工位集成)**。

## 从 GitHub 拉下来怎么用

仓库地址（示例）：`https://github.com/chelsea11100/ep-tube-meniscus.git`

```powershell
git clone https://github.com/chelsea11100/ep-tube-meniscus.git
cd ep-tube-meniscus
# 然后按下面「环境」建 venv，再按「本地测试」跑通
```

**两种常见角色**

| 角色 | 要做什么 | 看 README 哪一节 |
| --- | --- | --- |
| 工位/视觉同事 | clone → 装依赖 → 改 `config.yaml` → `run_inspect.py` 或 `run_server.py` | [环境](#环境)、[本地测试](#本地测试给他人上手) |
| 机器人/上位机同事 | **不必**会拍照；工位机已跑 `run_server.py` 时，只连 HTTP 或 `EpTubeHttpClient` | [调用方式](#调用方式机器人--工位集成)、[HTTP 服务](#http-服务) |

机器人侧最小对接：对工位 IP 发 **`POST http://<工位IP>:8000/v1/snapshot?include_image=false`**（或 `/v1/level` 上传 JPEG），读 JSON 里的 **`allow_grasp`**。

**Git 里 intentionally 不包含**（拉代码后需自行准备）：

| 不包含 | 原因 |
| --- | --- |
| `.venv/` | 每人本机 `python -m venv .venv` + `pip install -r requirements.txt` |
| `output/` | 运行结果，本地生成 |
| `.env`、API Key | 安全；用千问时才设 `DASHSCOPE_API_KEY` |
| `视觉摄像机/*.exe` | MVS 安装包太大；按 [视觉摄像机/README.md](视觉摄像机/README.md) 本机安装 |
| `*.pt` YOLO 权重 | 默认 OpenCV 检管不需要；开 YOLO 时自行下载 |

**拉代码后必改（若用海康 GigE）**：`config.yaml` 里 `mvs_development_dir`、`mvs_runtime_dir` 改成 **各自电脑** 的路径（仓库里可能是别人的 D 盘示例）。相机参数可用自带的 `config/mv_cu120_features.mfs`，也可在 MVS 里另存覆盖。

## 调用方式（机器人 / 工位集成）

对外统一返回 **dict / JSON**，字段与 HTTP 一致。机器人决策看 **`allow_grasp`**（建议再同时看 **`ok`**）；为 false 时读 **`error`**，不要直接夹取。

### 先选一种模式

| 场景 | 推荐做法 |
| --- | --- |
| 相机和识别都在 **同一台 Python 进程**（工位 PC、Windows + 海康 GigE） | **`EpTubeInspector`** |
| 识别服务 **常驻一台机器**，机器人/PLC 所在机 **只连网** | 工位上 **`run_server.py`**，调用方用 **`EpTubeHttpClient`** 或裸 HTTP |
| 相机在 **机器人侧**，识别在 **远端** | 机器人 **`POST /v1/level`** 上传 JPEG，或 **`EpTubeHttpClient.inspect_file`** |
| 不写代码、命令行联调 | **`run_inspect.py`**（本机 GigE 一键）或 **`examples/robot_one_call.py`** |

取图方式在 `config.yaml` 的 **`server.capture`**（`usb` / `v4l2` / `rtsp` / **`plugin`**）。当前工位海康 GigE 为 **`plugin`** + `examples.hikvision_gige`；**MVS 取图仅支持 Windows**，Linux 上请用上传图或 RTSP/USB。

### 模式 A：本机类 `EpTubeInspector`

识别逻辑与 `run_inspect.py` / HTTP 相同，**不经过网络**。相机按 `config.yaml` 里 `server.capture` 配置。

```python
from src.ep_tube_inspector import EpTubeInspector

inspector = EpTubeInspector()  # 默认读项目根 config.yaml
# inspector = EpTubeInspector("D:/path/to/config.yaml", verbose=False)

# 工位最常用：拍照 → 识别 → dict（字段同 inspect_result.json）
result = inspector.capture_and_inspect(include_image=False)

if result.get("allow_grasp"):
    # 允许夹取：fill_percent / volume_ml 可用
    pass
else:
    # result["ok"] 可能为 false；看 result["error"]
    pass

# 其它入口
result = inspector.inspect("照片test/某张.jpg")
result = inspector.inspect_bytes(jpeg_bytes, filename="frame.jpg")
result = inspector.snapshot(include_image=True)  # 与 capture_and_inspect 相同
cameras = inspector.list_cameras()  # USB 扫描；GigE plugin 时以 config 为准
```

| 方法 | 作用 |
| --- | --- |
| `capture_and_inspect()` / `snapshot()` | 按 config 取 **1 帧** → 识别 |
| `inspect(path)` / `inspect(ndarray)` | 已有图片路径或 BGR 数组 |
| `inspect_bytes(bytes)` | 内存里的 jpg/png |
| `list_cameras()` | 列出 USB 设备（OpenCV 索引） |

### 模式 B：远程类 `EpTubeHttpClient`

先在 **接相机或能取图的那台机** 启动服务（局域网可访问）：

```powershell
$env:PYTHONUTF8=1
.\.venv\Scripts\python.exe run_server.py
# 可选：.\.venv\Scripts\python.exe run_server.py --host 0.0.0.0 --port 8000
```

调用方（可以是 Linux 机器人控制器，只要网络通）：

```python
from src.ep_tube_inspector import EpTubeHttpClient

client = EpTubeHttpClient("http://192.168.1.10:8000", timeout=120.0)

# 服务端取图 + 识别（config 里 backend 为 plugin/usb/rtsp 等）
result = client.capture_and_inspect(include_image=False)

# 调用方自己拍照，只让服务端识别
result = client.inspect_file("frame.jpg", include_image=False)

devices = client.list_cameras()
```

### 模式 C：命令行（不写 Python）

工位 **GigE 一键**、输出文件说明、MVS 注意项见上文 **[本地测试 → 工位相机](#2-工位相机海康-gige--msvswindows推荐与产线一致)**。等价命令：

```powershell
.\.venv\Scripts\python.exe run_inspect.py -o output/inspect_result.json
```

**本机类 / 远程 HTTP 示例脚本**：

```powershell
# 本机 EpTubeInspector（读 config 取图）
.\.venv\Scripts\python.exe examples/robot_one_call.py --local --no-image -o output/result.json

# 远程 HTTP（改 --host 为 run_server 所在 IP）
.\.venv\Scripts\python.exe examples/robot_one_call.py --host 192.168.1.10 --no-image
```

### 模式 D：裸 HTTP（任意语言）

| 方法 | URL | 说明 |
| --- | --- | --- |
| GET | `/health` | 服务是否存活 |
| GET | `/v1/cameras` | USB 设备列表 |
| POST | `/v1/inspect` 或 `/v1/snapshot` | 服务端取 1 帧并识别；查询参数 `include_image=false`、`camera_index=1` |
| POST | `/v1/level` | 表单字段 **`image`**，上传一张 jpg/png |
| POST | `/v1/level/batch` | 表单字段 **`images`**（同名多文件） |

人工试接口可用 Swagger：`http://<服务IP>:8000/docs`。

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
| `image_base64` | 标注图 JPEG（`include_image=false` 时可省略） |

批量 HTTP 额外：`count`、`ok_count`、`results`（数组，每项结构同上）。

### 接到自己的 Python 工程

项目未打包成 pip 包时，任选其一：

1. 在机器人程序里把 **本项目根目录** 加入 `sys.path`（见 `examples/robot_one_call.py`）。
2. 设置环境变量 **`PYTHONPATH=项目根目录`** 后 `from src.ep_tube_inspector import EpTubeInspector`。
3. 不嵌代码：只调 **`run_server.py`**，用 HTTP 或 **`EpTubeHttpClient`**。

同一时间服务只处理 **一个** 识别请求（单张或批量）；高并发需自行排队。

## 环境

Python 3.10 或 3.11。默认用 OpenCV 检管，**不需要** YOLO 权重。只有 `config.yaml` 里打开 `yolo.fallback_yolo: true` 或 `backend: yolo` 时，才需在根目录放置 `yolov8s-worldv2.pt`（不进 git）。

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
```

说明：

- 请始终用 **`.venv\Scripts\python.exe`** 跑脚本。系统自带的 `python` 常缺依赖或版本不对。
- PowerShell 若无法执行 `Activate.ps1`，**不必激活 venv**，直接写全路径即可。
- 建议设 **`$env:PYTHONUTF8=1`**，避免中文路径/日志乱码。

Linux 示例：

```bash
python3 -m venv .venv
.venv/bin/pip install -r requirements.txt
export PYTHONUTF8=1
```

## 本地测试（给他人上手）

按下面顺序测即可；**集成与机器人对接**见下一节「调用方式」。

### 1. 没有相机：用已有照片

适合先确认算法与依赖是否正常。把 jpg/png 放进 `照片test/`（或任意路径）。

```powershell
cd "项目根目录"
$env:PYTHONUTF8=1
.\.venv\Scripts\python.exe run.py 照片test
```

看 `output/某图名/result.json` 里的 **`fill_percent`**、**`allow_grasp`**，以及 `level.jpg` 液面线是否合理。

单张文件：

```powershell
.\.venv\Scripts\python.exe run.py "照片test\某张.jpg"
```

### 2. 工位相机：海康 GigE + MVS（Windows，推荐与产线一致）

与 **`run_inspect.py`** 相同：**拍 1 帧 → 识别 → JSON + 两张 JPG**。当前仓库默认 `config.yaml` 里 `server.capture.backend: plugin`、`plugin_module: examples.hikvision_gige`。

**测前准备（一次性）**

1. 安装 [海康 MVS](视觉摄像机/README.md)（Runtime + Development；Development 里要有 `Samples/Python/MvImport`）。
2. 相机网口与工控机 **同一网段**（例如相机 `192.168.3.253`、电脑 `192.168.3.4`），MVS Client 里能 **单帧采图**。
3. 编辑 **`config.yaml`** → `server.capture`，按本机改路径（示例）：
   - `mvs_runtime_dir`：如 `C:\Program Files (x86)\Common Files\MVS\Runtime\Win64_x64`
   - `mvs_development_dir`：本机 MVS Development 目录
   - `mvs_feature_file`：如 `config/mv_cu120_features.mfs`（曝光等；可在 MVS 里 FeatureSave 后覆盖）
   - `roi_percent_x` / `roi_percent_y`：裁中心区域，减少手指和背景
4. **关闭 MVS Client**（否则会占用相机）。

**每次测试命令**

```powershell
cd "项目根目录"
$env:Path = "C:\Program Files (x86)\Common Files\MVS\Runtime\Win64_x64;" + $env:Path
$env:PYTHONUTF8=1
.\.venv\Scripts\python.exe run_inspect.py -o output/inspect_result.json
```

**输出文件**

| 文件 | 内容 |
| --- | --- |
| `output/inspect_result.json` | 完整结果；**前几行**看 **`allow_grasp`**、**`ok`**、**`fill_percent`**、**`error`** |
| `output/inspect_capture.jpg` | 送入识别的图（含 ROI 裁剪后） |
| `output/inspect_annotated.jpg` | 标注液面/管子的图 |

终端会打印一份 **不含超长 base64** 的 JSON 摘要；完整字段以 **`inspect_result.json`** 为准。

可选：只测能否取图（不跑识别）：

```powershell
$env:Path = "C:\Program Files (x86)\Common Files\MVS\Runtime\Win64_x64;" + $env:Path
.\.venv\Scripts\python.exe run_mvs_grab.py
# 默认写出 output/mvs_test.jpg
```

**怎么算测通**

- `inspect_result.json` 里 **`ok: true`** 且 **`allow_grasp: true/false`** 有明确布尔值（按液量与检出情况）。
- `class_name` 为 `tube_opencv` 等（不是整图 fallback 时更稳）。
- 打开 `inspect_annotated.jpg`，液面线与肉眼大致一致。

若 **`ok: false`** 或 **`allow_grasp: false`**：先看 `error` 和 `inspect_capture.jpg`，再调 ROI、背光、固定管子、或重新导出 `mfs`。

### 3. USB 相机（无 MVS）

在 `config.yaml` 把 `server.capture.backend` 改为 **`usb`**，设好 `camera_index`（外接 USB 常见 **1 或 2**），然后：

```powershell
$env:PYTHONUTF8=1
.\.venv\Scripts\python.exe run_inspect.py -o output/inspect_result.json
```

或：

```powershell
.\.venv\Scripts\python.exe examples/robot_one_call.py --local --no-image -o output/result.json
```

Linux 上可把 `backend` 设为 **`v4l2`**（逻辑同 USB）。

### 4. 本机 HTTP 服务（模拟机器人连网）

**终端 1** — 启动服务：

```powershell
cd "项目根目录"
$env:Path = "C:\Program Files (x86)\Common Files\MVS\Runtime\Win64_x64;" + $env:Path   # 若 backend 为 plugin 时需要
$env:PYTHONUTF8=1
.\.venv\Scripts\python.exe run_server.py
```

**终端 2** — 健康检查与一键识别：

```powershell
# 需已 pip install requests（requirements 里若有则跳过）
curl http://127.0.0.1:8000/health

.\.venv\Scripts\python.exe examples/robot_one_call.py --host 127.0.0.1 --no-image -o output/api_result.json
```

没有 GigE、只有照片时，可只测上传识别（不必改 backend）：

```powershell
.\.venv\Scripts\python.exe -c "import requests; r=requests.post('http://127.0.0.1:8000/v1/level', files={'image': open('照片test/某张.jpg','rb')}); print(r.json()['allow_grasp'], r.json()['fill_percent'])"
```

浏览器可看 Swagger：`http://127.0.0.1:8000/docs`（POST 试 `/v1/level` 或 `/v1/snapshot`）。

### 5. 常见问题（本地测）

| 现象 | 处理 |
| --- | --- |
| 找不到 `cv2` / 脚本无输出 | 用 `.venv\Scripts\python.exe`，不要用系统 `python` |
| MVS 报错 / 黑图 | 关 MVS Client；确认 PATH 含 Runtime；加大 `mvs_warmup_frames`；确认已 Load `mvs_feature_file` |
| JSON 里找不到能不能夹 | 打开 **`inspect_result.json` 文件开头**，字段名是 **`allow_grasp`**（不是中文） |
| 中文路径写 jpg 失败 | 本项目已用 `imencode` 写文件；仍失败则检查目录权限 |
| 服务端口被占用 | 换端口：`run_server.py --port 8001`，客户端 URL 一并修改 |

## 批量测试

照片放进 `照片test/`。不带参数会测这个文件夹里的全部 jpg/png。

```powershell
$env:PYTHONUTF8=1
.\.venv\Scripts\python.exe run.py
.\.venv\Scripts\python.exe run.py 照片test
.\.venv\Scripts\python.exe run.py a.jpg b.png
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

给外面的系统调用。启动一次，检测器常驻。**集成步骤见上文「调用方式」**；本节是服务地址与补充示例。

```powershell
$env:PYTHONUTF8=1
.\.venv\Scripts\python.exe run_server.py
```

默认监听 `0.0.0.0:8000`。改代码后要先关掉旧进程再启动。

### 地址一览

| 地址 | 用途 |
| --- | --- |
| http://127.0.0.1:8000/docs | 接口文档（Swagger，可选人工试 POST） |
| http://127.0.0.1:8000/health | 健康检查 |
| http://127.0.0.1:8000/ | 接口说明（JSON） |

根路径 `/` 和 `/v1/level` 在浏览器里直接打开可能显示 Not Found，这是正常的：识别接口是 **POST 上传图片**，不是网页。

### API 接口（给外部程序对接）

- 单张：`POST /v1/level`，表单字段 **`image`**（一个文件）
- 批量：`POST /v1/level/batch`，表单字段 **`images`**（同一字段名重复传多个文件）
- **一键取图 + 识别（推荐）**：`POST /v1/inspect` 或 `/v1/snapshot` — 抓 **1 帧** → 识别 → JSON。可选 `?camera_index=1`、`?include_image=false`。`server.capture.backend` 支持 `usb` / `v4l2` / `rtsp` / **`plugin`**（SDK 写在 `examples/camera_plugin_stub.py`）。
- 兼容旧名：`POST /v1/level/capture`（行为与 `/v1/snapshot` 相同）。
- 摄像头列表：`GET /v1/cameras`
- 示例脚本：`examples/robot_one_call.py`（`--local` / `--host`）

机器人相机若接在 **机器人控制器** 上、不在本机 USB，仍用 `POST /v1/level` 上传 JPEG；若相机 RTSP 网口对本机可达，可设 `server.capture.backend: rtsp` 后仍调 `/v1/snapshot`。

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
