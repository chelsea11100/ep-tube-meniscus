# ep-tube-level

1.5 mL EP 离心管液面识别。输入一张背光照片，检出管子和液面，给出液量百分比、约多少毫升、能不能夹，并返回画好线的图。

默认液面用本地算法。`config.yaml` 里 `liquid.backend` 改成 `qwen` 时，改走千问视觉模型，需要环境变量 `DASHSCOPE_API_KEY`。

可夹阈值是 `min_fill_percent`（默认 10%）。标称容量按 1.5 mL 换算。

## 环境

Python 3.10 或 3.11。模型权重 `yolov8s-worldv2.pt` 放在项目根目录（不进 git）。

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
```

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

## 界面

```powershell
$env:PYTHONUTF8=1
.\.venv\Scripts\python.exe run_gui.py
```

可以打开摄像头或图片，调颜色阈值，看液面线。改完代码要关掉窗口再开，旧进程不会加载新代码。

## HTTP 服务

给外面的系统调用。启动一次，检测器常驻。

```powershell
$env:PYTHONUTF8=1
.\.venv\Scripts\python.exe run_server.py
```

默认监听 `0.0.0.0:8000`。改代码后要先关掉旧进程再启动。

### 地址一览

| 地址 | 用途 |
| --- | --- |
| http://127.0.0.1:8000/ui | **浏览器测试页**（人工测试，推荐） |
| http://127.0.0.1:8000/docs | 接口文档（Swagger，给开发看） |
| http://127.0.0.1:8000/health | 健康检查 |
| http://127.0.0.1:8000/ | 接口说明（JSON） |

根路径 `/` 和 `/v1/level` 在浏览器里直接打开可能显示 Not Found，这是正常的：识别接口是 **POST 上传图片**，不是网页。

### 浏览器测试页 `/ui`

适合本机试效果，**不用在 /docs 里填任何文字**。

1. 打开 http://127.0.0.1:8000/ui
2. **单张**：选一张图 → 点「识别」
3. **批量**：
   - **选择多张**：手动挑若干 jpg/png
   - **选择文件夹**：点选整个文件夹（例如 `照片test`），会自动识别里面全部 jpg/png
   - 再点「批量识别」
4. 页面下方会显示 JSON 和画好线的结果图

`/docs` 里的批量接口在 Swagger 里可能显示成文本框，**不要在那里填路径**；批量请用 `/ui` 选文件夹，或用下面代码 POST。

### API 接口（给外部程序对接）

- 单张：`POST /v1/level`，表单字段 **`image`**（一个文件）
- 批量：`POST /v1/level/batch`，表单字段 **`images`**（同一字段名重复传多个文件）

响应是 JSON。单张直接在根上；批量在 `results` 数组里，每条结构和单张相同。图在 `image_base64`（JPEG）。主要字段：

| 字段 | 含义 |
| --- | --- |
| `ok` | 检出了管，并且有液面 |
| `fill_percent` | 液量百分比 |
| `volume_ml` | 按 1.5 mL 换算 |
| `allow_grasp` | 达到阈值且液面可靠时为 true |
| `meniscus_y` / `tube_top` / `tube_bottom` | 液面和管口、管底的像素位置 |
| `bbox` | 管在原图上的框 |
| `error` | 没检出时的说明，此时不要直接夹 |

批量额外字段：`count`（总数）、`ok_count`（检出液面的张数）、`results`（每张的结果，含 `filename`）。

单张调用示例：

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

同一时间只处理一个请求（单张或批量）。`allow_grasp` 为 false 或 `ok` 为 false 时，调用方应停下来，不要只看有没有返回图片。
