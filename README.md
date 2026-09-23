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

默认监听 `0.0.0.0:8000`。

- 健康检查：`GET http://127.0.0.1:8000/health`
- 识别：`POST http://127.0.0.1:8000/v1/level`，表单字段名 `image`

响应是 JSON。图在 `image_base64`（JPEG）。主要字段：

| 字段 | 含义 |
| --- | --- |
| `ok` | 检出了管，并且有液面 |
| `fill_percent` | 液量百分比 |
| `volume_ml` | 按 1.5 mL 换算 |
| `allow_grasp` | 达到阈值且液面可靠时为 true |
| `meniscus_y` / `tube_top` / `tube_bottom` | 液面和管口、管底的像素位置 |
| `bbox` | 管在原图上的框 |
| `error` | 没检出时的说明，此时不要直接夹 |

```python
import base64
import requests

with open("照片test/少.jpg", "rb") as f:
    resp = requests.post("http://127.0.0.1:8000/v1/level", files={"image": f})
data = resp.json()
open("result.jpg", "wb").write(base64.b64decode(data["image_base64"]))
print(data["fill_percent"], data["volume_ml"], data["allow_grasp"])
```

同一时间处理一张图。`allow_grasp` 为 false 或 `ok` 为 false 时，调用方应停下来，不要只看有没有返回图片。
