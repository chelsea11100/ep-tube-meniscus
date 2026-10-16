# 克隆仓库

```bash
cd "/home/tangger/Desktop/视觉界面" && git clone https://github.com/chelsea11100/ep-tube-meniscus.git
```

# 部署环境

```bash
cd "/home/tangger/Desktop/视觉界面/ep-tube-meniscus"
python3 -m venv .venv
.venv/bin/pip install -r requirements.txt
```

# 进入虚拟环境（若没进）

```bash
cd "/home/tangger/Desktop/视觉界面/ep-tube-meniscus"
source .venv/bin/activate
```

# 接口连接（总共 6 个接口，只需要发起到 /v1/）

```bash
python run_server.py
```

## Swagger 接口测试

http://127.0.0.1:8000/docs

## 可看看有摄像头

```bash
curl http://127.0.0.1:8000/v1/cameras
```

## 主流程

```bash
curl -X POST "http://127.0.0.1:8000/v1/snapshot?source=usb&include_image=true"
```

### 若将来要用 YOLO，则再单独

```bash
pip install 'ultralytics>=8.3.0,<8.4'
```

# GUI 连接

桌面界面：打开图片 / USB 摄像头，调 ROI、看液面线、阈值编辑器。

```bash
# Linux
.venv/bin/python run_gui.py
.venv/bin/python run_gui.py --camera 0 --open-camera

# Windows
.\.venv\Scripts\python.exe run_gui.py
.\.venv\Scripts\python.exe run_gui.py --camera 1 --open-camera
```

1. 右侧点 **扫描**，下拉框选外接 USB（不要选 Integrated / 内置）
2. 点 **打开 USB 摄像头**
3. 勾选 **摄像头开启时自动分析跟线**，或点 **分析当前画面**

`config.yaml` 的 `gui.camera_index` / `prefer_usb` / `auto_open_camera` 可设默认设备。改完代码要关掉窗口再开。

# 本地测试，返回读取液面图片

```bash
python run_inspect.py
```

在 output 里面看照片
