# 视觉摄像机（海康 MVS）

本目录用于存放 **海康机器人 MVS** 客户端安装包，供工业相机（GigE / USB3 Vision）驱动与取图。

## 文件

| 文件 | 说明 |
| --- | --- |
| `MVS_STD_5.1.0_260909.exe` | MVS 标准安装程序（体积大，**不要提交 Git**） |

## 安装（在本机或工控机执行一次）

1. 双击运行 `MVS_STD_5.1.0_260909.exe`，按向导安装 MVS 与相机驱动。
2. 相机接好线（USB 或网口），用 MVS 自带 **MVS Client** 确认能 **单帧采图**。
3. 关闭 MVS Client 后再跑本项目的 `run_inspect.py` / `run_server.py`，避免独占相机。

## 和 ep-tube 项目怎么接

**情况 A — 装驱动后，Windows 里出现普通 USB 相机**

- 在本项目里：`config.yaml` 设 `server.capture.backend: usb` 与 `camera_index`，再 `POST /v1/snapshot` 或 `EpTubeInspector().capture_and_inspect()`。
- 无需改代码。

**情况 B — 只有 MVS / GigE SDK 能取图（UVC 列表里没有）**

- 在 **`examples/camera_plugin_stub.py`** 的 `grab_frame()` 里接入海康/Linux SDK（返回 BGR `numpy`）。
- `config.yaml` 设 `backend: plugin`、`plugin_module: examples.camera_plugin_stub`。
- 然后 **一次调用** 即可：`POST /v1/inspect` 或 `EpTubeInspector().capture_and_inspect()`（内部取图→识别→JSON）。
- **不必装 Windows exe 才能跑 Linux 服务**；exe 仅 Windows 调试用，Linux 用 Linux 版 SDK + `plugin`。

## 机器人工位建议

```text
机器人移到拍照位 → 停稳 → 取 1 帧（MVS 或 USB）
→ EpTubeInspector.snapshot() / inspect() 或 HTTP /v1/snapshot
→ 根据 allow_grasp 决定是否夹取
```
