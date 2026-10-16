# 运行说明

程序读取视频，检测车辆和行人并识别属性，输出窗口画面及 `http://设备IP:8080/video_feed`。不保存视频文件。

## 环境

- Linux Rockchip 设备及匹配的 NPU 驱动、RKNN Toolkit Lite2 运行库。
- 当前程序使用 NPU CORE 0/1/2，要求设备支持这些核心；模型与芯片的兼容性需在设备上验证。
- Python、NumPy、OpenCV；显示窗口需要图形桌面，无桌面时使用 `--headless`。
- 三个模型放在脚本同目录，提供可解码的视频文件。

普通 Windows 环境可以做语法、参数和模拟测试，不能据此确认 RKNN 推理成功。请按板卡供应商的兼容性说明安装驱动及运行库。

## 命令

```sh
python det.py --help
python det.py /path/to/video.mp4 --check
python det.py /path/to/video.mp4
python det.py /path/to/video.mp4 --headless
```

不指定视频时，默认读取脚本旁的 `video1.mp4`。`--check` 仅检查文件与依赖导入，不初始化 NPU，不检查视频解码能力。实际运行才会验证视频打开和模型初始化。按 q 或 Ctrl+C 退出；视频结束后服务关闭。

类别表采用标准 COCO 80 类，必须与检测模型训练时的类别顺序一致。行人和车辆属性标签、预处理也必须与原始模型导出配置核对；当前目录没有提供这些训练/转换配置。

## 本地回归检查

```sh
python -m unittest -v test_runtime
```

测试使用模拟 RKNN 对象验证失败释放、线程异常和退出逻辑，不衡量检测准确率或硬件兼容性。
