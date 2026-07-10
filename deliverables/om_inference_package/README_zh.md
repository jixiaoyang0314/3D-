# OM 推理包说明

这个包包含当前 `weights/best.pt` 对应转换出的 Ascend OM 模型，以及配套 YOLO 解码/NMS 推理代码。

## 文件

- `weights/best.om`：Ascend OM 模型，转换目标为 `Ascend310B4`
- `infer_om.py`：pyACL 推理脚本
- `run_infer.sh`：启动脚本，会自动加载 CANN 环境
- `labels.txt`：类别编号到比赛物品编号的映射
- `class_mapping.json`：JSON 格式类别映射
- `model_info.json`：模型输入输出信息

## 输入输出

- 输入：`images [1,3,640,640]`
- 预处理：letterbox 到 640，BGR 转 RGB，float32，除以 255
- 输出：`output0 [1,15,8400]`
- 15 = 4 个框坐标 + 11 个类别分数

## 运行

在香橙派 AI Pro / Ascend 环境中：

```bash
cd om_inference_package
chmod +x run_infer.sh start.sh
./run_infer.sh --source /path/to/image_or_dir --conf 0.25 --iou 0.60 --save-vis
```

输出目录：

- `runs_om/predictions.json`
- `runs_om/labels/*.txt`
- `runs_om/vis/*`，只有加 `--save-vis` 时生成

如果 CANN 不在默认路径：

```bash
CANN_TOOLKIT_ROOT=/path/to/Ascend/ascend-toolkit ./run_infer.sh --source /path/to/images
```

如果板子上的 Python 命令不是 `python3`：

```bash
PYTHON_BIN=/path/to/python ./run_infer.sh --source /path/to/images
```

缺 OpenCV 时：

```bash
python3 -m pip install -r requirements.txt
```

`acl` 模块来自 CANN/pyACL，不是 pip 包。
