# 2026 3D 识别训练计划

更新时间：2026-07-08

## 规则要点

根据《赛事规则.机器人先进视觉赛-3D识别.2026专项赛.pdf》，训练策略重点覆盖：

- 两轮比赛均有随机背景纸、贴图干扰、场地干扰物、遮挡和叠放。
- 第二轮有 3 张目标台，其中 2 张会分别使用不同特定光源：黄光和白光。
- 摄像头到目标台距离为 1.0m-1.8m，视角由裁判临场决定。
- 圆形目标台可能旋转，物体姿态不限。
- 评分按官方 ID 和数量计分，因此训练类别必须使用官方编号。

## 当前数据

正式源数据集：

```text
D:\机器人识别大赛\yolo_dataset_official\data.yaml
```

增强后的训练数据集：

```text
D:\机器人识别大赛\yolo_dataset_official_aug\data.yaml
```

增强集生成命令：

```powershell
python "D:\机器人识别大赛\robot_vision_pipeline\augment_official_dataset.py" `
  --data "D:\机器人识别大赛\yolo_dataset_official\data.yaml" `
  --output "D:\机器人识别大赛\yolo_dataset_official_aug" `
  --yellow 0.10 `
  --white 0.10 `
  --dark 0.05 `
  --balance-target max `
  --copy-originals reencode `
  --max-side 1920 `
  --quality 90 `
  --overwrite
```

生成结果：

| 项目 | 数量 |
| --- | ---: |
| 原训练图 | 1640 |
| 原验证图 | 410 |
| 黄光增强训练图 | 164 |
| 白光增强训练图 | 164 |
| 暗光增强训练图 | 82 |
| 类别均衡补样图 | 1784 |
| 最终训练图 | 3834 |
| 最终验证图 | 410 |

## 类别分布

增强前训练框数：

| 训练索引 | 官方 ID | 物品 | 框数 |
| ---: | --- | --- | ---: |
| 0 | CA002 | 耳机 | 327 |
| 1 | CA003 | 水杯 | 243 |
| 2 | CA004 | 衣架 | 219 |
| 3 | CB002 | 瓜子 | 279 |
| 4 | CB003 | 火腿肠 | 653 |
| 5 | CB004 | 薯片 | 764 |
| 6 | CC001 | 罐装饮料 | 359 |
| 7 | CC002 | 瓶装饮料 | 243 |
| 8 | CC003 | 盒装牛奶 | 338 |
| 9 | CC004 | 瓶装饮用水 | 867 |
| 10 | CD003 | 香蕉 | 628 |

增强后训练框数：

| 训练索引 | 官方 ID | 物品 | 框数 |
| ---: | --- | --- | ---: |
| 0 | CA002 | 耳机 | 1306 |
| 1 | CA003 | 水杯 | 1087 |
| 2 | CA004 | 衣架 | 1086 |
| 3 | CB002 | 瓜子 | 1111 |
| 4 | CB003 | 火腿肠 | 1709 |
| 5 | CB004 | 薯片 | 1415 |
| 6 | CC001 | 罐装饮料 | 1541 |
| 7 | CC002 | 瓶装饮料 | 1088 |
| 8 | CC003 | 盒装牛奶 | 1087 |
| 9 | CC004 | 瓶装饮用水 | 2026 |
| 10 | CD003 | 香蕉 | 1087 |

说明：由于一张图里可能同时有多个类别，给少样本类别补图时，会连带增加共现类别，所以增强后不是数学上完全相等。现在少样本类别都已补到约 1086 框以上，适合训练；若要进一步压平 CC004/CB003 等高频类，需要采集更独立的单类图或做实例级裁剪增强。

## YOLO 基础增强

`train_yolo.py` 默认启用以下 Ultralytics 增强：

- `mosaic=0.8`：模拟多物体、多背景、多尺度混合场景，贴近比赛的随机背景和密集摆放。
- `close_mosaic=25`：最后 25 个 epoch 关闭 mosaic，让模型收敛到真实图像分布。
- `mixup=0.08`：轻量混合，增加泛化，但不宜太高，避免包装文字和边界变糊。
- `copy_paste=0.15`：增强遮挡/叠放场景。
- `hsv_h=0.03, hsv_s=0.65, hsv_v=0.45`：覆盖现场光照和相机自动曝光波动。
- `degrees=25, translate=0.15, scale=0.55, shear=4, perspective=0.0008`：覆盖姿态、距离和视角变化。
- `fliplr=0.5`：左右翻转增强。

## 推荐训练命令

首轮稳定训练，目标是学到遮挡、背景、光照和姿态泛化：

```powershell
cd "D:\机器人识别大赛\robot_vision_pipeline"

python train_yolo.py `
  --data "D:\机器人识别大赛\yolo_dataset_official_aug\data.yaml" `
  --weights yolo11s.pt `
  --preset balanced `
  --epochs 160 `
  --imgsz 960 `
  --batch 4 `
  --device 0 `
  --project runs/train `
  --name official_aug_yolo11s_960
```

效果优先训练，目标是保留更小目标细节：

```powershell
python train_yolo.py `
  --data "D:\机器人识别大赛\yolo_dataset_official_aug\data.yaml" `
  --weights yolo11s.pt `
  --preset precision `
  --epochs 220 `
  --imgsz 1280 `
  --batch 2 `
  --device 0 `
  --project runs/train `
  --name official_aug_yolo11s_1280
```

RTX 4060 Laptop 8GB 显存建议先用 `imgsz=960, batch=4`。如果显存溢出，把 `batch` 改为 2；如果仍不够，再把 `imgsz` 改为 832。

## 提高精准度的改进点

当前增强集是偏“强泛化”的版本。它适合先把模型训稳，但如果目标是比赛计分中的少误检、高精准度，还要注意以下问题：

1. `mosaic=0.8`、`mixup=0.08`、`copy_paste=0.15` 偏强。训练 batch 里会出现大片灰色填充和较不真实的拼接画面，前期有用，后期可能降低精准度。
2. 类别均衡补样会重复使用少数原图。当前增强集里最多一张源图出现 11 次，可能让少样本类别过拟合到少数背景。
3. 耳机 `CA002` 是明显小目标，训练分辨率太低会损失细节。建议最终模型至少做一次 `imgsz=1280` 训练或精调。
4. 比赛有贴图干扰和桌外干扰物，当前数据里缺少明确的“负样本/干扰样本”。这会影响精准度，尤其容易把印刷图片或非目标物误检成真物体。
5. 官方 5 个常见类仍缺标注：`CA001` 刷子、`CB001` 巧克力、`CD001` 桃子、`CD002` 苹果、`CD004` 梨。缺类是比增强参数更大的风险。

### 两阶段训练

建议正式训练分两阶段：

第一阶段用强增强训泛化：

```powershell
python train_yolo.py `
  --data "D:\机器人识别大赛\yolo_dataset_official_aug\data.yaml" `
  --weights yolo11s.pt `
  --preset balanced `
  --epochs 140 `
  --imgsz 960 `
  --batch 4 `
  --device 0 `
  --project runs/train `
  --name stage1_balanced_yolo11s_960
```

第二阶段用弱增强高精度精调：

```powershell
python train_yolo.py `
  --data "D:\机器人识别大赛\yolo_dataset_official_aug\data.yaml" `
  --weights "runs\detect\runs\train\stage1_balanced_yolo11s_960\weights\best.pt" `
  --preset fine_tune `
  --epochs 60 `
  --imgsz 1280 `
  --batch 2 `
  --device 0 `
  --lr0 0.001 `
  --project runs/train `
  --name stage2_finetune_yolo11s_1280
```

`fine_tune` 预设会把 `mosaic`、`mixup`、`copy_paste` 和几何扰动降得很低，让模型最后更贴近真实照片分布。

### 更适合精调的数据集

如果想进一步降低重复补样导致的过拟合，可以再生成一份“精调增强集”。它仍保留黄光、白光、暗光，但只做较温和的类别补样，并限制同一张源图的重复次数：

```powershell
python "D:\机器人识别大赛\robot_vision_pipeline\augment_official_dataset.py" `
  --data "D:\机器人识别大赛\yolo_dataset_official\data.yaml" `
  --output "D:\机器人识别大赛\yolo_dataset_official_precision_aug" `
  --yellow 0.10 `
  --white 0.10 `
  --dark 0.05 `
  --balance-target p75 `
  --balance-pick least-used `
  --max-source-repeats 5 `
  --copy-originals reencode `
  --max-side 1920 `
  --quality 90 `
  --overwrite
```

然后第二阶段精调可改用：

```powershell
--data "D:\机器人识别大赛\yolo_dataset_official_precision_aug\data.yaml"
```

### 赛前阈值策略

比赛计分里，识别到不存在的 ID 会扣分，所以不能只追求 recall。正式模型训练完成后，需要用验证集或现场录制回放调高置信度阈值：

- 全局 `conf_floor` 不建议长期维持 `0.12`，正式比赛可从 `0.25-0.35` 起试。
- 容易误检的类应单独提高 `class_thresholds`。
- 每类最多数量已按规则限制为 `max_count_per_id: 5`，这点保留。
- 有 RGBD 时应继续使用深度过滤，贴图干扰可以靠“离桌面高度/厚度”特征削弱。

## 仍需补强

当前有标注的官方类是 11 类。规则中的以下 5 个常见类仍没有可训练标注：

| 官方 ID | 物品 |
| --- | --- |
| CA001 | 刷子 |
| CB001 | 巧克力 |
| CD001 | 桃子 |
| CD002 | 苹果 |
| CD004 | 梨 |

这些物体比赛可能出现。若要比赛更稳，需要继续采集并标注这些类，然后重新生成官方数据集和增强集。
