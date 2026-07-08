# 3D Recognition Competition Pipeline

Precision-first RGBD recognition pipeline for the RoboCup China 3D recognition task.

The project is designed as a contest-ready skeleton:

- YOLO detection or segmentation as the main proposal model.
- RGBD plane fitting and point-cloud checks for picture distractors.
- 3D-aware multi-frame instance fusion.
- Conservative score-driven counting.
- Optional second-stage classifier for confusing classes.
- Optional OCR voting for unknown objects.
- Replay recording and automatic contest-style scoring.

## Quick Start

Install dependencies:

```bash
pip install -r requirements.txt
```

Train YOLO with your own dataset:

```bash
python train_yolo.py --data configs/data.example.yaml --weights yolo11s.pt --epochs 200 --imgsz 1280
```

For the local dataset layout used on this machine, first prepare a standard YOLO
dataset:

```bash
python prepare_dataset.py --source "D:/robot_contest_data/data" --output "D:/robot_contest_data/yolo_dataset"
```

On this Windows machine the actual prepared dataset is:

```text
D:\机器人识别大赛\yolo_dataset\data.yaml
```

Then train with that generated yaml:

```bash
python train_yolo.py --data "D:\机器人识别大赛\yolo_dataset\data.yaml" --weights yolo11s.pt --epochs 200 --imgsz 1280
```

Run a replay directory:

```bash
python run_competition.py --config configs/competition.yaml --source replay --replay-dir replay --weights weights/best.pt
```

Run a single-image smoke test:

```bash
python run_competition.py --config configs/competition.yaml --source replay --replay-dir replay --weights yolo11n.pt --frame-count 1 --warmup-frames 0 --stable-min-frames 1 --save-vis
```

## Replay Evaluation

Truth file format:

```text
START
CA003;2;1
CC002;1;1
END
```

Run replay evaluation:

```bash
python replay_evaluate.py --config configs/competition.yaml --replay-dir replay --truth truth.txt --weights weights/best.pt --frame-count 24
```

The session directory will contain:

- `run_record.json`: frames, detections, instances, outputs, truth and score.
- `validation_sample.json`: sample format accepted by `optimize_thresholds.py`.
- `frames/`: saved RGB and optional depth frames.

## Learn Per-Class Thresholds

After collecting validation sessions, merge their `validation_sample.json` files into one JSON file:

```json
{
  "samples": [
    {
      "instances": [
        {"class_name": "CA001", "table_id": 1, "score": 0.91, "visible_frames": 4}
      ],
      "truth": [
        {"object_id": "CA001", "num": 1, "table_id": 1}
      ]
    }
  ]
}
```

Optimize thresholds:

```bash
python optimize_thresholds.py --config configs/competition.yaml --validation validation.json --output configs/learned_thresholds.yaml
```

Copy the generated `runtime.class_thresholds` into `configs/competition.yaml`.

## Real Contest Adapters

Before a real run, finish these adapters:

- `src/vision_competition/camera.py`: implement `OrbbecCamera` for Astra Pro Plus RGBD capture.
- `src/vision_competition/judge.py`: replace placeholder TCP messages with the official judge-box protocol.
- `configs/competition.yaml`: set team name, model weights, camera intrinsics, table ROIs and learned thresholds.

## Important Runtime Strategy

The output layer is intentionally conservative:

- False IDs can lose points, so low-confidence classes are filtered.
- Over-counting can lose a class score, so count selection maximizes expected contest score.
- Picture-like flat depth regions are heavily penalized.
- Unknown objects are reported only when OCR voting maps text to a known `Wxxx` ID.
