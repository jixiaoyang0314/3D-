# Conversation Archive and Next-Step Context

Date: 2026-07-08

Project path:

```text
D:\机器人识别大赛\robot_vision_pipeline
```

GitHub remote:

```text
git@github.com:jixiaoyang0314/3D-.git
```

Current dataset:

```text
Raw data:      D:\机器人识别大赛\data\data
YOLO dataset:  D:\机器人识别大赛\yolo_dataset
YOLO yaml:     D:\机器人识别大赛\yolo_dataset\data.yaml
```

## Competition Context

The project targets the 2026 China Robot Competition / RoboCup China 3D recognition task.

Important rule points extracted from the PDFs:

- Hardware uses Orbbec Astra Pro Plus RGBD camera.
- Compute platform is OrangePi AIpro, 8T compute, 16GB memory.
- `start.sh` must launch the whole program with no extra parameters during contest.
- Recognition result must be written to the desktop `result` folder and sent to the judge box.
- Result txt format:

```text
START
CA003;2;1
CC002;1;1
END
```

- Each row is `ID;Num;Table`.
- Score is mainly recognition accuracy plus time score.
- If ID and Num are both correct, full score for that item is 3.
- If ID is not in the truth list, it can be penalized.
- If ID is correct but Num is less than truth, partial score is available.
- If ID is correct but Num is greater than truth, that item gets no score.
- Therefore the output strategy should be conservative: under-counting is safer than over-counting or false IDs.
- Round 1: one table, 7-15 objects, possible occlusion, picture distractors, field distractors, stacking.
- Round 2: three tables, 7-15 objects each, possible camera rotation, special lighting, picture distractors, field distractors, stacking.
- Unknown objects require text/OCR-based classification after categories are announced.

## Algorithm Direction

The current algorithm strategy is no longer "YOLO only". It is a score-driven multi-evidence system:

```text
YOLO detection/segmentation
  -> RGBD table plane fitting and point-cloud verification
  -> picture distractor suppression
  -> 3D-aware multi-frame instance fusion
  -> optional second-stage classifier for confusing classes
  -> optional OCR voting for unknown Wxxx objects
  -> score-driven conservative count decision
  -> result txt and judge-box output
```

Main design principle:

```text
YOLO finds candidates.
RGBD decides whether candidates are real 3D objects.
Multi-frame fusion counts stable instances.
The decision layer avoids risky outputs according to contest scoring.
```

## Implemented Code

Top-level scripts:

- `train_yolo.py`: train YOLO detect/segment model.
- `run_competition.py`: main inference entry for contest/replay.
- `prepare_dataset.py`: converts the local raw dataset into YOLO format.
- `replay_evaluate.py`: replay inference plus automatic scoring and record export.
- `optimize_thresholds.py`: learns per-class output thresholds from validation records.

Main modules:

- `src/vision_competition/camera.py`: replay/webcam camera plus placeholder `OrbbecCamera`.
- `src/vision_competition/detector.py`: Ultralytics YOLO wrapper.
- `src/vision_competition/depth.py`: RGBD depth normalization, RANSAC table plane fitting, point-cloud stats, picture-like filtering.
- `src/vision_competition/fusion.py`: 2D/3D multi-frame instance fusion.
- `src/vision_competition/decision.py`: score-driven conservative output decision.
- `src/vision_competition/classifier.py`: optional second-stage classifier verifier.
- `src/vision_competition/ocr.py`: optional OCR preprocessing, rotation, keyword voting for unknown objects.
- `src/vision_competition/recorder.py`: replay record export and truth parsing.
- `src/vision_competition/scorer.py`: contest-style scoring simulator.
- `src/vision_competition/threshold_optimizer.py`: per-class threshold search.
- `src/vision_competition/judge.py`: placeholder judge-box TCP client.

Tests:

- `tests/test_classifier.py`
- `tests/test_decision.py`
- `tests/test_depth.py`
- `tests/test_fusion.py`
- `tests/test_ocr.py`
- `tests/test_recorder.py`
- `tests/test_scorer.py`
- `tests/test_threshold_optimizer.py`

Last full test result:

```text
15 tests passed
compileall OK
single-image smoke test OK
replay evaluation OK
```

## Dataset Status

Raw data is under:

```text
D:\机器人识别大赛\data\data
```

Observed raw folders:

- `banana`
- `bb`
- `cup`
- `new`
- `part1`
- `part2`
- `pinzi`
- `shupian+huotuichang`
- `yijia`

Observed label zip files:

- `banana_with_label.zip`
- `bb.zip`
- `cup_with_label.zip`
- `new_with_label.zip`
- `part1.zip`
- `part2.zip`
- `shupian+huotuichang.zip`
- `yijia.zip`

`pinzi` currently has images but no matching label zip, so `prepare_dataset.py` skipped it.

Prepared YOLO dataset:

```text
D:\机器人识别大赛\yolo_dataset
```

Generated split:

```text
total labeled samples: 1806
train: 1445
val: 361
classes: 20
```

Generated class names:

```text
0: banana
1: bb_0
2: bb_1
3: bb_2
4: bb_3
5: cup
6: new_0
7: new_1
8: new_2
9: part_0
10: part_2
11: part_3
12: part_1
13: part_4
14: part_5
15: part_6
16: part_7
17: shupian+huotuichang_1
18: shupian+huotuichang_0
19: yijia
```

Important: these class names are dataset-derived placeholders. Before formal contest use, map them to official IDs such as `CA001`, `CB004`, `CD003`, etc.

## Useful Commands

Enter project directory:

```powershell
cd "D:\机器人识别大赛\robot_vision_pipeline"
```

Run tests:

```powershell
$env:PYTHONPATH = ".\src"
python -m unittest discover -s tests
```

Regenerate YOLO dataset:

```powershell
python prepare_dataset.py --source "D:/机器人识别大赛/data/data" --output "D:/机器人识别大赛/yolo_dataset"
```

If `D:\机器人识别大赛\yolo_dataset` already exists, delete or move it first.

Train a first YOLO model:

```powershell
python train_yolo.py --data "D:\机器人识别大赛\yolo_dataset\data.yaml" --weights yolo11s.pt --epochs 200 --imgsz 1280
```

Fast smoke test with pretrained COCO model:

```powershell
python run_competition.py --config configs/competition.yaml --source replay --replay-dir ..\yolo_dataset\images\val --weights yolo11n.pt --frame-count 1 --warmup-frames 0 --stable-min-frames 1 --save-vis
```

Replay evaluation with truth file:

```powershell
python replay_evaluate.py --config configs/competition.yaml --replay-dir replay --truth truth.txt --weights weights/best.pt --frame-count 24
```

Optimize per-class thresholds after collecting validation records:

```powershell
python optimize_thresholds.py --config configs/competition.yaml --validation validation.json --output configs/learned_thresholds.yaml
```

Git status:

```powershell
git status --short --branch
```

Commit and push:

```powershell
git add .
git commit -m "Describe the change"
git push
```

View history:

```powershell
git log --oneline --decorate
```

## Git History

Initial pushed commits:

```text
304e3c5 Initial competition pipeline
59d7a21 Add local dataset preparation
```

This archive file was added after those commits.

## Known Gaps

High-priority next tasks:

1. Train `weights/best.pt` using `D:\机器人识别大赛\yolo_dataset\data.yaml`.
2. Verify and rename class IDs to official contest IDs.
3. Add missing labels for `pinzi`, or intentionally exclude it.
4. Implement `OrbbecCamera` in `camera.py` for Astra Pro Plus RGBD capture.
5. Fill in official judge-box protocol in `judge.py` when released.
6. Test with real RGBD depth frames, not only RGB replay.
7. Tune `configs/competition.yaml` thresholds after replay validation.
8. Export/optimize model for OrangePi AIpro deployment.
9. Build a real `start.sh` launch path on the OrangePi desktop.
10. Measure model load time and total recognition time, because load time counts.

Medium-priority improvements:

- Add official object ID mapping config.
- Add confusion-specific classifier model for apple/pear/peach, bottled drink/water, milk/snacks.
- Enable and test OCR only after unknown object keywords are known.
- Add visual debug output for depth plane, 3D centers, and picture-like candidates.
- Add batch validation command over multiple replay sessions.

## Current Practical Next Step

The most useful immediate next step is training the first contest-specific YOLO model:

```powershell
cd "D:\机器人识别大赛\robot_vision_pipeline"
python train_yolo.py --data "D:\机器人识别大赛\yolo_dataset\data.yaml" --weights yolo11s.pt --epochs 100 --imgsz 960
```

Use `100` epochs and `960` image size for a first sanity run. If it trains correctly, run the stronger version:

```powershell
python train_yolo.py --data "D:\机器人识别大赛\yolo_dataset\data.yaml" --weights yolo11s.pt --epochs 200 --imgsz 1280
```

