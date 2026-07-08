from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from vision_competition.camera import create_camera
from vision_competition.classifier import ClassifierVerifier
from vision_competition.config import load_config
from vision_competition.decision import decide_outputs
from vision_competition.depth import DepthAnalyzer
from vision_competition.detector import YOLODetector
from vision_competition.fusion import MultiFrameFusion, draw_instances
from vision_competition.judge import JudgeClient
from vision_competition.ocr import UnknownOCR
from vision_competition.output import write_result_file
from vision_competition.recorder import ReplayRecorder
from vision_competition.types import Instance


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the precision-first competition pipeline.")
    parser.add_argument("--config", default="configs/competition.yaml")
    parser.add_argument("--source", choices=["replay", "webcam", "opencv", "orbbec", "astra"])
    parser.add_argument("--replay-dir")
    parser.add_argument("--round", type=int)
    parser.add_argument("--weights", help="Override YOLO weights in the config.")
    parser.add_argument("--device", help="Override runtime device, for example 0, cpu, or cuda:0.")
    parser.add_argument("--result-dir", help="Override result directory in the config.")
    parser.add_argument("--frame-count", type=int, help="Override number of frames to process.")
    parser.add_argument("--warmup-frames", type=int, help="Override number of warmup frames to skip.")
    parser.add_argument("--stable-min-frames", type=int, help="Override minimum stable frames.")
    parser.add_argument("--session-dir", help="Save replay frames, detections, instances, outputs and optional score.")
    parser.add_argument("--truth", help="Truth txt/json for automatic scoring when --session-dir is used.")
    parser.add_argument("--save-vis", action="store_true")
    return parser.parse_args()


def apply_cli_overrides(config_path: str, args: argparse.Namespace):
    config = load_config(config_path)
    if args.source:
        config.camera.source = args.source
    if args.replay_dir:
        config.camera.replay_dir = args.replay_dir
    if args.round is not None:
        config.competition.round_id = args.round
    if args.weights:
        config.models.yolo_weights = args.weights
    if args.device:
        config.runtime.device = args.device
    if args.result_dir:
        config.competition.result_dir = args.result_dir
    if args.frame_count is not None:
        config.runtime.frame_count = args.frame_count
    if args.warmup_frames is not None:
        config.runtime.warmup_frames = args.warmup_frames
    if args.stable_min_frames is not None:
        config.runtime.stable_min_frames = args.stable_min_frames
    return config


def attach_ocr(instances: list[Instance], ocr: UnknownOCR) -> None:
    for inst in instances:
        if not inst.class_name.startswith("W"):
            continue
        crops = inst.crops or ([] if inst.best_crop is None else [inst.best_crop])
        result = ocr.recognize_many(crops)
        if result is None:
            inst.score *= 0.45
            continue
        inst.class_name = result.object_id
        inst.ocr_text = result.text
        inst.ocr_score = result.confidence
        inst.score = min(1.0, max(inst.score, result.confidence))


def main() -> None:
    args = parse_args()
    config = apply_cli_overrides(args.config, args)

    judge = JudgeClient(config.judge)
    judge.start()

    detector = YOLODetector(
        weights=config.models.yolo_weights,
        device=config.runtime.device,
        imgsz=config.runtime.imgsz,
        conf=config.runtime.conf_floor,
        iou=config.runtime.yolo_iou,
    )
    camera = create_camera(config.camera)
    depth = DepthAnalyzer(config.depth, config.camera, config.runtime)
    fusion = MultiFrameFusion(config.runtime)
    ocr = UnknownOCR(config.ocr, enabled=config.models.ocr_enabled)
    classifier = ClassifierVerifier(config.models, config.runtime)
    recorder = ReplayRecorder(args.session_dir)

    start_time = time.monotonic()
    last_frame = None
    frame_seen = 0

    try:
        while frame_seen < config.runtime.frame_count:
            if time.monotonic() - start_time >= config.runtime.max_runtime_sec:
                break
            frame = camera.read()
            if frame is None:
                break
            last_frame = frame
            frame_seen += 1
            if frame_seen <= config.runtime.warmup_frames:
                continue

            detections = detector.predict(frame.rgb, frame.index)
            detections = depth.enrich(frame, detections)
            recorder.record_frame(frame, detections)
            fusion.add_frame(frame, detections)
    finally:
        camera.close()

    instances = fusion.finalize()
    instances = classifier.verify(instances)
    attach_ocr(instances, ocr)
    rows = decide_outputs(instances, config.runtime)
    result_path = write_result_file(config.competition, rows)
    judge.send_results(rows)
    judge.end()

    if args.save_vis and last_frame is not None:
        import cv2

        vis = draw_instances(last_frame.rgb, instances)
        vis_path = Path(config.competition.result_dir).expanduser() / "debug_instances.jpg"
        cv2.imwrite(str(vis_path), vis)

    elapsed = time.monotonic() - start_time
    record_payload = recorder.write_final(instances, rows, elapsed_sec=elapsed, truth_path=args.truth)
    print(f"frames={frame_seen} instances={len(instances)} outputs={len(rows)} elapsed={elapsed:.2f}s")
    print(f"result={result_path}")
    if record_payload and record_payload.get("score"):
        score = record_payload["score"]
        print(f"score={score['measure_score']:.3f}/{score['full_score']:.3f} avg={score['average_score']:.3f}")
    for row in rows:
        print(f"{row.object_id};{row.num};{row.table_id} expected={row.score:.3f}")


if __name__ == "__main__":
    main()
