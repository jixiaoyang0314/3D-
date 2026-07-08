from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run replay inference and write an evaluable session record.")
    parser.add_argument("--config", default="configs/competition.yaml")
    parser.add_argument("--replay-dir", required=True)
    parser.add_argument("--truth", required=True)
    parser.add_argument("--weights")
    parser.add_argument("--device", help="Override runtime device, for example 0, cpu, or cuda:0.")
    parser.add_argument("--result-dir", default="runs/replay_result")
    parser.add_argument("--session-dir", default="runs/replay_session")
    parser.add_argument("--frame-count", type=int)
    parser.add_argument("--warmup-frames", type=int, default=0)
    parser.add_argument("--stable-min-frames", type=int)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    command = [
        sys.executable,
        "run_competition.py",
        "--config",
        args.config,
        "--source",
        "replay",
        "--replay-dir",
        args.replay_dir,
        "--result-dir",
        args.result_dir,
        "--session-dir",
        args.session_dir,
        "--truth",
        args.truth,
        "--save-vis",
    ]
    if args.weights:
        command.extend(["--weights", args.weights])
    if args.device:
        command.extend(["--device", args.device])
    if args.frame_count is not None:
        command.extend(["--frame-count", str(args.frame_count)])
    if args.warmup_frames is not None:
        command.extend(["--warmup-frames", str(args.warmup_frames)])
    if args.stable_min_frames is not None:
        command.extend(["--stable-min-frames", str(args.stable_min_frames)])

    subprocess.run(command, cwd=Path(__file__).resolve().parent, check=True)
    print(f"record={Path(args.session_dir) / 'run_record.json'}")


if __name__ == "__main__":
    main()
