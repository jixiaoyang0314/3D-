# OM Inference Package

This package contains the Ascend OM model and matching YOLO postprocess code.

## Files

- `weights/best.om`: converted Ascend model for `Ascend310B4`
- `infer_om.py`: pyACL inference script
- `run_infer.sh`: convenience launcher that sources CANN environment
- `labels.txt`: class id to official object id mapping
- `class_mapping.json`: same mapping in JSON
- `model_info.json`: input/output shape and conversion metadata

## Model IO

- Input: `images`, shape `[1, 3, 640, 640]`, float32, RGB, normalized by `1/255`
- Output: `output0`, shape `[1, 15, 8400]`
- Classes: 11

## Run

On the Orange Pi AI Pro / Ascend environment:

```bash
cd om_inference_package
chmod +x run_infer.sh
./run_infer.sh --source /path/to/image_or_dir --conf 0.25 --iou 0.60 --save-vis
```

Outputs are written to `runs_om/`:

- `predictions.json`
- `labels/*.txt`
- `vis/*` when `--save-vis` is used

If CANN is installed in a non-standard path:

```bash
CANN_TOOLKIT_ROOT=/path/to/Ascend/ascend-toolkit ./run_infer.sh --source /path/to/images
```

If the board Python command is not `python3`:

```bash
PYTHON_BIN=/path/to/python ./run_infer.sh --source /path/to/images
```

Install Python dependencies if needed:

```bash
python3 -m pip install -r requirements.txt
```

The `acl` Python module is provided by CANN/pyACL, not by pip.
