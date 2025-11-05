# STG-STSG-Model (v0)
Minimal Spatio-Temporal Scene Grammar (ST-SG) inducer for synthetic facades/streets.

## What it does (v0)
- Detects floors via horizontal projection peaks (Split_y_k).
- Detects per-floor window repeats via autocorrelation (Repeat_x_c).
- Emits grammar JSON compatible with `stg-synthetic-eval`.

## Quick start
```bash
pip install -r requirements.txt
python -m src.infer_folder \
  --images ../stg-procedural-data/outputs/facades \
  --out    ../stg-synthetic-eval/datasets/synthetic_facades
```
It writes {scene}_pred.json next to your {scene}_gt.json.
