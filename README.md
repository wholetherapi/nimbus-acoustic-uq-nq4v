# Calibrated Uncertainty for Wildlife Acoustic Monitoring

Code for **Calibrated Uncertainty for Wildlife Acoustic Monitoring** (TMLR 2026).

The package implements the species-conditional conformal calibration layer and
the feature extraction pipeline used for the two field recordings in the paper.

## Layout

```
src/acoustic_uq/    calibration and feature extraction
scripts/            training and evaluation
configs/            experiment configuration
```

## Quick start

```bash
pip install -r requirements.txt
python scripts/train.py --config configs/default.yaml
python scripts/evaluate.py --config configs/default.yaml
```
