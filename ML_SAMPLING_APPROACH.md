# ML Sampling Approach (MVP)

This document captures the current ML sampling experiment for edge detection.
It is an MVP meant to validate sampling policy ideas before any full model is
trained for geometry classification or parameter regression.

## Goals

- Learn a sampling policy that chooses where to probe next.
- Keep per-vertex sample budgets under ~50.
- Use deterministic plane fitting (SVD/RANSAC) for geometry estimation.
- Compare learned policy variants against baseline attempts.

## Hybrid Architecture (Current)

- **Sampler policy**: chooses next probe direction from a fixed action set.
- **State**: small feature vector + simple exponential moving average "latent".
- **Measurement**: existing plane fits (SVD/RANSAC) on accumulated samples.

## Feature/State Inputs (MVP)

- Step index / budget fraction.
- Number of samples collected.
- Probe hit ratio (crossings found / attempts).
- Average sample distance.
- Dot(last_direction, hint_normal).
- Two latent slots (EMA of features).

## Action Spaces

- **Directional policy**: 32 fixed directions on a sphere (Fibonacci).
- **Octant policy**: 8 directions (octants).
- Octant evaluated with **argmax** and **lerp** variants.

## Reward Function (Training Utility)

At each step:

```
L = w_cls * CE + w_n * normal_error/90 + w_e * edge_dir_error/90
reward = (L_prev - L) - lambda
```

Where:
- `CE` uses oracle labels (face/edge/corner) during training.
- `lambda` penalizes extra samples.
- Errors are from deterministic fit results (face, edge, corner).

## Current Status (MVP Results)

- Accuracy is low (~27% on rotated cube).
- Budgets exceeded (~60 samples on average).
- Octant-lerp produced extreme sample counts without strong gains.
- Visualized sample clouds show erratic distributions (see `SAMPLE_CLOUD_DEBUG.md`).

## Next Steps

- Constrain sampling by **inside-start** and **cone hints** around normals.
- Add origin-offset head (surface vs inside vs +/- hint).
- Add a stop policy (confidence threshold) to enforce the sample budget.
- Expand training shapes (cylinder, chamfered box, CSG).

## References

- MVP implementation: `src/adaptive_surface_nets_2/stage4/research/experiments/ml_policy.rs`
- Sample cloud dumps: `sample_cloud_ml_*.cbor`
