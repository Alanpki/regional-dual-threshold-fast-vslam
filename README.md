# GPU regional dual-threshold FAST for monocular Jetson-SLAM

Source code, configurations, and scripts for the experiments of the letter *GPU-Parallel Regional Dual-Threshold Feature Detection for Monocular Visual SLAM under Low Contrast* (title provisional).

The final system is the monocular front end of Jetson-SLAM with an ORB-SLAM-style regional dual-threshold FAST detector executed on the GPU: a high threshold (τH = 20) is applied in every 32 × 32 cell, and cells without a corner are re-scanned with a low threshold (τL = 10); at most N = 1000 keypoints per frame are kept with a QuadTree distribution.

## Layout

| Path | Content |
|---|---|
| `code/` | Source of the evaluated system (Jetson-SLAM based, GPLv3). Detector: `code/src/cuda/orb_FAST_compute_score_v2.cu`, `code/src/cuda/orb_adaptive_kernels.cu`, `code/src/ORBextractor.cpp`; monocular drivers: `code/Examples/Monocular/`; settings: `code/Examples/Monocular/*.yaml` |
| `configs/` | Settings used for the baselines and variants (released-style baseline, global threshold 10) |
| `patches/` | Changes to the third-party baselines (Jetson-SLAM commit 6d47378, ORB-SLAM3 v1.0-release): drivers that write per-frame states, portrait ADVIO setting, matched initialization thresholds, and candidate-count output for the global-threshold cost test |
| `scripts/experiments/` | Batch scripts that ran the reported experiments inside Docker (EuRoC, contrast and region-wise degradation, ADVIO, memory, global-threshold cost, matched-initialization baseline) |
| `scripts/analysis/` | ATE evaluation (Sim(3)), degraded-image generators, texture statistics, and per-experiment analysis with permutation tests |

## Switches

The detector is selected by the environment variable `ABLATION` (four bits; only the first is used by the final system):

| Value | Meaning |
|---|---|
| `1000` | final system: regional dual-threshold detector |
| `0000` | same pipeline, single threshold 20 (`th_FAST_MIN` is set to `th_FAST_MAX`) |

A global threshold of 10 is run with `ABLATION=0000` and `configs/20261118 J EuRoC_G10_docker v1.yaml`. The other bits switch modules of an earlier development version that are not part of the final system.

## Environment

- Desktop: Intel i5-14500, NVIDIA RTX 4060 Ti, Ubuntu 22.04, Docker with the NVIDIA runtime.
- Container image: `jetson-slam:cuda11.6.1-cudnn8-devel-ubuntu20.04-cv4.10.0-pangolin0.6-eigen3.4.0` (Ubuntu 20.04, CUDA 11.6.1, gcc 9.4, OpenCV 4.10.0, Eigen 3.4.0, Pangolin 0.6). The scripts also use an ORB-SLAM3 image (`jetson-slam-orb3:v1`) built on the same base.
- Datasets (not included): EuRoC MAV, TUM RGB-D (fr3 structure/texture), ADVIO (downscaled to 360 × 640).
- ORB vocabulary (not included): put `ORBvoc.txt` (from ORB-SLAM3) into `code/Vocabulary/`.

## Build and run

```bash
# inside the container, with code/ mounted at /root/Jetson-SLAM
cd /root/Jetson-SLAM && bash build.sh      # builds DBoW2, g2o, and the library
# one EuRoC run: <output name> <run index> <sequence>
export ABLATION=1000 YAML=EuRoC_docker.yaml
bash "scripts/experiments/20261118 J run_euroc_idx v1.sh" J1_1000 1 MH_04_difficult
```

The scripts under `scripts/experiments/` assume the layout `~/slam_work/{code_v3,baselines,datasets,scripts,output}`; adjust the variables at their top. Each run writes the keyframe trajectory, a per-frame state file (`frametimes_*.txt`), and the log, which `scripts/analysis/` reads. Evaluation protocol: five runs per configuration (15 for confirmatory tests), medians with ranges, two-sided permutation tests (20 000 permutations) with Bonferroni correction, ATE on keyframes after Sim(3) alignment.

## Baselines

- Jetson-SLAM (commit 6d47378) and ORB-SLAM3 (v1.0-release) are cloned from their upstream repositories and patched with `patches/`.
- The evaluated system uses monocular initialization thresholds that differ from the released Jetson-SLAM (keypoints > 100, 100-pixel search window, ≥ 50 matches, ≥ 50 map points; released: 50, 50, 50, 100). The Jetson-SLAM baseline in the paper was re-run with the same thresholds using `patches/20261122 Jetson-SLAM對齊初始化 v1.diff.txt`.

## License

Derived from Jetson-SLAM and ORB-SLAM (GPLv3); the same license applies to the modified code. Third-party components under `code/Thirdparty/` keep their own licenses.
