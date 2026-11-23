#!/bin/bash
# [20261117] B 確認性重跑：1000 vs 0000 × MH04 MH05 V202 V203 × 全新 15 次（事先登記 20261117 退化掃描與確認性重跑 v1）
IMG=jetson-slam:cuda11.6.1-cudnn8-devel-ubuntu20.04-cv4.10.0-pangolin0.6-eigen3.4.0
W=$HOME/slam_work
COMMON="--rm --gpus all --ipc=host -e NVIDIA_DRIVER_CAPABILITIES=all -e CUDA_CACHE_MAXSIZE=4294967296 -v $W/output:/root/OutputTest -v $W/scripts:/root/scripts -v $W/nvcache_ours:/root/.nv -v $W/code_v2:/root/Jetson-SLAM -v $W/datasets/EuRoC:/root/Datasets/EuRoC"
echo "[$(date +%H:%M:%S)] B 開始"
for SEQ in MH_04_difficult V2_03_difficult MH_05_difficult V2_02_medium; do
  for B in 1000 0000; do
    docker run $COMMON -e YAML=EuRoC_docker.yaml -e ABLATION=$B --entrypoint /bin/bash $IMG /root/scripts/run_euroc.sh ABLB_$B 15 $SEQ
  done
done
echo "B_DONE $(date +%H:%M:%S)"
