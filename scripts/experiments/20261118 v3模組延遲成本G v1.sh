#!/bin/bash
# [20261118] G：v3 四個模組延遲成本（事先登記 20261118 v3模組延遲成本 v1）：0000 vs 1111 × MH01–05、V203 × 5，交替
IMG=jetson-slam:cuda11.6.1-cudnn8-devel-ubuntu20.04-cv4.10.0-pangolin0.6-eigen3.4.0
W=$HOME/slam_work
G="--rm --gpus all --ipc=host -e NVIDIA_DRIVER_CAPABILITIES=all -e CUDA_CACHE_MAXSIZE=4294967296 -v $W/datasets/EuRoC:/root/Datasets/EuRoC -v $W/output:/root/OutputTest -v $W/scripts:/root/scripts -e YAML=EuRoC_docker.yaml -v $W/nvcache_ours:/root/.nv -v $W/code_v3:/root/Jetson-SLAM"
echo "[$(date +%H:%M:%S)] G 開始"
for SEQ in MH_01_easy MH_02_easy MH_03_medium MH_04_difficult MH_05_difficult V2_03_difficult; do
  for AB in 0000 1111; do
    docker run $G -e ABLATION=$AB --entrypoint /bin/bash $IMG /root/scripts/run_euroc.sh ABLV3_$AB 5 $SEQ
  done
  echo "[$(date +%H:%M:%S)] $SEQ 完成"
done
echo "G_DONE $(date +%H:%M:%S)"
