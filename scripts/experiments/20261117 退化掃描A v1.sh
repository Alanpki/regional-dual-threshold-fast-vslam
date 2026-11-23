#!/bin/bash
# [20261117] A 影像退化掃描（事先登記 20261117 退化掃描與確認性重跑 v1）：等 B 結束後執行
#  條件 S0 原始、C50 C35 C25 對比降低、K2 K3 K4 跳幀 × 方法 ours1000、ours0000、Jetson-SLAM × MH03、V202 × 5 次
IMG=jetson-slam:cuda11.6.1-cudnn8-devel-ubuntu20.04-cv4.10.0-pangolin0.6-eigen3.4.0
W=$HOME/slam_work
until grep -q "B_DONE" $W/output/m10_B_progress.log; do sleep 60; done
echo "[$(date +%H:%M:%S)] A 開始"
run() { # $1 方法 $2 條件
  case $1 in
    ours1000) CODE=$W/code_v2; YAML=EuRoC_docker.yaml; AB="-e ABLATION=1000"; NV=$W/nvcache_ours;;
    ours0000) CODE=$W/code_v2; YAML=EuRoC_docker.yaml; AB="-e ABLATION=0000"; NV=$W/nvcache_ours;;
    base)     CODE=$W/baselines/Jetson-SLAM-master; YAML=EuRoC_baseline_docker.yaml; AB=""; NV=$W/nvcache_base;;
  esac
  case $2 in
    S0) DATA="-v $W/datasets/EuRoC:/root/Datasets/EuRoC";;
    C*) DATA="-v $W/datasets/EuRoC_deg/$2:/root/Datasets/EuRoC";;
    K*) DATA="-v $W/datasets/EuRoC:/root/Datasets/EuRoC -v $W/datasets/EuRoC_ts/$2:/root/Jetson-SLAM/Examples/Monocular/EuRoC_TimeStamps:ro";;
  esac
  docker run --rm --gpus all --ipc=host -e NVIDIA_DRIVER_CAPABILITIES=all -e CUDA_CACHE_MAXSIZE=4294967296 -e YAML=$YAML $AB \
    -v $NV:/root/.nv -v $CODE:/root/Jetson-SLAM $DATA -v $W/output:/root/OutputTest -v $W/scripts:/root/scripts \
    --entrypoint /bin/bash $IMG /root/scripts/run_euroc.sh DEG_${2}_$1 5 MH_03_medium V2_02_medium
}
for C in S0 C50 C35 C25 K2 K3 K4; do
  for M in ours1000 ours0000 base; do run $M $C; done
  echo "[$(date +%H:%M:%S)] 條件 $C 完成"
done
echo "A_DONE $(date +%H:%M:%S)"
