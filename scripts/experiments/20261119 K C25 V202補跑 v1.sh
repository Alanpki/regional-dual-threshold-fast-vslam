#!/bin/bash
# [20261119] K：事先登記 20261119 事先登記 C25 V202補跑 v1.md
#  C25 × V2_02_medium，ours1000 vs oursG10，run 6–15，逐次交替；輸出 K_C25_ours1000、K_C25_oursG10
IMG=jetson-slam:cuda11.6.1-cudnn8-devel-ubuntu20.04-cv4.10.0-pangolin0.6-eigen3.4.0
W=$HOME/slam_work
LOG=$W/output/m16_K_progress.log
BASE="--rm --gpus all --ipc=host -e NVIDIA_DRIVER_CAPABILITIES=all -e CUDA_CACHE_MAXSIZE=4294967296 -v $W/output:/root/OutputTest -v $W/scripts:/root/scripts"
DATA="-v $W/datasets/EuRoC_deg/C25:/root/Datasets/EuRoC"
echo "[$(date +%H:%M:%S)] K 開始" | tee -a $LOG
for i in 6 7 8 9 10 11 12 13 14 15; do
  docker run $BASE $DATA -e YAML=EuRoC_docker.yaml     -e ABLATION=1000 -v $W/nvcache_ours:/root/.nv -v $W/code_v3:/root/Jetson-SLAM --entrypoint /bin/bash $IMG /root/scripts/run_euroc_idx.sh K_C25_ours1000 $i V2_02_medium >> $LOG 2>&1
  docker run $BASE $DATA -e YAML=EuRoC_G10_docker.yaml -e ABLATION=0000 -v $W/nvcache_ours:/root/.nv -v $W/code_v3:/root/Jetson-SLAM --entrypoint /bin/bash $IMG /root/scripts/run_euroc_idx.sh K_C25_oursG10 $i V2_02_medium >> $LOG 2>&1
done
echo "K_DONE $(date +%H:%M:%S)" | tee -a $LOG
