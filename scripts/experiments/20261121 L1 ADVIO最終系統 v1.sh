#!/bin/bash
# [20261121] L1：事先登記 20261121 事先登記 補實驗L1-L4 v2.md
#  ADVIO 01–04 × {Final(1000)、Baseline-20(0000)、Jetson-SLAM、ORB-SLAM3} × N 次，逐次交替
# 用法：bash m18_L1.sh <次數> <輸出前綴> [序列...]   例：bash m18_L1.sh 5 L1 ；冒煙：bash m18_L1.sh 1 L1smoke advio-04
IMG=jetson-slam:cuda11.6.1-cudnn8-devel-ubuntu20.04-cv4.10.0-pangolin0.6-eigen3.4.0
IMG3=jetson-slam-orb3:v1
W=$HOME/slam_work; N=${1:-5}; P=${2:-L1}; shift 2
SEQS=${@:-advio-01 advio-02 advio-03 advio-04}
LOG=$W/output/${P}_progress.log
BASE="--rm --gpus all --ipc=host -e NVIDIA_DRIVER_CAPABILITIES=all -e CUDA_CACHE_MAXSIZE=4294967296 -v $W/output:/root/OutputTest -v $W/scripts:/root/scripts -v $W/datasets/ADVIO:/root/Datasets/ADVIO -e CSV=frames.csv"
run() { # $1 方法 $2 序列 $3 run 編號
  case $1 in
    final) docker run $BASE -e BIN=ADVIO/Monocular/mono_advio -e YAML=ADVIO/Monocular/ADVIO_B.yaml -e ABLATION=1000 -v $W/nvcache_ours:/root/.nv -v $W/code_v3:/root/Jetson-SLAM --entrypoint /bin/bash $IMG /root/scripts/run_advio_idx.sh ${P}_final $3 $2;;
    b20)   docker run $BASE -e BIN=ADVIO/Monocular/mono_advio -e YAML=ADVIO/Monocular/ADVIO_B.yaml -e ABLATION=0000 -v $W/nvcache_ours:/root/.nv -v $W/code_v3:/root/Jetson-SLAM --entrypoint /bin/bash $IMG /root/scripts/run_advio_idx.sh ${P}_b20 $3 $2;;
    jslam) docker run $BASE -e BIN=Examples/Monocular/mono_advio -e YAML=Examples/Monocular/ADVIO_baseline_B.yaml -v $W/nvcache_base:/root/.nv -v $W/baselines/Jetson-SLAM-master:/root/Jetson-SLAM --entrypoint /bin/bash $IMG /root/scripts/run_advio_idx.sh ${P}_jslam $3 $2;;
    orb3)  docker run $BASE -e BIN=Examples/Monocular/mono_advio -e YAML=Examples/Monocular/ADVIO_orb3_B.yaml -v $W/baselines/ORB_SLAM3:/root/Jetson-SLAM -v $W/baselines/ORB_SLAM3:/root/ORB_SLAM3 --entrypoint /bin/bash $IMG3 /root/scripts/run_advio_idx.sh ${P}_orb3 $3 $2;;
  esac
}
echo "[$(date +%F' '%T)] ${P} 開始" | tee -a $LOG
for SEQ in $SEQS; do
  for i in $(seq 1 $N); do
    for M in final b20 jslam orb3; do run $M $SEQ $i >> $LOG 2>&1; done
  done
  echo "[$(date +%F' '%T)] ${P} $SEQ 完成" | tee -a $LOG
done
echo "L1_DONE $(date +%F' '%T)" | tee -a $LOG
