#!/bin/bash
# [20261121] L3：事先登記 20261121 事先登記 補實驗L1-L4 v2.md
#  EuRoC MH04 × {Final(1000)、Baseline-20(0000)、Jetson-SLAM} × 每方法 N 次，逐次交替；每次 run 單獨取樣
#  GPU：nvidia-smi --query-compute-apps（依該行程 PID）每 0.1 秒；host：VmRSS（/proc/<pid>/status）每 0.1 秒
#  另用 /usr/bin/time -v 不方便（在容器內），改以取樣的 VmHWM（行程峰值 RSS）為 host 峰值
# 用法：bash m17_L3.sh <次數> <輸出前綴>   例：bash m17_L3.sh 5 L3 ；冒煙：bash m17_L3.sh 1 L3smoke
IMG=jetson-slam:cuda11.6.1-cudnn8-devel-ubuntu20.04-cv4.10.0-pangolin0.6-eigen3.4.0
W=$HOME/slam_work; N=${1:-5}; P=${2:-L3}
LOG=$W/output/${P}_progress.log
BASE="--rm --gpus all --ipc=host -e NVIDIA_DRIVER_CAPABILITIES=all -e CUDA_CACHE_MAXSIZE=4294967296 -v $W/datasets/EuRoC:/root/Datasets/EuRoC -v $W/output:/root/OutputTest -v $W/scripts:/root/scripts"
sample() { # $1 輸出檔
  while true; do
    t=$(date +%s.%N)
    apps=$(nvidia-smi --query-compute-apps=pid,used_memory --format=csv,noheader,nounits)
    for p in $(pgrep -x mono_euroc); do
      r=$(awk '/VmRSS/{print $2}' /proc/$p/status 2>/dev/null); h=$(awk '/VmHWM/{print $2}' /proc/$p/status 2>/dev/null)
      g=$(echo "$apps" | awk -F', ' -v p=$p '$1==p{print $2}')
      echo "$t $p ${r:-0} ${h:-0} ${g:-0}"
    done
    sleep 0.1
  done >> "$1"
}
run() { # $1 方法 $2 run 編號
  case $1 in
    final) docker run $BASE -e YAML=EuRoC_docker.yaml -e ABLATION=1000 -v $W/nvcache_ours:/root/.nv -v $W/code_v3:/root/Jetson-SLAM --entrypoint /bin/bash $IMG /root/scripts/run_euroc_idx.sh ${P}_$1 $2 MH_04_difficult;;
    b20)   docker run $BASE -e YAML=EuRoC_docker.yaml -e ABLATION=0000 -v $W/nvcache_ours:/root/.nv -v $W/code_v3:/root/Jetson-SLAM --entrypoint /bin/bash $IMG /root/scripts/run_euroc_idx.sh ${P}_$1 $2 MH_04_difficult;;
    jslam) docker run $BASE -e YAML=EuRoC_baseline_docker.yaml -v $W/nvcache_base:/root/.nv -v $W/baselines/Jetson-SLAM-master:/root/Jetson-SLAM --entrypoint /bin/bash $IMG /root/scripts/run_euroc_idx.sh ${P}_$1 $2 MH_04_difficult;;
  esac
}
mkdir -p $W/output/${P}_MEM
echo "[$(date +%F' '%T)] ${P} 開始；閒置 GPU 用量 $(nvidia-smi --query-gpu=memory.used --format=csv,noheader)" | tee -a $LOG
for i in $(seq 1 $N); do
  for M in final b20 jslam; do
    F=$W/output/${P}_MEM/mem_${M}_$i.txt; rm -f $F
    sample $F & SP=$!
    run $M $i >> $LOG 2>&1
    kill $SP; wait $SP 2>/dev/null
    echo "  [$M run $i] 取樣 $(wc -l < $F) 筆" | tee -a $LOG
  done
done
echo "L3_DONE $(date +%F' '%T)" | tee -a $LOG
