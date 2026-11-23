#!/bin/bash
# [20261122] L2：事先登記 20261121 事先登記 補實驗L1-L4 v2.md
#  區域退化 RC、RB × MH03、V202 × {final(1000)、g10、b20(0000)、jslam、orb3} × N 次，逐次交替
# 用法：bash m19_L2.sh <次數> <前綴>   冒煙：bash m19_L2.sh 1 L2smoke（只跑 RC MH03）
IMG=jetson-slam:cuda11.6.1-cudnn8-devel-ubuntu20.04-cv4.10.0-pangolin0.6-eigen3.4.0
IMG3=jetson-slam-orb3:v1
W=$HOME/slam_work; N=${1:-5}; P=${2:-L2}
LOG=$W/output/${P}_progress.log
BASE="--rm --gpus all --ipc=host -e NVIDIA_DRIVER_CAPABILITIES=all -e CUDA_CACHE_MAXSIZE=4294967296 -v $W/output:/root/OutputTest -v $W/scripts:/root/scripts"
run() { # $1 方法 $2 條件 $3 序列 $4 run 編號
  D="-v $W/datasets/EuRoC_deg/$2:/root/Datasets/EuRoC"; T=${P}_$2_$1
  case $1 in
    final) docker run $BASE $D -e YAML=EuRoC_docker.yaml     -e ABLATION=1000 -v $W/nvcache_ours:/root/.nv -v $W/code_v3:/root/Jetson-SLAM --entrypoint /bin/bash $IMG /root/scripts/run_euroc_idx.sh $T $4 $3;;
    g10)   docker run $BASE $D -e YAML=EuRoC_G10_docker.yaml -e ABLATION=0000 -v $W/nvcache_ours:/root/.nv -v $W/code_v3:/root/Jetson-SLAM --entrypoint /bin/bash $IMG /root/scripts/run_euroc_idx.sh $T $4 $3;;
    b20)   docker run $BASE $D -e YAML=EuRoC_docker.yaml     -e ABLATION=0000 -v $W/nvcache_ours:/root/.nv -v $W/code_v3:/root/Jetson-SLAM --entrypoint /bin/bash $IMG /root/scripts/run_euroc_idx.sh $T $4 $3;;
    jslam) docker run $BASE $D -e YAML=EuRoC_baseline_docker.yaml -v $W/nvcache_base:/root/.nv -v $W/baselines/Jetson-SLAM-master:/root/Jetson-SLAM --entrypoint /bin/bash $IMG /root/scripts/run_euroc_idx.sh $T $4 $3;;
    orb3)  docker run $BASE $D -e YAML=EuRoC.yaml -e IMGSUB= -v $W/baselines/ORB_SLAM3:/root/Jetson-SLAM -v $W/baselines/ORB_SLAM3:/root/ORB_SLAM3 --entrypoint /bin/bash $IMG3 /root/scripts/run_euroc_idx.sh $T $4 $3;;
  esac
}
CONDS="RC RB"; SEQS="MH_03_medium V2_02_medium"
[ "$P" = "L2smoke" ] && { CONDS="RC"; SEQS="MH_03_medium"; }
echo "[$(date +%F' '%T)] ${P} 開始" | tee -a $LOG
for C in $CONDS; do for SEQ in $SEQS; do
  for i in $(seq 1 $N); do for M in final g10 b20 jslam orb3; do run $M $C $SEQ $i >> $LOG 2>&1; done; done
  echo "[$(date +%F' '%T)] ${P} $C $SEQ 完成" | tee -a $LOG
done; done
echo "L2_DONE $(date +%F' '%T)" | tee -a $LOG
