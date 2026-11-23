#!/bin/bash
# [20261122] L4：事先登記 20261121 事先登記 補實驗L1-L4 v2.md
#  程式＝code_v3_L4（code_v3 複本，只加「候選點數／擷取延遲」輸出，見 code_v3_L4/L4_candidate_output.diff）
#  方法 {final(1000)、g10} × 場景 {S0：MH01–05、V203、V202；C25：MH03、V202} × N 次，逐次交替
# 用法：bash m20_L4.sh <次數> <前綴>   冒煙：bash m20_L4.sh 1 L4smoke（只跑 S0 MH03）
IMG=jetson-slam:cuda11.6.1-cudnn8-devel-ubuntu20.04-cv4.10.0-pangolin0.6-eigen3.4.0
W=$HOME/slam_work; N=${1:-5}; P=${2:-L4}
LOG=$W/output/${P}_progress.log
BASE="--rm --gpus all --ipc=host -e NVIDIA_DRIVER_CAPABILITIES=all -e CUDA_CACHE_MAXSIZE=4294967296 -v $W/output:/root/OutputTest -v $W/scripts:/root/scripts"
run() { # $1 方法 $2 條件 $3 序列 $4 run 編號
  case $2 in S0) D="-v $W/datasets/EuRoC:/root/Datasets/EuRoC";; C25) D="-v $W/datasets/EuRoC_deg/C25:/root/Datasets/EuRoC";; esac
  T=${P}_$2_$1; mkdir -p $W/output/$T/$3
  Y=EuRoC_docker.yaml; [ $1 = g10 ] && Y=EuRoC_G10_docker.yaml
  [ $1 = g10 ] && AB=0000 || AB=1000
  docker run $BASE $D -e YAML=$Y -e ABLATION=$AB -e CAND_LOG=/root/OutputTest/$T/$3/cand_$4.txt -v $W/nvcache_ours:/root/.nv -v $W/code_v3_L4:/root/Jetson-SLAM --entrypoint /bin/bash $IMG /root/scripts/run_euroc_idx.sh $T $4 $3
}
SC="S0:MH_01_easy S0:MH_02_easy S0:MH_03_medium S0:MH_04_difficult S0:MH_05_difficult S0:V2_03_difficult S0:V2_02_medium C25:MH_03_medium C25:V2_02_medium"
[ "$P" = "L4smoke" ] && SC="S0:MH_03_medium"
echo "[$(date +%F' '%T)] ${P} 開始" | tee -a $LOG
for s in $SC; do C=${s%%:*}; SEQ=${s##*:}
  for i in $(seq 1 $N); do for M in final g10; do run $M $C $SEQ $i >> $LOG 2>&1; done; done
  echo "[$(date +%F' '%T)] ${P} $C $SEQ 完成" | tee -a $LOG
done
echo "L4_DONE $(date +%F' '%T)" | tee -a $LOG
