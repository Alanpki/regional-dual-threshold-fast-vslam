#!/bin/bash
# [20261122] B：事先登記 20261122 事先登記 Jetson-SLAM初始化門檻對齊重跑 v1.md
#  JM＝baselines/Jetson-SLAM-matched（釋出版 Jetson-SLAM＋與 code_v3 相同的三處單眼初始化門檻）
# 用法：bash m21_B.sh <前綴> [批...]   批：B1 B2 B3 B4 B5 B6（預設全部依序）；冒煙：bash m21_B.sh Bsmoke SMOKE
IMG=jetson-slam:cuda11.6.1-cudnn8-devel-ubuntu20.04-cv4.10.0-pangolin0.6-eigen3.4.0
W=$HOME/slam_work; P=${1:-B}; shift; BATCHES=${@:-B1 B2 B3 B4 B5 B6}
LOG=$W/output/${P}_progress.log
CODE=$W/baselines/Jetson-SLAM-matched
BASE="--rm --gpus all --ipc=host -e NVIDIA_DRIVER_CAPABILITIES=all -e CUDA_CACHE_MAXSIZE=4294967296 -v $W/output:/root/OutputTest -v $W/scripts:/root/scripts"
euroc() { # $1 資料根目錄 $2 yaml $3 序列 $4 run $5 輸出名
  docker run $BASE -v $1:/root/Datasets/EuRoC -e YAML=$2 -v $W/nvcache_base:/root/.nv -v $CODE:/root/Jetson-SLAM --entrypoint /bin/bash $IMG /root/scripts/run_euroc_idx.sh $5 $4 $3
}
advio() { # $1 序列 $2 run $3 輸出名
  docker run $BASE -v $W/datasets/ADVIO:/root/Datasets/ADVIO -e CSV=frames.csv -e BIN=Examples/Monocular/mono_advio -e YAML=Examples/Monocular/ADVIO_baseline_B.yaml -v $W/nvcache_base:/root/.nv -v $CODE:/root/Jetson-SLAM --entrypoint /bin/bash $IMG /root/scripts/run_advio_idx.sh $3 $2 $1
}
sample() { while true; do t=$(date +%s.%N); apps=$(nvidia-smi --query-compute-apps=pid,used_memory --format=csv,noheader,nounits)
  for p in $(pgrep -x mono_euroc); do r=$(awk '/VmRSS/{print $2}' /proc/$p/status 2>/dev/null); h=$(awk '/VmHWM/{print $2}' /proc/$p/status 2>/dev/null); g=$(echo "$apps" | awk -F', ' -v p=$p '$1==p{print $2}'); echo "$t $p ${r:-0} ${h:-0} ${g:-0}"; done; sleep 0.1; done >> "$1"; }
DEG=$W/datasets/EuRoC_deg
echo "[$(date +%F' '%T)] ${P} 開始 批：$BATCHES" | tee -a $LOG
for B in $BATCHES; do
  case $B in
   SMOKE)
     euroc $DEG/C25 EuRoC_baseline_docker.yaml V2_02_medium 1 ${P}_C25 >> $LOG 2>&1
     advio advio-02 1 ${P}_ADVIO >> $LOG 2>&1
     euroc $W/datasets/EuRoC EuRoC_baseline_docker.yaml MH_04_difficult 1 ${P}_MH04 >> $LOG 2>&1;;
   B1) for C in S0 C50 C35 C25; do R=$DEG/$C; [ $C = S0 ] && R=$W/datasets/EuRoC
         for S in MH_03_medium V2_02_medium; do for i in 1 2 3 4 5; do euroc $R EuRoC_baseline_docker.yaml $S $i JM_B1_$C >> $LOG 2>&1; done
           echo "[$(date +%F' '%T)] B1 $C $S 完成" | tee -a $LOG; done; done;;
   B2) for S in advio-01 advio-02 advio-03 advio-04; do for i in 1 2 3 4 5; do advio $S $i JM_B2 >> $LOG 2>&1; done
         echo "[$(date +%F' '%T)] B2 $S 完成" | tee -a $LOG; done;;
   B3) for S in MH_01_easy MH_02_easy MH_03_medium MH_04_difficult MH_05_difficult; do for i in 1 2 3 4 5; do euroc $W/datasets/EuRoC EuRoC_baseline_docker.yaml $S $i JM_B3 >> $LOG 2>&1; done
         echo "[$(date +%F' '%T)] B3 $S 完成" | tee -a $LOG; done;;
   B4) for C in RC RB; do for S in MH_03_medium V2_02_medium; do for i in 1 2 3 4 5; do euroc $DEG/$C EuRoC_baseline_docker.yaml $S $i JM_B4_$C >> $LOG 2>&1; done
         echo "[$(date +%F' '%T)] B4 $C $S 完成" | tee -a $LOG; done; done;;
   B5) mkdir -p $W/output/JM_B5_MEM; for i in 1 2 3 4 5; do F=$W/output/JM_B5_MEM/mem_$i.txt; rm -f $F; sample $F & SP=$!
         euroc $W/datasets/EuRoC EuRoC_baseline_docker.yaml MH_04_difficult $i JM_B5 >> $LOG 2>&1; kill $SP; wait $SP 2>/dev/null; done
       echo "[$(date +%F' '%T)] B5 完成" | tee -a $LOG;;
   B6) for S in MH_01_easy MH_03_medium V2_02_medium; do R=$W/datasets/EuRoC; for i in 1 2 3; do euroc $R EuRoC_baseline_t48_docker.yaml $S $i JM_B6 >> $LOG 2>&1; done
         echo "[$(date +%F' '%T)] B6 $S 完成" | tee -a $LOG; done;;
  esac
done
echo "B_DONE $(date +%F' '%T)" | tee -a $LOG
