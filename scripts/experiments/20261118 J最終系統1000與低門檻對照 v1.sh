#!/bin/bash
# [20261118] J：事先登記 20261118 事先登記 最終系統1000與低門檻對照 v1.md
#  J1 最終系統 1000 vs 0000：MH01–05、V203 × 5，逐次交替（code_v3）
#  J2 低對比對照：S0、C35、C25 × MH03、V202 × 5，方法 ours1000（雙門檻）、oursG10（全域單一門檻 10）、ORB-SLAM3，逐次交替
# 準備（Ubuntu，執行前一次）：
#   cp "<NAS>/4 第二次投稿/1 實驗腳本/20261118 J run_euroc_idx v1.sh" ~/slam_work/scripts/run_euroc_idx.sh
#   cp "<NAS>/4 第二次投稿/1 實驗腳本/20261118 J EuRoC_G10_docker v1.yaml" ~/slam_work/code_v3/Examples/Monocular/EuRoC_G10_docker.yaml
#   ORB-SLAM3 容器裡沒有 run_euroc_idx.sh 以外的需求（同一個 scripts 掛載）
# 執行期間：這台電腦不要做其他工作（含備份、安裝套件、轉檔）
IMG=jetson-slam:cuda11.6.1-cudnn8-devel-ubuntu20.04-cv4.10.0-pangolin0.6-eigen3.4.0
IMG3=jetson-slam-orb3:v1
W=$HOME/slam_work
LOG=$W/output/m16_J_progress.log
BASE="--rm --gpus all --ipc=host -e NVIDIA_DRIVER_CAPABILITIES=all -e CUDA_CACHE_MAXSIZE=4294967296 -v $W/output:/root/OutputTest -v $W/scripts:/root/scripts"
data() { case $1 in S0) echo "-v $W/datasets/EuRoC:/root/Datasets/EuRoC";; C*) echo "-v $W/datasets/EuRoC_deg/$1:/root/Datasets/EuRoC";; esac; }
run() { # $1 方法 $2 條件 $3 序列 $4 run 編號 $5 輸出名
  case $1 in
    ours1000) docker run $BASE $(data $2) -e YAML=EuRoC_docker.yaml     -e ABLATION=1000 -v $W/nvcache_ours:/root/.nv -v $W/code_v3:/root/Jetson-SLAM --entrypoint /bin/bash $IMG /root/scripts/run_euroc_idx.sh $5 $4 $3;;
    ours0000) docker run $BASE $(data $2) -e YAML=EuRoC_docker.yaml     -e ABLATION=0000 -v $W/nvcache_ours:/root/.nv -v $W/code_v3:/root/Jetson-SLAM --entrypoint /bin/bash $IMG /root/scripts/run_euroc_idx.sh $5 $4 $3;;
    oursG10)  docker run $BASE $(data $2) -e YAML=EuRoC_G10_docker.yaml -e ABLATION=0000 -v $W/nvcache_ours:/root/.nv -v $W/code_v3:/root/Jetson-SLAM --entrypoint /bin/bash $IMG /root/scripts/run_euroc_idx.sh $5 $4 $3;;
    orb3)     docker run $BASE $(data $2) -e YAML=EuRoC.yaml -e IMGSUB= -v $W/baselines/ORB_SLAM3:/root/Jetson-SLAM -v $W/baselines/ORB_SLAM3:/root/ORB_SLAM3 --entrypoint /bin/bash $IMG3 /root/scripts/run_euroc_idx.sh $5 $4 $3;;
  esac
}
echo "[$(date +%H:%M:%S)] J 開始" | tee -a $LOG
# J1：每個序列 run i：1000 → 0000（逐次交替）
for SEQ in MH_01_easy MH_02_easy MH_03_medium MH_04_difficult MH_05_difficult V2_03_difficult; do
  for i in 1 2 3 4 5; do
    run ours1000 S0 $SEQ $i J1_1000 >> $LOG 2>&1
    run ours0000 S0 $SEQ $i J1_0000 >> $LOG 2>&1
  done
  echo "[$(date +%H:%M:%S)] J1 $SEQ 完成" | tee -a $LOG
done
# J2：每個條件、序列、run i：1000 → G10 → ORB-SLAM3
for C in C25 C35 S0; do
  for SEQ in MH_03_medium V2_02_medium; do
    for i in 1 2 3 4 5; do
      for M in ours1000 oursG10 orb3; do run $M $C $SEQ $i J2_${C}_$M >> $LOG 2>&1; done
    done
    echo "[$(date +%H:%M:%S)] J2 $C $SEQ 完成" | tee -a $LOG
  done
done
echo "J_DONE $(date +%H:%M:%S)" | tee -a $LOG
