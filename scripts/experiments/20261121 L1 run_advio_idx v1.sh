#!/bin/bash
# [20261121] 在容器內執行 ADVIO 單次 run（指定 run 編號，供逐次交替）；其餘同 run_advio.sh
# 用法：bash run_advio_idx.sh <輸出資料夾名> <run 編號> <序列>   環境變數：BIN、YAML、CSV
set -u
export USE_VIEWER=0
TAG=$1; i=$2; SEQ=$3
J=/root/Jetson-SLAM
OUT=/root/OutputTest/$TAG/$SEQ
mkdir -p "$OUT"; cd "$OUT"
echo "[$(date +%H:%M:%S)] $TAG $SEQ run $i"
rm -f KeyFrameTrajectory.txt CameraTrajectory.txt
for attempt in 1 2 3; do
  FRAME_TIMES="$OUT/frametimes_$i.txt" "$J/$BIN" "$J/Vocabulary/ORBvoc.txt" "$J/$YAML" \
    "/root/Datasets/ADVIO/$SEQ/iphone/img360" "/root/Datasets/ADVIO/$SEQ/iphone/$CSV" > "run_$i.log" 2>&1
  rc=$?
  if [ $rc -ne 0 ] && ! grep -q "Start processing" "run_$i.log"; then
    echo "  [環境 crash，未開始處理] exit=$rc attempt=$attempt → 重試" | tee -a "$OUT/init_crash.log"; sleep 10; continue
  fi
  break
done
echo "  exit=$rc"
if [ -f KeyFrameTrajectory.txt ]; then mv KeyFrameTrajectory.txt "KeyFrameTrajectory_$i.txt"; else echo "  (no trajectory)"; fi
[ -f CameraTrajectory.txt ] && mv CameraTrajectory.txt "CameraTrajectory_$i.txt"
grep -iE "median tracking time|mean tracking time" "run_$i.log" | sed 's/^/  /'
