#!/bin/bash
# [20261118] 在容器內執行單次 run，可指定 run 編號（為了逐次交替）；其餘同 run_euroc.sh
# 用法：bash run_euroc_idx.sh <輸出資料夾名> <run 編號> <序列>
# 放到 ~/slam_work/scripts/run_euroc_idx.sh
set -u
export USE_VIEWER=0
TAG=$1; i=$2; SEQ=$3
YAML=${YAML:-EuRoC_docker.yaml}
J=/root/Jetson-SLAM
case "$SEQ" in
  MH_0*) SHORT="MH0${SEQ:4:1}";;
  V1_0*) SHORT="V10${SEQ:4:1}";;
  V2_0*) SHORT="V20${SEQ:4:1}";;
esac
OUT=/root/OutputTest/$TAG/$SEQ
mkdir -p "$OUT"
echo "[$(date +%H:%M:%S)] $TAG $SEQ run $i"
cd "$OUT"
rm -f KeyFrameTrajectory.txt
for attempt in 1 2 3; do
  FRAME_TIMES="$OUT/frametimes_$i.txt" "$J/Examples/Monocular/mono_euroc" "$J/Vocabulary/ORBvoc.txt" "$J/Examples/Monocular/$YAML" \
    "/root/Datasets/EuRoC/$SEQ/${IMGSUB-mav0/cam0/data}" "$J/Examples/Monocular/EuRoC_TimeStamps/$SHORT.txt" \
    > "run_$i.log" 2>&1
  rc=$?
  if [ $rc -ne 0 ] && ! grep -q "Start processing" "run_$i.log"; then
    echo "  [環境 crash，未開始處理] exit=$rc attempt=$attempt → 重試" | tee -a "$OUT/init_crash.log"; sleep 10; continue
  fi
  break
done
echo "  exit=$rc"
if [ -f KeyFrameTrajectory.txt ]; then mv KeyFrameTrajectory.txt "KeyFrameTrajectory_$i.txt"; else echo "  (no trajectory)"; fi
grep -iE "median tracking time|mean tracking time|Peak Memory" "run_$i.log" | sed 's/^/  /'
