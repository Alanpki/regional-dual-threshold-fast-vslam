#!/bin/bash
# M1 正式：MH01–05 × 5 次，兩方法；JIT 快取持久化（已預熱）；不限定 CPU；輸出每幀追蹤時間
IMG=jetson-slam:cuda11.6.1-cudnn8-devel-ubuntu20.04-cv4.10.0-pangolin0.6-eigen3.4.0
W=$HOME/slam_work
run() { # $1 code dir, $2 tag, $3 yaml, $4 cache dir
  docker run --rm --gpus all --ipc=host -e NVIDIA_DRIVER_CAPABILITIES=all -e YAML=$3 \
    -e CUDA_CACHE_MAXSIZE=4294967296 -v $4:/root/.nv \
    -v $1:/root/Jetson-SLAM -v $W/datasets/EuRoC:/root/Datasets/EuRoC \
    -v $W/output:/root/OutputTest -v $W/scripts:/root/scripts \
    --entrypoint /bin/bash $IMG /root/scripts/run_euroc.sh $2 5 MH_01_easy MH_02_easy MH_03_medium MH_04_difficult MH_05_difficult
}
run $W/code M1_ours EuRoC_docker.yaml $W/nvcache_ours
run $W/baselines/Jetson-SLAM-master M1_baseline EuRoC_baseline_docker.yaml $W/nvcache_base
echo ALLDONE
