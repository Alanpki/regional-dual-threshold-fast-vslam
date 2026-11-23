# Set X display variables
set_x_display() {
    MOUNT_X="-e DISPLAY=$DISPLAY -v /tmp/.X11-unix/:/tmp/.X11-unix"
    xhost +local:root >/dev/null
}

# Set Container WorkSpace
WS=/root

# Main script execution
main() {
    set_x_display

    # Launch the container
    set -x
    docker run \
        -it \
        --rm \
        --net=host \
        --ipc=host \
        --pid=host \
        --gpus all \
        ${MOUNT_X} \
        -e NVIDIA_DRIVER_CAPABILITIES=all \
        -e NVIDIA_VISIBLE_DEVICES=all \
        -e=DISPLAY \
        -e=ROS_DOMAIN_ID \
        -v ~/SLAM/Jetson-SLAM:/root/Jetson-SLAM \
        -v ~/Datasets:/root/Datasets \
        -v ~/OutputTest:/root/OutputTest \
        --entrypoint /bin/bash \
        jetson-slam:cuda11.6.1-cudnn8-devel-ubuntu20.04-cv4.10.0-pangolin0.6-eigen3.4.0
}

# Execute the main script
main

# 本機跟容器使用圖應個使用者
        # -e HOME=$HOME \
        # -v $HOME/SLAM/Jetson-SLAM:$HOME/Jetson-SLAM \
        # -v $HOME/Datasets:$HOME/Datasets \
        # -v $HOME/OutputTest/MatchFrame:$HOME/OutputTest/MatchFrame \
        # -v /etc/passwd:/etc/passwd:ro \
        # -v /etc/group:/etc/group:ro \
        # --user $(id -u):$(id -g) \



        # -v $HOME/SLAM/Jetson-SLAM:$WS/Jetson-SLAM \
        # -v $HOME/Datasets:$WS/Datasets \
        # -v $HOME/OutputTest/MatchFrame:$WS/OutputTest/MatchFrame \
