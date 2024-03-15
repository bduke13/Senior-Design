FROM ros:humble-ros-core

# Install build essentials and other dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    cmake \
    python3-colcon-common-extensions \
    python3-rosdep \
    python3-pip \
    git \
    sudo \
    && rm -rf /var/lib/apt/lists/*

RUN apt-get update && apt-get install -y \
    ros-humble-rqt \
    ros-humble-rqt-common-plugins \
    && rm -rf /var/lib/apt/lists/*

RUN pip3 install numpy pandas matplotlib

# Create a symbolic link for python to python3
RUN ln -s /usr/bin/python3 /usr/bin/python

# Initialize rosdep
RUN rosdep init && rosdep update

# Add vscode user with same UID and GID as your host system
ARG USERNAME=vscode
ARG USER_UID=1000
ARG USER_GID=$USER_UID
RUN groupadd --gid $USER_GID $USERNAME \
    && useradd -s /bin/bash --uid $USER_UID --gid $USER_GID -m $USERNAME \
    && echo $USERNAME ALL=\(root\) NOPASSWD:ALL > /etc/sudoers.d/$USERNAME \
    && chmod 0440 /etc/sudoers.d/$USERNAME

# Switch from root to user
#USER $USERNAME

# Add user to video group to allow access to webcam
RUN sudo usermod --append --groups video $USERNAME

# Set ROS environment variables
RUN echo "source /opt/ros/${ROS_DISTRO}/setup.bash" >> ~/.bashrc

# Prepare the ROS workspace
WORKDIR /home/${USERNAME}/ros2_ws/src

# Clone the sllidar_ros2 repository
RUN git clone https://github.com/Slamtec/sllidar_ros2.git

# Move back to the workspace root
WORKDIR /home/${USERNAME}/ros2_ws

# Install ROS package dependencies with rosdep
RUN sudo rosdep install --from-paths src --ignore-src -r -y

# Build the ROS package
RUN /bin/bash -c '. /opt/ros/humble/setup.bash; colcon build --symlink-install'

# Source the workspace in bashrc
RUN echo "source /home/${USERNAME}/ros2_ws/install/setup.bash" >> ~/.bashrc

RUN sudo apt update
RUN sudo apt install -y ros-humble-rviz2
RUN sudo apt install -y ros-humble-teleop-twist-keyboard
RUN sudo apt install -y ros-humble-irobot-create-msgs
