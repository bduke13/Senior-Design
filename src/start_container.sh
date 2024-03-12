#documentation found at https://github.com/iRobotEducation/create3-docker/tree/main
#runs an example ros2 humble docker container. See the .devconatiner/Dockerfile for what
#   container we are running and .devcontainer/devcontainer.json to see the launch arguments

docker run -it --net=host --device=/dev/ttyUSB1 ros:humble-ros-core

