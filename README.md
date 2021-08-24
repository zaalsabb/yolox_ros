# yolox_ros

ROS wrapper for [YOLOX](https://github.com/MACILLAS/YOLOX) for spalling detection.

## Installation
This package uses python 3, so you must create a separate catkin workspace that's compatible with python 3 if your ROS version still uses python 2. (Based on this guide [How to setup ROS with Python 3](https://medium.com/@beta_b0t/how-to-setup-ros-with-python-3-44a69ca36674))

1) Install these dependencies:

```bash
sudo apt-get install python3-pip python3-yaml python-catkin-tools python3-dev python3-numpy
sudo pip3 install rospkg catkin_pkg
```

2) recursively clone this repository to your src folder in your new catkin workspace.

``` bash
mkdir ~/catkin3_ws && cd ~/catkin3_ws
mkdir src && cd src
git clone --recursive https://github.com/zaalsabb/yolox_ros.git
```

3) Install all the requirement for the [YOLOX](https://github.com/MACILLAS/YOLOX) repository.

``` bash
cd ~/catkin3_ws/src/yolox_ros/src/YOLOX
pip3 install -r requirements.txt
```

4) Configure and build the catkin workspace using catkin_make.

``` bash
catkin config -DPYTHON_EXECUTABLE=/usr/bin/python3 -DPYTHON_INCLUDE_DIR=/usr/include/python3.6m -DPYTHON_LIBRARY=/usr/lib/x86_64-linux-gnu/libpython3.6m.so
catkin config --install
catkin_make
```

Note: you need to run the setup file each time you open a new terminal, since yolox_ros cannot be started from a launch file in other catkin workspaces.
``` bash
cd ~/catkin3_ws
source devel/setup.sh
```

## Usage

```bash
roslaunch yolox_ros yolox.launch
```

### Topics and Parameters:

| Topic | Type | Description |
|---|---|---|
| /image | sensor_msgs/Image | Input image. |
| /bounding_boxes | BoundingBoxes | The set of bounding boxes of objects detected in the image. See [darknet_ros_msgs](https://github.com/leggedrobotics/darknet_ros/tree/master/darknet_ros_msgs/msg) |
| /debug/image | sensor_msgs/Image | Image with all the detected bounding boxes. Used for debugging. |

| Parameter | Description |
|---|---|
| ~model | The absolute path to the ONNX model used for detection. |

