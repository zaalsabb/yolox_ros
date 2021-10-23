# yolox_ros

ROS wrapper for [YOLOX](https://github.com/MACILLAS/YOLOX) for spalling detection.

## Installation
This package uses python 3, so you must create a separate catkin workspace that's compatible with python 3 if your ROS version still uses python 2. (Based on this guide [How to setup ROS with Python 3](https://medium.com/@beta_b0t/how-to-setup-ros-with-python-3-44a69ca36674))

1) Install these dependencies:

First, let's install some tools we’ll need for the build process
```bash
sudo apt-get install python3-pip python3-yaml python-catkin-tools python3-dev python3-numpy python3-catkin-pkg-modules python3-rospkg-modules
sudo pip3 install rospkg catkin_pkg
```
Now, create new catkin_ws_py3 to avoid any future problems with catkin_make(assuming you are using it) and config catkin to use your python 3(3.6 in my case) when building packages:
```bash
mkdir ~/catkin_ws_py3 && cd ~/catkin_ws_py3 && mkdir src
catkin config -DPYTHON_EXECUTABLE=/usr/bin/python3 -DPYTHON_INCLUDE_DIR=/usr/include/python3.6m -DPYTHON_LIBRARY=/usr/lib/x86_64-linux-gnu/libpython3.6m.so
catkin config --install
```
clone official vision_opencv repo:
```bash
cd ~/catkin_ws_py3
cd src
git clone -b melodic https://github.com/ros-perception/vision_opencv.git
```
Build tf2_ros using Python3
```bash
wstool init
wstool set -y src/geometry2 --git https://github.com/ros/geometry2 -v 0.6.5
wstool up
rosdep install --from-paths src --ignore-src -y -r
```

2) recursively clone the [YOLOX ROS wrapper repository](https://github.com/zaalsabb/yolox_ros) to your src folder in your new catkin workspace.

```bash
cd ~/catkin_ws_py3/src
git clone --recursive https://github.com/zaalsabb/yolox_ros.git
```
3) Install all the requirement for the [YOLOX](https://github.com/zaalsabb/YOLOX) repository.

```bash
cd ~/catkin_ws_py3/src/yolox_ros/src/YOLOX
pip3 install -r requirements.txt
```

4) Configure and build the catkin workspace using catkin_make.

```bash
cd ~/catkin_ws_py3
catkin_make --cmake-args \
            -DCMAKE_BUILD_TYPE=Release \
            -DPYTHON_EXECUTABLE=/usr/bin/python3 \
            -DPYTHON_INCLUDE_DIR=/usr/include/python3.6m \
            -DPYTHON_LIBRARY=/usr/lib/x86_64-linux-gnu/libpython3.6m.so
catkin config --install
catkin_make
```
Note: you need to run the setup file each time you open a new terminal, since yolox_ros cannot be started from a launch file in other catkin workspaces.
```bash
cd ~/catkin_ws_py3
source devel/setup.sh
```
Edit the ```~/.bashrc``` file and add the following lines (replace melodic with kinetic if you are on Ubuntu 16):
```bash
source /opt/ros/melodic/setup.bash
source ~/catkin_ws_py3/devel/setup.bash
```

#### Usage
Open a new terminal, then enter the following:
```bash
source ~/catkin_ws_py3/devel/setup.sh
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

