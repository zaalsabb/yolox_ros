cmake_minimum_required(VERSION 3.0.2)
project(yolox_ros)

find_package(catkin REQUIRED COMPONENTS
  roscpp
  rospy
  sensor_msgs
  std_msgs
  message_generation
  tf2
  tf2_ros
  geometry_msgs
)

add_message_files(
   FILES
   BoundingBox.msg
   BoundingBoxes2D.msg
)

## Generate services in the 'srv' folder
# add_service_files(
#   FILES
#   Service1.srv
#   Service2.srv
# )

## Generate actions in the 'action' folder
# add_action_files(
#   FILES
#   Action1.action
#   Action2.action
# )

## Generate added messages and services with any dependencies listed here
generate_messages(
   DEPENDENCIES
   std_msgs
   sensor_msgs
 )


catkin_package(
   CATKIN_DEPENDS roscpp rospy std_msgs geometry_msgs sensor_msgs tf2_ros message_runtime
)


#install(FILES
#  yolox3D.launch
#  DESTINATION ${CATKIN_PACKAGE_SHARE_DESTINATION}
#)


