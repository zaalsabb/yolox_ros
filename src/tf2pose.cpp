#include <ros/ros.h>
#include <tf2_ros/transform_listener.h>
#include <geometry_msgs/TransformStamped.h>
#include <geometry_msgs/PoseStamped.h>

int main(int argc, char** argv){
  ros::init(argc, argv, "tf2pose");

  ros::NodeHandle node;

  ros::Publisher pub =
    node.advertise<geometry_msgs::PoseStamped>("/pose", 60);

  std::string frame_id, map_id;
  node.getParam("frame_id", frame_id);  
  node.getParam("map_id", map_id);  

  tf2_ros::Buffer tfBuffer;
  tf2_ros::TransformListener tfListener(tfBuffer);

  ros::Rate rate(60.0);
  while (node.ok()){
    geometry_msgs::TransformStamped transformStamped;
    try{
      transformStamped = tfBuffer.lookupTransform(frame_id, map_id,
                               ros::Time(0));
    }
    catch (tf2::TransformException &ex) {
      ROS_WARN("%s",ex.what());
      ros::Duration(1.0).sleep();
      continue;
    }

    geometry_msgs::PoseStamped pose_msg;

    pose_msg.pose.position.x = transformStamped.transform.translation.x;
    pose_msg.pose.position.y = transformStamped.transform.translation.y;
    pose_msg.pose.position.z = transformStamped.transform.translation.z;
    pose_msg.pose.orientation.x = transformStamped.transform.rotation.x;
    pose_msg.pose.orientation.y = transformStamped.transform.rotation.y;
    pose_msg.pose.orientation.z = transformStamped.transform.rotation.z;
    pose_msg.pose.orientation.w = transformStamped.transform.rotation.w;

    pub.publish(pose_msg);

    rate.sleep();
  }
  return 0;
};
