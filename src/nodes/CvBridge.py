"""
    Source: Answer from https://answers.ros.org/question/350904/cv_bridge-throws-boost-import-error-in-python-3-and-ros-melodic/

    Provides conversions between OpenCV and ROS image formats in a hard-coded way.  
    CV_Bridge, the module usually responsible for doing this, is not compatible with Python 3,
     - the language this all is written in.  So we create this module, and all is... well, all is not well,
     - but all works.  :-/
"""
import sys
import numpy as np
import rospy
from sensor_msgs.msg import Image
import cv2

def imgmsg_to_cv2(img_msg):
    if img_msg.encoding != "rgb8" and img_msg.encoding != "bgr8":
        rospy.logerr("This Coral detect node has been hardcoded to the 'rgb8/bgr8' encoding.  Come change the code if you're actually trying to implement a new camera")
    dtype = np.dtype("uint8") # Hardcode to 8 bits...
    dtype = dtype.newbyteorder('>' if img_msg.is_bigendian else '<')
    image_opencv = np.ndarray(shape=(img_msg.height, img_msg.width, 3), # and three channels of data.
                    dtype=dtype, buffer=img_msg.data)

    if img_msg.encoding == "rgb8": # Since OpenCV works with bgr natively, we don't need to reorder the channels unless it is rgb.
        image_opencv = cv2.cvtColor(image_opencv, cv2.COLOR_RGB2BGR)
    # If the byt order is different between the message and the system.
    if img_msg.is_bigendian == (sys.byteorder == 'little'):
        image_opencv = image_opencv.byteswap().newbyteorder()
    return image_opencv

def cv2_to_imgmsg(cv_image,encoding="bgr8"):
    img_msg = Image()
    img_msg.height = cv_image.shape[0]
    img_msg.width = cv_image.shape[1]
    img_msg.encoding = encoding
    img_msg.is_bigendian = 0
    img_msg.data = cv_image.tostring()
    img_msg.step = len(img_msg.data) // img_msg.height # That double line is actually integer division, not a comment
    return img_msg