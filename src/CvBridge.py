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
from sensor_msgs.msg import Image, CompressedImage
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

def cv2_to_compressed_imgmsg(cvim, dst_format = "jpg"):

    import cv2
    import numpy as np
    if not isinstance(cvim, (np.ndarray, np.generic)):
        raise TypeError('Your input type is not a numpy array')
    cmprs_img_msg = CompressedImage()
    cmprs_img_msg.format = dst_format
    ext_format = '.' + dst_format
    try:
        cmprs_img_msg.data = np.array(cv2.imencode(ext_format, cvim)[1]).tostring()
    except cv2.error as e:
        print("Could not compress image: {}".format(e))
        return None

    return cmprs_img_msg

def compressed_imgmsg_to_cv2(cmprs_img_msg, desired_encoding = "passthrough"):

    import cv2
    import numpy as np

    str_msg = cmprs_img_msg.data
    buf = np.ndarray(shape=(1, len(str_msg)),
                        dtype=np.uint8, buffer=cmprs_img_msg.data)
    im = cv2.imdecode(buf, cv2.IMREAD_UNCHANGED)

    if desired_encoding == "passthrough":
        return im
    elif desired_encoding == "rgb8":
        return cv2.cvtColor(im, cv2.COLOR_BGR2RGB)
    try:
        res = cv2.cvtColor(im, "bgr8", desired_encoding)
    except RuntimeError as e:
        print("Could not convert image: {}".format(e))
        return None

    return res    