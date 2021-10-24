#!/usr/bin/env python3
import os
import sys
import rospy
import cv2
import CvBridge
from sensor_msgs.msg import Image, CameraInfo, CompressedImage
from std_msgs.msg import Empty
import message_filters
import numpy as np
import pandas as pd
import tf2_ros

YOLOX_module=os.path.join(os.path.dirname(os.path.realpath(__file__)),'YOLOX')
sys.path.insert(0,YOLOX_module)

from YOLOX import detspall

from yolox_ros.msg import BoundingBox, SegmentImage

class Node:

    def __init__(self):

        rospy.init_node('yolox')
        self.tfBuffer = tf2_ros.Buffer()        
        ls = tf2_ros.TransformListener(self.tfBuffer)
        
        self.model = rospy.get_param('~model', default=os.path.join(YOLOX_module,'yolox.onnx'))       
        self.segment_rate = rospy.get_param('~segment_rate', default=1) 
        self.num_images = rospy.get_param('~num_images', default=8)         
        self.map_frame_id = rospy.get_param('~map_frame_id', default='map')       

        sub1=message_filters.Subscriber('/image', Image)
        sub2=message_filters.Subscriber('/camera_info', CameraInfo)

        self.pub1 = rospy.Publisher('/bounding_box',BoundingBox,queue_size=1)
        self.pub2 = rospy.Publisher('/debug/image',Image,queue_size=1)
        self.pub3 = rospy.Publisher('/segment_image',SegmentImage,queue_size=1)
        self.pub4 = rospy.Publisher('/segment_req',Empty,queue_size=1)

        ts = message_filters.ApproximateTimeSynchronizer([sub1, sub2], 1, 0.5) 
        ts.registerCallback(self.callback)

        self.spallDetector = detspall(model=self.model)
        # cache dataframe
        # self.bb_df = pd.DataFrame(columns=['file', 'x1', 'y1', 'x2', 'y2', 'track'])
        self.lastTrackIdx = 0
        self.i = 0
        self.t1 = rospy.get_time()
        self.t2 = None
        self.new_object_thres = 0.5
        self.current_object_thres = self.new_object_thres

        rospy.spin()

    def callback(self,*args):
        img = CvBridge.imgmsg_to_cv2(args[0])
        caminfo = args[1]
        bbsTrack,conf = self.spallDetector.detAndTrack(img)
        
        if bbsTrack.size > 0 and conf > self.current_object_thres: #check if returned a bounding box            
            bbsTrack = bbsTrack[0]
            bb_msg = self.sendBoundingBoxMsg(bbsTrack,conf,caminfo)
            if bbsTrack[4] == self.lastTrackIdx: #still tracking same object
                self.current_object_thres = 0
                self.i += 1
                self.t2 = rospy.get_time()
                if (self.t2-self.t1) > 1/self.segment_rate:
                    self.sendSegmentImg(img,bb_msg,self.i,caminfo.header.frame_id)
                    self.t1 = self.t2
                                    
            else: #starting to track the next object (TRIGGER)
                self.current_object_thres = self.new_object_thres
                self.lastTrackIdx = bbsTrack[4]
                # start segmentation
                self.pub4.publish(Empty())
            
            self.sendDebugImg(img,bbsTrack)     


    def sendSegmentImg(self,img,bb_msg,i,camera_frame_id):

        msg = SegmentImage()
        try:
            trans = self.tfBuffer.lookup_transform(self.map_frame_id, camera_frame_id, bb_msg.header.stamp)
        except (tf2_ros.LookupException, tf2_ros.ConnectivityException, tf2_ros.ExtrapolationException):
            return 

        img_msg = CvBridge.cv2_to_compressed_imgmsg(img)
        msg.header.stamp = bb_msg.header.stamp
        msg.header.frame_id = camera_frame_id
        msg.image = img_msg
        msg.bbox = bb_msg
        msg.transform = trans.transform
        self.pub3.publish(msg) 

    def bb_to_msg(self,bbsTrack,conf,caminfo):
        bb = BoundingBox()
        bb.header.stamp = caminfo.header.stamp
        bb.header.frame_id = self.map_frame_id
        bb.camera_info = caminfo
        bb.x1 = int(bbsTrack[0])
        bb.y1 = int(bbsTrack[1])
        bb.x2 = int(bbsTrack[2])
        bb.y2 = int(bbsTrack[3])
        bb.confidence = conf
        bb.track = int(bbsTrack[4])
        bb.Class = "spalling"
        return bb        
 
    def sendBoundingBoxMsg(self,bbsTrack,conf,caminfo):
        bb=self.bb_to_msg(bbsTrack,conf,caminfo)
        self.pub1.publish(bb)
        return bb

    def sendDebugImg(self,img,bbsTrack):
        if bbsTrack[4] % 2 == 0:
            color = (255, 0, 0)
        else:
            color = (0, 0, 255)
        img = cv2.rectangle(img, (int(bbsTrack[0]),int(bbsTrack[1])), (int(bbsTrack[2]),int(bbsTrack[3])), color, 3)        
        msg = CvBridge.cv2_to_imgmsg(img)
        self.pub2.publish(msg)      


if __name__ == '__main__':
    Node() 

    
       
