#!/usr/bin/env python3
import os
import sys
import rospy
import cv2
from CvBridge import imgmsg_to_cv2, cv2_to_imgmsg
from sensor_msgs.msg import Image, CameraInfo
import message_filters
import numpy as np
import pandas as pd


YOLOX_module=os.path.join(os.path.dirname(os.path.realpath(__file__)),'YOLOX')
sys.path.insert(0,YOLOX_module)

from YOLOX import detspall

from yolox_ros.msg import BoundingBox, BoundingBoxes2D

class Node:

    def __init__(self):

        rospy.init_node('yolox')
        self.model = rospy.get_param('~model', default=os.path.join(YOLOX_module,'yolox.onnx'))       
        self.datadir = rospy.get_param('~datadir', default=os.path.join(YOLOX_module,'datasets/tmp')) 
        self.num_images = rospy.get_param('~num_images', default=8)         

        sub1=message_filters.Subscriber('/image', Image)
        sub2=message_filters.Subscriber('/camera_info', CameraInfo)

        self.pub1 = rospy.Publisher('/bounding_box',BoundingBox,queue_size=1)
        self.pub2 = rospy.Publisher('/debug/image',Image,queue_size=1)

        ts = message_filters.ApproximateTimeSynchronizer([sub1, sub2], 1, 0.5) 
        ts.registerCallback(self.callback)

        self.spallDetector = detspall(model=self.model)
        # cache dataframe
        self.bb_df = pd.DataFrame(columns=['file', 'x1', 'y1', 'x2', 'y2', 'track'])
        self.lastTrackIdx = 0
        self.i = 0
        self.new_object_thres = 0.5
        self.current_object_thres = self.new_object_thres

        rospy.spin()

    def callback(self,*args):
        img = imgmsg_to_cv2(args[0])
        caminfo = args[1]
        bbsTrack,conf = self.spallDetector.detAndTrack(img)

        if bbsTrack.size > 0 and conf > self.current_object_thres: #check if returned a bounding box            
            bbsTrack = bbsTrack[0]
            #print(bbsTrack)
            if bbsTrack[4] == self.lastTrackIdx: #still tracking same object
                self.current_object_thres = 0
                imgFile = str(self.i)+'.png'
                entry = pd.DataFrame(
                        {'file': imgFile, 'x1': bbsTrack[0], 'y1': bbsTrack[1], 'x2': bbsTrack[2], 'y2': bbsTrack[3], 'track': bbsTrack[4]}, index=[self.i])
                self.bb_df = self.bb_df.append(entry)
                self.i += 1
                # real implementation we will need to save the images in a different directory
            else: #starting to track the next object (TRIGGER)
                self.current_object_thres = self.new_object_thres
                self.lastTrackIdx = bbsTrack[4]
                # need to sample the dataframe for number of images
                #bb_df_elements = self.bb_df.sample(n=self.num_images) #randomly sample 8 elements
            self.sendMsg(bbsTrack,conf,caminfo)
            self.sendDebugImg(img,bbsTrack)

    def SegmentationRequest(self,imgs_list):
        pass

    def sendMsg(self,bbsTrack,conf,caminfo):
        bb = BoundingBox()
        bb.header.stamp = caminfo.header.stamp
        bb.header.frame_id = "detection"
        bb.camera_info = caminfo        
        bb.xmin = int(bbsTrack[0])
        bb.ymin = int(bbsTrack[1])
        bb.xmax = int(bbsTrack[2])
        bb.ymax = int(bbsTrack[3])
        bb.id = bbsTrack[4]
        bb.probability = conf
        bb.Class = "spalling"        

        self.pub1.publish(bb)

    def sendDebugImg(self,img,bbsTrack):
        if bbsTrack[4] % 2 == 0:
            color = (255, 0, 0)
        else:
            color = (0, 0, 255)
        img = cv2.rectangle(img, (int(bbsTrack[0]),int(bbsTrack[1])), (int(bbsTrack[2]),int(bbsTrack[3])), color, 3)        
        msg = cv2_to_imgmsg(img)
        self.pub2.publish(msg)      

    #     final_boxes, final_scores, final_cls_inds = self.run(sess=self.session, img=image, visual=False)
    #     msg_bb = BoundingBoxes2D()
    #     msg_bb.header.stamp = rospy.Time()
    #     msg_bb.header.frame_id = "detection"
    #     msg_bb.camera_info = caminfo_msg
    #     if final_boxes is not None:
    #         for box,score in zip(final_boxes,final_scores):
    #             bb = BoundingBox()
    #             bb.Class = "spalling"
    #             bb.xmin = int(box[0])
    #             bb.ymin = int(box[1])
    #             bb.xmax = int(box[2])
    #             bb.ymax = int(box[3])
    #             bb.probability = score
    #             msg_bb.bounding_boxes.append(bb)

    #             if score > 0.5:
    #                 image = cv2.rectangle(image, (int(bb.xmin),int(bb.ymin)), (int(bb.xmax),int(bb.ymax)), (255, 0, 0), 3)
    #             else:
    #                 image = image                
        
    #     self.pub1.publish(msg_bb)
    #     msg = cv2_to_imgmsg(image)
    #     self.pub2.publish(msg)       
        
    # def open_sess(self,model='yolox.onnx'):
    #     return onnxruntime.InferenceSession(model)

    # def run(self,sess=None, img=None, input_shape="416,416", score=0.3, visual=False):
    #     input_shape = tuple(map(int, input_shape.split(',')))
    #     origin_img = img
    #     mean = (0.485, 0.456, 0.406)
    #     std = (0.229, 0.224, 0.225)
    #     img, ratio = preprocess(origin_img, input_shape, mean, std)
    #     session = sess

    #     ort_inputs = {session.get_inputs()[0].name: img[None, :, :, :]}
    #     output = session.run(None, ort_inputs)
    #     predictions = demo_postprocess(output[0], input_shape)[0]

    #     boxes = predictions[:, :4]
    #     scores = predictions[:, 4:5] * predictions[:, 5:]

    #     boxes_xyxy = np.ones_like(boxes)
    #     boxes_xyxy[:, 0] = boxes[:, 0] - boxes[:, 2] / 2.
    #     boxes_xyxy[:, 1] = boxes[:, 1] - boxes[:, 3] / 2.
    #     boxes_xyxy[:, 2] = boxes[:, 0] + boxes[:, 2] / 2.
    #     boxes_xyxy[:, 3] = boxes[:, 1] + boxes[:, 3] / 2.
    #     boxes_xyxy /= ratio
    #     dets = multiclass_nms(boxes_xyxy, scores, nms_thr=0.45, score_thr=0.1)

    #     if dets is not None:
    #         final_boxes, final_scores, final_cls_inds = dets[:, :4], dets[:, 4], dets[:, 5]
    #     else:
    #         return None, None, None

    #     if visual:
    #         COCO_CLASSES = '0'
    #         origin_img = vis(origin_img, final_boxes, final_scores, final_cls_inds,
    #                         conf=score, class_names=COCO_CLASSES)
    #         cv2.imwrite('output.png', origin_img)

    #     return final_boxes, final_scores, final_cls_inds

      

if __name__ == '__main__':
    Node() 

    
       
