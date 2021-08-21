#!/usr/bin/env python3
import os
import sys
import rospy
import cv2
from CvBridge import imgmsg_to_cv2, cv2_to_imgmsg
from sensor_msgs.msg import Image
import numpy as np

import onnxruntime

yolox_module_dir = '/'.join(os.path.dirname(os.path.realpath(__file__)).split('/')[:-1])+'/YOLOX'
sys.path.insert(0,yolox_module_dir)

from yolox.data.data_augment import preproc as preprocess
from yolox.utils import multiclass_nms, demo_postprocess, vis

from yolox_ros.msg import BoundingBox, BoundingBoxes

class Node:

    def __init__(self):

        rospy.init_node('yolox')
        self.model = rospy.get_param('~model', default=os.path.join(yolox_module_dir,'yolox.onnx'))        
        self.session = self.open_sess(model=self.model)

        rospy.Subscriber('/image', Image,self.callback)
        self.pub1 = rospy.Publisher('/bounding_boxes',BoundingBoxes,queue_size=1)
        self.pub2 = rospy.Publisher('/debug/image',Image,queue_size=1)

        rospy.spin()

    def callback(self,*args):
        image = imgmsg_to_cv2(args[0])
        final_boxes, final_scores, final_cls_inds = self.run(sess=self.session, img=image, visual=False)
        if final_boxes is not None:
            msg_bb = BoundingBoxes()
            msg_bb.header.stamp = rospy.Time()
            msg_bb.header.frame_id = "detection"
            msg_bb.image_header = args[0].header
            for box,score in zip(final_boxes,final_scores):
                bb = BoundingBox()
                bb.Class = "spalling"
                bb.xmin = box[0]
                bb.ymin = box[1]
                bb.xmax = box[2]
                bb.ymax = box[3]
                bb.probability = score
                msg_bb.bounding_boxes.append(bb)

                if score > 0.5:
                    image = cv2.rectangle(image, (int(bb.xmin),int(bb.ymin)), (int(bb.xmax),int(bb.ymax)), (255, 0, 0), 3)
                else:
                    image = image                
        
            self.pub1.publish(msg_bb)

        msg = cv2_to_imgmsg(image)
        self.pub2.publish(msg)       
        
    def open_sess(self,model='yolox.onnx'):
        return onnxruntime.InferenceSession(model)

    def run(self,sess=None, img=None, input_shape="416,416", score=0.3, visual=False):
        input_shape = tuple(map(int, input_shape.split(',')))
        origin_img = img
        mean = (0.485, 0.456, 0.406)
        std = (0.229, 0.224, 0.225)
        img, ratio = preprocess(origin_img, input_shape, mean, std)
        session = sess

        ort_inputs = {session.get_inputs()[0].name: img[None, :, :, :]}
        output = session.run(None, ort_inputs)
        predictions = demo_postprocess(output[0], input_shape)[0]

        boxes = predictions[:, :4]
        scores = predictions[:, 4:5] * predictions[:, 5:]

        boxes_xyxy = np.ones_like(boxes)
        boxes_xyxy[:, 0] = boxes[:, 0] - boxes[:, 2] / 2.
        boxes_xyxy[:, 1] = boxes[:, 1] - boxes[:, 3] / 2.
        boxes_xyxy[:, 2] = boxes[:, 0] + boxes[:, 2] / 2.
        boxes_xyxy[:, 3] = boxes[:, 1] + boxes[:, 3] / 2.
        boxes_xyxy /= ratio
        dets = multiclass_nms(boxes_xyxy, scores, nms_thr=0.45, score_thr=0.1)

        if dets is not None:
            final_boxes, final_scores, final_cls_inds = dets[:, :4], dets[:, 4], dets[:, 5]
        else:
            return None, None, None

        if visual:
            COCO_CLASSES = '0'
            origin_img = vis(origin_img, final_boxes, final_scores, final_cls_inds,
                            conf=score, class_names=COCO_CLASSES)
            cv2.imwrite('output.png', origin_img)

        return final_boxes, final_scores, final_cls_inds



if __name__ == '__main__':
    Node() 

    
       
