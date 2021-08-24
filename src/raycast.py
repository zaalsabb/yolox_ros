#!/usr/bin/env python3

import open3d as o3d
import numpy as np
import rospy
from yolox_ros.msg import BoundingBoxes2D
from utils import *
from tf2pose.srv import *
from scipy.spatial.transform import Rotation as Rot
from geometry_msgs.msg import PolygonStamped, Point32, PoseStamped
import message_filters
import time

class Node:

    def __init__(self):

        rospy.init_node('raycast')
        self.pointcloud_file = rospy.get_param('~pointcloud_file', default=None)  
        self.map_id = rospy.get_param('~map_id',default='world')

        self.load_pointcloud()

        self.pub1 = rospy.Publisher('/bounding_boxes3D/polygon',PolygonStamped,queue_size=1)
        #self.pub2 = rospy.Publisher('/bounding_boxes3D',BoundingBoxes3D,queue_size=1)

        sub1=message_filters.Subscriber('/bounding_boxes2D',BoundingBoxes2D)
        sub2=message_filters.Subscriber('/pose', PoseStamped)
        ts = message_filters.ApproximateTimeSynchronizer([sub1, sub2], 1000, 0.5) 
        ts.registerCallback(self.callback) 

        rospy.spin()

    def load_pointcloud(self):
        self.pcd = o3d.io.read_point_cloud(self.pointcloud_file)
        self.pcd.paint_uniform_color([0.5, 0.5, 0.5])
        self.pcd_tree = o3d.geometry.KDTreeFlann(self.pcd)
        self.pcd_bound = self.pcd.get_max_bound()-self.pcd.get_min_bound()

    def raycast_search(self,origin,ray):
        ray = ray/np.linalg.norm(ray)
        p = origin
        R = 0.1
        intersection = None
        i = 0    
        while True:
            [k, idx, _] = self.pcd_tree.search_knn_vector_3d(p, 1)
            closest_p = np.asarray(self.pcd.points)[idx[0], :]
            d = np.linalg.norm(p-closest_p)
            p = p + d*ray
            [k, idx, _] = self.pcd_tree.search_hybrid_vector_3d(p, radius=R, max_nn=1000)
            
            if k > 0:
                points=np.asarray(self.pcd.points)[idx[0:], :]
                inlier_ratio = 0.80
                max_dist = R/2
                max_iteration = 50
                #A,B,C,D,r,ii,_ = ransac_plane(points,inlier_ratio,max_dist,max_iteration)    
                #A,B,C,D = planeNpoint(points)
                A,B,C,D = plane_leastsq(points[:,0],points[:,1],points[:,2])
                intersection=PlanePointIntersect(origin,ray,A,B,C,D)      
                break
            
            if np.any(np.abs(p)>self.pcd_bound):
                break
            i += 1
            if i > 50:
                break   

        return intersection

    def callback(self,*args):

        try:
            t1 = time.time()
            camera_info = args[0].camera_info
            K = np.array(camera_info.K,dtype=np.float32).reshape(3,3)
            frame_id = camera_info.header.frame_id
            #time = camera_info.header.stamp

            # Construct projection matrix (P)
            #p,q = self.tf2pose(frame_id,time)
            p,q = unpack_pose(args[1].pose)
            
            r = Rot.from_quat(q)
            r = r.as_matrix()

            R = r.T
            t = R.dot(-p)
            t = t.reshape(-1,1)

            P = K @ np.hstack([R, t])  
            
            # raycast bounding box pixels
            bounding_boxes_uv = []
            for bb in args[0].bounding_boxes:
                bounding_boxes_uv.append([bb.xmin,bb.ymin])
                bounding_boxes_uv.append([bb.xmax,bb.ymin])
                bounding_boxes_uv.append([bb.xmax,bb.ymax])
                bounding_boxes_uv.append([bb.xmin,bb.ymax])

            bounding_boxes_uv = np.array(bounding_boxes_uv).T

            rays=ProjectToWorld(p,R,P,bounding_boxes_uv)
            rays = rays.T

            msg = PolygonStamped()
            msg.header.frame_id= self.map_id      

            for ray in rays:
                point=self.raycast_search(p,ray)
                if point is not None:
                    point32 = Point32()
                    point32.x = point[0]
                    point32.y = point[1]
                    point32.z = point[2]
                    msg.polygon.points.append(point32)              
                
            self.pub1.publish(msg)
            t2 = time.time()
            rospy.loginfo('%.10f' % (t2-t1))            
        except Exception as e:
            pass

    def tf2pose(self,frame_id,t):
        srv = rospy.ServiceProxy('/request_tf2', tf2req)
        req = tf2reqRequest()
        req.header.frame_id = frame_id
        req.header.stamp = t
        res=srv(req)
        return unpack_pose(res.pose)

if __name__ == '__main__':
    Node() 

