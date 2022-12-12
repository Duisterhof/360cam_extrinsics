#!/usr/bin/env python
# license removed for brevity

# Publish a point cloud using keyboard control
import sys
sys.path.append('../')

# import copy
import rospy
import tf
import time
import math
import numpy as np
import expo_utility as expo_util

import airsim
from airsim.types import Pose, Vector3r, Quaternionr
from airsim import utils as sim_util
from airsim.utils import to_quaternion

from nav_msgs.msg import Odometry
from sensor_msgs.msg import PointCloud2
from geometry_msgs.msg import Point

from settings import get_args
from os.path import isdir, join
from os import mkdir, system

import cv2
from signal import signal, SIGINT

class ExpoController:
    def __init__(self, args):
        rospy.init_node('expo_control', anonymous=True)

        self.args = args
        self.cmd_client = airsim.VehicleClient()
        self.cmd_client.confirmConnection()
        self.tf_broad_ = tf.TransformBroadcaster()
        self.odom_pub_ = rospy.Publisher('pose', Odometry, queue_size=1)
        self.cloud_pub_ = rospy.Publisher('cloud_in', PointCloud2, queue_size=1)
        self.camid = 0
        self.img_type = [airsim.ImageRequest(self.camid, airsim.ImageType.DepthPlanner, True)]
        self.FAR_POINT = args.far_point
        self.cam_pos = [0., 0., 0.]
        self.fov = args.camera_fov
        self.path_skip = args.path_skip
        self.last_list_len = 10
        self.last_ten_goals = [[0.,0.,0.]]*self.last_list_len # detect and avoid occilation
        self.lfg_ind = 0
        self.replan_step = 1
        # self.mapfilename = self.args.map_filename
        # self.mapfiledir = self.args.map_dir
        # self.try_round = self.args.try_round

    def get_depth_campos(self):
        '''
        cam_pose: 0: [x_val, y_val, z_val] 1: [x_val, y_val, z_val, w_val]
        '''
        img_res = self.cmd_client.simGetImages(self.img_type)
        
        if img_res is None or img_res[0].width==0: # Sometime the request returns no image
            return None, None

        # import ipdb;ipdb.set_trace()
        depth_front = sim_util.list_to_2d_float_array(img_res[0].image_data_float,
                                                      img_res[0].width, img_res[0].height)
        depth_front[depth_front>self.FAR_POINT] = self.FAR_POINT
        cam_pose = (img_res[0].camera_position, img_res[0].camera_orientation)

        return depth_front, cam_pose

    def collect_points_6dir(self, tgt):
        # must be in CV mode, otherwise the point clouds won't align
        scan_config = [0, -1, 2, 1, 4, 5] # front, left, back, right, up, down
        points_6dir = np.zeros((0, self.imgwidth, 3), dtype=np.float32)
        for k,face in enumerate(scan_config): 
            if face == 4:  # look upwards at tgt
                pose = Pose(Vector3r(tgt[0], tgt[1], tgt[2]), to_quaternion(math.pi / 2, 0, 0)) # up
            elif face == 5:  # look downwards
                pose = Pose(Vector3r(tgt[0], tgt[1], tgt[2]), to_quaternion(-math.pi / 2, 0, 0)) # down - pitch, roll, yaw
            else:  # rotate from [-90, 0, 90, 180]
                yaw = math.pi / 2 * face
                pose = Pose(Vector3r(tgt[0], tgt[1], tgt[2]), to_quaternion(0, 0, yaw))

            self.set_vehicle_pose(pose)
            depth_front, _ = self.get_depth_campos()
            if depth_front is None:
                rospy.logwarn('Missing image {}: {}'.format(k, tgt))
                continue
            # import ipdb;ipdb.set_trace()
            point_array = expo_util.depth_to_point_cloud(depth_front, self.focal, self.pu, self.pv, mode = k)
            points_6dir = np.concatenate((points_6dir, point_array), axis=0)
        # reset the pose for fun
        pose = Pose(Vector3r(tgt[0], tgt[1], tgt[2]), to_quaternion(0, 0, 0))
        self.set_vehicle_pose(pose)
        # print 'points:', points_6dir.shape            
        return points_6dir

    def publish_lidar_scans_6dir(self, tgt=None):
        rostime = rospy.Time.now()
        if tgt is None:
            pose = self.cmd_client.simGetVehiclePose()
            tgt = [pose.position.x_val, pose.position.y_val, pose.position.z_val]
        points = self.collect_points_6dir(tgt)
        pc_msg = expo_util.xyz_array_to_point_cloud_msg(points, rostime)
        odom_msg = expo_util.trans_to_ros_odometry(tgt, rostime)
        self.cam_pose = tgt

        self.tf_broad_.sendTransform(translation=tgt, rotation=[0.,0.,0.,1.],
                                     time=rostime, child='map', parent='world')
        self.odom_pub_.publish(odom_msg)
        self.cloud_pub_.publish(pc_msg)

        rospy.loginfo('Published {} points'.format(len(points)*self.imgwidth))


    def set_vehicle_pose(self, pose, ignore_collison=True, vehicle_name=''):
        self.cmd_client.simSetVehiclePose(pose, ignore_collison, vehicle_name) # amigo: this is supposed to be used in CV mode
        time.sleep(0.1)

    def init_exploration(self):
        # cloud_msg, cloud_odom_msg = self.get_point_cloud_msg(cam_id=0)
        # cam_trans, cam_rot = expo_util.odom_to_trans_and_rot(cloud_odom_msg)
        # cam_pose = self.cmd_client.simGetCameraInfo(camera_name=0).pose
        cam_info = self.cmd_client.simGetCameraInfo(camera_name=self.camid)
        img_res = self.cmd_client.simGetImages(self.img_type)
        img_res = img_res[0]
        cam_pose = Pose(img_res.camera_position, img_res.camera_orientation)
        cam_trans, cam_rot = expo_util.sim_pose_to_trans_and_rot(cam_pose)
        self.imgwidth = img_res.width
        self.imgheight = img_res.height
        self.focal, self.pu, self.pv = expo_util.get_intrinsic(img_res, cam_info, self.fov)
        self.cam_pos = cam_trans

        rospy.loginfo('Initialized img ({},{}) focal {}, ({},{})'.format(self.imgwidth, self.imgheight, self.focal, self.pu, self.pv))
        # self.publish_lidar_scans_6dir(cam_trans)
        time.sleep(1)


def handler(signal_received, frame):
    # Handle any cleanup here
    pass
    exit(0)

if __name__ == '__main__':

    rospy.init_node('expo_control', anonymous=True)
    args = get_args()
    controller = ExpoController(args)
    controller.init_exploration()


    # Tell Python to run the handler() function when SIGINT is recieved
    signal(SIGINT, handler)


    print('Running. Press SPACE to publish point cloud. \nPress q to exit.')
    count = 0
    key = 0

    while key != 113: # 'q'
        cv2.imshow('img',np.zeros((320,320,3),dtype=np.uint8))
        key = cv2.waitKey(10)
        # print key
        if key==32: # space key
            controller.publish_lidar_scans_6dir()
