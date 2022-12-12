#!/usr/bin/env python
# license removed for brevity

# import sys
# import os
# curdir = os.path.dirname(os.path.realpath(__file__))
# print curdir+'/..'
# sys.path.insert(0,curdir+'/..')
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
from planner_base.srv import path as PathSrv
from planner_base.srv import nearfrontier as NearfrontierSrv
from geometry_msgs.msg import Point

from settings import get_args
from os.path import isdir, join
from os import mkdir, system


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
        self.mapfilename = self.args.map_filename
        self.mapfiledir = self.args.map_dir
        self.try_round = self.args.try_round

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

    def publish_lidar_scans_6dir(self, tgt):
        rostime = rospy.Time.now()
        points = self.collect_points_6dir(tgt)
        pc_msg = expo_util.xyz_array_to_point_cloud_msg(points, rostime)
        odom_msg = expo_util.trans_to_ros_odometry(tgt, rostime)
        self.cam_pose = tgt

        self.tf_broad_.sendTransform(translation=tgt, rotation=[0.,0.,0.,1.],
                                     time=rostime, child='map', parent='world')
        self.odom_pub_.publish(odom_msg)
        self.cloud_pub_.publish(pc_msg)


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
        self.publish_lidar_scans_6dir(cam_trans)
        time.sleep(5)

    def points_dist(self, pt1, pt2):
        dist = (pt1[0] - pt2[0])**2 + (pt1[1] - pt2[1])**2 + (pt1[2] - pt2[2])**2
        dist = math.sqrt(dist)
        return dist

    def is_same_point(self, pt1, pt2):
        if abs(pt1[0]-pt2[0]) > 1e-3 or abs(pt1[1]-pt2[1]) > 1e-3 or abs(pt1[2]-pt2[2]) > 1e-3:
            return False
        return True

    def explore(self):
        """
        We have to handle two cases: oscillation and local frontier disappear
        :param try_round: -1 for all the frontiers
        :path_skip: skip way points on the path
        :return:
        """
        # import ipdb;ipdb.set_trace()
        # A star path finding for local exploration
        local_path = self.call_local_planning_service(self.try_round)
        if local_path is None: 
            return False
        # import ipdb;ipdb.set_trace()
        # No feasible local_path is found
        if len(local_path) == 0: 
            return False

        # insert the goal point to the list for occilation detection
        target_pt = local_path[0]
        occilation = False
        if not self.is_same_point(target_pt, self.last_ten_goals[self.lfg_ind]): # target point changes
            for k,pt in enumerate(self.last_ten_goals): # check if the target already exist in the goal list
                if self.is_same_point(target_pt, pt): # this is a occilation
                    occilation = True
                    tmp = self.last_ten_goals[self.lfg_ind]
                    self.last_ten_goals[self.lfg_ind] = pt
                    self.last_ten_goals[k] = tmp
                    rospy.logwarn('Occilation detected, increase replan step! %d - %d', self.lfg_ind, k)
                    break
            if occilation:
                self.replan_step = self.replan_step * 3
                if self.replan_step>7: # serious occilation
                    rospy.logwarn('Escapte occilation by going the the goal directly!!')
                    local_path = [local_path[0]]
            else:
                rospy.loginfo("new target, reset replan step..")
                self.lfg_ind = (self.lfg_ind+1)%self.last_list_len
                self.last_ten_goals[self.lfg_ind] = target_pt
                self.replan_step = 1
        else:
            rospy.loginfo("flying to the same target")

        path_len = len(local_path)
        for i in range(self.replan_step):
            if path_len < self.path_skip:
                next_ind = path_len - (i+1)*((path_len+1)/2) 
            else:
                next_ind = len(local_path) - (i+1)*self.path_skip 
            if next_ind<0:
                break
            rospy.loginfo('Path len {}, move to waypoint {}'.format(len(local_path),local_path[next_ind]))
            next_way_point = local_path[next_ind]
            # self.move_to_tgt(next_way_point)
            self.publish_lidar_scans_6dir(next_way_point)
        return True

    def explore_global_frontier(self,):
        # import ipdb;ipdb.set_trace()
        next_point = self.get_nearest_frontier()
        if next_point is None:
            return False
        rospy.loginfo('Next global frontier ({}, {}, {})'.format(next_point[0], next_point[1], next_point[2]))
        self.publish_lidar_scans_6dir(next_point)
        return True

    def get_nearest_frontier(self):
        rospy.wait_for_service('near_frontier_srv')

        robot_pos = Point(self.cam_pose[0], self.cam_pose[1], self.cam_pose[2])
        try:
            global_frontier_srv = rospy.ServiceProxy('near_frontier_srv', NearfrontierSrv)
            global_frontier = global_frontier_srv(robot_pos)
        except rospy.ServiceException:
            print("No frontier returned..")
            return None

        return [global_frontier.nearfrontier.x, global_frontier.nearfrontier.y, global_frontier.nearfrontier.z]

    def call_local_planning_service(self, try_round):
        target_path = []
        rospy.wait_for_service('bbx_path_srv')
        try:
            feasible_path = rospy.ServiceProxy('bbx_path_srv', PathSrv)
            resp = feasible_path(try_round)
            for item in resp.path.points:
                target_path.append([item.x, item.y, item.z])
            return target_path
        except rospy.ServiceException:
            print("service call failed")
            return None


    def save_map(self, mapfiledir, mapfilename):
        # save map
        timestr = time.strftime('%m%d_%H%M%S',time.localtime())
        filepath = mapfiledir+'/OccMap'
        if not isdir(filepath):
            mkdir(filepath)
        filepathname = filepath+'/'+mapfilename+'_'+timestr+'.ot'
        cmd = 'rosrun octomap_server octomap_saver ' + filepathname
        system(cmd)
        rospy.loginfo('Save map {}'.format(filepathname))

    def mapping(self):
        self.init_exploration()
        count = 0
        while not rospy.is_shutdown():
            count += 1
            if self.args.global_only:
                if self.explore_global_frontier():
                    time.sleep(2.0)
                else: # mapping finished
                    break 
            else: # A star planning on local map, move to nearest global frontier when no local frontiers
                if self.explore(): 
                    time.sleep(2.0)
                else: # no local frontier
                    if self.explore_global_frontier():
                        time.sleep(2.0)
                    else:
                        break 
            if count%100==0:
                self.save_map(self.mapfiledir, self.mapfilename)
            # import ipdb;ipdb.set_trace()

        self.save_map(self.mapfiledir, self.mapfilename)

if __name__ == '__main__':
    rospy.init_node('expo_control', anonymous=True)
    args = get_args()

    controller = ExpoController(args)
    controller.mapping()

