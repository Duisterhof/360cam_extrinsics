#!/usr/bin/env python
import rospy
import numpy as np
from std_msgs.msg import Header
from nav_msgs.msg import Odometry
from sensor_msgs.msg import PointCloud2, PointField
from math import pi, tan

def get_intrinsic(response, caminfo, fov=None):
    """
    The intrinsic parameters are related to image size,
    response and caminfo are both from airsim api
    """
    imgwidth = response.width
    imgheight = response.height
    if fov is None:
        fov = caminfo.fov
    focal = imgwidth/2.0/tan(fov/2.0*pi/180.0)
    pu, pv = imgwidth/2.0, imgheight/2.0
    return focal, pu, pv

def get_intrinsic_matrix(focal, pu, pv):
    """
    put values in matrix
    """
    intrinsic = np.asarray([[focal, 0, pu],
                            [0, focal, pv],
                            [0, 0.0, 1.0]]).astype(np.float32)
    return intrinsic


def xyz_array_to_point_cloud_msg(points, timestamp=None):
    """
    Please refer to this ros answer about the usage of point cloud message:
        https://answers.ros.org/question/234455/pointcloud2-and-pointfield/
    :param points:
    :param header:
    :return:
    """
    header = Header()
    header.frame_id = 'map'
    if timestamp is None:
        timestamp = rospy.Time().now()
    header.stamp = timestamp
    msg = PointCloud2()
    msg.header = header
    msg.width = points.shape[0]
    msg.height = points.shape[1]
    msg.fields = [PointField('x', 0, PointField.FLOAT32, 1),
                  PointField('y', 4, PointField.FLOAT32, 1),
                  PointField('z', 8, PointField.FLOAT32, 1), ]
    msg.is_bigendian = False
    msg.point_step = 12
    msg.row_step = msg.point_step * msg.width
    # organized clouds are non-dense, since we have to use std::numeric_limits<float>::quiet_NaN()
    # to fill all x/y/z value; un-organized clouds are dense, since we can filter out unfavored ones
    msg.is_dense = False
    xyz = points.astype(np.float32)
    msg.data = xyz.tostring()
    return msg

def rotate_points(x, y, z, mode): # TODO: understand why!
    if mode == 0: # front
        return np.stack([x, y, z], axis=2)
    if mode == 3: # left
        return np.stack([-y, x, z], axis=2)
    if mode == 2: # back
        return np.stack([-x, -y, z], axis=2)
    if mode == 1: # right
        return np.stack([y, -x, z], axis=2)
    if mode == 5: # up
        return np.stack([-z, y, x], axis=2)
    if mode == 4: # down
        return np.stack([z, y, -x], axis=2)

def depth_to_point_cloud(depth, focal, pu, pv, mode = 0):
    """
    Todo: change the hard stack to transformation matrix
    Convert depth image to point cloud based on intrinsic parameters
    :param depth: depth image
    :return: xyz point array
    """
    h, w = depth.shape
    depth64 = depth.astype(np.float64)
    wIdx = np.linspace(0, w - 1, w, dtype=np.float64) + 0.5 # put the optical center at the middle of the image
    hIdx = np.linspace(0, h - 1, h, dtype=np.float64) + 0.5 # put the optical center at the middle of the image
    u, v = np.meshgrid(wIdx, hIdx)
    x = (u - pu) * depth64 / focal
    y = (v - pv) * depth64 / focal
    x = x.astype(np.float32)
    y = y.astype(np.float32)
    points = rotate_points(depth, x, y, mode) # amigo: this is in NED coordinate
    return points


def odom_to_trans_and_rot(odometry):
    """
    Convert pose object to translation list and rotation list
    :param odometry: Pose object defined in airsim types
    :return: translation and rotation
    """
    assert isinstance(odometry, Odometry)
    tans = [odometry.pose.pose.position.x,
            odometry.pose.pose.position.y,
            odometry.pose.pose.position.z]
    rot = [odometry.pose.pose.orientation.x,
           odometry.pose.pose.orientation.y,
           odometry.pose.pose.orientation.z,
           odometry.pose.pose.orientation.w]
    return tans, rot

def sim_pose_to_trans_and_rot(pose):
    trans = [pose.position.x_val, pose.position.y_val, pose.position.z_val]
    rot = [pose.orientation.x_val, pose.orientation.y_val, pose.orientation.z_val, pose.orientation.w_val,]
    return trans, rot

def sim_pose_to_ros_odometry(position, orientation):
    """
    Convert the pose message from airsim to ros
    :param pose_sim: airsim Pose message
    :return: ros Odometry message
    """
    pose_ros = Odometry()
    pose_ros.header.frame_id = 'world'
    pose_ros.header.stamp = rospy.Time().now()
    pose_ros.pose.pose.position.x = position.x_val
    pose_ros.pose.pose.position.y = position.y_val
    pose_ros.pose.pose.position.z = position.z_val
    pose_ros.pose.pose.orientation.x = orientation.x_val
    pose_ros.pose.pose.orientation.y = orientation.y_val
    pose_ros.pose.pose.orientation.z = orientation.z_val
    pose_ros.pose.pose.orientation.w = orientation.w_val
    return pose_ros

def trans_to_ros_odometry(transition, timestamp=None):
    """
    Convert the pose message from airsim to ros
    :param pose_sim: airsim Pose message
    :return: ros Odometry message
    """
    pose_ros = Odometry()
    pose_ros.header.frame_id = 'world'
    if timestamp is None:
        timestamp = rospy.Time().now()
    pose_ros.header.stamp = timestamp
    pose_ros.pose.pose.position.x = transition[0]
    pose_ros.pose.pose.position.y = transition[1]
    pose_ros.pose.pose.position.z = transition[2]
    pose_ros.pose.pose.orientation.x = 0.
    pose_ros.pose.pose.orientation.y = 0.
    pose_ros.pose.pose.orientation.z = 0.
    pose_ros.pose.pose.orientation.w = 1.0
    return pose_ros
