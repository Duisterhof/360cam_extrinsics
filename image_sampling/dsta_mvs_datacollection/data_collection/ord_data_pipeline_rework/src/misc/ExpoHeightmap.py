#!/usr/bin/env python
# license removed for brevity

'''
Build a height map in a rectagular open aera
python ExpoHeightmap.py --far-point 10 --path-skip 2 --map-dir /home/wenshan/tmp/maps/offroad/ --map-filename offroad
'''
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


from settings import get_args
from os.path import isdir, join
from os import mkdir, system

class ExpoHeightmap:
    def __init__(self, args):
        self.args = args
        self.cmd_client = airsim.VehicleClient()
        self.cmd_client.confirmConnection()
        self.camid = 0
        self.img_type = [airsim.ImageRequest(self.camid, airsim.ImageType.DepthPlanner, True)]
        self.FAR_POINT = args.far_point
        self.cam_pos = [0., 0., 0.]
        self.fov = args.camera_fov
        self.path_skip = args.path_skip
        self.mapfilename = self.args.map_filename
        self.mapfiledir = self.args.map_dir

        INITVALUE = 100000
        # the range is [min, max)
        # change these values
        self.xmin = -10
        self.xmax = 10
        self.ymin = -20
        self.ymax = 20
        self.z = -3
        self.grid_resolution = 0.1

        self.xmin_ind = int(math.floor(self.xmin/self.grid_resolution))
        self.xmax_ind = int(math.floor((self.xmax-1e-6)/self.grid_resolution))
        self.ymin_ind = int(math.floor(self.ymin/self.grid_resolution))
        self.ymax_ind = int(math.floor((self.ymax-1e-6)/self.grid_resolution))
        self.grid_num_x = self.xmax_ind - self.xmin_ind + 1
        self.grid_num_y = self.ymax_ind - self.ymin_ind + 1
        self.heightmap = np.ones((self.grid_num_x, self.grid_num_y),dtype=np.float32) * INITVALUE
        print('Gridnum ({}, {})'.format(self.grid_num_x, self.grid_num_y))

    def coord2ind(self, x, y):
        x_ind = int(math.floor(x/self.grid_resolution)) - self.xmin_ind
        y_ind = int(math.floor(y/self.grid_resolution)) - self.ymin_ind
        return x_ind, y_ind

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
        mask = depth_front<self.FAR_POINT
        cam_pose = (img_res[0].camera_position, img_res[0].camera_orientation)

        return depth_front, cam_pose, mask

    def collect_points_6dir(self, tgt):
        # must be in CV mode, otherwise the point clouds won't align
        scan_config = [0, -1, 2, 1, 4, 5] # front, left, back, right, up, down
        points_6dir = np.zeros((0, 3), dtype=np.float32)
        for k,face in enumerate(scan_config): 
            if face == 4:  # look upwards at tgt
                pose = Pose(Vector3r(tgt[0], tgt[1], tgt[2]), to_quaternion(math.pi / 2, 0, 0)) # up
            elif face == 5:  # look downwards
                pose = Pose(Vector3r(tgt[0], tgt[1], tgt[2]), to_quaternion(-math.pi / 2, 0, 0)) # down - pitch, roll, yaw
            else:  # rotate from [-90, 0, 90, 180]
                yaw = math.pi / 2 * face
                pose = Pose(Vector3r(tgt[0], tgt[1], tgt[2]), to_quaternion(0, 0, yaw))

            self.set_vehicle_pose(pose)
            depth_front, _, mask = self.get_depth_campos()
            if depth_front is None:
                print('Missing image {}: {}'.format(k, tgt))
                continue
            point_array = expo_util.depth_to_point_cloud(depth_front, self.focal, self.pu, self.pv, mode = k)
            point_array_filtered = point_array[mask,:]
            print point_array_filtered.shape
            points_6dir = np.concatenate((points_6dir, point_array_filtered), axis=0)
        # reset the pose for fun
        pose = Pose(Vector3r(tgt[0], tgt[1], tgt[2]), to_quaternion(0, 0, 0))
        self.set_vehicle_pose(pose)
        # print 'points:', points_6dir.shape    
        return points_6dir

    def convert_points_to_height(self, tgt):
        points = self.collect_points_6dir(tgt)
        points_global = points + tgt # transform local coord to global coord
        for pt in points_global:
            x, y, z = pt
            xind, yind = self.coord2ind(x, y)
            if xind<self.grid_num_x and yind<self.grid_num_y and xind>=0 and yind>=0:
                if self.heightmap[xind, yind] > z:
                    self.heightmap[xind, yind] = z
            # else:
            #     print('Point Out of Bounds ({}, {})'.format(xind, yind))

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

        print('Initialized img ({},{}) focal {}, ({},{})'.format(self.imgwidth, self.imgheight, self.focal, self.pu, self.pv))
        time.sleep(5)
        self.convert_points_to_height(cam_trans)


    def points_dist(self, pt1, pt2):
        dist = (pt1[0] - pt2[0])**2 + (pt1[1] - pt2[1])**2 + (pt1[2] - pt2[2])**2
        dist = math.sqrt(dist)
        return dist

    def is_same_point(self, pt1, pt2):
        if abs(pt1[0]-pt2[0]) > 1e-3 or abs(pt1[1]-pt2[1]) > 1e-3 or abs(pt1[2]-pt2[2]) > 1e-3:
            return False
        return True

    def save_map(self):
        # save map
        timestr = time.strftime('%m%d_%H%M%S',time.localtime())
        filepath = self.mapfiledir+'/OccMap'
        if not isdir(filepath):
            mkdir(filepath)
        filepathname = filepath+'/'+self.mapfilename+'_height_'+timestr+'.npy'
        np.save(filepathname, self.heightmap)
        print('Save height map {}'.format(filepathname))

    def mapping(self):
        self.init_exploration()
        xtgts = np.arange(self.xmin, self.xmax, self.path_skip)
        ytgts = np.arange(self.ymin, self.ymax, self.path_skip)

        count = 0
        for tgtx in xtgts:
            for tgty in ytgts:
                tgt = [tgtx, tgty, self.z]
                self.convert_points_to_height(tgt)

        self.heightmap[self.heightmap==1e+5]=self.heightmap[self.heightmap<1e+5].max()
        self.save_map()
        import cv2
        disp = (self.heightmap-self.heightmap.min())/(self.heightmap.max()-self.heightmap.min())
        cv2.imshow('img',disp)
        cv2.waitKey(0)
        import ipdb;ipdb.set_trace()

if __name__ == '__main__':
    args = get_args()

    controller = ExpoHeightmap(args)
    controller.mapping()

