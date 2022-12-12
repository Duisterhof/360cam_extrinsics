# collect images in cv mode
import os
import sys
# # The path of the current Python script.
# _CURRENT_PATH = os.path.dirname(os.path.realpath(__file__))

# sys.path.append(os.path.join(_CURRENT_PATH, '..'))

# print(sys.path)

from airsim.types import Pose, Vector3r , Quaternionr
from airsim.utils import to_eularian_angles, to_quaternion
from .ImageClient import ImageClient

import cv2 # debug
import numpy as np
from math import cos, sin, tanh, pi
import time

from os import mkdir, listdir
from os.path import isdir, join
import sys
import random
from settings import get_args

from pyquaternion import Quaternion as Quaternionpy # quaternion multiplication

from transform import Rotation, RotationSpline # quaternion spline
import numpy as np
# import matplotlib.pyplot as plt

import json

np.set_printoptions(precision=3, suppress=True, threshold=10000)

class QuaternionSampler(object):
    '''
    Quaternionpy: qyquternion, multiplication and linear interpolation
    Quaternionr: airsim, eular-quaternion translation
    scipy Rotation: spline smoothing
    '''
    def __init__(self, args):
        self.args = args

        self.MaxRandAngle = np.clip(self.args.rand_degree, 0, 90)
        self.MaxRandRad = self.MaxRandAngle * pi / 180.
        self.SmoothCount =  self.args.smooth_count
        self.MaxYaw = min(self.args.max_yaw, 180) * pi / 180.
        self.MinYaw = max(self.args.min_yaw, -180) * pi / 180.
        self.MaxPitch = min(self.args.max_pitch, 90) * pi / 180.
        self.MinPitch = max(self.args.min_pitch, -90) * pi / 180.
        self.MaxRoll = min(self.args.max_roll, 180) * pi / 180.
        self.MinRoll = max(self.args.min_roll, -180) * pi / 180.


    def next_quaternion(self,idx):
        pass

    def init_random_yaw(self):
        randomyaw = np.random.uniform(self.MinYaw, self.MaxYaw)
        print ('Random yaw {}, angle {}'.format(randomyaw, randomyaw*180.0/pi))
        qtn = to_quaternion(0., 0., randomyaw)
        return Quaternionpy(qtn.w_val, qtn.x_val, qtn.y_val, qtn.z_val), randomyaw

    def random_quaternion(self):
        '''
        return Quaternionpy for multiplication
        '''
        theta =  np.random.random()*self.MaxRandRad*2 - self.MaxRandRad
        axi = np.random.random(3)
        axi = axi/np.linalg.norm(axi)
        return Quaternionpy(axis=axi, angle=theta)    

    def clip_quaternion(self, quatpy):
        '''
        Input quatpy: Quaternionpy
        Return new_ori_clip: Quaternionpy
        '''
        quatr = Quaternionr(quatpy.x, quatpy.y, quatpy.z, quatpy.w)
        (pitch, roll, yaw) = to_eularian_angles(quatr)
        pitch_clip = np.clip(pitch, self.MinPitch, self.MaxPitch)
        roll_clip = np.clip(roll, self.MinRoll, self.MaxRoll)
        yaw_clip = np.clip(yaw, self.MinYaw, self.MaxYaw)
        quatr_clip = to_quaternion(pitch_clip, roll_clip, yaw_clip)
        quatpy_clip = Quaternionpy(quatr_clip.w_val, quatr_clip.x_val, quatr_clip.y_val, quatr_clip.z_val)

        return quatpy_clip, (roll_clip, pitch_clip, yaw_clip)


class RandQuaternionSampler(QuaternionSampler):

    def reset(self, posenum):
        self.orientation, _ = self.init_random_yaw()
        self.orilist = []
        self.oriind = self.SmoothCount


    def next_quaternion(self,idx):
        if self.oriind >= self.SmoothCount: # sample a new orientation
            rand_ori = self.random_quaternion()
            new_ori = rand_ori * self.orientation

            quatpy_clip, (roll, pitch, yaw) = self.clip_quaternion(new_ori)

            qtnlist = Quaternionpy.intermediates(self.orientation, quatpy_clip, self.SmoothCount-1, include_endpoints=True)
            self.orientation = quatpy_clip
            self.orilist = list(qtnlist)
            self.oriind = 1
            # print "sampled new", new_ori, ', after clip', self.orientation #, 'list', self.orilist

        next_qtn = self.orilist[self.oriind]
        self.oriind += 1
        # print "  return next", next_qtn
        return Quaternionr(next_qtn.x, next_qtn.y, next_qtn.z, next_qtn.w)

def quatpy2eular(quatpy):
    qqq=Quaternionr(quatpy.x, quatpy.y, quatpy.z, quatpy.w)
    rrr= to_eularian_angles(qqq)
    return np.array(rrr)*180/pi # pitch, roll, yaw

def rpy_diff(rpy1, rpy2, degree=False):
    rpydiff = np.array(rpy1) - np.array(rpy2)
    rpydiff = np.array(rpy1) - np.array(rpy2)
    if degree:
        thresh = 180
    else:
        thresh = pi
    rpydiff[rpydiff>thresh] = rpydiff[rpydiff>thresh] - 2*thresh
    rpydiff[rpydiff<-thresh] = rpydiff[rpydiff<-thresh] + 2*thresh
    return np.abs(rpydiff)

def quatarray2eular(quatarray):
    qqq=Quaternionr(quatarray[0], quatarray[1], quatarray[2], quatarray[3])
    (pitch, roll, yaw)= to_eularian_angles(qqq)
    return np.array([roll, pitch, yaw])*180/pi # roll, pitch, yaw

def array2quatpy(quatarray):
    return Quaternionpy(quatarray[3], quatarray[0], quatarray[1], quatarray[2])

class RandQuaternionSplineSampler(QuaternionSampler):
    '''
    Calculate and smooth the quaternion in reset function
    Assume the trajectory is not super long: a few thousands is reasonable
    '''
    def reset(self, posenum):
        # timestamp for key frames
        times = range(0, posenum + self.SmoothCount, self.SmoothCount)
        keynum = len(times)

        # generate key frame orientations for the whole sequence
        orientation, yaw = self.init_random_yaw()
        # angles = [[0.,0.,yaw]]
        last_angle = [0.,0.,yaw]
        quats = [[orientation.x, orientation.y, orientation.z, orientation.w]]

        # import ipdb;ipdb.set_trace()
        k = 1
        while k < keynum:
            rand_ori = self.random_quaternion()
            orientation = rand_ori * orientation #orientation
            orientation, (roll, pitch, yaw) = self.clip_quaternion(orientation)
            if np.all(rpy_diff([roll, pitch, yaw], last_angle)<self.MaxRandRad):
                last_angle = [roll, pitch, yaw]
                # angles.append([roll, pitch, yaw])
                quats.append([orientation.x, orientation.y, orientation.z, orientation.w])
                k += 1
                # quat_debug = to_quaternion(pitch, roll, yaw)
                # assert quat_debug.w_val == orientation.w and \
                #  quat_debug.x_val == orientation.x and \
                #  quat_debug.y_val == orientation.y and \
                #  quat_debug.z_val == orientation.z 
            # else:
            #     print "%.2f %.2f %.2f (%.2f, %.2f, %.2f)" % (abs(angle_diff(roll,last_angle[0]))-self.MaxRandRad, \
            #         abs(angle_diff(pitch,last_angle[1]))-self.MaxRandRad, \
            #         abs(angle_diff(yaw,last_angle[2]))-self.MaxRandRad, \
            #         roll, pitch, yaw)

            #     angleplot = np.array(angles) * 180 / pi
            #     plt.plot(angleplot[:,0],'.-')
            #     plt.plot(angleplot[:,1],'.-')
            #     plt.plot(angleplot[:,2],'.-')
            #     plt.scatter([len(angleplot)-1,len(angleplot)], [angleplot[-1,0],roll* 180 / pi])
            #     plt.scatter([len(angleplot)-1,len(angleplot)], [angleplot[-1,1],pitch* 180 / pi])
            #     plt.scatter([len(angleplot)-1,len(angleplot)], [angleplot[-1,2],yaw* 180 / pi])
            #     plt.legend(['roll', 'pitch', 'yaw'])
            #     plt.grid()
            #     plt.show(block=False)
            #     plt.pause(0.5)
            #     plt.close()
        # import ipdb;ipdb.set_trace()

        # # spline interpolation
        # rotations_angs = Rotation.from_euler('XYZ', angles, degrees=False)
        # spline_angs = RotationSpline(times, rotations_angs)
        # self.orilist_angs = spline_angs(np.array(range(posenum))).as_quat()
        # self.anglist_angs = spline_angs(np.array(range(posenum))).as_euler('XYZ', degrees=True)

        rotations_quats = Rotation.from_quat(np.array(quats), normalized=False)
        spline_quats = RotationSpline(times, rotations_quats)
        self.orilist_quats = spline_quats(np.array(range(posenum))).as_quat()
        self.anglist_quats = spline_quats(np.array(range(posenum))).as_euler('ZYX', degrees=True)

        # self.orilist = self.orilist_quats.copy()
        StepThresh = self.MaxRandAngle/self.SmoothCount
        for k in range(len(self.anglist_quats)-1):
            anglediff = rpy_diff(self.anglist_quats[k], self.anglist_quats[k+1], degree=True)
            if np.any(anglediff>StepThresh*1.5):
                print ('{} angle diff above thresh: {}'.format(k, anglediff))
                # a hacking solution - use linear interpolation instead
                quatsind = int(k/self.SmoothCount)
                quatsind_start = quatsind * self.SmoothCount
                quatsind_end = min(quatsind_start + self.SmoothCount, len(self.orilist_quats))
                qtns = Quaternionpy.intermediates(array2quatpy(quats[quatsind]), array2quatpy(quats[quatsind+1]), self.SmoothCount-1, include_endpoints=True)
                qtnlist = []
                for qtn in qtns:
                    qtnlist.append([qtn.x, qtn.y, qtn.z, qtn.w])
                # check again the angle difference
                # it turns out that the slerp results will also violate the degree threshold
                angles=[]
                inthresh = True
                for qtn in qtnlist:
                    angles.append(quatarray2eular(qtn))
                for w in range(len(angles)-1):
                    anglediff2 = rpy_diff(angles[w], angles[w+1], degree=True)
                    # print '  ',anglediff2
                    if np.any(anglediff2 > StepThresh*1.5):
                        inthresh = False
                self.orilist_quats[quatsind_start:quatsind_end] = qtnlist[0:quatsind_end-quatsind_start]
                if not inthresh:
                    print('slerp interpolation still above thresh..')
                    print(anglediff2)
            # else:
            #     print k, anglediff

        # clip the quaternion


        # print("Posenum {}, key frame number {}, spline num {}".format(posenum, keynum, len(self.orilist)))
        return quats # for debugging angles, 

    def next_quaternion(self, idx):
        assert(idx<len(self.orilist_quats))
        quat = self.orilist_quats[idx]
        quat_py = Quaternionpy(quat[3], quat[0], quat[1], quat[2])
        quat_clip, _ = self.clip_quaternion(quat_py)

        return Quaternionr(quat_clip.x, quat_clip.y, quat_clip.z, quat_clip.w)

class DataSampler(object):
    def __init__(self, data_dir, args):

        self.args = args
        self.datadir = data_dir

        self.imgtypelist = self.args.img_type.split('_')
        self.camlist = self.args.cam_list.split('_')
        self.camlist_name = {'0': 'front', '1': 'right', '2': 'left', '3': 'back', '4': 'bottom'} 

        self.imgclient = ImageClient(self.camlist, self.imgtypelist)
        self.imgclient.cube_cuda = not args.disable_cube_cuda
        # self.randquaternionsampler = RandQuaternionSampler(self.args)
        self.randquaternionsampler = RandQuaternionSplineSampler(self.args)

        self.logfile = self.datadir + '/sample.log'

        self.imgdirs = []
        self.depthdirs = []
        self.segdirs = []
        self.imgdirs_cube = []
        self.distdirs_cube = []

        self.posefilelist = [] # save incremental pose files
        self.posenplist = [] # save pose in numpy files

    def create_folders(self, trajdir):
        self.imgdirs = []
        self.depthdirs = []
        self.segdirs = []
        self.posefilelist = [] # save incremental pose files
        self.posenplist = [] # save pose in numpy files
        for camind in self.camlist:
            camname = self.camlist_name[camind]
            if 'Scene' in self.imgtypelist:
                self.imgdirs.append(trajdir+'/image_'+camname)
                mkdir(self.imgdirs[-1])
            if 'DepthPlanner' in self.imgtypelist:
                self.depthdirs.append(trajdir+'/depth_'+camname)
                mkdir(self.depthdirs[-1])
            if 'Segmentation' in self.imgtypelist:
                self.segdirs.append(trajdir+'/seg_'+camname)
                mkdir(self.segdirs[-1])
            if 'CubeScene' in self.imgtypelist:
                self.imgdirs_cube.append(join(trajdir, 'cube_image_%s' % camname))
                mkdir(self.imgdirs_cube[-1])
            if 'CubeDistance' in self.imgtypelist:
                self.distdirs_cube.append(join(trajdir, 'cube_dist_%s' % camname))
                mkdir(self.distdirs_cube[-1])
            # create pose file
            self.posefilelist.append(trajdir+'/pose_'+camname+'.txt')
            self.posenplist.append([])

    def init_folders(self, traj_folder, disable_timestr=False):
        '''
        traj_folder: string that denotes the folder name, e.g. T000
        '''
        if not isdir(self.datadir):
            mkdir(self.datadir)
        else: 
            print ('Data folder already exists.. {}'.format(self.datadir))

        trajdir = join(self.datadir, traj_folder)
        if not isdir(trajdir):
            mkdir(trajdir)
            self.create_folders(trajdir)
        elif disable_timestr:
            print ('Trajectory folder already exists! {}, but forced to overwrite the data.'.format(trajdir))
            self.create_folders(trajdir)
        else:
            print ('Trajectory folder already exists! {}, create folder with time stamp.'.format(trajdir))
            trajdir = join(self.datadir, traj_folder + '_' + time.strftime('%m%d_%H%M%S',time.localtime()))
            mkdir(trajdir)
            self.create_folders(trajdir)

    def data_sampling(self, positions, trajname, random_orientation=True, save_data=True, MAXTRY=3, disable_timestr=False): 

        self.init_folders(trajname, disable_timestr)

        with open(self.logfile,'a') as f:
            f.write('Sample trajname '+ trajname+'\n')

        if random_orientation:
            self.randquaternionsampler.reset(len(positions))
        start = time.time()
        for k,pose in enumerate(positions):
            position = Vector3r(pose[0], pose[1], pose[2])

            if random_orientation:
                orientation = self.randquaternionsampler.next_quaternion(k)
            else:
                orientation = Quaternionr(pose[3], pose[4], pose[5], pose[6])

            dronepose = Pose(position, orientation)
            self.imgclient.setpose(dronepose)

            if save_data:

                time.sleep(0.02)
                self.imgclient.simPause(True)
                rgblist, depthlist, seglist, rgblist_cube, distlist_cube, camposelist = \
                    self.imgclient.readimgs()
                # handle read image error
                if rgblist is None:
                    # try read the images again
                    print ('  !!Error read image: {}-{}: {}'.format(trajname, k, pose))
                    for s in range(MAXTRY):
                        time.sleep(0.01)
                        rgblist, depthlist, seglist, camposelist = self.imgclient.readimgs()
                        if rgblist is not None:
                            break
                        else:
                            print ('    !!Error retry read image: Retry {}'.format(s))

                self.imgclient.simPause(False)
            else:
                camposelist = []
                for camind in self.camlist:
                    camposelist.append([pose[0], pose[1], pose[2],orientation.x_val,orientation.y_val,orientation.z_val,orientation.w_val])

            if rgblist is None:
                print ('  !!Can not recover from read image error {}'.format(trajname))
                return

            # save images and poses
            imgprefix = str(k).zfill(6)+'_'
            for w,camind in enumerate(self.camlist):
                camname = self.camlist_name[camind]
                if save_data:
                    # save RGB image
                    if 'Scene' in self.imgtypelist:
                        img = rgblist[w] # change bgr to rgb
                        cv2.imwrite(join(self.imgdirs[w], imgprefix+camname+'.png'), img)
                    # save depth image
                    if 'DepthPlanner' in self.imgtypelist:
                        depthimg = depthlist[w]
                        np.save(join(self.depthdirs[w], imgprefix+camname+'_depth.npy'), depthimg)
                    # save segmentation image
                    if 'Segmentation' in self.imgtypelist:
                        segimg = seglist[w]
                        np.save(join(self.segdirs[w],imgprefix+camname+'_seg.npy'),segimg)
                    # save cube scene.
                    if 'CubeScene' in self.imgtypelist:
                        img = rgblist_cube[w]
                        cv2.imwrite(
                            join(self.imgdirs_cube[w], '%s%s_cube.png' % (imgprefix, camname)), 
                            img)
                    # save cube distance.
                    if 'CubeDistance' in self.imgtypelist:
                        distimg = distlist_cube[w]
                        cv2.imwrite(
                            join(self.distdirs_cube[w], '%s%s_dist_cube.png' % (imgprefix, camname)),
                            np.expand_dims(distimg, axis=-1).view('<u1'))
                # write pose to file
                self.posenplist[w].append(np.array(camposelist[w]))

                # imgshow = np.concatenate((leftimg,rightimg),axis=1)
            print ('  {0}, pose {1}, orientation ({2:.2f},{3:.2f},{4:.2f},{5:.2f})'.format(k, pose, orientation.x_val, orientation.y_val, orientation.z_val, orientation.w_val))
            # cv2.imshow('img',imgshow)
            # cv2.waitKey(1)

        for w in range(len(self.camlist)):
            # save poses into numpy txt
            np.savetxt(self.posefilelist[w], self.posenplist[w])

        end = time.time()
        print('Trajectory sample time {}'.format(end - start))

        with open(self.logfile,'a') as f:
            f.write('    Success! Traj len: {}, time {} min. \n'.format(len(positions), (end-start)/60.0))

    def close(self,):
        self.imgclient.close()

class MultiviewDataSampler(DataSampler):
    def __init__(self, data_dir, args):

        super(MultiviewDataSampler, self).__init__(data_dir, args)

        self.randquaternionsampler = RandQuaternionSampler(self.args)
        self.rigtypelist = self.args.rig_type.split('_')
        self.parallel_multiview = self.args.load_posefile_parallelmultiview
        self.nbatch = self.args.nbatch
        self.idxid = self.args.idxid

    def initMVConfig(self, configPath):
        
        with open(configPath) as configFile:
            mv_config = json.load(configFile)
            assert(len(mv_config["upPos"]) == len(mv_config["downPos"]))

            self.upPosList = []
            self.downPosList = []
            self.numPairs = len(mv_config["upPos"])

            self.z_bounds = mv_config["zbound"]
            self.noise_radius = mv_config["radius"]
            self.minbaseline = mv_config["minbaseline"]

            #All top and then all bot
            for pair in mv_config["upPos"]:
                self.upPosList.append(np.array(pair))

            for pair in mv_config["downPos"]:
                self.downPosList.append(np.array(pair))

    '''
    def sampleExtrinsics(self, rigpose, addNoise = True, MAXTRIES = 5):

        for 

        return pose_list
    '''

    def sampleExtrinsics(self, rigpose, addNoise = True, MAXTRIES = 5):

        six_oris = dict(
                f = Quaternionr(0, 0, 0, 1),                    #Forward Image Orientation
                b = Quaternionr(0, 0, 1, 0),                    #Backward Image Orientation
                l = Quaternionr(0, 0, -0.7071068, 0.7071068),   #Left Image Orientation
                r = Quaternionr(0, 0, 0.7071068, 0.7071068),    #Right Image Orientation
                u = Quaternionr(0, 0.7071068, 0, 0.7071068),    #Up Image Orientation
                d = Quaternionr(0, -0.7071068, 0, 0.7071068)    #Down Image Orientation
            )

        pose_list = []
        for k in six_oris:
            pose_list.append(Pose(rigpose.position, six_oris[k]))

        return pose_list

    def create_folders(self, trajdir):
        self.imgdirs = []
        self.depthdirs = []
        self.segdirs = []
        self.posefilelist = [] # save incremental pose files
        self.posenplist = [] # save pose in numpy files
        
        for c in range(2*self.numPairs + 1):
            typelist = self.imgtypelist if c < 2*self.numPairs else self.rigtypelist
            topbotname = "top_" if c < self.numPairs else "bot_"
            camname = topbotname + "cam" + str(c%self.numPairs) if c < 2*self.numPairs else "rig"
            
            if 'Scene' in typelist:
                self.imgdirs.append(trajdir+'/image_'+camname)
                if not os.path.exists(self.imgdirs[-1]):
                    mkdir(self.imgdirs[-1])
            if 'DepthPlanner' in typelist:
                self.depthdirs.append(trajdir+'/depth_'+camname)
                if not os.path.exists(self.depthdirs[-1]):
                    mkdir(self.depthdirs[-1])
            if 'Segmentation' in typelist:
                self.segdirs.append(trajdir+'/seg_'+camname)
                if not os.path.exists(self.segdirs[-1]):
                    mkdir(self.segdirs[-1])
            if 'CubeScene' in typelist:
                self.imgdirs_cube.append(join(trajdir, 'cube_image_%s' % camname))
                if not os.path.exists(self.imgdirs_cube[-1]):
                    mkdir(self.imgdirs_cube[-1])
            if 'CubeDistance' in typelist:
                self.distdirs_cube.append(join(trajdir, 'cube_dist_%s' % camname))
                if not os.path.exists(self.distdirs_cube[-1]):
                    mkdir(self.distdirs_cube[-1])
            # create pose file
            if self.parallel_multiview:
                if self.stopIdx % self.nbatch != 0:
                    tempstopIdx = self.stopIdx + (self.nbatch - (self.stopIdx % self.nbatch))
                else:
                    tempstopIdx = self.stopIdx
                self.batch_idx = (self.numPos//self.nbatch)-((self.numPos-tempstopIdx)//self.nbatch)-1
                self.posefilelist.append(trajdir+'/poses/pose_'+camname+"_{}.txt".format(self.batch_idx))
            else:
                self.posefilelist.append(trajdir+'/poses/pose_'+camname+".txt")
            self.posenplist.append([])
        
        if not os.path.exists(trajdir + "/poses"):
            mkdir(trajdir + "/poses")

        print("------------------------------")
        print(self.imgdirs_cube)
        print(self.distdirs_cube)
        print("------------------------------")
            

    def data_sampling(self, positions, trajname, random_orientation=True, save_data=True, MAXTRY=3, disable_timestr=False,
                      stepsize = 5, extNoiseOn = False, startIdx = 0, stopIdx = None):

        self.startIdx = startIdx
        self.stopIdx = stopIdx
        self.numPos = len(positions)
        random_orientation = True

        configPath = join(join(self.datadir, trajname), "multiview.json")
        self.initMVConfig(configPath)
        self.init_folders(trajname, disable_timestr)

        with open(self.logfile,'a') as f:
            f.write('Sample trajname '+ trajname+'\n')

        if random_orientation:
            self.randquaternionsampler.reset(len(positions))

        if self.stopIdx is None:
            self.stopIdx = len(positions)

        start = time.time()
        for k,pose in enumerate(positions[self.startIdx:self.stopIdx]):
            if k % stepsize == 0:
                position = Vector3r(pose[0], pose[1], pose[2])

                if random_orientation:
                    randquat = self.randquaternionsampler.random_quaternion()
                    randyaw, _ = self.randquaternionsampler.init_random_yaw()
                    rq = randquat * randyaw
                    rq, _ = self.randquaternionsampler.clip_quaternion(randquat)
                    orientation = Quaternionr(rq.x, rq.y, rq.z, rq.w)
                else:
                    #orientation = Quaternionr(pose[3], pose[4], pose[5], pose[6])
                    orientation = Quaternionr(0, 0, 0, 1)

                rigpose = Pose(position, orientation)
                cam_extrin_list = self.sampleExtrinsics(rigpose, addNoise = extNoiseOn)

                for w, multi_pose in enumerate([*cam_extrin_list, rigpose]):
                    typelist = self.imgtypelist if w < 2*self.numPairs else self.rigtypelist
                    topbotname = "top_" if w < self.numPairs else "bot_"
                    camname = topbotname + "cam" + str(w%self.numPairs) if w < 2*self.numPairs else "rig"

                    if w == 2*self.numPairs:
                        self.imgclient.populateRequests(self.imgclient.CAMLIST, self.rigtypelist)
                    elif w == 0:
                        self.imgclient.populateRequests(self.imgclient.CAMLIST, self.imgtypelist)
                    

                    self.imgclient.setpose(multi_pose)

                    start_retr = time.time()
                    if save_data:

                        time.sleep(0.02)
                        self.imgclient.simPause(True)
                        rgblist, depthlist, seglist, rgblist_cube, distlist_cube, camposelist = \
                            self.imgclient.readimgs()
                        # handle read image error
                        if rgblist is None:
                            # try read the images again
                            print ('  !!Error read image: {}-{}: {}'.format(trajname, k, multi_pose))
                            for s in range(MAXTRY):
                                time.sleep(0.01)
                                rgblist, depthlist, seglist, camposelist = self.imgclient.readimgs()
                                if rgblist is not None:
                                    break
                                else:
                                    print ('    !!Error retry read image: Retry {}'.format(s))

                        if rgblist is None:
                            print ('  !!Can not recover from read image error {}'.format(trajname))
                            return

                        self.imgclient.simPause(False)
                    print("Retrieving Time: ", time.time()-start_retr)

                    camposelist = []
                    for camind in self.camlist:
                        p = multi_pose.position
                        o = multi_pose.orientation
                        camposelist.append([p.x_val, p.y_val, p.z_val, o.x_val, o.y_val, o.z_val, o.w_val])

                    imgprefix = str(k+self.startIdx).zfill(6)+'_'
                    start_write = time.time()
                    if save_data:
                        # save RGB image
                        if 'Scene' in typelist:
                            img = rgblist[0] # change bgr to rgb
                            cv2.imwrite(join(self.imgdirs[w], imgprefix+camname+'.png'), img)
                        # save depth image
                        if 'DepthPlanner' in typelist:
                            depthimg = depthlist[0]
                            np.save(join(self.depthdirs[w], imgprefix+camname+'_depth.npy'), depthimg)
                        # save segmentation image
                        if 'Segmentation' in typelist:
                            segimg = seglist[0]
                            np.save(join(self.segdirs[w],imgprefix+camname+'_seg.npy'),segimg)
                        # save cube scene.
                        if 'CubeScene' in typelist:
                            img = rgblist_cube[0]
                            cv2.imwrite(
                                join(self.imgdirs_cube[w], '%s%s_cube.png' % (imgprefix, camname)), 
                                img)
                        # save cube distance.
                        if 'CubeDistance' in typelist:
                            distimg = distlist_cube[0]
                            print("{} To {}|Writing... ".format(self.startIdx,self.stopIdx),
                                    '%s%s_dist_cube.png' % (imgprefix, camname))
                            cv2.imwrite(
                                join(self.distdirs_cube[0], '%s%s_dist_cube.png' % (imgprefix, camname)),
                                np.expand_dims(distimg, axis=-1).view('<u1'))
                    
                    # write pose to file
                    self.posenplist[w].append(np.array(camposelist[0]))
                    print("Writing Time: ", time.time()-start_write)

                    # imgshow = np.concatenate((leftimg,rightimg),axis=1)
                    print ('~{0}:  {1}, pose {2}, orientation ({3:.2f},{4:.2f},{5:.2f},{6:.2f})'.format(camname, k+self.startIdx, multi_pose, multi_pose.orientation.x_val, 
                                                                                                    multi_pose.orientation.y_val, multi_pose.orientation.z_val,
                                                                                                    multi_pose.orientation.w_val))
                print ('  {0}, pose {1}, orientation ({2:.2f},{3:.2f},{4:.2f},{5:.2f})'.format(k, pose, orientation.x_val, orientation.y_val, orientation.z_val, orientation.w_val))
            # cv2.imshow('img',imgshow)
            # cv2.waitKey(1)

        for w in range(2*self.numPairs +1):
             # save poses into numpy txt
             np.savetxt(self.posefilelist[w], self.posenplist[w])

        end = time.time()
        print('Trajectory sample time {}'.format(end - start))

        with open("/Flags/flag_python_{}.txt".format(self.idxid),'w') as sampling_flag:
            sampling_flag.write("\n")

        with open(self.logfile,'a') as f:
            f.write('    Success! Traj len: {}, time {} min. \n'.format(len(positions), (end-start)/60.0))


def collect_data_files(args):
    '''
    1. args.load_posefile = False: collect data from position file, using randomly generated orientation
    2. args.load_posefile = True:  collect data from pose file
    Example: python collect_images.py --environment-dir oldtown --position-folder position --rand-degree 60 --smooth-count 20 --max-pitch 30 --min-pitch -45 --max-roll 20 --min-roll -20    
    '''
    metadir = args.environment_dir
    posefolder = args.position_folder

    # output data folder
    if args.data_folder_suffix != '':
        foldername = 'Data_'+args.data_folder_suffix
    else:
        foldername = 'Data'
    datasampler = DataSampler(join(metadir,foldername), args)

    posfiles = listdir(join(metadir, posefolder))
    posfiles = [ff for ff in posfiles if ff[-3:]=='txt']
    posfiles.sort()

    for posfilename in posfiles:
        outfoldername = posfilename.split('.txt')[0]
        print ('*** {} ***'.format(outfoldername))

        positions = np.loadtxt(join(metadir, posefolder, posfilename))
        datasampler.data_sampling(positions, outfoldername, 
            random_orientation=(not args.load_posefile), save_data=(not args.save_posefile_only))

    datasampler.close()

def collect_data_from_leftright_poses(args):
    '''
    Used for resample images along same trajectory according to pose_left.txt and pose_right.txt
    args.load_posefile = True and args.load_posefile_left_right = True: collect data from left right posefiles
    left right pose file are located in [foldername]/P000/pose_left.txt and [foldername]/P000/pose_right.txt
    Example: python collect_images.py --environment-dir oldtown --load-posefile --load-posefile-left-right
    '''
    metadir = args.environment_dir
    posefolder = args.position_folder

    if args.data_folder_suffix != '':
        foldername = 'Data_'+args.data_folder_suffix
    else:
        foldername = 'Data'
    datasampler = DataSampler(join(metadir,foldername), args)

    subfolders = listdir(join(metadir, foldername))
    subfolders = [ff for ff in subfolders if ff[0]=='P']
    subfolders.sort()

    print(subfolders)

    for subfolder in subfolders:
        print ('*** {} ***'.format(subfolder))
        subfolderpath = join(metadir, foldername, subfolder)
        leftposes = np.loadtxt(join(subfolderpath, 'pose_left.txt'))
        rightposes = np.loadtxt(join(subfolderpath, 'pose_right.txt'))
        positions = (leftposes+rightposes)/2.0

        datasampler.data_sampling(positions, subfolder, 
            random_orientation=(not args.load_posefile), save_data=(not args.save_posefile_only),
            disable_timestr=True)

    datasampler.close()

def collect_data_from_multiview(args):
    '''
    Used for resample images along same trajectory according to pose_left.txt and pose_right.txt
    Will collect multiple top and bottom images, simulating the openresearch drone. Use only 'Front' camera
    args.load_posefile = True and args.load_posefile_left_right = True: collect data from left right posefiles
    left right pose file are located in [foldername]/P000/pose_left.txt and [foldername]/P000/pose_right.txt
    Example: python collect_images.py --environment-dir oldtown --load-posefile --load-posefile-left-right
    '''
    metadir = args.environment_dir
    posefolder = args.position_folder

    if args.data_folder_suffix != '':
        foldername = 'Data_'+args.data_folder_suffix
    else:
        foldername = 'Data'
    datasampler = MultiviewDataSampler(join(metadir,foldername), args)

    subfolders = listdir(join(metadir, foldername))
    subfolders = [ff for ff in subfolders if ff[0]=='P']
    subfolders.sort()

    print(subfolders)

    for subfolder in subfolders:
        print ('*** {} ***'.format(subfolder))
        subfolderpath = join(metadir, foldername, subfolder)

        if os.path.exists(join(subfolderpath, 'pose_left.txt')):
            leftposes = np.loadtxt(join(subfolderpath, 'pose_left.txt'))
            rightposes = np.loadtxt(join(subfolderpath, 'pose_right.txt'))
            positions = (leftposes+rightposes)/2.0
        else:
            positions = np.loadtxt(join(subfolderpath, '{}.txt'.format(subfolder)))

        datasampler.data_sampling(positions, subfolder, 
            random_orientation=True, save_data=(not args.save_posefile_only),
            disable_timestr=True, extNoiseOn = args.add_extrinsic_noise, stepsize = args.frame_stepsize)

    datasampler.close()

def collect_data_from_parallelmultiview(args):
    '''
    Used for resample images along same trajectory according to pose_left.txt and pose_right.txt
    Will collect multiple top and bottom images, simulating the openresearch drone. Use only 'Front' camera
    args.load_posefile = True and args.load_posefile_left_right = True: collect data from left right posefiles
    left right pose file are located in [foldername]/P000/pose_left.txt and [foldername]/P000/pose_right.txt
    Example: python collect_images.py --environment-dir oldtown --load-posefile --load-posefile-left-right
    '''
    trajdir = args.environment_dir
    datadir = trajdir[:(len(trajdir)-4)]
    trajname = trajdir[(len(trajdir)-4):]
    datasampler = MultiviewDataSampler(datadir, args)

    if os.path.exists(join(trajdir, 'pose_left.txt')):
        leftposes = np.loadtxt(join(trajdir, 'pose_left.txt'))
        rightposes = np.loadtxt(join(trajdir, 'pose_right.txt'))
        positions = (leftposes+rightposes)/2.0
    else:
        positions = np.loadtxt(join(trajdir, '{}.txt'.format(trajname)))


    datasampler.data_sampling(positions, trajname, 
        random_orientation=True, save_data=(not args.save_posefile_only),startIdx=args.start,
        stopIdx=args.stop,disable_timestr=True, extNoiseOn = args.add_extrinsic_noise,
        stepsize = args.frame_stepsize)

    datasampler.close()

# Sample commands.
# python collection/collect_images.py \
# --environment-dir <output_dir> \
# --load-posefile \
# --load-posefile-left-right \
# --cam-list 1_2 \
# --img-typ Scene_DepthPlanner_Segmentation_CubeScene_CubeDistance

if __name__ == '__main__':

    args = get_args()

    if args.load_posefile and args.load_posefile_left_right:
        collect_data_from_leftright_poses(args)
    if args.load_posefile and args.load_posefile_multiview:
        collect_data_from_multiview(args)
    if args.load_posefile and args.load_posefile_parallelmultiview:
        collect_data_from_parallelmultiview(args)
    else:
        collect_data_files(args)
        

    # # test the RandQuaternionSampler
    # randquaternionsampler = RandQuaternionSampler(args)
    # randquaternionsampler = RandQuaternionSplineSampler(args)
    # anglelist = []
    # randquaternionsampler.reset(1000)
    # for k in range(1000):
    #     quatr = randquaternionsampler.next_quaternion(k)
    #     (pitch, roll, yaw) = to_eularian_angles(quatr)
    #     anglelist.append([roll,pitch, yaw])

    # anglelist = np.array(anglelist) * 180.0/pi
    # plt.plot(anglelist[:,0],'.-')
    # plt.plot(anglelist[:,1],'.-')
    # plt.plot(anglelist[:,2],'.-')     
    # plt.legend(['roll', 'pitch', 'yaw']) 
    # plt.show()  

    # test the RandQuaternionSampler again
    # import matplotlib.pyplot as plt
    # length = 1000
    # smoothcount = args.smooth_count
    # randquaternionsampler = RandQuaternionSplineSampler(args)
    # angles = randquaternionsampler.reset(length)
    # anglelist = []
    # for k in range(length):
    #     quatr = randquaternionsampler.next_quaternion(k)
    #     (pitch, roll, yaw) = to_eularian_angles(quatr)
    #     anglelist.append([roll,pitch, yaw])
    # anglelist = np.array(anglelist)*180/pi
    # anglelist2 = randquaternionsampler.anglist

    # angles = np.array(angles)*180/pi
    # yawdiff = anglelist[1:,2] - anglelist[:-1,2]
    # yawdiff[yawdiff>180] = yawdiff[yawdiff>180]-360
    # yawdiff[yawdiff<-180] = yawdiff[yawdiff<-180]+360
    # plt.plot(range(0, length + smoothcount, smoothcount),angles[:,0], 'rx')
    # plt.plot(range(0, length + smoothcount, smoothcount),angles[:,1], 'gx')
    # plt.plot(range(0, length + smoothcount, smoothcount),angles[:,2], 'bx')
    # plt.plot(anglelist[:,0], 'r.')
    # plt.plot(anglelist[:,1], 'g.')
    # plt.plot(anglelist[:,2], 'b.')

    # plt.plot(anglelist2[:,0], 'r-')
    # plt.plot(anglelist2[:,1], 'g-')
    # plt.plot(anglelist2[:,2], 'b-')
    # # plt.plot(yawdiff, '.-')    
    # plt.grid()
    # plt.show()


    # import matplotlib.pyplot as plt
    # length = 1000
    # smoothcount = args.smooth_count
    # randquaternionsampler = RandQuaternionSplineSampler(args)
    # quats = randquaternionsampler.reset(length)
    # # angles = np.array(angles)*180/pi

    # quats_euler_sparse = []
    # for quat in quats:
    #     quats_euler_sparse.append(quatarray2eular(quat))
    # quats_euler_sparse = np.array(quats_euler_sparse)

    # plt.figure(figsize=(18,10))
    # plt.subplot(221)
    # plt.plot(range(0, length + smoothcount, smoothcount),angles[:,0], 'rx')
    # plt.plot(range(0, length + smoothcount, smoothcount),angles[:,1], 'gx')
    # plt.plot(range(0, length + smoothcount, smoothcount),angles[:,2], 'bx')
    # plt.plot(range(0, length + smoothcount, smoothcount),quats_euler[:,0], 'r.')
    # plt.plot(range(0, length + smoothcount, smoothcount),quats_euler[:,1], 'g.')
    # plt.plot(range(0, length + smoothcount, smoothcount),quats_euler[:,2], 'b.')

    # quatlist = randquaternionsampler.orilist_angs
    # anglelist = randquaternionsampler.anglist_angs
    # quats_euler = []
    # for quat in quatlist:
    #     quats_euler.append(quatarray2eular(quat))
    # quats_euler = np.array(quats_euler)
    # plt.subplot(222)
    # plt.plot(range(0, length + smoothcount, smoothcount),angles[:,0], 'rx')
    # plt.plot(range(0, length + smoothcount, smoothcount),angles[:,1], 'gx')
    # plt.plot(range(0, length + smoothcount, smoothcount),angles[:,2], 'bx')
    # plt.plot(anglelist[:,0], 'r.')
    # plt.plot(anglelist[:,1], 'g.')
    # plt.plot(anglelist[:,2], 'b.')
    # plt.plot(quats_euler[:,0], 'r-')
    # plt.plot(quats_euler[:,1], 'g-')
    # plt.plot(quats_euler[:,2], 'b-')


    # quatlist = randquaternionsampler.orilist_quats
    # # anglelist = randquaternionsampler.anglist_quats
    # quats_euler = []
    # for quat in quatlist:
    #     quats_euler.append(quatarray2eular(quat))
    # quats_euler = np.array(quats_euler)
    # # quatlist = randquaternionsampler.orilist
    # # quats_euler2 = []
    # # for quat in quatlist:
    # #     quats_euler2.append(quatarray2eular(quat))
    # # quats_euler2 = np.array(quats_euler2)
    # # # plt.subplot(223)
    # plt.plot(range(0, length + smoothcount, smoothcount),quats_euler_sparse[:,0], 'rx')
    # plt.plot(range(0, length + smoothcount, smoothcount),quats_euler_sparse[:,1], 'gx')
    # plt.plot(range(0, length + smoothcount, smoothcount),quats_euler_sparse[:,2], 'bx')
    # # plt.plot(quats_euler2[:,0], 'r.')
    # # plt.plot(quats_euler2[:,1], 'g.')
    # # plt.plot(quats_euler2[:,2], 'b.')
    # plt.plot(quats_euler[:,0], 'r-')
    # plt.plot(quats_euler[:,1], 'g-')
    # plt.plot(quats_euler[:,2], 'b-')
    
    # plt.legend(['roll', 'pitch', 'yaw']) 
    # plt.grid()
    # plt.show()

