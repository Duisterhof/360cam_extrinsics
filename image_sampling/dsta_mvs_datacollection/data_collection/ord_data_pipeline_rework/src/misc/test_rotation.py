import numpy as np
# from airsim.types import Pose, Vector3r, Quaternionr
# from airsim.utils import to_eularian_angles, to_quaternion
# from os.path import join
import matplotlib.pyplot as plt
from os.path import join
# from math import pi
import cv2

import airsim

# pathdir = '/home/wenshan/tmp/maps/seasonsforest/Data_test2/P002/'
# posfile = 'pose_left.txt'

# poses = np.loadtxt(join(pathdir, posfile))

# anglelist = []
# for pose in poses:
# 	orientation = pose[3:]
# 	quat = Quaternionr(orientation[0], orientation[1], orientation[2], orientation[3])

# 	(pitch, roll, yaw) = to_eularian_angles(quat)
# 	anglelist.append([roll*180/pi, pitch*180/pi, yaw*180/pi])

# anglelist = np.array(anglelist)
# plt.plot(anglelist[:,0],'.-')
# plt.plot(anglelist[:,1],'.-')
# plt.plot(anglelist[:,2],'.-')

# # plt.plot(anglelist[:-1,2] - anglelist[1:,2],'.-')

# plt.grid()
# plt.show()
# # import ipdb;ipdb.set_trace()


# # test left right posefiles
# posedir = '/home/wenshan/tmp/maps/seasonsforest/pose_iros/P000'
# positions = np.loadtxt(join(posedir,'P001.txt'))
# leftposes = np.loadtxt(join(posedir,'pose_left.txt'))
# rightposes = np.loadtxt(join(posedir,'pose_right.txt'))

# showlen = 100
# plt.figure(figsize=(10,10))
# plt.plot(positions[:showlen,0],positions[:showlen,1],'x-')
# plt.plot(leftposes[:showlen,0],leftposes[:showlen,1],'.-')
# plt.plot(rightposes[:showlen,0],rightposes[:showlen,1],'.-')
# plt.grid()
# plt.show()

# xx=(leftposes[:,0]+rightposes[:,0])/2.0
# yy=(leftposes[:,1]+rightposes[:,1])/2.0
# zz=(leftposes[:,2]+rightposes[:,2])/2.0
# xxdiff=xx-positions[:,0]
# yydiff=yy-positions[:,1]
# zzdiff=zz-positions[:,2]
# angdiff=leftposes[:,3:]-rightposes[:,3:]

# import ipdb;ipdb.set_trace()

# # fix analysis file name caused by bug
# from settings import get_args
# from os.path import isfile, join, isdir
# from os import listdir, mkdir, environ, system

# args = get_args()
# data_root_dir = args.data_root
# data_folders = args.data_folders.split(',')

# if args.env_folders=='': # read all available folders in the data_root_dir
#     env_folders = listdir(data_root_dir)    
# else:
#     env_folders = args.env_folders.split(',')
# print('Detected envs {}'.format(env_folders))

# filelist = ['_disp_hist.png',
# 			'_disp_max.npy',
# 			'_disp_mean.npy',
# 			'_disp_min.npy',
# 			'_left_file_index_all.txt',
# 			'_rgb_mean.npy',
# 			'_rgb_mean_std.png',
# 			'_rgb_std.npy']

# for env_folder in env_folders:
#     env_dir = join(data_root_dir, env_folder)
#     ana_dir = join(env_dir, 'analyze')
#     datapath = join(env_dir, data_folders[0])

#     trajfolders = listdir(datapath)
#     trajfolders = [ tf for tf in trajfolders if tf[0]=='P' ]
#     trajfolders.sort()
#     print('    Found {} trajectories'.format(len(trajfolders)))

#     for trajfolder in trajfolders:
#     	for ff in filelist:
#     		fff = join(ana_dir, trajfolder+ff)
#     		target_ff = join(ana_dir, trajfolder+'_'+data_folders[0]+ff)
#     		if isfile(fff):
#     			system('mv '+fff+' '+ target_ff)
#     		else:
#     			print('ERROR: MISSING FILE {}'.format(fff))

vidfile = '/home/wenshan/Videos/test_airsim_cams.mp4'
cap = cv2.VideoCapture(vidfile)

while(cap.isOpened()):
    ret, frame = cap.read()
    cv2.imshow('frame',frame)
    if cv2.waitKey(0) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()