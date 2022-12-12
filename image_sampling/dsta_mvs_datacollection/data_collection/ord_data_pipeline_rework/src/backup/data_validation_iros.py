'''
Generates video for data overview
Analyze depth info mostly for stereo problem
'''
import cv2
import numpy as np
from os.path import isfile, join, isdir
from os import listdir, mkdir, environ
import time
from data_validation import FileLogger,ImageReader

import matplotlib.pyplot as plt
if ( not ( "DISPLAY" in environ ) ):
    plt.switch_backend('agg')
    print("Environment variable DISPLAY is not present in the system.")
    print("Switch the backend of matplotlib to agg.")


class DataVerifier(object):
    '''
    inputdir - the root data folder, which contains 'image_left', 'image_right',
    'depth_left', 'depth_right', 'seg_left', 'seg_right'
    - generate a video
    - depth statistics
    '''
    def __init__(self, inputdir):
        self.inputdir = inputdir
        self.leftfolder = join(inputdir, 'image_left')
        self.rightfolder = join(inputdir, 'image_right')
        self.leftdepthfolder = join(inputdir, 'depth_left')
        self.rightdepthfolder = join(inputdir, 'depth_right') 
        self.leftsegfolder = join(inputdir, 'seg_left')
        self.rightsegfolder = join(inputdir, 'seg_right') 

        self.leftsuffix = '_left.png' # left rgb image
        self.rightsuffix = '_right.png' # right rgb image
        self.leftdepthsuffix = '_left_depth.npy' # left depth image
        self.rightdepthsuffix = '_right_depth.npy' # right depth image
        self.leftsegsuffix = '_left_seg.npy' # left segmentation image
        self.rightsegsuffix = '_right_seg.npy' # right segmentation image

        self.imgreader = ImageReader()

        self.leftlist = listdir(self.leftfolder)
        self.leftlist = [ff for ff in self.leftlist if ff[-3:]=='png']
        self.imgnum = len(self.leftlist)


    def save_vid_with_depth_statistics(self, outvidfile, logf, scale=0.5, startind=0, check_depth=True, skip_seg=False, skip_depth=False): 
        '''
        outvidfile: xxx.mp4
        scale: scale the image in the video
        startind: the image index does not start from 0
        check_depth: put text on the image about the depth statistics
        '''
        dummyfn = join(self.leftfolder, self.leftlist[0])
        imgleft = self.imgreader.read_rgb(dummyfn, scale)
        (imgh, imgw, _) = imgleft.shape

        fourcc = cv2.VideoWriter_fourcc(*'MP4V')
        if skip_depth:
            fout=cv2.VideoWriter(outvidfile, fourcc, 20.0, (imgw, imgh*2))
        else:
            fout=cv2.VideoWriter(outvidfile, fourcc, 20.0, (imgw*2, imgh*2))

        self.imgreader.reset_colormap()

        for k in range(startind, startind+self.imgnum): # it should be a concequtive sequence starting from 0
            indstr = str(k).zfill(6)
            leftfile = join(self.leftfolder,indstr+self.leftsuffix)
            if isfile(leftfile):
                imgleft = self.imgreader.read_rgb(leftfile, scale)
                if imgleft is None:
                    logf.logline('Left file error ' + leftfile)
                    print 'left file error', leftfile
                    continue
            else:
                logf.logline('Left file missing ' + leftfile)
                print 'left file missing', leftfile
                continue

            rightfile = join(self.rightfolder,indstr+self.rightsuffix)
            if isfile(rightfile):
                imgright = self.imgreader.read_rgb(rightfile, scale)
                if imgright is None:
                    logf.logline('Rright file error ' + rightfile)
                    print 'right file error', rightfile
                    continue
            else:
                logf.logline('Rright file missing ' + rightfile)
                print 'right file missing', rightfile
                continue

            if not skip_depth:
                leftdepthfile = join(self.leftdepthfolder,indstr+self.leftdepthsuffix)
                if isfile(leftdepthfile):
                    depthleft = self.imgreader.read_depth(leftdepthfile, scale, maxthresh = 50)
                else:
                    logf.logline('Left depth file missing ' + leftdepthfile)
                    print 'left depth file missing', leftdepthfile
                    continue

                rightdepthfile = join(self.rightdepthfolder,indstr+self.rightdepthsuffix)
                if isfile(rightdepthfile):
                    depthright = self.imgreader.read_depth(rightdepthfile, scale, maxthresh = 50)
                else:
                    logf.logline('Left depth file missing ' + rightdepthfile)
                    print 'left depth file missing', rightdepthfile
                    continue

            if not skip_seg:
                segfile = join(self.leftsegfolder,indstr+self.leftsegsuffix)
                if isfile(segfile):
                    segleft = self.imgreader.read_seg(segfile, scale)
                else:
                    logf.logline('Seg file missing ' + segfile)
                    print 'seg file missing', segfile
                    continue

            if check_depth:
                # do statistic on depth image, and put on the image
                displeft = self.imgreader.read_disparity(leftdepthfile)

                rightdepthfile = join(self.rightdepthfolder,indstr+self.rightdepthsuffix)
                if isfile(rightdepthfile):                
                    dispright = self.imgreader.read_disparity(rightdepthfile)
                else:
                    logf.logline('Right depth file missing ' + rightdepthfile)
                    print 'right depth file missing', rightdepthfile
                    continue

                pts = np.array([[0,0],[300,0],[300,20],[0,20]],np.int32)
                cv2.fillConvexPoly(imgleft,pts,(70,30,10))
                cv2.putText(imgleft,'{}. meand = {:.2f}, maxd = {:.2f}, mind = {:.2f}'.format(str(k), displeft.mean(), displeft.max(), displeft.min()),
                            (5,15),cv2.FONT_HERSHEY_SIMPLEX,0.4,(0,255,255),thickness=1)

                cv2.fillConvexPoly(imgright,pts,(70,30,10))
                cv2.putText(imgright,'{}. meand = {:.2f}, maxd = {:.2f}, mind = {:.2f}'.format(str(k), dispright.mean(), dispright.max(), dispright.min()),
                            (5,15),cv2.FONT_HERSHEY_SIMPLEX,0.4,(0,255,255),thickness=1)

            imgdisp0 = np.concatenate((imgleft, imgright), 0)
            if skip_depth:
                imgdisp = imgdisp0
            else:
                if skip_seg:
                    imgdisp1 = np.concatenate((depthleft, depthright), 0)
                else:
                    imgdisp1 = np.concatenate((depthleft, segleft), 0)
                imgdisp = np.concatenate((imgdisp0, imgdisp1), 1)
            fout.write(imgdisp)
            # cv2.imshow('img', imgdisp)
            # cv2.waitKey(0)
        fout.release()

    def depth_statistic(self, logf, startind=0):
        '''
        return statistics on depth image
        '''
        dmax = []
        dmin = []
        dmean = []
        leftfileindlist = []
        for k in range(startind, startind+self.imgnum): # it should be a concequtive sequence starting from 0
            indstr = str(k).zfill(6)

            leftdepthfile = join(self.leftdepthfolder,indstr+self.leftdepthsuffix)
            if isfile(leftdepthfile):
                displeft = self.imgreader.read_disparity(leftdepthfile)
            else:
                logf.logline('Left depth file missing ' + leftdepthfile)
                print 'left depth file missing', leftdepthfile
                continue

            rightdepthfile = join(self.rightdepthfolder,indstr+self.rightdepthsuffix)
            if isfile(rightdepthfile):
                dispright = self.imgreader.read_disparity(rightdepthfile)
            else:
                logf.logline('Right depth file missing ' + rightdepthfile)
                print 'right depth file missing', rightdepthfile
                continue

            dmax.append(displeft.max())
            dmin.append(displeft.min())
            dmean.append(displeft.mean())
            dmax.append(dispright.max())
            dmin.append(dispright.min())
            dmean.append(dispright.mean())
            leftfileindlist.append(join(self.leftfolder,indstr))

        return dmean, dmax, dmin, leftfileindlist

def plot_depth_info(dispmean, dispmax, dispmin, disphistfig):
    # save depth statistic figures
    plt.figure(figsize=(12,12))
    showlog = True
    binnum = 500
    plt.subplot(3,1,1)
    plt.hist(dispmean,bins=binnum, log=showlog)
    plt.title('disp mean')
    plt.grid()

    plt.subplot(3,1,2)
    plt.hist(dispmax,bins=binnum, log=showlog)
    plt.title('disp max')
    plt.grid()

    plt.subplot(3,1,3)
    plt.hist(dispmin,bins=binnum, log=showlog)
    plt.title('disp min')
    plt.grid()

    # plt.show()
    plt.savefig(disphistfig)

def save_preview_video(env_root_dir, vid_out_dir = 'video'):
    '''
    Input: Trajectory folder is organized in <env_root_dir>/data_folder[k]/P00X
    In each trajectory folder, image data are in 'image_left', 'image_right',
    'depth_left', 'depth_right', 'seg_left', 'seg_right'. 

    Output: a video for each trajectory: <vid_out_dir>/<datafolder>_P00X.mp4
    '''
    if not isdir(vid_out_dir):
        mkdir(vid_out_dir)

    logf = FileLogger(join(vid_out_dir, 'error.log'))
    if not isdir(env_root_dir):
        logf.logline('Data folder missing ' + env_root_dir)
        print '!!data folder missing', env_root_dir
        return
    print('    Opened data folder {}'.format(env_root_dir))

    trajfolders = listdir(env_root_dir)
    trajfolders = [ tf for tf in trajfolders if tf[0]=='P' ]
    trajfolders.sort()
    print('    Found {} trajectories'.format(len(trajfolders)))

    env_name = env_root_dir.split('/')[-1]
    for trajfolder in trajfolders:
        # generate a video for each trajectory
        outvidfile = join(vid_out_dir, env_name+'_'+trajfolder+'.mp4')
        datavarifier = DataVerifier(join(env_root_dir,trajfolder))
        datavarifier.save_vid_with_depth_statistics(outvidfile, logf, scale=0.5, startind=0, check_depth=False, skip_seg=True, skip_depth=True)
    logf.close()

def analyze_depth_data(env_root_dir, ana_out_dir = 'analyze', info_from_file=False):
    '''
    Input: Trajectory folder is organized in env_root_dir/data_folder[k]/P00X
    In each trajectory folder, image data are in 'image_left', 'image_right',
    'depth_left', 'depth_right', 'seg_left', 'seg_right'. 

    Output: 1. depth info: <env_root_dir>/<ana_out_dir>/disp_mean(max,min).npy
            2. depth histogram: <env_root_dir>/<ana_out_dir>/disp_hist.jpg
            3. index file: <env_root_dir>/<ana_out_dir>/left_file_index_all.txt 
                - content: each line correspond to a file index: <env_root_dir>/<datafolder>/<trajfolder>/<image_left>/000xxx
                - each line does not contain image suffix
    '''
    # check and create output folders
    anaoutdir = join(env_root_dir, ana_out_dir)
    if not isdir(anaoutdir):
        mkdir(anaoutdir)

    # depth statistics
    dispmax = []
    dispmin = []
    dispmean = []
    leftindexlist = []
    env_name = env_root_dir.split('/')[-1]
    dispmaxfile = join(anaoutdir, env_name+'_disp_max.npy')
    dispminfile = join(anaoutdir, env_name+'_disp_min.npy')
    dispmeanfile = join(anaoutdir, env_name+'_disp_mean.npy')
    leftindfile = join(anaoutdir, env_name+'_left_file_index_all.txt') 
    disphistfig = join(anaoutdir, env_name+'_disp_hist.png')

    if info_from_file:
        dispmax = np.load(dispmaxfile)
        dispmean = np.load(dispmeanfile)
        dispmin = np.load(dispminfile)
        print ('    depth info file loaded from file: {}, {}, {}'.format(dispmax.shape, dispmean.shape, dispmin.shape))

    else: # calculate the statistics from depth image
        logf = FileLogger(join(anaoutdir,'error.log'))
        if not isdir(env_root_dir):
            logf.logline('Data folder missing '+ env_root_dir)
            print 'data folder missing', env_root_dir
            return
        print('    Opened data folder {}'.format(env_root_dir))

        trajfolders = listdir(env_root_dir)
        trajfolders = [ tf for tf in trajfolders if tf[0]=='P' ]
        trajfolders.sort()
        print('    Found {} trajectories'.format(len(trajfolders)))

        for trajfolder in trajfolders:
            # generate a video for each trajectory
            datavarifier = DataVerifier(join(env_root_dir,trajfolder))
            # get depth info
            dmean, dmax, dmin, indlist = datavarifier.depth_statistic(logf, startind=0)
            dispmean.extend(dmean)
            dispmax.extend(dmax)
            dispmin.extend(dmin)
            leftindexlist.extend(indlist)

        # generate depth statistic files
        numFileInEnv = len(leftindexlist)
        assert len(dispmean) == numFileInEnv * 2
        assert len(dispmax) == numFileInEnv * 2
        assert len(dispmin) == numFileInEnv * 2

        dispmean = np.array(dispmean)
        dispmax = np.array(dispmax)
        dispmin = np.array(dispmin)
        np.save(dispmeanfile, dispmean)
        np.save(dispmaxfile, dispmax)
        np.save(dispminfile, dispmin)

        with open(leftindfile, 'w') as f:
            for leftind in leftindexlist:
                f.write('%s\n' % leftind)
        print ('    saved depth info file: {}, {}, {}'.format(dispmeanfile, dispmaxfile, dispminfile))
        logf.close()

    plot_depth_info(dispmean, dispmax, dispmin, disphistfig)

def stereo_depth_filter(rootdir, leftimg_ind_file, disp_max_file, disp_mean_file, out_stereo_file, out_stereo_error_file, maxmax = 400, maxmean = 200, minmax=0.4):
    '''
    Input: npy files of depth info 
           list of image indexes
    Output: list of files for the stereo task 
            - left_image_file_path right_image_file_path left_depth_file_path
    Filtering condition
        - Nothing too close (maxmax=400 -> mindisp=80/400=0.2m)
        - No big thing too close (maxmean=200 -> meandisp=80/200=0.4m)
        - No all background (minmax=0.4 -> maxdisp=80/0.4=200m) 
    '''
    # save the filtered file
    f = open(join(rootdir,leftimg_ind_file), 'r')
    lines = f.readlines()
    f.close()

    maxlist = np.load(join(rootdir, disp_max_file))
    meanlist = np.load(join(rootdir, disp_mean_file))
    print ('    input file loaded (index, maxdisp, meandisp): {}, {}, {}'.format(len(lines), maxlist.shape, meanlist.shape))
    assert len(lines) * 2 == maxlist.shape[0]
    assert len(lines) * 2 == meanlist.shape[0]

    logf = FileLogger(join(rootdir,out_stereo_file))
    logfe = FileLogger(join(rootdir,out_stereo_error_file))
    count = 0
    for k, line in enumerate(lines):
        line = line.strip()
        leftind = k*2
        rightind = k*2 + 1

        if maxlist[leftind] < maxmax and maxlist[rightind] < maxmax and \
            meanlist[leftind] < maxmean and meanlist[rightind] < maxmean and \
            maxlist[leftind] > minmax and maxlist[rightind] > minmax: 
            leftimgfile = line + '_left.png'
            rightimgfile = leftimgfile.replace('left', 'right')
            leftdepthfile = line.replace('image_left', 'depth_left') + '_left_depth.npy'
            logf.log(leftimgfile+' '+rightimgfile+' '+leftdepthfile+'\n')
            count += 1
        else: 
            logfe.log(line+'\n')

    logf.close()    
    logfe.close()
    print ('    After filtering: {}'.format(count))

from settings import get_args

if __name__ == '__main__':
    args = get_args()
    data_root_dir = args.data_root

    if args.env_folders=='': # read all available folders in the data_root_dir
        env_folders = listdir(data_root_dir)    
        env_folders = [ff for ff in env_folders if (ff!='video' and ff!='analyze')]
    else:
        env_folders = args.env_folders.split(',')
    print('Detected envs {}'.format(env_folders))

    videodir = join(data_root_dir, 'video')
    mkdir(videodir)
    analyzedir = join(data_root_dir, 'analyze')
    # mkdir(analyzedir)
    create_video = args.create_video
    analyze_depth = args.analyze_depth
    depth_from_file = args.depth_from_file
    depth_filter = args.depth_filter

    for env_folder in env_folders:
        env_dir = join(data_root_dir, env_folder)
        print('Working on env {}'.format(env_dir))
        if create_video:
            print ('  creating video..')
            save_preview_video(env_dir,  vid_out_dir = videodir)
        if analyze_depth:
            print ('  analyzing depth..')
            analyze_depth_data(env_dir,  ana_out_dir = analyzedir, info_from_file=depth_from_file)
        if depth_filter:
            print ('  filtering depth..')
            stereo_depth_filter(join(env_dir, 'analyze'), 
                                    leftimg_ind_file = 'left_file_index_all.txt', 
                                    disp_max_file = 'disp_max.npy', 
                                    disp_mean_file = 'disp_mean.npy', 
                                    out_stereo_file = env_folder+'.txt',
                                    out_stereo_error_file = env_folder+'_error.txt')
