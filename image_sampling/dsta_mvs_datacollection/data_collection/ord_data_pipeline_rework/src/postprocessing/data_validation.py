'''
Generates video for data overview
Analyze depth info mostly for stereo problem
'''
import cv2
import numpy as np
from os.path import isfile, join, isdir
from os import listdir, mkdir, environ
import time
from data_visualization import visflow

import matplotlib.pyplot as plt
if ( not ( "DISPLAY" in environ ) ):
    plt.switch_backend('agg')
    print("Environment variable DISPLAY is not present in the system.")
    print("Switch the backend of matplotlib to agg.")

class FileLogger():
    def __init__(self, filename, overwrite=False):
        if isfile(filename):
            if overwrite:
                print('Overwrite existing file {}'.format(filename))
            else:
                timestr = time.strftime('%m%d_%H%M%S',time.localtime())
                filename = filename+'_'+timestr
        self.f = open(filename, 'w')

    def log(self, logstr):
        self.f.write(logstr)

    def logline(self, logstr):
        self.f.write(logstr+'\n')

    def close(self,):
        self.f.close()

class ImageReader():
    '''
    Read images from files, and return visulizable data 

    '''
    def __init__(self, ):
        self.COLORS = [(128, 0, 0),(165, 42, 42),(220, 20, 60),(255, 99, 71),(205, 92, 92),(233, 150, 122),(255, 160, 122),(255, 140, 0),(255, 215, 0),(218, 165, 32),(128, 128, 0),(154, 205, 50),(107, 142, 35),(127, 255, 0),(102, 205, 170),(32, 178, 170),(0, 128, 128),(0, 255, 255),(127, 255, 212),(95, 158, 160),(100, 149, 237),(30, 144, 255),(135, 206, 235),(25, 25, 112),(0, 0, 255),(138, 43, 226),(72, 61, 139),(123, 104, 238),(139, 0, 139),(255, 0, 255),(199, 21, 133),(255, 20, 147),(255, 182, 193),(160, 82, 45),(205, 133, 63),(222, 184, 135),(112, 128, 144),(176, 196, 222),(255,0,0), (0,255,0), (0,0,255), (255,255,0), (0,255,255), (255,0,255), (128,0,0), (128,128,0), (0,128,0), (128,0,128), (0,128,128), (0,0,128)] 
        self.colors = [self.COLORS[k] for k in [  30, 15, 5, 40, 44, 38, 43, 37, 28, 14, 47, 17, 42, 18, 35, 10,  3, 46,  9, 20, 21, 29, 36, 
        26,  1, 16, 41, 32, 12, 49, 22,  0, 24, 33, 48, 45, 23, 8, 11, 34,  4,  7,  2, 31, 19, 25, 13, 27,  6, 39] ]
        # np.random.permutation(len(self.COLORS))
        self.colormap = {}

    def reset_colormap(self,):
        '''
        reset the color map if there's new env or sequence
        '''
        self.colors = [self.COLORS[k] for k in np.random.permutation(len(self.COLORS)) ]
        self.colormap = {}

    def read_rgb(self, imgpath, scale = 1):
        img = cv2.imread(imgpath)
        if img is None or img.size==0:
            return None
        img = cv2.resize(img, (0,0), fx=scale, fy=scale)

        return img

    def depth2vis(self, depth, maxthresh = 50):
        depthvis = np.clip(depth,0,maxthresh)
        depthvis = depthvis/maxthresh*255
        depthvis = depthvis.astype(np.uint8)
        depthvis = np.tile(depthvis.reshape(depthvis.shape+(1,)), (1,1,3))

        return depthvis

    def read_depth(self, depthpath, scale = 1, maxthresh = 50):
        depth = np.load(depthpath)
        depth = self.depth2vis(depth, maxthresh)
        depth = cv2.resize(depth, (0,0), fx=scale, fy=scale)

        return depth

    def read_disparity(self, depthpath, p = 80.0):
        depth = np.load(depthpath)
        return p/depth

    def seg2vis(self, segnp):
        segvis = np.zeros(segnp.shape+(3,), dtype=np.uint8)

        colorind = 0
        for k in range(256):
            mask = segnp==k
            if np.sum(mask)>0:
                if self.colormap.has_key(k):
                    cind = self.colormap[k]
                else:
                    self.colormap[k] = colorind
                    cind = colorind
                    colorind += 1
                    if colorind>= len(self.colors):
                        colorind = 0
                segvis[mask,:] = self.colors[cind]

        return segvis

    def read_seg(self, segpath, scale = 1):
        seg = np.load(segpath)
        seg = self.seg2vis(seg)
        seg = cv2.resize(seg, (0,0), fx=scale, fy=scale)

        return seg

    def read_flow(self, flowpath, scale = 1):
        flownp = np.load(flowpath)
        flowvis = visflow(flownp)
        flowvis = cv2.resize(flowvis, (0,0), fx=scale, fy=scale)

        return flowvis

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
        self.flowfolder = join(inputdir, 'flow')

        self.leftsuffix = '_left.png' # left rgb image
        self.rightsuffix = '_right.png' # right rgb image
        self.leftdepthsuffix = '_left_depth.npy' # left depth image
        self.rightdepthsuffix = '_right_depth.npy' # right depth image
        self.leftsegsuffix = '_left_seg.npy' # left segmentation image
        self.rightsegsuffix = '_right_seg.npy' # right segmentation image
        self.flowsuffix = '_flow.npy' # right segmentation image

        self.imgreader = ImageReader()

        self.leftlist = listdir(self.leftfolder)
        self.leftlist = [ff for ff in self.leftlist if ff[-3:]=='png']
        self.imgnum = len(self.leftlist)
        self.flownum = self.imgnum-1


    def save_vid_with_depth_statistics(self, outvidfile, logf, scale=0.5, startind=0, check_depth=True, video_with_flow=False): 
        '''
        outvidfile: xxx.mp4
        scale: scale the image in the video
        startind: the image index does not start from 0
        check_depth: put text on the image about the depth statistics
        '''
        dummyfn = join(self.leftfolder, self.leftlist[0])
        imgleft = self.imgreader.read_rgb(dummyfn, scale)
        (imgh, imgw, _) = imgleft.shape
        flow_end_ind = startind + self.flownum

        fourcc = cv2.VideoWriter_fourcc(*'MP4V')
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

            if not video_with_flow:
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
            else: 
                if k<flow_end_ind: 
                    indstr2 = str(k+1).zfill(6)
                    flowfile = join(self.flowfolder, indstr+'_'+indstr2+self.flowsuffix)
                    if isfile(flowfile):
                        imgright = self.imgreader.read_flow(flowfile, scale)
                    else:
                        logf.logline('Flow file error ' + flowfile)
                        print 'flow file error', flowfile
                        continue
                else: # flow file is one frame less than other files
                    imgright = np.zeros_like(imgleft)

            leftdepthfile = join(self.leftdepthfolder,indstr+self.leftdepthsuffix)
            if isfile(leftdepthfile):
                depthleft = self.imgreader.read_depth(leftdepthfile, scale, maxthresh = 50)
            else:
                logf.logline('Left depth file missing ' + leftdepthfile)
                print 'left depth file missing', leftdepthfile
                continue

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

                pts = np.array([[0,0],[320,0],[320,20],[0,20]],np.int32)
                cv2.fillConvexPoly(imgleft,pts,(70,30,10))
                cv2.putText(imgleft,'{}. meand={:.2f}, maxd={:.2f}, mind={:.2f}'.format(str(k), displeft.mean(), displeft.max(), displeft.min()),
                            (5,15),cv2.FONT_HERSHEY_SIMPLEX,0.4,(0,255,255),thickness=1)

                cv2.fillConvexPoly(imgright,pts,(70,30,10))
                cv2.putText(imgright,'{}. meand={:.2f}, maxd={:.2f}, mind={:.2f}'.format(str(k), dispright.mean(), dispright.max(), dispright.min()),
                            (5,15),cv2.FONT_HERSHEY_SIMPLEX,0.4,(0,255,255),thickness=1)

            imgdisp0 = np.concatenate((imgleft, imgright), 0)
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

            if k%100==0:
                print("        Read {} depth files...".format(k))

        return dmean, dmax, dmin, leftfileindlist

    def rgb_validate(self, logf, startind=0):
        '''
        return rgb values in order to detect images too dark or too bright
        '''
        rgbmeanlist = []
        rgbstdlist = []
        for k in range(startind, startind+self.imgnum): # it should be a concequtive sequence starting from 0
            indstr = str(k).zfill(6)
            leftfile = join(self.leftfolder,indstr+self.leftsuffix)
            if isfile(leftfile):
                imgleft = self.imgreader.read_rgb(leftfile, 1)
                if imgleft is None:
                    logf.logline('Left file error ' + leftfile)
                    print 'left file error', leftfile
                    continue
            else:
                logf.logline('Left file missing ' + leftfile)
                print 'left file missing', leftfile
                continue


            rgbmeanlist.append(imgleft.mean())
            rgbstdlist.append(imgleft.std())

        return rgbmeanlist, rgbstdlist

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

def save_preview_video(env_root_dir, data_folder=['Data', 'Data_fast'], vid_out_dir = 'video', video_with_flow = False):
    '''
    Input: Trajectory folder is organized in <env_root_dir>/data_folder[k]/P00X
    In each trajectory folder, image data are in 'image_left', 'image_right',
    'depth_left', 'depth_right', 'seg_left', 'seg_right'. 

    Output: a video for each trajectory: <vid_out_dir>/<datafolder>_P00X.mp4
    '''
    vidoutdir = join(env_root_dir, vid_out_dir)
    if not isdir(vidoutdir):
        mkdir(vidoutdir)

    logf = FileLogger(join(vidoutdir, 'error.log'))
    for datafolder in data_folder:
        datapath = join(env_root_dir, datafolder)
        if not isdir(datapath):
            logf.logline('Data folder missing ' + datapath)
            print '!!data folder missing', datapath
            continue
        print('    Opened data folder {}'.format(datapath))

        trajfolders = listdir(datapath)
        trajfolders = [ tf for tf in trajfolders if tf[0]=='P' ]
        trajfolders.sort()
        print('    Found {} trajectories'.format(len(trajfolders)))

        for trajfolder in trajfolders:
            # generate a video for each trajectory
            outvidfile = join(vidoutdir, datafolder+'_'+trajfolder+'.mp4')
            datavarifier = DataVerifier(join(datapath,trajfolder))
            datavarifier.save_vid_with_depth_statistics(outvidfile, logf, scale=0.5, startind=0, check_depth=True, video_with_flow=video_with_flow)
    logf.close()

def analyze_depth_data(env_root_dir, data_folder=['Data', 'Data_fast'], ana_out_dir = 'analyze', info_from_file=False):
    '''
    Input: Trajectory folder is organized in env_root_dir/data_folder[k]/P00X
    In each trajectory folder, image data are in 'image_left', 'image_right',
    'depth_left', 'depth_right', 'seg_left', 'seg_right'. 

    Output: 1. depth info: <env_root_dir>/<ana_out_dir>/disp_mean(max,min).npy
            2. depth histogram: <env_root_dir>/<ana_out_dir>/disp_hist.jpg
            3. index file: <env_root_dir>/<ana_out_dir>/left_file_index_all.txt 
                - content: each line correspond to a file index: <env_root_dir>/<datafolder>/<trajfolder>/<image_left>/000xxx
                - each line does not contain image suffix
            4. also save individule file for each trajectory
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
    dispmaxfile = join(anaoutdir, 'disp_max.npy')
    dispminfile = join(anaoutdir, 'disp_min.npy')
    dispmeanfile = join(anaoutdir, 'disp_mean.npy')
    leftindfile = join(anaoutdir, 'left_file_index_all.txt') 
    disphistfig = join(anaoutdir, 'disp_hist.png')

    if info_from_file:
        dispmax = np.load(dispmaxfile)
        dispmean = np.load(dispmeanfile)
        dispmin = np.load(dispminfile)
        print ('    depth info file loaded from file: {}, {}, {}'.format(dispmax.shape, dispmean.shape, dispmin.shape))

    else: # calculate the statistics from depth image
        logf = FileLogger(join(anaoutdir,'error.log'))
        for datafolder in data_folder:
            datapath = join(env_root_dir, datafolder)
            if not isdir(datapath):
                logf.logline('Data folder missing '+ datapath)
                print 'data folder missing', datapath
                continue
            print('    Opened data folder {}'.format(datapath))

            trajfolders = listdir(datapath)
            trajfolders = [ tf for tf in trajfolders if tf[0]=='P' ]
            trajfolders.sort()
            print('    Found {} trajectories'.format(len(trajfolders)))

            for trajfolder in trajfolders:
                print('      Move into trajectory {}'.format(trajfolder))
                # generate a video for each trajectory
                datavarifier = DataVerifier(join(datapath,trajfolder))
                # get depth info
                dmean, dmax, dmin, indlist = datavarifier.depth_statistic(logf, startind=0)
                dispmean.extend(dmean)
                dispmax.extend(dmax)
                dispmin.extend(dmin)
                leftindexlist.extend(indlist)

                # also save statistics for each trajectory
                dispmaxfile_traj = join(anaoutdir, trajfolder+'_'+datafolder+'_disp_max.npy')
                dispminfile_traj = join(anaoutdir, trajfolder+'_'+datafolder+'_disp_min.npy')
                dispmeanfile_traj = join(anaoutdir, trajfolder+'_'+datafolder+'_disp_mean.npy')
                leftindfile_traj = join(anaoutdir, trajfolder+'_'+datafolder+'_left_file_index_all.txt') 
                disphistfig_traj = join(anaoutdir, trajfolder+'_'+datafolder+'_disp_hist.png')

                np.save(dispmeanfile_traj, np.array(dmean))
                np.save(dispmaxfile_traj, np.array(dmax))
                np.save(dispminfile_traj, np.array(dmin))
                with open(leftindfile_traj, 'w') as f:
                    for leftind in indlist:
                        f.write('%s\n' % leftind)
                plot_depth_info(dmean, dmax, dmin, disphistfig_traj)

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

def analyze_rgb_data(env_root_dir, data_folder=['Data', 'Data_fast'], ana_out_dir = 'analyze'):
    '''
    Input: Trajectory folder is organized in env_root_dir/data_folder[k]/P00X
    In each trajectory folder, image data are in 'image_left', 'image_right',
    'depth_left', 'depth_right', 'seg_left', 'seg_right'. 

    Output: 1. rgb info: <env_root_dir>/<ana_out_dir>/rgb_mean(std).npy
            2. also save individule file for each trajectory
    '''
    # check and create output folders
    anaoutdir = join(env_root_dir, ana_out_dir)
    if not isdir(anaoutdir):
        mkdir(anaoutdir)

    # depth statistics
    rgbmean = []
    rgbstd = []
    rgbmeanfile = join(anaoutdir, 'rgb_mean.npy')
    rgbstdfile = join(anaoutdir, 'rgb_std.npy')
    rgbfig = join(anaoutdir, 'rgb_mean_std.png')

    logf = FileLogger(join(anaoutdir,'rgb_error.log'))
    for datafolder in data_folder:
        datapath = join(env_root_dir, datafolder)
        if not isdir(datapath):
            logf.logline('Data folder missing '+ datapath)
            print 'data folder missing', datapath
            continue
        print('    Opened data folder {}'.format(datapath))

        trajfolders = listdir(datapath)
        trajfolders = [ tf for tf in trajfolders if tf[0]=='P' ]
        trajfolders.sort()
        print('    Found {} trajectories'.format(len(trajfolders)))

        for trajfolder in trajfolders:
            # generate a video for each trajectory
            datavarifier = DataVerifier(join(datapath,trajfolder))
            # get rgb info
            
            rgbmeanlist, rgbstdlist = datavarifier.rgb_validate(logf, startind=0)
            rgbmean.extend(rgbmeanlist)
            rgbstd.extend(rgbstdlist)

            # also save statistics for each trajectory
            rgbmeanfile_traj = join(anaoutdir, trajfolder+'_'+datafolder+'_rgb_mean.npy')
            rgbstdfile_traj = join(anaoutdir, trajfolder+'_'+datafolder+'_rgb_std.npy') 
            rgbfig_traj = join(anaoutdir, trajfolder+'_'+datafolder+'_rgb_mean_std.png')
            np.save(rgbmeanfile_traj, np.array(rgbmeanlist))
            np.save(rgbstdfile_traj, np.array(rgbstdlist))
            plt.figure(figsize=(12,6))
            plt.plot(np.array(rgbmeanlist))
            plt.plot(np.array(rgbstdlist))
            plt.legend(['mean', 'std'])
            plt.grid()
            plt.savefig(rgbfig_traj)

    rgbmean = np.array(rgbmean)
    rgbstd = np.array(rgbstd)
    np.save(rgbmeanfile, rgbmean)
    np.save(rgbstdfile, rgbstd)
    plt.figure(figsize=(12,6))
    plt.plot(rgbmean)
    plt.plot(rgbstd)
    plt.legend(['mean', 'std'])
    plt.grid()
    plt.savefig(rgbfig)

    print ('    saved rgb info file: {}, {}'.format(rgbmeanfile, rgbstdfile))
    logf.close()

def stereo_depth_filter(rootdir, leftimg_ind_file, disp_max_file, disp_mean_file, out_stereo_file, out_stereo_error_file, maxmax = 400, maxmean = 200, minmax=0.4):
    '''
    Input: npy files of depth info 
           list of image indexes
    Output: list of files for the stereo task 
            - left_image_file_path right_image_file_path left_depth_file_path
    Filtering condition
        - Nothing too close (maxmax=400 -> min_dist=80/400=0.2m)
        - No big thing too close (maxmean=200 -> mean_dist=80/200=0.4m)
        - No all background (minmax=0.4 -> max_dist=80/0.4=200m) 
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

def stereo_depth_rgb_filter(rootdir, leftimg_ind_file, 
                        disp_max_file, disp_mean_file, 
                        rgb_mean_file, rgb_std_file, 
                        # flow_report_file, 
                        out_traj_file, out_traj_error_file, 
                        max_disp_max = 500, max_disp_mean = 250, # the objects can not be too close
                        min_disp_max = 4, min_disp_mean = 0.4, # the objects can not be all far away (e.g. looking into sky)
                        min_rgb_mean = 2, min_rgb_std = 5, # the scene can not be too dark
                        max_rgb_mean = 240, min_rgb_std2 = 5, # the scene can not be too bright (e.g. facing a white wall) 
                        # max_flow_error = 10, max_invalid_num = 150000,
                        ):
    '''
    Input: npy files of depth info 
           list of image indexes
    Output: list of files for the stereo task 
            - left_image_file_path right_image_file_path left_depth_file_path
    Filtering condition
        - Nothing too close (maxmax=500 -> min_dist=80/500=0.16m)
        - No big thing too close (maxmean=250 -> mean_dist=80/250=0.32m)
        - No all background (minmax=4 -> max_dist=80/4=20m) 
        - Large enough close object (minmean=0.4 -> mean_dist=80/0.4=200m)
    Error code:
        - 1: object too close
        - 2: scene too far
        - 3: rgb too dark
        - 4: rgb too bright
    '''
    # save the filtered file
    f = open(join(rootdir,leftimg_ind_file), 'r')
    lines = f.readlines()
    f.close()
    framenum = len(lines)

    dispmaxlist = np.load(join(rootdir, disp_max_file))
    dispmeanlist = np.load(join(rootdir, disp_mean_file))
    rgbmeanlist = np.load(join(rootdir, rgb_mean_file))
    rgbstdlist = np.load(join(rootdir, rgb_std_file))
    print ('    input file loaded (index, maxdisp, meandisp, rgbmean, rgbstd): {}, {}, {}, {}, {}'.format(len(lines), 
                            dispmaxlist.shape, dispmeanlist.shape, rgbmeanlist.shape, rgbstdlist.shape))
    assert framenum * 2 == dispmaxlist.shape[0]
    assert framenum * 2 == dispmeanlist.shape[0]
    assert framenum  == rgbmeanlist.shape[0]
    assert framenum  == rgbstdlist.shape[0]

    logf = FileLogger(join(rootdir,out_traj_file), overwrite=True)
    logfe = FileLogger(join(rootdir,out_traj_error_file), overwrite=True)
    count = 0
    for k, line in enumerate(lines):
        line = line.strip()
        leftind = k*2
        rightind = k*2 + 1

        errorlist = []

        if dispmaxlist[leftind] > max_disp_max or dispmaxlist[rightind] > max_disp_max or \
            dispmeanlist[leftind] > max_disp_mean or dispmeanlist[rightind] > max_disp_mean: 
            errorlist.append(1)

        if dispmaxlist[leftind] < min_disp_max or dispmaxlist[rightind] < min_disp_max or \
            dispmeanlist[leftind] < min_disp_mean or dispmeanlist[rightind] < min_disp_mean: 
            errorlist.append(2)

        if rgbmeanlist[k] < min_rgb_mean and rgbstdlist[k] < min_rgb_std:
            errorlist.append(3)

        if rgbmeanlist[k] > max_rgb_mean and rgbstdlist[k] < min_rgb_std:
            errorlist.append(4)

        if len(errorlist)==0:
            logf.logline(line)
            count += 1
        else: 
            logfe.log(line)
            for ec in errorlist:
                logfe.log(' '+str(ec))
            logfe.log('\n')

    logf.close()    
    logfe.close()
    print ('    After filtering: {}'.format(count))

def stereo_depth_rgb_filter_all_trajs(env_root_dir, data_folder=['Data', 'Data_fast'], ana_out_dir = 'analyze'):
    '''
    Iterate the trajectory and call stereo_depth_rgb_filter
    '''
    # check and create output folders
    anaoutdir = join(env_root_dir, ana_out_dir)
    envname = env_root_dir.split('/')[-1]

    out_env_file = join(anaoutdir, envname+'_good_frames.txt')
    out_env_bad_file = join(anaoutdir, envname+'_bad_frames.txt')
    env_file_list = []
    env_bad_file_list = []

    logf = FileLogger(join(anaoutdir,'filter_error.log'))
    for datafolder in data_folder:
        datapath = join(env_root_dir, datafolder)
        if not isdir(datapath):
            logf.logline('Data folder missing '+ datapath)
            print 'data folder missing', datapath
            continue
        print('    Opened data folder {}'.format(datapath))

        trajfolders = listdir(datapath)
        trajfolders = [ tf for tf in trajfolders if tf[0]=='P' ]
        trajfolders.sort()
        print('    Found {} trajectories'.format(len(trajfolders)))

        for trajfolder in trajfolders:
            traj_disp_max_file = join(anaoutdir, trajfolder+'_'+datafolder+'_disp_max.npy')
            traj_disp_mean_file = join(anaoutdir, trajfolder+'_'+datafolder+'_disp_mean.npy')
            traj_rgb_mean_file = join(anaoutdir, trajfolder+ '_'+datafolder+'_rgb_mean.npy')
            traj_rgb_std_file = join(anaoutdir, trajfolder+ '_'+datafolder+'_rgb_std.npy')
            traj_leftimg_ind_file = join(anaoutdir, trajfolder+ '_'+datafolder+'_left_file_index_all.txt')
            out_traj_file = join(anaoutdir, trajfolder+ '_'+datafolder+'_good_frames.txt')
            out_traj_error_file = join(anaoutdir, trajfolder+ '_'+datafolder+'_bad_frames.txt')

            stereo_depth_rgb_filter(anaoutdir, traj_leftimg_ind_file, 
                        traj_disp_max_file, traj_disp_mean_file, 
                        traj_rgb_mean_file, traj_rgb_std_file, 
                        out_traj_file, out_traj_error_file)

            env_file_list.append(out_traj_file)
            env_bad_file_list.append(out_traj_error_file)

    envgoodframe = FileLogger(out_env_file)
    for fn in env_file_list:
        with open(fn, 'r') as ff:
            lines = ff.readlines()
        for ll in lines:
            envgoodframe.log(ll)     
    envgoodframe.close()

    envbadframe = FileLogger(out_env_bad_file)
    for fn in env_bad_file_list:
        with open(fn, 'r') as ff:
            lines = ff.readlines()
        for ll in lines:
            envbadframe.log(ll)     
    envbadframe.close()

    logf.close()


from settings import get_args

if __name__ == '__main__':
    args = get_args()
    data_root_dir = args.data_root
    data_folders = args.data_folders.split(',')

    if args.env_folders=='': # read all available folders in the data_root_dir
        env_folders = listdir(data_root_dir)    
    else:
        env_folders = args.env_folders.split(',')
    print('Detected envs {}'.format(env_folders))

    create_video = args.create_video
    video_with_flow = args.video_with_flow
    analyze_depth = args.analyze_depth
    depth_from_file = args.depth_from_file
    rgb_validate = args.rgb_validate
    depth_filter = args.depth_filter
    rgb_depth_filter = args.rgb_depth_filter

    for env_folder in env_folders:
        env_dir = join(data_root_dir, env_folder)
        print('Working on env {}'.format(env_dir))
        if create_video:
            print ('  creating video..')
            save_preview_video(env_dir, data_folder=data_folders, vid_out_dir = 'video', video_with_flow = video_with_flow)
        if analyze_depth:
            print ('  analyzing depth..')
            analyze_depth_data(env_dir, data_folder=data_folders, ana_out_dir = 'analyze', info_from_file=depth_from_file)
        if rgb_validate:
            print('  validate rgb..')
            analyze_rgb_data(env_dir, data_folder=data_folders, ana_out_dir = 'analyze')
        if depth_filter:
            print ('  filtering depth..')
            stereo_depth_filter(join(env_dir, 'analyze'), 
                                    leftimg_ind_file = 'left_file_index_all.txt', 
                                    disp_max_file = 'disp_max.npy', 
                                    disp_mean_file = 'disp_mean.npy', 
                                    out_stereo_file = env_folder+'.txt',
                                    out_stereo_error_file = env_folder+'_error.txt')
        if rgb_depth_filter:
            print ('  filtering depth and rgb..')
            stereo_depth_rgb_filter_all_trajs(env_dir, data_folder=data_folders, ana_out_dir = 'analyze')