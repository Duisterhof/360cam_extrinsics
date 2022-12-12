import cv2
import numpy as np
import time
from ImageClient import ImageClient
from os import mkdir
from os.path import isdir, join
from signal import signal, SIGINT
from sys import exit
from data_validation import ImageReader
import sys
import time

outdir = '/home/wenshan/tmp/data/tartan'
imgclient = ImageClient(['1','2'], [ 'Scene' ,'DepthPlanner' ]) #,'Segmentation']) , '3', '0', '4'
imgreader = ImageReader()
leftposefile = 'pose_left.txt'
# rightposefile = 'pose_right.txt'

def handler(signal_received, frame):
    # Handle any cleanup here
    pass
    exit(0)

if __name__ == '__main__':
    # Tell Python to run the handler() function when SIGINT is recieved
    signal(SIGINT, handler)

    envname = sys.argv[1] # 
    # outputdir = sys.argv[2]

    envdir = join(outdir, envname)
    if not isdir(envdir):
        mkdir(envdir)
        # mkdir(envdir+'/image_left')
        # mkdir(envdir+'/depth_front')
        # mkdir(envdir+'/image_right')
        # mkdir(envdir+'/depth_right')
        # mkdir(envdir+'/depth_right')
        # mkdir(envdir+'/depth_back')
        # mkdir(envdir+'/depth_left')
        # mkdir(envdir+'/seg_left')

    print('Running. Press SPACE to save images. \nPress q to exit.')
    count = 28
    key = 0

    leftposelist = []
    rightposelist = []
    time.sleep(3)

    while key != 113: # 'q'
        # if key==107 or key==32 or key==0:
        # import ipdb; ipdb.set_trace()
        rgblist, depthlist, seglist, camposelist = imgclient.readimgs()
        if rgblist is None: # reading error
            continue
        rgb_left = rgblist[1]
        rgb_right = rgblist[0]
        # rgb = cv2.cvtColor(rgb,cv2.COLOR_RGB2BGR) 
        depth_left = imgreader.depth2vis(depthlist[1], maxthresh = 5)
        depth_right = imgreader.depth2vis(depthlist[0], maxthresh = 5)
        # segvis = imgreader.seg2vis(seglist[0])
        # depthall = np.concatenate((depthlist[0],depthlist[1],depthlist[2],depthlist[3]),axis=1)
        # depthvis = imgreader.depth2vis(depthall, maxthresh=50)
        # depthvis = cv2.resize(depthvis, (0,0), fx=0.5, fy=0.5)

        # depthmax = 80.0/depthlist[1].max()
        # depthmin = 80.0/depthlist[1].min()
        # depthmean = 80.0/depthlist[1].mean()

        # imdisp = np.concatenate((rgb_left,depth_left), axis=1)
        # imdisp = cv2.resize(imdisp, (0,0), fx=0.15, fy=0.15)
        # pts = np.array([[0,0],[320,0],[320,20],[0,20]],np.int32)
        # cv2.fillConvexPoly(imdisp,pts,(70,30,10))
        # cv2.putText(imdisp,'meand={:.2f}, maxd={:.2f}, mind={:.2f}'.format(depthmean, depthmax, depthmin),
        #             (5,15),cv2.FONT_HERSHEY_SIMPLEX,0.4,(0,255,255),thickness=1)
        cv2.imshow('img',depth_right)
        key = cv2.waitKey(10)
        # print key
        if key==32: # space key
            indstr = str(count).zfill(6)
            cv2.imwrite(join(envdir,  indstr + '_left.png'), rgb_left, [cv2.IMWRITE_PNG_COMPRESSION, 0])
            cv2.imwrite(join(envdir,  indstr + '_left_depth_vis.png'), depth_left)
            cv2.imwrite(join(envdir,  indstr + '_right.png'), rgb_right, [cv2.IMWRITE_PNG_COMPRESSION, 0])
            cv2.imwrite(join(envdir,  indstr + '_right_depth_vis.png'), depth_right)
            # cv2.imwrite(join(envdir, '' + '_seg.png'), seg)
            np.save(join(envdir, indstr  + '_left_depth.npy'),depthlist[1])
            np.save(join(envdir, indstr  + '_right_depth.npy'),depthlist[0])
            # np.save(join(envdir, 'depth_right/' + indstr  + '_right_depth.npy'),depthlist[1])
            # np.save(join(envdir, 'depth_back/' + indstr  + '_back_depth.npy'),depthlist[2])
            # np.save(join(envdir, 'depth_left/' + indstr  + '_left_depth.npy'),depthlist[3])
            # np.save(join(envdir, str(count)+'_seg.npy'),seglist[0])
            # leftposelist.append(camposelist[1])
            # rightposelist.append(camposelist[0])
            print('Saved files {} to {}'.format(count, envdir))
            count += 1

            # np.savetxt(join(envdir, leftposefile), np.array(leftposelist))
            # np.savetxt(join(envdir, rightposefile), np.array(rightposelist))
