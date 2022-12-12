from ImageClient import ImageClient
from airsim.types import Pose, Vector3r , Quaternionr
from airsim.utils import to_eularian_angles, to_quaternion
import cv2 # debug
import numpy as np
from math import cos, sin, tanh, pi
import time

class OmniRGBSampler(object):
    '''
    Sample 360-direction image 
    '''
    def __init__(self):

        self.imgtypelist = ['Scene','DepthPlanner']
        self.camlist = ['0']
        self.camlist_name = {0: 'front', 1: 'left', 2: 'back', 3: 'right', 4: 'up', 5: 'bottom'} 

        self.imgclient = ImageClient(self.camlist, self.imgtypelist)
        # must be in CV mode, otherwise the point clouds won't align
        self.scan_config = [0, -1, 2, 1, 4, 5] #[0, 1, 2, -1] # front, left, back, right, up, down

    def sample(self, position):

        imglist = []
        deplist = []
        self.imgclient.simPause(True)
        for k,face in enumerate(self.scan_config): 
            if face == 4:  # look upwards at tgt
                pose = Pose(Vector3r(position[0], position[1], position[2]), to_quaternion(pi / 2, 0, 0)) # up
            elif face == 5:  # look downwards
                pose = Pose(Vector3r(position[0], position[1], position[2]), to_quaternion(-pi / 2, 0, 0)) # down - pitch, roll, yaw
            else:
                yaw = pi / 2 * face
                pose = Pose(Vector3r(position[0], position[1], position[2]), to_quaternion(0, 0, yaw))

            self.imgclient.setpose(pose)
            time.sleep(0.02)
            # import ipdb;ipdb.set_trace()
            rgblist, depthlist, seglist, camposelist = self.imgclient.readimgs()
            if rgblist is None:
                # try read the images again
                print ('  !!Error read image: {}-{}: {}'.format(trajname, k, pose))
                return None, None
            else: 
                img = rgblist[0] # change bgr to rgb
                imglist.append(img)
                deplist.append(depthlist[0])
                # cv2.imwrite(join(self.imgdirs[w], imgprefix+camname+'.png'), img, [cv2.IMWRITE_PNG_COMPRESSION, 0])

        self.imgclient.simPause(False)
        return imglist, deplist

if __name__ == '__main__':
    sampler = OmniRGBSampler()
    imglist, deplist = sampler.sample(position=[0.,-1.0,0.0])
    ind = 0
    for k, (img, depth) in enumerate(zip(imglist, deplist)):
        cv2.imwrite('/home/wenshan/tmp/data/omni_image/'+str(ind)+'_'+sampler.camlist_name[k]+'.png', img)
        np.save('/home/wenshan/tmp/data/omni_image/'+str(ind)+'_'+sampler.camlist_name[k]+'_depth.npy', depth)
    blank = np.zeros_like(imglist[0])
    combine1 = np.concatenate((blank, imglist[4], blank, blank), axis=1)
    combine2 = np.concatenate([imglist[1],imglist[0],imglist[3],imglist[2]], axis=1)
    combine3 = np.concatenate((blank, imglist[5], blank, blank), axis=1)
    combine = np.concatenate((combine1, combine2, combine3), axis=0)
    cv2.imwrite('/home/wenshan/tmp/data/omni_image/combine.png', combine)
