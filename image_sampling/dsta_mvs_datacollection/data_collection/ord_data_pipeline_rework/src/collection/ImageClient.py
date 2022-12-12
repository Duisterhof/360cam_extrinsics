import airsim
from airsim.types import Pose, Vector3r, Quaternionr
import time

import cv2 # debug
import numpy as np

from .PanoramaDepth2Distance import (
    meshgrid_from_img, depth_2_distance)

np.set_printoptions(precision=3, suppress=True, threshold=10000)

  # Scene = 0, 
  # DepthPlanner = 1, 
  # DepthPerspective = 2,
  # DepthVis = 3, 
  # DisparityNormalized = 4,
  # Segmentation = 5,
  # SurfaceNormals = 6,
  # Infrared = 7
class ImageClient(object):
    def __init__(self, camlist, typelist, ip=''):
        self.client = airsim.MultirotorClient(ip=ip,timeout_value=14400)
        self.client.confirmConnection()

        self.populateRequests(camlist, typelist)

        # Meshtrid coordinates for the panorama images.
        # Will be populated upon receiving the first panorama image.
        self.panorama_xx = None
        self.panorama_yy = None
        self.cube_cuda = False # Set True to use CUDA.

    def populateRequests(self, camlist, typelist):

        self.IMGTYPELIST = typelist
        self.CAMLIST = camlist

        self.imgRequest = []
        for k in self.CAMLIST:
            for imgtype in self.IMGTYPELIST:
                if imgtype == 'Scene':
                    self.imgRequest.append(airsim.ImageRequest(str(k), airsim.ImageType.Scene, False, False))

                elif imgtype == 'DepthPlanner':
                    self.imgRequest.append(airsim.ImageRequest(str(k), airsim.ImageType.DepthPlanner, True))

                elif imgtype == 'Segmentation':
                    self.imgRequest.append(airsim.ImageRequest(str(k), airsim.ImageType.Segmentation, False, False))

                elif imgtype == 'CubeScene':
                    self.imgRequest.append(airsim.ImageRequest(str(k), airsim.ImageType.CubeScene, False, True))

                elif imgtype == 'CubeDistance':
                    self.imgRequest.append(airsim.ImageRequest(str(k), airsim.ImageType.CubeDepth, True, False))

                else:
                    print ('Error image type: {}'.format(imgtype))

    def get_cam_pose(self, response):
        cam_pos = response.camera_position # Vector3r
        cam_ori = response.camera_orientation # Quaternionr

        cam_pos_vec = [cam_pos.x_val, cam_pos.y_val, cam_pos.z_val]
        cam_ori_vec = [cam_ori.x_val, cam_ori.y_val, cam_ori.z_val, cam_ori.w_val]

        # print cam_pos_vec, cam_ori_vec
        return cam_pos_vec + cam_ori_vec

    def readimgs(self):
        start = time.time()
        responses = self.client.simGetImages(self.imgRequest) # discard the first query because of AirSim error

        responses = self.client.simGetImages(self.imgRequest)

        camposelist = []
        rgblist, depthlist, seglist = [], [], []
        rgblist_cube, distlist_cube = [], []
        idx = 0
        for k in range(len(self.CAMLIST)):
            for imgtype in self.IMGTYPELIST:
                response = responses[idx]
                hh, ww = response.height, response.width
                if hh==0 or ww==0:
                    print ('Error read image: {}'.format(imgtype))
                    return None, None, None, None
                # response_nsec = response.time_stamp
                # response_time = rospy.rostime.Time(int(response_nsec/1000000000),response_nsec%1000000000)
                if imgtype == 'DepthPlanner': #response.pixels_as_float:  # for depth data
                    img1d = np.array(response.image_data_float, dtype=np.float32)
                    depthimg = img1d.reshape(hh, ww)
                    depthlist.append(depthimg)

                elif imgtype == 'Scene':  # raw image data
                    img1d = np.fromstring(response.image_data_uint8, dtype=np.uint8)  # get numpy array
                    rgbimg = img1d.reshape(hh, ww, -1)
                    rgblist.append(rgbimg[:,:,:3])

                elif imgtype == 'Segmentation': # TODO: should map the RGB back to index
                    img1d = np.fromstring(response.image_data_uint8, dtype=np.uint8) #get numpy array
                    img_rgba = img1d.reshape(hh, ww, -1)
                    # import ipdb;ipdb.set_trace()
                    img_seg = img_rgba[:,:,0]
                    seglist.append(img_seg)

                elif imgtype == 'CubeScene':
                    # Decode the image directly from the bytes.
                    decoded = cv2.imdecode(np.frombuffer(response.image_data_uint8, np.uint8), -1)
                    #print("Time Stamp Decode: ", time.time() - start)
                    rgblist_cube.append(decoded[:, :, :3])

                elif imgtype == 'CubeDistance':
                    # Get a PFM format array.
                    pfm = np.reshape(
                        np.asarray(response.image_data_float, np.float32), 
                        (hh, ww))
                    
                    # Convert the depth image to distance image.
                    dist = np.zeros_like(pfm)
                    self.cube_cuda = False
                    if ( self.cube_cuda ):
                    	raise Exception('u18c11vulkan Docker Image does not support CUDA distance conversion. ')
                    else:
                        # Meshgrid coordinates.
                        if ( self.panorama_xx is None or self.panorama_yy is None ):
                            self.panorama_xx, self.panorama_yy = \
                                meshgrid_from_img(pfm)

                        if ( not depth_2_distance(pfm, self.panorama_xx, self.panorama_yy, dist) ):
                            raise Exception('Failed to convert cube depth image to distance image (CPU)')

                    distlist_cube.append(dist)

                idx += 1
            # import ipdb;ipdb.set_trace()
            cam_pose_img = self.get_cam_pose(response) # get the cam pose for each camera
            camposelist.append(cam_pose_img)

        return rgblist, depthlist, seglist, rgblist_cube, distlist_cube, camposelist

    def listobjs(self):
        object_list = sorted(self.client.simListSceneObjects())
        object_seg_ids = [self.client.simGetSegmentationObjectID(object_name.lower()) for object_name in object_list]

        for object_name, object_seg_id in zip(object_list, object_seg_ids):
            if object_seg_id!=-1:
                print("object_name: {}, object_seg_id: {}".format(object_name, object_seg_id))


    def setpose(self, pose):
        self.client.simSetVehiclePose(pose, ignore_collision=True)

    def getpose(self):
        return self.client.simGetVehiclePose()

    def simPause(self, pause): # this is valid for customized AirSim
        return self.client.simPause(pause)

    def close(self):
        self.client.simPause(False)
        self.client.reset()
