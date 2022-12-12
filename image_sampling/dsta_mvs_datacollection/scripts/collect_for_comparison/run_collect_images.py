# collect images in cv mode

import os
import sys

# The path of the current Python script.
_CURRENT_PATH       = os.path.dirname(os.path.realpath(__file__))
_TOP_PATH           = os.path.join(_CURRENT_PATH, '..', '..', 'data_collection')
_DATA_PIPELINE_PATH = os.path.join(_TOP_PATH, 'ord_data_pipeline_rework', 'src')
sys.path.insert( 0, _DATA_PIPELINE_PATH )
sys.path.insert( 0, _TOP_PATH )
for i, p in enumerate(sys.path):
    print(f'{i}: {p}')

# System-wede packages.
import cv2 # debug
import json
import numpy as np
from os.path import join
from scipy.spatial.transform import Rotation as R
from subprocess import run
import time
# from pyquaternion import Quaternion as Quaternionpy # quaternion multiplication

# Datacollection pipeline.
from settings import get_args
from collection.collect_images import RandQuaternionSampler
from transform import Rotation, RotationSpline # quaternion spline

# AirSim client Python API.
from airsim.types import Pose, Vector3r , Quaternionr
from airsim.utils import to_eularian_angles, to_quaternion
from collection.ImageClient import ImageClient

# The MVS utilities.
from collection.data_sampler import RawDataSampler

class FlexibleDataSampler(RawDataSampler):

    def __init__(self, args, data_path):
        self.args = args
        self.data_dir = data_path

        #Initialize the file structure and relevant data structures for data collection,
        #as specified in the metadata.json file.
        super().__init__(self.data_dir)
        self.read_metadata_and_initialize_dirs(args.metadata_path, args.frame_graph_path)

        #Initialize the ImageClient that connects to and collects from the Unreal Environment.
        #Also initialize the Quaternion Sampler for use during data collection, specified by
        #the settings.py file and relevant command line arguments.
        self.imgclient = ImageClient(self.init_cam_list, self.init_imgtype_list)
        self.imgclient.cube_cuda = not args.disable_cube_cuda
        self.randquaternionsampler = RandQuaternionSampler(self.args) 

        self.six_pane_oris = six_oris = dict(
                front = Quaternionr(0, 0, 0, 1.0),                    #Forward Image Orientation
                back = Quaternionr(0, 0, 1.0, 0),                    #Backward Image Orientation
                left = Quaternionr(0, 0, -0.7071068, 0.7071068),   #Left Image Orientation
                right = Quaternionr(0, 0, 0.7071068, 0.7071068),    #Right Image Orientation
                top = Quaternionr(0, 0.7071068, 0, 0.7071068),    #Up Image Orientation
                bot = Quaternionr(0, -0.7071068, 0, 0.7071068)    #Down Image Orientation
            )

    
    def generate_rand_orientation(self):
        '''
        Generates a random Quaternion
        '''
        rq = self.randquaternionsampler.random_quaternion()
        return Quaternionr(rq.x, rq.y, rq.z, rq.w)

    def rotate_w_quat(self, curr_rot, rotby):

        curr = R.from_quat(curr_rot.to_numpy_array())
        by = R.from_quat(rotby.to_numpy_array())
        curr = curr * by
        return Quaternionr(*curr.as_quat())

    def transform_pose(self, rigpos, campos, ori):
        '''
        Transforms a local campos w.r.t. to the rig frame into a point in the global NED AirSim Frame.
        '''

        #Rotate the campos 3D vector by the ori quaternion
        rotcam = ori * Quaternionr(*campos, 0.0) * ori.inverse()
        rotcam = rotcam.to_numpy_array()

        #Shift the rotated local camera pose vector with the rigpos to transform it to the NED AirSim global frame.
        newpos = rigpos + rotcam[:3]
        return Vector3r(*newpos)

    def read_and_format_6_image_pane(self, rand_ori, tpos):
        '''
        Gets a 6 image pane from the AirSim unreal environment for 6 image to fisheye image conversion
        '''

        time.sleep(0.02)

        for idx, k in enumerate(self.six_pane_oris):
            print(f"Taking {k} View now...")

            #pane_ori = rand_ori * self.six_pane_oris[k] * rand_ori.inverse()
            pane_ori = self.rotate_w_quat(rand_ori, self.six_pane_oris[k])

            tpose = Pose(tpos, pane_ori)
            self.imgclient.setpose(tpose)

            time.sleep(0.02)

            if idx == 0:
                rgblist_all, depthlist_all, seglist_all, rgblist_cube_all, distlist_cube_all, camposelist_all = \
                    self.imgclient.readimgs()

            else:
                rgblist, depthlist, seglist, rgblist_cube, distlist_cube, camposelist = \
                    self.imgclient.readimgs()

                rgblist_all.extend(rgblist)
                depthlist_all.extend(depthlist)
                seglist_all.extend(seglist)
                rgblist_cube_all.extend(rgblist_cube)
                distlist_cube_all.extend(distlist_cube)

        return {
            "Scene":rgblist_all,
            "DepthPlanner":depthlist_all,
            "Segmentation":seglist_all,
            "CubeScene":rgblist_cube_all,
            "CubeDistance":[np.expand_dims(dimg, axis=-1).view('<u1') for dimg in distlist_cube_all]
        }

    def read_and_format_airsim_imgs(self):
        '''
        Gets the images from the AirSim Unreal environment and formats the results into a easily indexable dictionary.
        '''
        rgblist, depthlist, seglist, rgblist_cube, distlist_cube, camposelist = \
                self.imgclient.readimgs()

        return {
            "Scene":rgblist,
            "DepthPlanner":depthlist,
            "Segmentation":seglist,
            "CubeScene":rgblist_cube,
            "CubeDistance":[np.expand_dims(dimg, axis=-1).view('<u1') for dimg in distlist_cube]
        }

    def save_raw_data_img(self, path_wo_type, t, img_of_type):

        print(img_of_type.min(), img_of_type.max(), img_of_type.shape)

        if t == 'Segmentation' or t == 'DepthPlanner':
            file_type = ".npy"
            np.save(path_wo_type + file_type, img_of_type)

        else:
            file_type = ".png"
            cv2.imwrite(path_wo_type + file_type, img_of_type)

        return file_type

    def save_imgs_at_pose(self, pos, i):
        '''
        Saves an image at the given pose into the relevant directory and indexes the image in the relevant csv files
        '''

        #Initialize the empty camera path lists
        temp_cam_paths = list()
        temp_rig_paths = list()

        #For each mapped point, provide a random orientation
        if self.flag_random_orientation:
            rand_ori = self.generate_rand_orientation()
        else:
            rand_ori = Quaternionr(x_val=0.0, y_val=0.0, z_val=0.0, w_val=1.0)

        #For each camera registered, sample the image from the current Unreal Environment
        for k in self.cam_to_camdata:
            #Extract camera data and create the name of the images taken in this position for the index
            cam_data = self.cam_to_camdata[k]
             
            #Add the indexed name to the temporary list for the camera index.
            temp_cam_paths.append(f"{i:06}")

            #Change the ImageClient's image requests
            self.imgclient.populateRequests(self.imgclient.CAMLIST, cam_data["types"])
            
            #Find the global position of the camera and set the pose in the simulation
            cam_pos = np.array(cam_data["data"]["pos"]) if k != "rig" else np.array([0.0,0.0,0.0])
            tpos = self.transform_pose(pos, cam_pos, rand_ori)
            tpose = Pose(tpos, rand_ori)

            start_pose = time.time()
            self.imgclient.setpose(tpose)
            print(f"For cam {k}, time to change pose was: {time.time() - start_pose}")
           
            #Give time for the simulation to update the pose and pause the simulation so no lighting effects 
            #or otherwise changes.
            time.sleep(0.02)
            self.imgclient.simPause(True)
            
            #Read the images from AirSim and format into a dictonary where the types are the keys and 
            #the resulting images are the values.
            start_reqimgs = time.time()

            airsim_imgs = self.read_and_format_6_image_pane(rand_ori, tpos) if args.save_as_six_imgs else self.read_and_format_airsim_imgs()

            print(f"For cam {k}, time to request was: {time.time() - start_reqimgs}")

            #For each type that is collected by the camera, save the image to the corresponding directory as specified in
            #the cam_data
            for t in cam_data["types"]:
                imgs_of_type = airsim_imgs[t]

                img_name = f"{i:06}_{t}"
                path_wo_type = join(cam_data["path"], img_name)

                start_saveimg = time.time()
                if args.save_as_six_imgs and t == "Scene":

                    for idx, kpane in enumerate(self.six_pane_oris):
                        file_type = self.save_raw_data_img(path_wo_type + f"_{kpane}", t, imgs_of_type[idx])
                        
                else:
                    img_of_type = imgs_of_type[0]
                    file_type = self.save_raw_data_img(path_wo_type, t, img_of_type)

                print(f"For cam {k}, time to save images was: {time.time() - start_saveimg}")
                    
                #If the camera is the rig or is indexed as the rig,
                #add the image name for each of the types
                if k == "rig" or "rig_is_cam" in cam_data:
                    temp_rig_paths.append(img_name + file_type)
                
            #Unpause the simulation
            self.imgclient.simPause(False)

            #Save the pose for each camera to the corresponding camera
            self.cam_to_poses_dict[k].append([
                *tpose.position.to_numpy_array(), *tpose.orientation.to_numpy_array()
            ])
        
        #Add the image names to the index
        self.cam_paths_list.append(temp_cam_paths)
        self.rig_paths_list.append(temp_rig_paths)

        
    def data_sampling(self, args, traj):
        '''
        Samples a given trajectory in an open unreal engine environment
        '''
        
        #Save number of points and sampling start time for reporting uses
        numpnts = len(traj[0:-1:args.frame_stepsize])
        start_smpl = time.time()

        #Save all images that are needed (as specified in metadata.json) for all trajectory points
        for i, pos in enumerate(traj[0:-1:args.frame_stepsize]):
            start_retr = time.time()
            self.save_imgs_at_pose(pos, i)

            #Report successful sampling 
            smpl_time = time.time()-start_retr
            print(f"Sample #{i} | {round((i/numpnts) * 100.0)}% Collected of {numpnts} samples | Sampling Time: {smpl_time} \
                    \n Estimated to finish in {round(((numpnts- i) * smpl_time) / 60, 3)} minutes \
                    \n --------------------------------------------")

        #Report total sampling time   
        print(f"\n ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ \
                \n Total Sampling time of trajectory: {time.time()-start_smpl} seconds \
                \n ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ \n")
    
    def close(self,):
        '''
        Closes the imgclient after sampling is done
        '''

        #Close the ImageClient
        self.imgclient.close()

        #Save the index lists to csv files for later use in organization
        np.savetxt(join(self.data_dir,"cam_paths.csv"), self.cam_paths_list, delimiter=",", fmt="%s")
        np.savetxt(join(self.data_dir,"rig_paths.csv"), self.rig_paths_list, delimiter=",", fmt="%s")

        #Save each of the cameras' poses over the whole trajectory
        for k in self.cam_to_poses_dict:
            np.savetxt(join(self.data_dir,f"cam_{k}_poses.csv"), self.cam_to_poses_dict[k], delimiter=",", fmt="%s")


def open_unreal_env_windows(envname, envdir_path, args):
    '''
    Opens a correctly packaged Unreal Environment given a correct environment directory path and the name of the environment.
    '''

    env_exe_name = join(envdir_path, envname + ".exe")
    run(f"START /B {env_exe_name} -ResX=2048 -ResY=1024 -WINDOWED", shell=True)
    run("timeout /t 10 /nobreak", shell=True)


def close_unreal_env_windows(envname):
    '''
    Closes a correctly packaged Unreal Environment given a correct environment name.
    IMPORTANT: The Unreal Environment must be packaged as a Windows 64-bit environment. The configuration is up to user choice but the Shipping 
           configuration is recommended.
    '''

    run(f"wmic process where \"name like \'{envname}.exe\'\" delete", shell=True)
    run(f"Taskkill /IM \"{envname}-Win64-*\" /F", shell=True)


def collect_data_from_requests(args):
    '''
    Header function for data collection, reads in a requests.json file as given by argparse and samples from each environment.
    '''
 
    #Path of the requests.json file, used to specify where the unreal environment executable is 
    #and where each environment's trajectories are and where the data should be saved.
    requests_path = args.requests_path

    #Open the requests.json file and for each environment in the requests.json file...
    with open(requests_path) as requests_file:
        requests = json.load(requests_file)

        for envname in requests:
            #Extract the relevant paths
            data_paths = requests[envname]
            envdir_path = data_paths["envdir"]
            trajdir_path = data_paths["trajdir"]
            args.frame_stepsize = data_paths["stepsz"]
            args.save_as_six_imgs = data_paths["as_six_imgs"]

            #Open the Unreal Environment
            open_unreal_env_windows(envname, envdir_path, args)

            #For each trajectory file (Starts with P and is not a directory) ...
            for trajpath in [p for p in os.listdir(trajdir_path) if p.startswith("P") and
                             os.path.isfile(join(trajdir_path, p))]:

                #Load the trajectory
                traj = np.loadtxt(join(trajdir_path, trajpath))

                #Make a directory to store the data in
                data_path = join(trajdir_path, trajpath.split('.')[0])
                if not os.path.exists(data_path):
                    os.makedirs(data_path)

                #Initialize the data sampler, start sampling the data, and close after sampling is complete.
                sampler = FlexibleDataSampler(args, data_path)
                if args.disable_meta_random_orientation:
                    print(f'Disable random orientation. ')
                    sampler.flag_random_orientation = False
                sampler.data_sampling(args, traj)
                sampler.close()      
            
            #Close the unreal environment after all trajectories for that environment have been collected.
            close_unreal_env_windows(envname)

'''
Please see this Google Document for more instructions/information about using this pipeline:
https://docs.google.com/document/d/1KjhCUnSugCfQJa_XYzeB0sTaklofoqA1yesa0GQoCss/edit?usp=sharing
'''
if __name__ == "__main__":

    args = get_args()
    collect_data_from_requests(args)