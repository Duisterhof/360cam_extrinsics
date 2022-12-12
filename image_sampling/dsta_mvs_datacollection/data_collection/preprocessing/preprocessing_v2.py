
import copy
import cv2
import datetime
import json
import numpy as np
import os, shutil
from os.path import join
from scipy.spatial.transform import Rotation as R
#from sklearn.model_selection import StratifiedKFold, StratifiedShuffleSplit
from tqdm import tqdm
import time

# For FTensor.
import torch

# Local package.
from ..image_sampler import SAMPLERS
from ..image_sampler import make_object as make_sampler

from ..mvs_utils.image_io import ( read_compressed_float, write_compressed_float )

from ..mvs_utils.metadata_reader import MetadataReader
from ..mvs_utils.camera_models import CAMERA_MODELS
from ..mvs_utils.camera_models import make_object as make_camera_model

from ..mvs_utils.ftensor import FTensor
from ..mvs_utils.frame_io import parse_orientation

class Fisheye_Preprocessor(MetadataReader):

    def __init__(self, args):
        self.args = args

        self.read_preprocessing_manifest(args)
 
        # Parse the camera models.
        camera_models = dict()
        for key, value in self.manifest["camera_models"].items():
            camera_models[key] = make_camera_model(CAMERA_MODELS, value)

        self.gt_needs_resampling = "rig" in self.manifest["samplers"]

        # Parse the samplers.
        self.cam_to_PanoramaConverter = dict()
        self.cam_to_SixImgConverter = dict()
        for k in self.manifest["samplers"]:
            
            # Get the camera specification.
            cam_spec = copy.deepcopy( self.manifest["samplers"][k] )
            
            # Convert orientatioin to rotation matrix.
            q = cam_spec["orientation"]
            R_raw_fisheye = FTensor( parse_orientation(q), f0=q['f0'], f1=q['f1'], rotation=True ).to(dtype=torch.float32)
            # cam_spec["R_raw_fisheye"] = R.from_quat([q["x"], q["y"], q["z"], q["w"]]).as_matrix()
            cam_spec["R_raw_fisheye"] = R_raw_fisheye
            cam_spec.pop("orientation")
            
            # Get the camera model.
            cam_spec["camera_model"] = camera_models[ cam_spec["cam_model_key"] ]
            cam_spec.pop("cam_model_key")
            
            # Create the the sampler.
            sampler = make_sampler(SAMPLERS, cam_spec)
            self.cam_to_PanoramaConverter.update({
                k: sampler
            })

            # TODO: This needs better implementation.
            #self.cam_to_SixImgConverter.update({
            #    k: sampler
            #})

    def read_preprocessing_manifest(self, args):

        print("Reading Manifest File...")
        with open(args.manifest_path) as manifest_file:
            self.manifest = json.load(manifest_file)

            #Extract important paths and the datasets to process from the loaded JSON
            self.dataset_name = self.manifest["dataset_name"] + "_" + datetime.date.today().strftime("%m%d%Y")
            self.storage_path = self.manifest["storage_path"]
            self.dataset_data = self.manifest["processing_manifest"]

            self.processed_dataset_path = join(self.storage_path, self.dataset_name)
            if not os.path.exists(self.processed_dataset_path):
                os.makedirs(self.processed_dataset_path)

            # Use shutil.copy instead of json.dump to preserve the formatting of the original manifest file.
            # manifest_out = open(join(self.processed_dataset_path,"manifest.json"), 'w')
            # json.dump(self.manifest, manifest_out)
            shutil.copy( args.manifest_path, join(self.processed_dataset_path,"manifest.json") )

            #Initialize important dictionaries. Cam_To_Paths will have all the image paths from all datasets that need to be processed for each camera into fisheye images.
            #Rig paths will have all the ground truth images that need to be copied to the processed dataset.
            self.dataset_num_indx = dict()

            #For each dataset in the manifest.json...
            #E.g., training and testing.
            for dataset in self.dataset_data:
                self.cam_to_paths = dict()
                self.rig_paths = list()
                self.cam_to_header = dict()

                #Initialize the processed directory structure in the "storage_path", which is where the processed data will be stored.
                #E.g., training and testing.
                subset_path = join(self.processed_dataset_path, dataset)
                if not os.path.exists(subset_path):
                    os.makedirs(subset_path)

                # TODO: This is really unconventional. We need a way that does not call the parent's
                # __init__() method multiple times.
                super().__init__(subset_path)

                #Extract out the paths to the raw data for each environment
                sets_to_process = self.manifest["processing_manifest"][dataset]
                cindx = 0

                self.dataset_num_indx.update({dataset:list()})

                # Loop over all the directories listed under the "datasets" key.
                for pre_sets in sets_to_process["datasets"]:
                    
                    #Set important metadata variables. On first pass, this will also initialize the output directory structure.
                    #args.metadata_path = join(pre_sets,"metadata.json")
                    # TODO: Should every pre_set (envrionment) have its own metadata and frame_graph?
                    # TODO: How to deal with multiple trajectories in a single environment?
                    self.read_metadata_and_initialize_dirs(args.metadata_path, args.frame_graph_path)

                    #Dump important metadata for each dataset at the subset level
                    # TODO: Similar to the above. metadata and frame graph might be directly associated with an environment.
                    # Use shutil.copy instead of json.dump to preserve the formatting of the original matadata file.
                    # with open(join(subset_path, "metadata.json"), 'w') as metadata_out:
                        # json.dump(self.metadata, metadata_out)
                    shutil.copy( args.metadata_path, join(subset_path, "metadata.json") )
                    shutil.copy( args.frame_graph_path, join(subset_path, "frame_graph.json") )

                    #Load the image names and paths from the cam_paths.csv file
                    # It's a string array like:
                    # [ [  'cam0'   'cam1'   'cam2' ]
                    #   ['000000' '000000' '000000' ]
                    #   ['000001' '000001' '000001' ]
                    #   ['000002' '000002' '000002' ] ]
                    img_name_arr = np.genfromtxt(
                                join(pre_sets,"cam_paths.csv"), delimiter=',', dtype=str
                            )       

                    #Create an index from camera name to a list of partial absolute paths to the images that need processing. 
                    # The missing parts of the image path include the ImageType (CubeDistance or CubeScene) and 
                    # the file type (.png or .npy).                        
                    for img_num in img_name_arr[1:,0]:
                        self.dataset_num_indx[dataset].append((pre_sets, img_num, cindx))
                        cindx += 1

                # print("----------------------")
                # print(self.dataset_num_indx)
    '''
    WIP Reading and Writing Images with pyvip. Benchmarks show VIPS might be more than twice as fast as openCV for I/O of images.
    https://github.com/libvips/pyvips/issues/179#issuecomment-618936358

    def open_img_with_vips(self, img_path):
        img = pyvips.Image.new_from_file(img_path, access="sequential")
        img = img.colourspace("srgb")
        mem_img = img.write_to_memory() 
        imgnp=np.frombuffer(mem_img, dtype=np.uint8).reshape(img.height, img.width, 3)  
    
        return imgnp

    def write_img_with_vips(self, img, img_path):
        img = pyvips.Image.new_from_memory(img.data, img.width, img.height, bands=3, format="uchar")
        img.write_to_file(img_path)
    '''

    def preprocess_requested_imgs(self):

        # dataset is the key, e.g., training, self.dataset_num_indx is a dict.
        for dataset, all_paths_for_preproc in self.dataset_num_indx.items():

            count_per_dataset = 0
            for img_tuple in tqdm(all_paths_for_preproc, desc = f"Converting for {dataset}..."):

                start_time = time.time()
                for k in self.cam_to_camdata:

                    # NOTE: k is an integer for ordinary cameras, but a string for the rig.
                    cam_data = self.cam_to_camdata[k]
                    
                    # Test if the camera is the rig.
                    # rig_process = (k == "rig") or ("is_rig" in cam_data)

                    if k=="rig":
                        rig_process = True
                        cam_name = "rig"
                    elif "is_rig" in cam_data:
                        rig_process = True
                        cam_name = f"cam{k}"
                    else:
                        rig_process = False
                        cam_name = f"cam{k}"
                    
                    dpath       = img_tuple[0] # The dir of the environment + trajectory sub dir.
                    img_num     = img_tuple[1] # The number string of the sample.
                    preproc_num = img_tuple[2] # The unified index of the dataset.

                    # Is this a good syntax?
                    img_path = join(dpath, cam_name, img_num)

                    # #TODO Finish six camera support.
                    # converter = self.cam_to_SixImgConverter[k]
                    # raise NotImplementedError("Six Image to Fisheye Conversion not Implemented Yet")

                    if k != "rig":
                        # Convert the raw image to a fisheye image and write to the filesystem.
                        converter = self.cam_to_PanoramaConverter["cam%d" % (k)]
                        img = cv2.imread(img_path + "_CubeScene.png", cv2.IMREAD_COLOR)
                        img_conv, valid_mask = converter(img)
                        cv2.imwrite(join(self.processed_dataset_path, dataset, cam_name, str(preproc_num).zfill(6) + "_Fisheye.png"),
                                    img_conv)
                    
                        if count_per_dataset == 0:
                            # First image for the dataset. Write the mask image.
                            valid_mask = valid_mask.astype(np.uint8) * 255
                            cv2.imwrite( join( self.processed_dataset_path, dataset, cam_name, 'mask.png' ),
                                        valid_mask )
                    
                    if rig_process:
                        for t in cam_data["types"]:
                            in_rig_path = img_path + f"_{t}.png"
                            out_rig_path = join(self.processed_dataset_path, dataset, "rig", str(preproc_num).zfill(6) + f"_{t}.png")

                            if self.gt_needs_resampling:
                                #Apply a frame rotation to the panorama, resample, and save to the target directory.
                                converter = self.cam_to_PanoramaConverter["rig"]
                                
                                if t == "CubeDistance":
                                    img = read_compressed_float( in_rig_path )
                                    inter_method = 'nearest'
                                    img_conv, _ = converter(img, interpolation=inter_method)
                                    write_compressed_float( out_rig_path, img_conv )
                                elif t == "CubeScene":
                                    img = cv2.imread(in_rig_path, cv2.IMREAD_COLOR)
                                    inter_method = 'linear'
                                    img_conv, _ = converter(img, interpolation=inter_method)
                                    cv2.imwrite(out_rig_path, img_conv)
                                else:
                                    raise Exception(f'Unexpected image type {t}. ')

                            else:
                                # Copy the ground truth to the target directory.
                                shutil.copy(in_rig_path, out_rig_path)