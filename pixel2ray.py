import sys 
sys.path.insert(1,'image_sampling')
sys.path.insert(1,'image_sampling/dsta_mvs_datacollection')
import yaml 

# Local.
from data_collection.multiproc.process_pool import ( ReplicatedArgument, PoolWithLogger )
from data_collection.multiproc.utils import ( compose_name_from_process_name, job_print_info )
from data_collection.mvs_utils.shape_struct import ShapeStruct
from data_collection.mvs_utils.camera_models import DoubleSphere, Equirectangular
from data_collection.mvs_utils.point_cloud_helper import write_PLY
from data_collection.image_sampler.six_images import (
    SixPlanarNumba, SixPlanarTorch )
from data_collection.image_sampler.blend_function import (BlendBy2ndOrderGradTorch, BlendBy2ndOrderGradOcv)


class Pixel2Ray:
    def __init__(self,cam_params):
        with open(cam_params,'r') as stream:
            self.parsed_cam = yaml.safe_load(stream)

        self.cam_ids = list(self.parsed_cam.keys())
        self.cam_ids.sort()
    

    def load_cams(self):
        self.cams = []
        for cam_id in self.cam_ids:
            distortion = self.parsed_cam[cam_id]['distortion_model']
            projection = self.parsed_cam[cam_id]['camera_model']
            
            if (projection == 'ds' and distortion == 'none'):
                print("Double sphere model selected")

                fov = self.parsed_cam[cam_id]['fov']
                output_res = self.parsed_cam[cam_id]['resolution']
                ss = ShapeStruct( H=output_res[0], W=output_res[1] )
                camera_params = self.parsed_cam[cam_id]['intrinsics']+[fov, ss]
                self.cams.append(DoubleSphere(
                *camera_params,
                in_to_tensor=True,
                out_to_numpy=False))

            
            else:
                print("Camera model unknown")
                exit()
