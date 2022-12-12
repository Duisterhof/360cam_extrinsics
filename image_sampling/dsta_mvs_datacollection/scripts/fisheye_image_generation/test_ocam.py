
import os
import sys

# Configure the Python search path.
_TOP_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
sys.path.insert(0, _TOP_DIR)
for i, p in enumerate(sys.path):
    print(f'{i}: {p}')

import cv2
import numpy as np
import yaml

import torch

# # Configure matplotlib for Windows WSL.
# import matplotlib
# matplotlib.use('agg')
# import matplotlib.pyplot as plt

from data_collection.mvs_utils.shape_struct import ShapeStruct
from data_collection.mvs_utils.camera_models import Ocam
from data_collection.image_sampler.six_images import (
    SixPlanarAsBase,
    FRONT, BACK, LEFT, RIGHT, TOP, BOTTOM )

from utils import get_all_orientations

def read_camera_config(fn):
    with open(fn, 'r') as f:
        config = yaml.safe_load(f)
    
    camera_list = []
    for cc in config['cameras']:
        camera_list.append({
            'cam_id': cc['cam_id'],
            'poly': cc['poly'],
            'inv_poly': cc['inv_poly'],
            'center': cc['center'],
            'affine': cc['affine'],
            'image_size': cc['image_size'],
            'max_fov': cc['max_fov'],
        })
        
    return camera_list

def main_rgb():
    data_dir = 'data/20220620_04_w896'
    out_dir = 'output/omnimvs/20220620_04_w896'
    img_names = ['front.png','right.png','bottom.png','left.png','top.png','back.png']
    camera_config_fn = './data/omnimvs/camera_config.yaml'
    
    if not os.path.isdir(out_dir):
        os.makedirs(out_dir)
    
    # Read the cameras.
    cameras = read_camera_config(camera_config_fn)
    # for camera in cameras:
    #     print(camera)
        
    # Selecte the first ocam.
    ocam  = cameras[0]
    fov   = ocam['max_fov']
    shape = ocam['image_size']
    ss = ShapeStruct( H=shape[0], W=shape[1] )
    print(f'ocam = \n{ocam}')
    
    # Input images.
    img_keys = [FRONT, RIGHT, BOTTOM, LEFT, TOP, BACK]
    img_dict = dict()
    
    for fn, key in zip(img_names, img_keys):
        in_fn = os.path.join(data_dir, fn)
        img = cv2.imread(in_fn)
        assert img is not None, f'{in_fn} does not exist. '
        
        img_dict[key] = img

    R_raw_fisheye_list, case_names_fisheye = get_all_orientations()
    
    # Create the camera model.
    # Note we need to flip the x and y coordinates.
    camera_model = Ocam(
        poly_coeff=ocam['poly'][-1:0:-1], # Skip the first element.
        inv_poly_coeff=ocam['inv_poly'][-1:0:-1], # Skip the first element.
        cx=ocam['center'][1], 
        cy=ocam['center'][0], 
        affine_coeff=ocam['affine'],
        fov_degree=fov,
        shape_struct=ss,
        in_to_tensor=True,
        out_to_numpy=True )
    
    for case_name, R_raw_fisheye in zip( case_names_fisheye, R_raw_fisheye_list ):
        print(case_name)
        converter = SixPlanarAsBase(fov, camera_model, R_raw_fisheye)
        converter.enable_cuda()

        sampled, _ = converter(img_dict)
        out_fn = os.path.join(out_dir, 'result_six_fisheye_%s.png' % (case_name))
        cv2.imwrite(out_fn, sampled)

if __name__ == '__main__':
    with torch.no_grad():
        main_rgb()