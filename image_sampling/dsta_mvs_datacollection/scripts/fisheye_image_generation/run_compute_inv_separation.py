
import os
import sys

# Configure the Python search path.
_TOP_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
sys.path.insert(0, _TOP_DIR)
for i, p in enumerate(sys.path):
    print(f'{i}: {p}')

import cv2
import numpy as np

import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt

from data_collection.mvs_utils.shape_struct import ShapeStruct
from data_collection.mvs_utils.camera_models import DoubleSphere, Equirectangular, CameraModel
from data_collection.image_sampler.six_images import SixPlanarAsBase
from data_collection.image_sampler.six_images import (
    FRONT, BACK, LEFT, RIGHT, TOP, BOTTOM )

from data_collection.mvs_utils.ftensor import f_eye

def write_heat_map(fn, mat, title='No title'):
    fig, ax = plt.subplots(1, 1, figsize=(10, 10))
    im = ax.imshow( mat, cmap=plt.get_cmap('jet'), vmin=0.5, vmax=10 )
    fig.colorbar(im)
    ax.set_title(title)
    fig.savefig(fn)

def read_image( fn ):
    image = cv2.imread(fn, cv2.IMREAD_UNCHANGED)
    assert image is not None, \
        f'{fn} read error. '
    return image

def read_images( dir_name ):
    '''
    Read the input images based on the hardcoded names.
    
    Returns:
    A dictionary of all the six images.
    '''
    global FRONT, BACK, LEFT, RIGHT, TOP, BOTTOM
    
    input_names = {
        FRONT:  os.path.join( dir_name, '000013_front.png'  ),
        BACK:   os.path.join( dir_name, '000013_back.png'   ),
        LEFT:   os.path.join( dir_name, '000013_left.png'   ),
        RIGHT:  os.path.join( dir_name, '000013_right.png'  ),
        TOP:    os.path.join( dir_name, '000013_up.png'     ),
        BOTTOM: os.path.join( dir_name, '000013_bottom.png' ), }
    
    # Read all the images.
    images = dict()
    for key, value in input_names.items():
        images[key] = read_image(value)
        
    return images

def create_scaled_double_sphere(ori_model: DoubleSphere, new_ss: ShapeStruct) -> DoubleSphere:
    '''
    ori_model: The original camera model. 
    new_ss: The shape of the scaled camera model.
    '''
    
    # Get the focal length and principle point of the origianl camera model.
    fx, fy = ori_model.fx, ori_model.fy
    cx, cy = ori_model.cx, ori_model.cy
    
    # The scale factor.
    q = new_ss.W / ori_model.ss.W
    
    # Scale the focal length and principle point.
    new_fx, new_fy = q * fx, q * fy
    new_cx, new_cy = q * cx, q * cy
    
    # Create a new camera model.
    return DoubleSphere(
        ori_model.xi, ori_model.alpha, 
        new_fx, new_fy,
        new_cx, new_cy, 
        ori_model.fov_degree, 
        new_ss,
        in_to_tensor=ori_model.in_to_tensor, 
        out_to_numpy=ori_model.out_to_numpy )

def create_scaled_equirectangular(ori_model, new_ss: ShapeStruct) -> Equirectangular:
    '''
    ori_model: The original camera model. 
    new_ss: The shape of the scaled camera model.
    '''
    
    # Get the focal length and principle point of the origianl camera model.
    cx, cy = ori_model.cx, ori_model.cy
    
    # The scale factor.
    q = new_ss.W / ori_model.ss.W
    
    # Scale the focal length and principle point.
    new_cx, new_cy = q * cx, q * cy
    
    # Create a new camera model.
    return Equirectangular(
        new_cx, new_cy,  
        new_ss,
        in_to_tensor=ori_model.in_to_tensor, 
        out_to_numpy=ori_model.out_to_numpy )

def main_fisheye(out_dir, image_dict):
    print('=== Fisheye. ===')
    
    if not os.path.isdir(out_dir):
        os.makedirs(out_dir)
        
    fov = 195
    lower_W, upper_W = 600, 1224
    
    # Create an initial camera model.
    ss = ShapeStruct(H=1000, W=1000)
    camera_model = DoubleSphere(
        -0.196, 0.589, 
        235, 235,
        499.5, 499.5, 
        fov, ss,
        in_to_tensor=True, 
        out_to_numpy=False)

    # Sample one image.
    converter = SixPlanarAsBase(fov, camera_model, f_eye(3, f0='cbf', f1='fisheye'))
    converter.enable_cuda()
    
    sampled, _ = converter(image_dict)
    out_fn = os.path.join(out_dir, 'result_six_fisheye_%s.png' % ('sample_H%d_W%d' % ( ss.H, ss.W ) ) )
    cv2.imwrite(out_fn, sampled)

    # Get the shape of the raw image.
    raw_image_shape = image_dict[FRONT].shape[:2]

    # Loop over all target fisheye image size.
    for w in range( lower_W, upper_W + 1, 100 ):
        # The shape.
        ss = ShapeStruct( H=w, W=w )
        
        # Create a scaled camera model.
        scaled_camera_model = create_scaled_double_sphere(
            camera_model, ss )

        converter = SixPlanarAsBase(fov, scaled_camera_model, f_eye(3, f0='cbf', f1='fisheye'))
        converter.enable_cuda()
    
        mean_sampling_diff, valid_mask = converter.compute_mean_samping_diff( raw_image_shape )
        single_mean = mean_sampling_diff[valid_mask].mean()
        print(f'w = {w}. Global mean sampling diff = {single_mean}')
        
        out_fn = os.path.join(out_dir, f'mean_diff_{w:03d}.png')
        title = f'fisheye {ss.shape} from 6 support at {raw_image_shape}, global mean = {single_mean:.2f}'
        print(title)
        write_heat_map(out_fn, mean_sampling_diff, title)

def main_equirectangular(out_dir, image_dict):
    print('=== Equirectangular. ===')
    if not os.path.isdir(out_dir):
        os.makedirs(out_dir)
        
    lower_W, upper_W = 1000, 2400
    
    # Create an initial camera model.
    ss = ShapeStruct(H=1024, W=2048)
    camera_model = Equirectangular(
        1023.5, 511.5, 
        ss,
        in_to_tensor=True, 
        out_to_numpy=False)

    # Sample one image.
    converter = SixPlanarAsBase(360, camera_model, f_eye(3, f0='cbf', f1='fisheye'))
    converter.enable_cuda()
    
    sampled, _ = converter(image_dict)
    out_fn = os.path.join(out_dir, 'result_six_equirect_%s.png' % ('sample_H%d_W%d' % ( ss.H, ss.W ) ) )
    cv2.imwrite(out_fn, sampled)

    # Get the shape of the raw image.
    raw_image_shape = image_dict[FRONT].shape[:2]

    # Loop over all target fisheye image size.
    for w in range( lower_W, upper_W + 1, 100 ):
        # The shape.
        ss = ShapeStruct( H=w//2, W=w )
        
        # Create a scaled camera model.
        scaled_camera_model = create_scaled_equirectangular(
            camera_model, ss )

        converter = SixPlanarAsBase(360, scaled_camera_model, f_eye(3, f0='cbf', f1='fisheye'))
        converter.enable_cuda()
    
        mean_sampling_diff, valid_mask = converter.compute_mean_samping_diff( raw_image_shape )
        single_mean = mean_sampling_diff[valid_mask].mean()
        print(f'w = {w}. Global mean sampling diff = {single_mean}')
        
        out_fn = os.path.join(out_dir, f'mean_diff_{w:03d}.png')
        title = f'equirectangular {ss.shape} from 6 support at {raw_image_shape}, global mean = {single_mean:.2f}'
        print(title)
        write_heat_map(out_fn, mean_sampling_diff, title)

if __name__ == '__main__':
    data_dir = './data/20220827_sample_fisheye_wenshan'

    out_dir = './output/20220827_tartanair_v2'
    
    # Read the input images.
    image_dict = read_images(os.path.join(data_dir, 'image'))
    
    # Fisheye.
    main_fisheye(os.path.join(out_dir, 'fisheye'), image_dict)
    main_equirectangular(os.path.join(out_dir, 'equirectangular'), image_dict)
        