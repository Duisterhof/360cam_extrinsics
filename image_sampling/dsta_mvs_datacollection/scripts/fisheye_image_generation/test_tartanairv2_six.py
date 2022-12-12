
import os
import sys

# Configure the Python search path.
_TOP_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
sys.path.insert(0, _TOP_DIR)
for i, p in enumerate(sys.path):
    print(f'{i}: {p}')

import cv2
from functools import partial
import numpy as np

import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt

from data_collection.mvs_utils.shape_struct import ShapeStruct
from data_collection.mvs_utils.camera_models import DoubleSphere, Equirectangular
from data_collection.image_sampler.six_images import SixPlanarAsBase
from data_collection.image_sampler.six_images import (
    FRONT, BACK, LEFT, RIGHT, TOP, BOTTOM )

from data_collection.mvs_utils.ftensor import f_eye

def ocv_read( fn ):
    image = cv2.imread(fn, cv2.IMREAD_UNCHANGED)
    assert image is not None, \
        f'{fn} read error. '
    return image

def read_rgb( fn ):
    return ocv_read(fn)

def read_dep( fn ):
    image = ocv_read(fn)
    return np.squeeze( image.view('<f4'), axis=-1 )

def read_seg( fn ):
    image = ocv_read(fn)
    return image.astype(np.uint8)

def ocv_write( fn, image ):
    cv2.imwrite(fn, image)
    
def write_as_is( fn, image ):
    ocv_write( fn, image )
    
def write_float_compressed(fn, image):
    assert(image.ndim == 2), 'image.ndim = {}'.format(image.ndim)
    
    # Check if the input array is contiguous.
    if ( not image.flags['C_CONTIGUOUS'] ):
        image = np.ascontiguousarray(image)

    dummy = np.expand_dims( image, 2 )
    ocv_write( fn, dummy )
    
def write_float_grayscale(fn, image, m0, m1):
    # Clip values.
    image = np.clip( image, m0, m1 )
    
    # Normalize img.
    image = image.astype(np.float32)
    image = image - m0
    image = image / ( m1 - m0 )
    image = np.clip(image * 255.0, 0.0, 255.0).astype(np.uint8)
    
    ocv_write( fn, image )

class SegmentationVisualizer(object):
    def __init__(self, color_code_fn):
        super().__init__()
        
        # Read the color code table.
        self.color_code_array = np.loadtxt( color_code_fn, delimiter=',', dtype=np.uint8 )
        
    def __call__(self, fn, image):
        assert image.dtype == np.uint8, \
            f'Expected np.uint8 got {image.dtype}. '
            
        H, W = image.shape[:2]
        image = image.reshape( (-1, ) )
        
        out = self.color_code_array[ image, : ]
        out = out.reshape( ( H, W, 3 ) )
        
        ocv_write( fn, out )

def read_images( dir_name, reader, prefix, suffix='' ):
    '''
    Read the input images based on the hardcoded names.
    
    dir_name (str): The input directory.
    reader (callable): The funciton for reading the images.
    suffix (str): The suffix to put at the end of the hardcoded image names.
    
    Returns:
    A dictionary of all the six images.
    '''
    global FRONT, BACK, LEFT, RIGHT, TOP, BOTTOM
    
    input_names = {
        FRONT:  os.path.join( dir_name, '%sfront%s.png'  % (prefix, suffix) ),
        BACK:   os.path.join( dir_name, '%sback%s.png'   % (prefix, suffix) ),
        LEFT:   os.path.join( dir_name, '%sleft%s.png'   % (prefix, suffix) ),
        RIGHT:  os.path.join( dir_name, '%sright%s.png'  % (prefix, suffix) ),
        TOP:    os.path.join( dir_name, '%stop%s.png'    % (prefix, suffix) ),
        BOTTOM: os.path.join( dir_name, '%sbottom%s.png' % (prefix, suffix) ), }
    
    # Read all the images.
    images = dict()
    for key, value in input_names.items():
        images[key] = reader(value)
        
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

def sample_image(out_dir, out_fn_base, image_writer, camera_model, image_dict, interpolation='linear'):
    print(f'=== Sample fisheye images to {out_dir}. ===')
    
    if not os.path.isdir(out_dir):
        os.makedirs(out_dir)

    # The sampler.
    sampler = SixPlanarAsBase(camera_model.fov_degree, camera_model, f_eye(3, f0='cbf', f1='fisheye'))
    sampler.enable_cuda()
    
    # Sample one image.
    sampled, _ = sampler(image_dict, interpolation=interpolation)
    out_fn = os.path.join(out_dir, '%s.png' % (out_fn_base) )
    image_writer(out_fn, sampled)

if __name__ == '__main__':
    data_dir = './data/20220827_sample_fisheye_wenshan'
    out_dir  = './output/20220827_tartanair_v2'
    
    # Read the images.
    prefix = '000013_'
    image_dict_rgb = read_images( os.path.join(data_dir, 'image'), read_rgb, prefix, suffix='' )
    image_dict_dep = read_images( os.path.join(data_dir, 'depth'), read_dep, prefix, suffix='_depth' )
    image_dict_seg = read_images( os.path.join(data_dir, 'seg'),   read_seg, prefix, suffix='_seg' )
    
    # The fisheye camera model.
    camera_model_fisheye = DoubleSphere(
        -0.196, 0.589, 
        235, 235,
        499.5, 499.5, 
        195, ShapeStruct(H=1000, W=1000),
        in_to_tensor=True, 
        out_to_numpy=False)
    
    # The equirectangular mdoel.
    camera_model_equirect = Equirectangular(
        1023.5, 511.5, 
        ShapeStruct(H=1024, W=2048),
        in_to_tensor=True, 
        out_to_numpy=False)
    
    # The visualizer for the segmentation image.
    sv = SegmentationVisualizer( os.path.join( data_dir, 'seg_rgbs.txt' ) )
    
    # Prepare other arguments.
    sample_args = [
        {
            'out_dir': os.path.join(out_dir, 'fisheys_rgb'),
            'out_fn_base': f'{prefix}rgb',
            'camera_model': camera_model_fisheye,
            'image_dict': image_dict_rgb,
            'interpolation': 'linear',
            'image_writer': write_as_is,
        },
        {
            'out_dir': os.path.join(out_dir, 'fisheye_depth'),
            'out_fn_base': f'{prefix}depth',
            'camera_model': camera_model_fisheye,
            'image_dict': image_dict_dep,
            'interpolation': 'nearest',
            'image_writer': partial(write_float_grayscale, m0=0.5, m1=50),
        },
        {
            'out_dir': os.path.join(out_dir, 'fisheye_seg'),
            'out_fn_base': f'{prefix}seg',
            'camera_model': camera_model_fisheye,
            'image_dict': image_dict_seg,
            'interpolation': 'nearest',
            'image_writer': sv,
        },
        {
            'out_dir': os.path.join(out_dir, 'equirect_rgb'),
            'out_fn_base': f'{prefix}rgb',
            'camera_model': camera_model_equirect,
            'image_dict': image_dict_rgb,
            'interpolation': 'linear',
            'image_writer': write_as_is,
        },
        {
            'out_dir': os.path.join(out_dir, 'equirect_depth'),
            'out_fn_base': f'{prefix}depth',
            'camera_model': camera_model_equirect,
            'image_dict': image_dict_dep,
            'interpolation': 'nearest',
            'image_writer': partial(write_float_grayscale, m0=0.5, m1=50),
        },
        {
            'out_dir': os.path.join(out_dir, 'equirect_seg'),
            'out_fn_base': f'{prefix}seg',
            'camera_model': camera_model_equirect,
            'image_dict': image_dict_seg,
            'interpolation': 'nearest',
            'image_writer': sv,
        },
    ]
    
    # Do the sampling.
    for args in sample_args:
        sample_image(**args)
        