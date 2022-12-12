
import os
import sys

# Configure the Python search path.
_TOP_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
sys.path.insert(0, _TOP_DIR)
for i, p in enumerate(sys.path):
    print(f'{i}: {p}')

import cv2
from datetime import datetime
import multiprocessing
import numpy as np
import re

import torch

# Configure matplotlib for Windows WSL.
import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt

# Local.
from data_collection.multiproc.process_pool import ( ReplicatedArgument, PoolWithLogger )
from data_collection.multiproc.utils import ( compose_name_from_process_name, job_print_info )
from data_collection.mvs_utils.shape_struct import ShapeStruct
from data_collection.mvs_utils.camera_models import DoubleSphere, Equirectangular
from data_collection.mvs_utils.camera_models import Velodyne
from data_collection.mvs_utils.simulated_lidar_pre_def_models import VELODYNE_VLP_32C
from data_collection.mvs_utils.point_cloud_helper import write_PLY
from data_collection.image_sampler.six_images import (
    SixPlanarNumba, SixPlanarTorch )
from data_collection.image_sampler.blend_function import (BlendBy2ndOrderGradTorch, BlendBy2ndOrderGradOcv)

SixPlanarSampler = None
BlendFactorCalculator = None
debug_out_dir_suffix = None

# SixPlanarSampler = SixPlanarTorch
# debug_out_dir_suffix = '_torch'

# SixPlanarSampler = SixPlanarNumba
# debug_out_dir_suffix = '_numba'

from data_collection.image_sampler.six_images_common import (
    FRONT, BACK, LEFT, RIGHT, TOP, BOTTOM)

from utils import get_all_orientations

def write_heat_map(fn, mat, title='No title'):
    fig, ax = plt.subplots(1, 1, figsize=(10, 10))
    im = ax.imshow( mat, cmap=plt.get_cmap('jet'), vmin=0.5, vmax=10 )
    fig.colorbar(im)
    ax.set_title(title)
    fig.savefig(fn)

def process_do_sampling(data_dir, img_names, out_dir):
    global FRONT, RIGHT, BOTTOM, LEFT, TOP, BACK
    global P_JOB_LOGGER
    global P_SIX_PLANAR_SAMPLER_TYPE
        
    proc_name = multiprocessing.current_process().name
    proc_name = compose_name_from_process_name('P', proc_name)
    
    if not os.path.isdir(out_dir):
        os.makedirs(out_dir)
    
    img_keys = [FRONT, RIGHT, BOTTOM, LEFT, TOP, BACK]
    img_dict = dict()
    
    for fn, key in zip(img_names, img_keys):
        in_fn = os.path.join(data_dir, fn)
        img = cv2.imread(in_fn)
        assert img is not None, f'{in_fn} does not exist. '
        
        img_dict[key] = img

    fov = 195
    # shape = np.array([686,686])
    # shape = np.array([1028, 1224])
    ss = ShapeStruct( H=1028, W=1224 )
    # dsc = [-0.17023409,0.59679147,156.96507623, 157.72873153, 343,343]
    # camera_model = DoubleSphere(-0.17023409,0.59679147,156.96507623, 157.72873153, 343, 343, fov)
    camera_model = DoubleSphere(
        -0.196, 0.589, 
        235, 235,
        612, 514, fov, ss, 
        in_to_tensor=True,
        out_to_numpy=False)

    R_raw_fisheye_list, case_names_fisheye = get_all_orientations()

    flag_mask_output_done = False
    for case_name, R_raw_fisheye in zip( case_names_fisheye, R_raw_fisheye_list ):
        job_print_info(P_JOB_LOGGER, proc_name, case_name)
        converter = P_SIX_PLANAR_SAMPLER_TYPE(fov, camera_model, R_raw_fisheye)

        sampled, valid_mask = converter(img_dict)
        out_fn = os.path.join(out_dir, 'result_six_fisheye_%s.png' % (case_name))
        cv2.imwrite(out_fn, sampled)
        
        if not flag_mask_output_done:
            valid_mask = valid_mask.astype(np.uint8) * 255
            out_fn = os.path.join(out_dir, 'valid_mask.png')
            cv2.imwrite(out_fn, valid_mask)
            job_print_info(P_JOB_LOGGER, proc_name, f'Write mask to {out_fn}. ')
            flag_mask_output_done = True
    
    # Mean sampling difference.
    job_print_info(P_JOB_LOGGER, proc_name, 'Mean sampling difference. ')
    converter = P_SIX_PLANAR_SAMPLER_TYPE(fov, camera_model, R_raw_fisheye_list[0])
    
    support_shape = img_dict[FRONT].shape[:2]
    mean_sampling_diff, valid_mask = converter.compute_mean_samping_diff(support_shape)
    single_mean = mean_sampling_diff[valid_mask].mean()
    job_print_info(P_JOB_LOGGER, proc_name, f'Global mean sampling diff = {single_mean}')
    out_fn = os.path.join(out_dir, 'mean_diff.png')
    title = f'fisheye {ss.shape[:2]} from 6 support at {support_shape[:2]}, global mean = {single_mean:.2f}'
    job_print_info(P_JOB_LOGGER, proc_name, title)
    write_heat_map(out_fn, mean_sampling_diff, title)
    
    job_print_info(P_JOB_LOGGER, proc_name, 'equirectangular')
    # Equirectangular.
    
    # New shape.
    nss = ShapeStruct( H=1024, W=2048 )
    
    # lon_shift=0
    # lon_shift=(-np.pi/2)
    # camera_equi_rect = Equirectangular(1023.5, 511.5, nss, lon_shift=lon_shift, open_span=False, 
    #                                    in_to_tensor=True, out_to_numpy=True)
    camera_equi_rect = Equirectangular(
        nss, 
        longitude_span=(-np.pi * 3 /2, np.pi/2), 
        latitude_span=(-np.pi/2, np.pi/2),
        open_span=False, 
        in_to_tensor=True, 
        out_to_numpy=False)
    converter_equi_rect = P_SIX_PLANAR_SAMPLER_TYPE( 360, camera_equi_rect )
    
    sampled_equi_rect, _ = converter_equi_rect(img_dict)
    out_fn = os.path.join(out_dir, 'result_six_equi_rect.png')
    cv2.imwrite(out_fn, sampled_equi_rect)
    
    return 0

def job_initializer(logger_name, log_queue, *args):
    # Use global variables to transfer variables to the job process.
    # This funciton is called in the job process.
    # https://superfastpython.com/multiprocessing-pool-initializer/
    global P_JOB_LOGGER
    
    # The logger.
    P_JOB_LOGGER = PoolWithLogger.job_prepare_logger(logger_name, log_queue)
    
    # print(P_JOB_LOGGER.handlers)
    
    # It seems on Windows, global variables are not updated after spawning the child process.
    # So we need to create dedicated ones for each worker process.
    global P_SIX_PLANAR_SAMPLER_TYPE
    
    if args[0] == 'torch':
        P_SIX_PLANAR_SAMPLER_TYPE = SixPlanarTorch
    else:
        P_SIX_PLANAR_SAMPLER_TYPE = SixPlanarNumba

def main_rgb(backend, blend_threshold_factor=0):
    global FRONT, RIGHT, BOTTOM, LEFT, TOP, BACK
    global debug_out_dir_suffix
    
    print('main_rgb')
    
    # data_dir = 'data'
    # out_dir = 'output'
    # img_names = ['front.png','right.png','down.png','left.png','up.png','back.png']
    
    # data_dir = 'data/20220618'
    # out_dir = 'output/20220618'
    # img_names = ['front.png','right.png','bottom.png','left.png','top.png','back.png']
    
    # data_dir = 'data/20220618_02'
    # out_dir = 'output/20220618_02'
    # img_names = ['front.png','right.png','bottom.png','left.png','top.png','back.png']
    
    # data_dir = 'data/20220618_03_fov89'
    # out_dir = 'output/20220618_03_fov89'
    # img_names = ['front.png','right.png','bottom.png','left.png','top.png','back.png']
    
    # data_dir = 'data/20220618_04_boxes_fov89'
    # out_dir = 'output/20220618_04_boxes_fov89'
    # img_names = ['front.png','right.png','bottom.png','left.png','top.png','back.png']
    
    # data_dir = 'data/20220618_05_boxes_fov90'
    # out_dir = 'output/20220618_05_boxes_fov90'
    # img_names = ['front.png','right.png','bottom.png','left.png','top.png','back.png']
    
    # data_dir = 'data/20220620_01_w640'
    # out_dir = 'output/20220620_01_w640'
    # img_names = ['front.png','right.png','bottom.png','left.png','top.png','back.png']
    
    # data_dir = 'data/20220620_03_w768'
    # out_dir = 'output/20220620_03_w768'
    # img_names = ['front.png','right.png','bottom.png','left.png','top.png','back.png']
    
    # data_dir = 'data/20220620_04_w896'
    data_dir = [
        'data/20220620_04_w896',
        'data/20220620_04_w896_mp' # MP for multi-processing.
    ]
    
    out_dir_base = [ 
        'output/20220620_04_w896',
        'output/20220620_04_w896_mp',
    ]
    out_dir = [ f'{d}{debug_out_dir_suffix}' for d in out_dir_base ]
    
    img_names = [ 
                 ['front.png','right.png','bottom.png','left.png','top.png','back.png']
    ] * 2
    
    # data_dir = 'data/20220620_02_w1024'
    # out_dir = 'output/20220620_02_w1024'
    # img_names = ['front.png','right.png','bottom.png','left.png','top.png','back.png']
    
    # for data_dir_p, img_names_p, out_dir_p in zip( data_dir, img_names, out_dir ):
    #     process_do_sampling( data_dir_p, img_names_p, out_dir_p )
        
    mp_args = zip( data_dir, img_names, out_dir )
    with PoolWithLogger(2, job_initializer, 'tartanair', './logger_output.log', (backend)) as pool:
        results = pool.map( process_do_sampling, mp_args )
        print(f'Main: \nMultiprocessing')

def convert_bgra_2_float(img):
    assert img.shape[2] == 4, f'img.shape = {img.shape}'
    return np.squeeze( img.view('<f4') )

def normalize_float_image(img, min_value=None, max_value=None):
    assert img.dtype == 'float32', f'img.dtype = {img.dtype}'
    
    # Figure out the min and max values.
    if min_value is None:
        min_value = img.min()
    
    if max_value is None:
        max_value = img.max()
        
    if min_value >= max_value:
        raise ValueError(f'Wrong min and max values: min_value = {min_value}, max_value = {max_value}')
    
    # First, clip the input image.
    img = np.clip(img, min_value, max_value)
    
    return ( img - min_value ) / ( max_value - min_value )

def write_distance_image_as_ply( out_fn, camera_model, dist_img, max_dist=100 ):
    pixel_coordinates = camera_model.pixel_coordinates() # 2xN.
    pixel_rays, ray_valid_mask = camera_model.pixel_2_ray( pixel_coordinates )
    
    if isinstance(pixel_rays, torch.Tensor):
        pixel_rays = pixel_rays.cpu().numpy()
    
    points = pixel_rays * dist_img.reshape((-1,)) # Broadcast automatically.
    
    mask = np.linalg.norm(points, axis=0) < max_dist
    
    write_PLY(out_fn, points[:, mask])

def write_points(out_fn, points, max_dist):
    '''
    points: 3xN.
    '''

    # Compute the distance of each point.
    d = np.linalg.norm(points, axis=0)
    
    # Mask.
    mask = d <= max_dist
    
    write_PLY(out_fn, points[:, mask])
class BlendFactorDebugCallback(object):
    def __init__(self, out_dir, fn_base) -> None:
        super().__init__()
        
        self.out_dir = out_dir
        self.fn_base = fn_base # Filename base.
    
    @staticmethod
    def extract_first_batch_as_numpy(f):
        if not isinstance(f, torch.Tensor):
            return f
        else:
            return f[0].squeeze(0).cpu().numpy()
    
    def save_graysacle(self, fn_suffix, blend_factor):
        # Convert to NumPy array if inputs are PyTorch Tensors.
        blend_factor = BlendFactorDebugCallback.extract_first_batch_as_numpy(blend_factor)
        
        # Convert to grayscale images.
        to_gray = lambda x : np.clip(x.astype(np.float32) * 255, 0, 255).astype(np.uint8)
        blend_factor = to_gray( blend_factor )
        
        # Composet the filename.
        fn = os.path.join( self.out_dir, f'{self.fn_base}{fn_suffix}.png' )
        
        # Save.
        cv2.imwrite(fn, blend_factor)
    
    def __call__(self, blend_factor_ori, blend_factor_sampled):
        self.save_graysacle( '_ori', blend_factor_ori )
        self.save_graysacle( '_smp', blend_factor_sampled )

def main_distance(backend, blend_threshold_factor=0):
    def write_heat_map(fn, mat, title='No title'):
        fig, ax = plt.subplots(1, 1, figsize=(10, 10))
        im = ax.imshow( mat, cmap=plt.get_cmap('jet'), vmin=0.5, vmax=10 )
        fig.colorbar(im)
        ax.set_title(title)
        fig.savefig(fn)
    
    global FRONT, BACK, LEFT, RIGHT, TOP, BOTTOM
    global debug_out_dir_suffix
    global SixPlanarSampler
    global BlendFactorCalculator
    
    print('main_distance')
    
    # data_dir = 'data/20220703_distance_w896'
    # out_dir = 'output/20220703_distance_w896'
    
    data_dir = 'data/20220703_distance_w896_oldtown'
    out_dir = 'output/20220703_distance_w896_oldtown'
    out_dir = f'{out_dir}{debug_out_dir_suffix}'
    
    img_names = ['front.png','right.png','bottom.png','left.png','top.png','back.png']
    img_names_distance = ['front_distance.png','right_distance.png','bottom_distance.png','left_distance.png','top_distance.png','back_distance.png']
    
    if not os.path.isdir(out_dir):
        os.makedirs(out_dir)
    
    img_keys = [FRONT, RIGHT, BOTTOM, LEFT, TOP, BACK]
    img_dict = dict()
    img_dict_distance = dict()
    
    for fn, fn_distance, key in zip(img_names, img_names_distance, img_keys):
        in_fn = os.path.join(data_dir, fn)
        img = cv2.imread(in_fn, cv2.IMREAD_UNCHANGED)
        assert img is not None, f'{in_fn} does not exist. '
        
        img_dict[key] = img
        
        in_fn_distance = os.path.join(data_dir, fn_distance)
        img_distance = cv2.imread(in_fn_distance, cv2.IMREAD_UNCHANGED)
        assert img_distance is not None, f'{in_fn_distance} does not exist. '
        
        img_dict_distance[key] = convert_bgra_2_float(img_distance)

    fov = 195
    # shape = np.array([686,686])
    # shape = np.array([1028, 1224])
    ss = ShapeStruct(H=1028, W=1224)
    # dsc = [-0.17023409,0.59679147,156.96507623, 157.72873153, 343,343]
    # camera_model = DoubleSphere(-0.17023409,0.59679147,156.96507623, 157.72873153, 343, 343, fov)
    camera_model = DoubleSphere(
        -0.196, 0.589, 
        235, 235,
        612, 514, fov, ss, 
        in_to_tensor=True,
        out_to_numpy=False) # Was True for the sampler before torch.

    R_raw_fisheye_list, case_names_fisheye = get_all_orientations()

    flag_mask_output_done = False
    for case_name, R_raw_fisheye in zip( case_names_fisheye, R_raw_fisheye_list ):
        print(case_name)
        converter = SixPlanarSampler(fov, camera_model, R_raw_fisheye)

        sampled, valid_mask = converter(img_dict)

        out_fn = os.path.join(out_dir, 'result_six_fisheye_%s.png' % (case_name))
        cv2.imwrite(out_fn, sampled)
        
        if not flag_mask_output_done:
            valid_mask = valid_mask.astype(np.uint8) * 255
            out_fn = os.path.join(out_dir, 'valid_mask.png')
            cv2.imwrite(out_fn, valid_mask)
            print(f'Write mask to {out_fn}. ')
            flag_mask_output_done = True
        
    # ========== The distance image. ==========
    
    # New shape.
    nss = ShapeStruct( H=1024, W=2048 )
    
    # lon_shift=(-np.pi/2)
    # camera_equi_rect = Equirectangular(1023.5, 511.5, nss, lon_shift=lon_shift, open_span=False,
    #                                    in_to_tensor=True, out_to_numpy=True)
    camera_equi_rect = Equirectangular(
        nss, 
        longitude_span=(-np.pi * 3 /2, np.pi/2), 
        latitude_span=(-np.pi/2, np.pi/2),
        open_span=False,
        in_to_tensor=True, 
        out_to_numpy=False)
    
    converter_equi_rect = SixPlanarSampler( 360, camera_equi_rect )
    
    sampled_equi_rect, _ = converter_equi_rect(img_dict)
    out_fn = os.path.join(out_dir, 'result_six_equi_rect.png')
    cv2.imwrite(out_fn, sampled_equi_rect)

    sampled_equi_rect_distance, _ = converter_equi_rect(
        img_dict_distance, interpolation='nearest', invalid_pixel_value=-1)
    
    # Create a PLY point cloud file.
    ply_fn = os.path.join(out_dir, 'result_six_equi_rect_distance_points.ply')
    write_distance_image_as_ply( ply_fn, camera_equi_rect, sampled_equi_rect_distance )
    
    # Normalize the image.
    normalized = ( normalize_float_image(sampled_equi_rect_distance, 0, 100) * 255 ).astype(np.uint8)
    
    out_fn = os.path.join(out_dir, 'result_six_equi_rect_distance_vis.png')
    cv2.imwrite(out_fn, normalized)
    
    if blend_threshold_factor > 0:
        debug_callback = BlendFactorDebugCallback(out_dir, 'blend_factor')
        blend_func = BlendFactorCalculator(blend_threshold_factor)
        sampled_equi_rect_distance, _ = converter_equi_rect.blend_interpolation(
            img_dict_distance, blend_func, invalid_pixel_value=-1, debug_callback=debug_callback)
        
        # Create a PLY point cloud file.
        ply_fn = os.path.join(out_dir, 'result_six_equi_rect_distance_blend_points.ply')
        write_distance_image_as_ply( ply_fn, camera_equi_rect, sampled_equi_rect_distance )
        
        # Normalize the image.
        normalized = ( normalize_float_image(sampled_equi_rect_distance, 0, 100) * 255 ).astype(np.uint8)
        
        out_fn = os.path.join(out_dir, 'result_six_equi_rect_distance_blend_vis.png')
        cv2.imwrite(out_fn, normalized)

def main_lidar(backend, blend_threshold_factor=0.1):
    global FRONT, BACK, LEFT, RIGHT, TOP, BOTTOM
    global debug_out_dir_suffix
    global SixPlanarSampler
    global BlendFactorCalculator
    
    print('main_lidar')
    
    data_dir = 'data/20220703_distance_w896_oldtown'
    out_dir = 'output/20220703_distance_w896_oldtown'
    out_dir = f'{out_dir}{debug_out_dir_suffix}'
    
    img_names = ['front.png','right.png','bottom.png','left.png','top.png','back.png']
    img_names_distance = ['front_distance.png','right_distance.png','bottom_distance.png','left_distance.png','top_distance.png','back_distance.png']
    
    if not os.path.isdir(out_dir):
        os.makedirs(out_dir)
    
    img_keys = [FRONT, RIGHT, BOTTOM, LEFT, TOP, BACK]
    img_dict = dict()
    img_dict_distance = dict()
    
    for fn, fn_distance, key in zip(img_names, img_names_distance, img_keys):
        in_fn = os.path.join(data_dir, fn)
        img = cv2.imread(in_fn, cv2.IMREAD_UNCHANGED)
        assert img is not None, f'{in_fn} does not exist. '
        
        img_dict[key] = img
        
        in_fn_distance = os.path.join(data_dir, fn_distance)
        img_distance = cv2.imread(in_fn_distance, cv2.IMREAD_UNCHANGED)
        assert img_distance is not None, f'{in_fn_distance} does not exist. '
        
        img_dict_distance[key] = convert_bgra_2_float(img_distance)
    
    # Create a Velodyne model.
    velodyne = Velodyne(
        description=VELODYNE_VLP_32C,
        in_to_tensor=True,
        out_to_numpy=False)
    
    converter_velodyne = SixPlanarSampler( 360, velodyne )
    
    sampled_velodyne, _ = converter_velodyne(
        img_dict_distance, interpolation='nearest', invalid_pixel_value=-1)

    points = velodyne.measure_wrt_lidar_frame( torch.from_numpy(sampled_velodyne).view( (1, 1, -1) ) )
    points = points.cpu().numpy().squeeze(0)
    
    # Create a PLY point cloud file.
    ply_fn = os.path.join(out_dir, 'result_six_velodyne_points.ply')
    write_points( ply_fn, points, max_dist=100 )
        
    if blend_threshold_factor > 0:
        blend_func = BlendFactorCalculator(blend_threshold_factor)
        sampled_velodyne, _ = converter_velodyne.blend_interpolation(
            img_dict_distance, blend_func, invalid_pixel_value=-1)
        
        points = velodyne.measure_wrt_lidar_frame( torch.from_numpy(sampled_velodyne).view((1, 1, -1)) )
        points = points.cpu().numpy().squeeze(0)
        
        # Create a PLY point cloud file.
        ply_fn = os.path.join(out_dir, 'result_six_velodyne_blend_points.ply')
        write_points( ply_fn, points, max_dist=100 )

TESTS = {
    'rgb': main_rgb,
    'distance': main_distance,
    'lidar': main_lidar,
}

if __name__ == '__main__':
    multiprocessing.set_start_method('spawn')
    
    import argparse
    
    parser = argparse.ArgumentParser()
    
    parser.add_argument('--testtype', type=str, choices=TESTS.keys(), default='rgb',
                        help='The type of test.')
    
    parser.add_argument('--backend', type=str, choices=['opencv', 'torch'],
                        help='The backend to use. ')
    
    parser.add_argument('--blend-threshold-factor', type=float, default=0,
                        help='Set positive value to enable blending between linear and nearest interpolation. ')
    
    args = parser.parse_args()
    
    if args.backend == 'torch':
        SixPlanarSampler = SixPlanarTorch
        BlendFactorCalculator = BlendBy2ndOrderGradTorch
        debug_out_dir_suffix = '_torch'
    else:
        SixPlanarSampler = SixPlanarNumba
        BlenderFactorCalculator = BlendBy2ndOrderGradOcv
        debug_out_dir_suffix = '_opencv'
    
    with torch.no_grad():
        TESTS[args.testtype](args.backend, args.blend_threshold_factor)