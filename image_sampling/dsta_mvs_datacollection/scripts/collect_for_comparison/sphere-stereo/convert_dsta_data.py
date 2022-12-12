import os
import sys

# The path of the current Python script.
_CURRENT_PATH       = os.path.dirname(os.path.realpath(__file__))
_TOP_PATH           = os.path.join(_CURRENT_PATH, '..', '..', '..')
_DATA_PIPELINE_PATH = os.path.join(_TOP_PATH, 'data_collection', 'ord_data_pipeline_rework', 'src')
sys.path.insert( 0, _DATA_PIPELINE_PATH )
sys.path.insert( 0, _TOP_PATH )
for i, p in enumerate(sys.path):
    print(f'{i}: {p}')

import argparse
import json
from pyquaternion import Quaternion
import re

from data_collection.mvs_utils.ftensor import (RefFrame, f_eye)
from data_collection.mvs_utils.frame_io import (parse_orientation, read_frame_graph)
# from data_collection.mvs_utils.metadata_reader import MetadataReader
from data_collection.mvs_utils.camera_models import CAMERA_MODELS
from data_collection.mvs_utils.camera_models import make_object as make_camera_model
from data_collection.mvs_utils.pretty_dict import (
    PrettyDict, PlainPrinter, DictPrinter, ListPrinter )

def read_manifest(fn):
    with open(fn, 'r') as fp:
        j_obj = json.load(fp)
        
    # Figure out the preprocessed subsets.
    subsets = j_obj['processing_manifest'].keys()
        
    # Create the camera models.
    camera_model_specs = j_obj['camera_models']
    camera_models = dict()
    for key, value in camera_model_specs.items():
        camera_model = make_camera_model(CAMERA_MODELS, value)
        camera_models[key] = camera_model
        
    # Traverse the samplers to assign camera models to each camera.
    samplers = j_obj['samplers']
    camera_camera_model_map = dict()
    camera_number_filter = re.compile(r'cam(\d+)')
    for key, value in samplers.items():
        if key == 'rig':
            continue
        
        orientation_dict = value['orientation']
        
        # Filter the camera number.
        matches = camera_number_filter.search(key)
        assert matches is not None, f'Sampler key {key} has no camera number suffix. '
        camera_number = int(matches.group(1))
        
        # Create an ftensor.
        pose_ft = f_eye(
            4, 
            f0=f'{orientation_dict["f0"]}{camera_number}', 
            f1=f'{orientation_dict["f1"]}{camera_number}',
            rotation=True )
        pose_ft.rotation = parse_orientation(orientation_dict)
        
        # 
        camera_camera_model_map[key] = dict(
            camera_model=camera_models[ value['cam_model_key'] ],
            orientation=orientation_dict,
            pose_ft=pose_ft,
        )

    return subsets, camera_camera_model_map

def transformation_ftensor_2_plain_pose_dict(ft):
    assert ft.shape[-2:] == (4, 4), f'ft.shape = {ft.shape}'

    # Translation.
    t = ft.translation.cpu().numpy()

    # Rotation.
    r = ft.rotation.cpu().numpy()
    q = Quaternion(matrix=r)

    return { 
        'px': float(t[0]),
        'py': float(t[1]),
        'pz': float(t[2]),
        'qx': float(q[1]),
        'qy': float(q[2]),
        'qz': float(q[3]),
        'qw': float(q[0]) }

def convert_camera_model_info_2_sphere_stereo_calib_list_entries(frame_graph, camera_model_info):
    # The pose. Assuming fisheye0 is the reference.
    this_fisheye_frame = camera_model_info['pose_ft'].f1
    t = frame_graph.query_transform( f0='fisheye0', f1=this_fisheye_frame )
    pose_dict = transformation_ftensor_2_plain_pose_dict(t)

    # The intrinsics.
    camera_model = camera_model_info['camera_model']
    intrinsics_dict = {
        'camera_type': 'ds',
        'intrinsics': {
            'fx': camera_model.fx,
            'fy': camera_model.fy,
            'cx': camera_model.cx,
            'cy': camera_model.cy,
            'xi': camera_model.xi,
            'alpha': camera_model.alpha
        }
    }

    # The resolution in (W, H) order. 
    # Check https://github.com/castacks/dsta-sphere-stereo-cvpr2021/blob/e484b34b2d6957ac2f25446c88752b15f06226bc/python/utils.py#L172
    resolution_list = camera_model.ss.size

    return pose_dict, intrinsics_dict, resolution_list

def write_sphere_stereo_conf(fn, conf):
    # # Construct the PrettyDict.
    # pd = PrettyDict()

    # # Printer for "value0".
    # dp_value0 = DictPrinter()
    # dp_value0['T_imu_cam'] = ListPrinter()
    # dp_value0['intrinsics'] = ListPrinter()
    # dp_value0['']

    with open(fn, 'w') as fp:
        json.dump(conf, fp, indent=4)

def handle_args():
    parser = argparse.ArgumentParser(description='Convert the DSTA data to the format of sphere-stereo. ')
    
    parser.add_argument('--data-dir', type=str, 
                        help='The root directory of the collected data. ')
     
    return parser.parse_args()

if __name__ == '__main__':
    # Handle the arguments.
    args = handle_args()
    
    # === Read the manifest file. ===
    subsets, camera_camera_model_map = \
        read_manifest(os.path.join( args.data_dir, 'manifest.json' ))
    
    print('Discovered subsets: ')
    for s in subsets:
        print(f'{s}')
        
    print('Cameras: ')
    for key, value in camera_camera_model_map.items():
        print(f'{key}: \n{value}')    
    
    # === Process each subset. ===
    for s in subsets:
        print(f'Processing subset {s}... ')
        
        frame_graph = \
            read_frame_graph(os.path.join( args.data_dir, s, 'frame_graph.json' ))
        
        # Frame fisheyex is associated with camx.
        # Add frame fisheyex to the frame graph by referencing the camera panorama frame (cpfx).
        for key, camera_model_info in camera_camera_model_map.items():
            fisheye_pose = camera_model_info['pose_ft']
            frame_graph.add_frame( RefFrame( fisheye_pose.f1, fisheye_pose.f1 ) )
            frame_graph.add_or_update_pose_edge( fisheye_pose )

        # Computing the relative pose between the cameras.
        out_list_T_imu_cam  = []
        out_list_intrinsics = []
        out_list_resolution = []

        sorted_keys = sorted( camera_camera_model_map.keys() )
        for key in sorted_keys:
            pose_dict, intrinsics_dict, resolution_list = \
                convert_camera_model_info_2_sphere_stereo_calib_list_entries(frame_graph, camera_camera_model_map[key])

            out_list_T_imu_cam.append(pose_dict)
            out_list_intrinsics.append(intrinsics_dict)
            out_list_resolution.append(resolution_list)

        # Compose the final dictionary.
        sphere_stereo_conf = {
            'value0': {
                'T_imu_cam': out_list_T_imu_cam,
                'intrinsics': out_list_intrinsics,
                'resolution': out_list_resolution,
                'gt_loader': { 'type': 'DSTAGTLoader' }
            }
        }

        # Write the conf file.
        print(sphere_stereo_conf)
        write_sphere_stereo_conf( 
            os.path.join( args.data_dir, s, 'sphere_stereo_calibration.json' ),
            sphere_stereo_conf )