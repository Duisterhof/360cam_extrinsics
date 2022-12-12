import os
import sys

# The path of the current Python script.
_CURRENT_PATH       = os.path.dirname(os.path.realpath(__file__))
_TOP_PATH           = os.path.join(_CURRENT_PATH, '..', '..', '..', 'data_collection')
_DATA_PIPELINE_PATH = os.path.join(_TOP_PATH, 'ord_data_pipeline_rework', 'src')
sys.path.insert( 0, _DATA_PIPELINE_PATH )
sys.path.insert( 0, _TOP_PATH )
for i, p in enumerate(sys.path):
    print(f'{i}: {p}')

import cv2
import glob
import numpy as np
import pandas as pd
from pyquaternion import Quaternion
import re
import shutil
import yaml

from data_collection.mvs_utils.metadata_reader import MetadataReader
from data_collection.mvs_utils.camera_models import Ocam
from data_collection.mvs_utils.shape_struct import ShapeStruct
from data_collection.image_sampler.full_view_rotation import FullViewRotation

def read_camera_config(fn):
    with open(fn, 'r') as f:
        config = yaml.safe_load(f)
    
    assert len(config['cameras']) != 0, f'No cameras specified in {fn}. '
    
    camera_list = []
    for cc in config['cameras']:
        camera_list.append({
            'cam_id': cc['cam_id'],
            'img_dir': cc['img_dir'],
            'poly': cc['poly'],
            'inv_poly': cc['inv_poly'],
            'center': cc['center'],
            'affine': cc['affine'],
            'image_size': cc['image_size'],
            'max_fov': cc['max_fov'],
        })
        
    return camera_list

def get_file_parts(fn):
    s0 = os.path.split(fn)
    s1 = os.path.splitext(s0[1])
    return s0[0], s1[0], s1[1]

def search_trajectory_dirs(root_dir, file_pattern='cam_paths.csv'):
    search_pattern = os.path.join( root_dir, '**', file_pattern )
    files = sorted( glob.glob( search_pattern, recursive=True ) )
    assert len(files) > 0, f'No files found by {search_pattern}. '
    
    # Get the directory part.
    return [ get_file_parts(f)[0] for f in files ]

class DummyArgs(object):
    def __init__(self, metadata_path, frame_graph_path):
        super().__init__()
        self.metadata_path = metadata_path
        self.frame_graph_path = frame_graph_path

def read_metadata(metadata_fn, frame_graph_fn, traj_data_dir):
    reader = MetadataReader(traj_data_dir)
    dummy_args = DummyArgs(metadata_fn, frame_graph_fn)
    reader.read_metadata_and_initialize_dirs(
        dummy_args.metadata_path, dummy_args.frame_graph_path, create_dirs=False)
    return reader.cam_to_camdata, reader.frame_graph

def create_image_out_dirs(root_dir, cameras):
    for camera in cameras:
        img_dir = os.path.join( root_dir, camera['img_dir'] )
        if not os.path.isdir(img_dir):
            os.makedirs(img_dir)
            
    # Make the ground truth directory.
    true_dir = os.path.join(root_dir, 'rig')
    if not os.path.isdir(true_dir):
        os.makedirs(true_dir)

def copy_camera_config_to_out_dir(out_dir, camera_config_fn):
    parts = get_file_parts(camera_config_fn)
    out_config_fn = os.path.join( out_dir, '%s%s' % (parts[1], parts[2]) )
    shutil.copy( camera_config_fn, out_config_fn )

def read_cam_paths(fn, suffix_cam, suffix_rig_list):
    '''
    This function parse the information saved in the cam_paths.csv file.
    
    Currently, the header of the cam_paths.csv is the absolute path of the 
    individual camera image directories. However, we cannot use the absolute 
    path since this script may be run inside a WSL or a Docker container, where
    filesystem structure is different with that used for raw data collection.
    We will assume that the last part of the absolute path is the subfolder name
    for the raw images.
    
    This function composes a dictionary. The keys are 'cam0', 'cam1', and 'rig_rgb', 'rig_dist', etc.
    The values are the relative path or the raw images.
    
    suffix_cam is the filename suffix for the camera images.
    suffix_rig_list is a list of suffixes for the rig RGB images and distance images.
    It is assumed that the RGB suffix of the rig images comes first.
    '''
    
    # Prepare for the subfoler name filter.
    cam_name_search = re.compile(r'(cam\d+)$')
    rig_name_search = re.compile(r'(rig)$')
    
    # Parse the CSV file by Pandas.
    df = pd.read_csv(fn, header=0, dtype=str)

    raw_image_dict = {}
    for name, values in df.iteritems():
        # Get the last part of the absolute path.
        # Because we do not know the what operating system it is,
        # then we just filter the last par of name.
        if rig_name_search.search(name) is not None:
            raw_image_dict['rig_rgb'] = \
                [ os.path.join('rig', '%s_%s.png' % (f, suffix_rig_list[0])) for f in values ]
            raw_image_dict['rig_dist'] = \
                [ os.path.join('rig', '%s_%s.png' % (f, suffix_rig_list[1])) for f in values ]
            continue
        
        cam_search_result = cam_name_search.search(name)
        if cam_search_result is not None:
            sub_dir = cam_search_result[0]
            raw_image_dict[sub_dir] = [ os.path.join(sub_dir, '%s_%s.png' % (f, suffix_cam)) for f in values.tolist() ]    
    return raw_image_dict

def create_camera_model(camera_config):
    # Prepare the shape struct.
    # OmniMVS uses HxW ordering.
    shape = camera_config['image_size']
    # Not using *shape for explicity.
    ss = ShapeStruct(H=shape[0], W=shape[1])
    
    # Create the camera model.
    # Note we need to flip the x and y coordinates.
    return Ocam(
        poly_coeff=camera_config['poly'][-1:0:-1], # Skip the first element.
        inv_poly_coeff=camera_config['inv_poly'][-1:0:-1], # Skip the first element.
        cx=camera_config['center'][1], 
        cy=camera_config['center'][0], 
        affine_coeff=camera_config['affine'],
        fov_degree=camera_config['max_fov'],
        shape_struct=ss,
        in_to_tensor=True,
        out_to_numpy=True )

def read_image(fn):
    assert( os.path.isfile(fn) ), \
        f'{fn} does not exist. '

    return cv2.imread(fn, cv2.IMREAD_UNCHANGED)

def sample_fisheye_images( startd_index, out_dir, raw_image_root_dir, raw_image_fn_list, sampler ):
    for i, fn in enumerate(raw_image_fn_list):
        # Compose the input raw image path.
        raw_fn = os.path.join( raw_image_root_dir, fn )
        
        # Read the raw image.
        raw_image = read_image(raw_fn)
        
        # Sample.
        sampled, _ = sampler(raw_image)
        
        # Compose the output filename.
        out_fn = os.path.join( out_dir, '%04d.png' % ( startd_index + i ) )
        
        # Write.
        cv2.imwrite(out_fn, sampled)
        print(f'Fisheye image written to {out_fn}')
    return len(raw_image_fn_list)

def process_trajectory(start_index, traj_dir, out_dir, metadata, cameras):
    '''
    start_index is the image index for the first generated image.
    Note that omnimvs starts from 1. However, start_index is zero-based.
    cameras is the camera data read from the config file.
    '''
    
    # Get the image name suffixes.
    suffix_cam = metadata[0]['types'][0]
    suffix_rig_list = metadata['rig']['types']
    
    # Parse the cam_paths.csv file.
    cam_paths_csv = os.path.join( traj_dir, 'cam_paths.csv' )
    raw_image_dict = read_cam_paths(cam_paths_csv, suffix_cam=suffix_cam, suffix_rig_list=suffix_rig_list)
    
    # Prepare the filter for the number suffix of img_dir saved in the camera config file.
    cam_suffix_search = re.compile(r'cam(\d+)$')
    
    # Loop for all the cameras.
    last_n_images_written = None
    for camera in cameras:
        img_dir = camera['img_dir']
        
        # Get the number suffix.
        # Omnimvs counts cameras from 1.
        number_suffix = int(cam_suffix_search.search(img_dir)[1]) - 1
        
        # Compose the key for the raw image dictionary.
        key = 'cam%d' % number_suffix
        
        # Get the raw image list.
        raw_image_fn_list = raw_image_dict[key]
        
        # Get the camera orientation.
        ori = metadata[ number_suffix ]['data']['ori']
        
        # Create the camera model.
        camera_model = create_camera_model(camera)
        
        # Compose the quaternion and the rotation matrix.
        R_rbf_cbf = Quaternion( w=ori[3], x=ori[0], y=ori[1], z=ori[2] ).rotation_matrix
        
        # Since when we collect the raw images, the orientation of the camera is the same
        # as the orientation of the rig. We can write.
        R_cbf0_cbf1 = R_rbf_cbf
        # Where, the rotation describes how the camera body has rotated with respect to 
        # its original orientation. However, we need R_cpf_cif. We can start from
        # R_cpf_cbf0.
        R_cpf_cbf0 = np.array(
            [ [  0, -1,  0 ],
              [  0,  0,  1 ],
              [ -1,  0,  0 ] ], dtype=np.float32)
        # Then we need R_cbf1_cif.
        R_cbf1_cif = np.array(
            [ [ 0, 0, 1 ],
              [ 1, 0, 0 ], 
              [ 0, 1, 0 ] ], dtype=np.float32)
        # Finally, we can get R_cpf_cif.
        R_cpf_cif = R_cpf_cbf0 @ R_cbf0_cbf1 @ R_cbf1_cif
        
        # Create the fisheye image sampler.
        sampler = FullViewRotation( camera_model, R_cpf_cif )
        
        # Compose the output directory.
        cam_out_dir = os.path.join(out_dir, img_dir)
        
        # Sample fisheye images.
        # Note that 
        n_images_written = sample_fisheye_images(start_index + 1, cam_out_dir, traj_dir, raw_image_fn_list, sampler)
        
        if last_n_images_written is not None:
            assert n_images_written == last_n_images_written, \
                f'n_images_written = {n_images_written}, last_n_images_written = {last_n_images_written}, img_dir = {img_dir}, traj_dir = {traj_dir}. '
           
    raw_rig_rgb_fn_list = raw_image_dict['rig_rgb']
    assert len(raw_rig_rgb_fn_list) == n_images_written, \
        f'n_images_written = {n_images_written}, len(raw_rig_rgb_fn_list) = {len(raw_rig_rgb_fn_list)}, traj_dir = {traj_dir}.'
         
    # Handle the rig.
    for i, fns in enumerate(zip(raw_rig_rgb_fn_list, raw_image_dict['rig_dist'])) :
        # RGB.
        target = os.path.join( out_dir, 'rig', '%04d.png' % (i) )
        true_rgb_fn = os.path.join( traj_dir, fns[0] )
        shutil.copy( true_rgb_fn, target )
        print(f'Copy {true_rgb_fn} to \n{target}')
        
        # Distance.
        target = os.path.join( out_dir, 'rig', '%04d_dist.png' % (i) )
        true_dist_fn = os.path.join( traj_dir, fns[1] )
        shutil.copy( true_dist_fn, target )
        print(f'Copy {true_dist_fn} to \n{target}')
    
    return n_images_written # The total number of new generated samples.
    
def main(args):
    # Prepare the output directory.
    if not os.path.isdir(args.out_dir):
        os.makedirs(args.out_dir)
    
    # Read the camera configurations from the config file.
    cameras = read_camera_config(args.camera_config)
    n_cameras_config = len(cameras)
    print(f'{n_cameras_config} cameras read from {args.camera_config}. ')
    
    # Create the output directories for the images.
    create_image_out_dirs( args.out_dir, cameras )
    
    # Copyt the camera config file to the output directory.
    copy_camera_config_to_out_dir( args.out_dir, args.camera_config )
    
    # Search for all the trajectory directories.
    traj_dirs = search_trajectory_dirs(args.data_dir)
    n_traj_dirs = len(traj_dirs)
    print(f'Find {n_traj_dirs} trajectory directories. ')
    
    # The total number of samples.
    n_total_samples = 0
    
    for t in traj_dirs:
        print(t)
    
        # Read the data collection metadata.
        metadata_fn = os.path.join( args.data_dir, args.data_collection_metadata )
        frame_graph_fn = os.path.join( args.data_dir, args.data_collection_frame_graph )
        metadata, _ = read_metadata(metadata_fn, frame_graph_fn, t)
        # print(metadata) # Already printed by the reader.
        
        # Check the number of cameras from different sources.
        n_cameras_metadata = len(metadata)
        assert n_cameras_config + 1 == n_cameras_metadata, \
            f'n_cameras_config={n_cameras_config}, n_cameras_metadata={n_cameras_metadata}. '
            
        # Process this trajectory.
        new_samples = process_trajectory( n_total_samples, t, args.out_dir, metadata, cameras )
        n_total_samples += new_samples
    
    return 0

if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser(description='Generate fisheye images for the omnimvs model.')
    
    parser.add_argument('--data-dir', type=str, 
                        help='The root directory of the collected data. ')
    # parser.add_argument('--trajectory-dir', type=str, default='Trajectories',
    #                     help='The directory saves the collected data under data_dir. ')
    
    # Thsi might need to be chaanged due to the fact that the metadata will be saved in the
    # individual trajectory directories. 
    parser.add_argument('--data-collection-metadata', type=str, default='metadata.json',
                        help='The metadata used for data collection. ')
    parser.add_argument('--data-collection-frame-graph', type=str, default='frame_graph.json',
                        help='The frame graph JSON file used for data collection. ')
    parser.add_argument('--camera-config', type=str, 
                        help='The camera config file. ')
    parser.add_argument('--out-dir', type=str, 
                        help='The output directory. ')
    
    args = parser.parse_args()
    
    sys.exit(main(args))