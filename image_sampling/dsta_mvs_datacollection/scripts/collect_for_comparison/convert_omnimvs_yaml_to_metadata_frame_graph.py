
import os
import sys

# The path of the current Python script.
_CURRENT_PATH       = os.path.dirname(os.path.realpath(__file__))
_TOP_PATH           = os.path.join(_CURRENT_PATH, '..', '..', 'data_collection')
_DATA_PIPELINE_PATH = os.path.join(_TOP_PATH, 'ord_data_pipeline_rework', 'src')
sys.path.insert( 0, _DATA_PIPELINE_PATH )
sys.path.insert( 0, _TOP_PATH )
for p in sys.path:
    print(p)

from typing import List

import copy
import numpy as np
from scipy.spatial.transform import Rotation as R
import yaml

# # Configure matplotlib for Windows WSL.
# import matplotlib
# matplotlib.use('agg')
import matplotlib.pyplot as plt

from pytransform3d import rotations as pr
from pytransform3d import transformations as pt
from pytransform3d.transform_manager import TransformManager

# Local package.
from mvs_utils.pretty_dict import (
    PrettyDict, PlainPrinter, DictPrinter, ListPrinter, NumPyPrinter, NumPyLineBreakPrinter )

# This function is from omnimvs.
def rodrigues(r: np.ndarray) -> np.ndarray:
    if r.size == 3: 
        return R.from_rotvec(r.squeeze()).as_matrix()
    else: 
        return R.from_matrix(r).as_rotvec().reshape((3, 1))

def read_camera_config(fn):
    with open(fn, 'r') as f:
        config = yaml.safe_load(f)
    
    camera_list = []
    for cc in config['cameras']:
        pose = np.array(cc['pose']).astype(np.float32)
        
        # Convert pose to rotation matrix and translation vector.
        rot_mat = rodrigues(pose[:3])
        trans_vec = pose[3:6]
        
        # The quaternion.
        q = R.from_rotvec(pose[:3]).as_quat()
        
        # # AirSim needs w to be the first element.
        # q_a = np.zeros((4,), dtype=np.float32)
        # q_a[0]  = q[-1]
        # q_a[1:] = q[:-1]
        q_a = q
        
        # Save.
        camera_list.append({
            'cam_id': cc['cam_id'],
            'R': rot_mat,
            't': trans_vec, 
            'q': q_a,
        })
        
    return camera_list

def visualize_frames(camera_list: List[dict]):
    transforms = {}
    for camera in camera_list:
        transforms[camera['cam_id']] = \
            pt.transform_from(R=camera['R'], p=camera['t'])
            
    tm = TransformManager()
    for key, value in transforms.items():
        tm.add_transform(key, 'rig', value)
        
    ax = tm.plot_frames_in('rig', s=0.1)
    ax.set_xlim(-0.6, 0.6)
    ax.set_ylim(-0.6, 0.6)
    ax.set_zlim(-0.6, 0.6)
    plt.show()

def rot_mat_to_airsim_quat(m):
    # The quaternion.
    q = R.from_matrix(m).as_quat()
    
    # # AirSim needs w to be the first element.
    # q_a = np.zeros((4,), dtype=np.float32)
    # q_a[0]  = q[-1]
    # q_a[1:] = q[:-1]
    q_a = q
    
    return q_a

def convert_to_ned(camera_list: List[dict]):
    '''
    The frame of the OmniVMS model typical Camera Image Frame, 
    where z-forward, x-right, y-downward.
    
    We need to convert the poses to a NED frame, the Rig Body Frame.
    '''
    
    # TODO: Use FTensor.
    
    T_cif_cbf = np.zeros((4, 4), dtype=np.float32)
    T_cif_cbf[2, 0] = 1
    T_cif_cbf[0, 1] = 1
    T_cif_cbf[1, 2] = 1
    T_cif_cbf[3, 3] = 1
    
    T_rbf_rif = np.zeros((4, 4), dtype=np.float32)
    T_rbf_rif[1, 0] = 1
    T_rbf_rif[2, 1] = 1
    T_rbf_rif[0, 2] = 1
    T_rbf_rif[3, 3] = 1
    
    camera_list = copy.deepcopy(camera_list)
    
    for camera in camera_list:
        # Construct T_rif_cif
        T_rif_cif = np.eye(4, dtype=np.float32)
        T_rif_cif[:3, :3] = camera['R']
        T_rif_cif[:3,  3] = camera['t']
        
        # New transformation matrix.
        T_rbf_cbf = T_rbf_rif @ T_rif_cif @ T_cif_cbf
        
        # Update.
        camera['R'] = T_rbf_cbf[:3, :3]
        camera['t'] = T_rbf_cbf[:3,  3]
        camera['q'] = rot_mat_to_airsim_quat(camera['R'])
        
    return camera_list

def write_metadata(fn: str, camera_list: List[dict]):
    # Construct the PrettyDict.
    pd = PrettyDict()
    # pd['__README__'] = 'Ensure that the positions of each camera are written w.r.t respect to the rig being the origin point'
    # pd['__README2__'] = 'and the frame being an NED Frame.'
    
    # The Printer for the camera values.
    lp = ListPrinter(linear=False)
    li = ListPrinter()
    li.append(PlainPrinter())
    
    dp = DictPrinter()
    # dp['pos'] = NumPyPrinter()
    # dp['ori'] = NumPyPrinter()
    # dp['dsc'] = PlainPrinter()
    dp['img_types'] = li
    dp['airsim_cam_nums'] = PlainPrinter()
    dp['frame'] = PlainPrinter()
    dp['image_frame'] = PlainPrinter()
    
    cams = []
    # Compose the camera PrettyDict objects.
    for i, camera in enumerate( camera_list ):
        pdc = PrettyDict()
        # pdc['pos'] = camera['t']
        # pdc['ori'] = camera['q']
        # pdc['dsc'] = [0, 0, 0, 0, 0, 0] # dummy values.
        pdc['img_types'] = ['CubeScene']
        pdc['airsim_cam_nums'] = [0]
        pdc['frame'] = f'cbf{i}'
        pdc['image_frame'] = f'cif{i}'
        cams.append(pdc)
        lp.append(dp)
    
    # Add cams to pd.    
    pd['cams'] = cams
    
    # rig.
    pd['rig_img_types'] = ['CubeScene', 'CubeDistance']
    
    # Misc.
    pd["randomize_orientation"] = False
    
    # Update the Printer objects.
    pd.auto_update_printer()
    pd.update_printer('cams', lp)
    
    # Visualize.
    s = pd.make_str()
    print(s)

    # Write.
    with open(fn, 'w') as fp:
        fp.write( s )

def construct_frame_graph_dict(camera_list: List[dict]):
    frames = [
        dict( name='awf', comment='AirSim World NED Frame. ' ),
        dict( name='rbf', comment='Rig Body Frame. ' ),
        dict( name='rpf', comment='Rig Ponorama Frame. ' )
    ]
    
    typical_poses = dict(
        T_body_panorama=dict(
            position=[0.0, 0.0, 0.0],
            orientation=dict(
                type='rotation_matrix',
                data=np.array([
                     0.0,  0.0, -1.0,
					-1.0,  0.0,  0.0,
					 0.0,  1.0,  0.0 ]) ) ),
        T_body_image=dict(
            position=[0.0, 0.0, 0.0],
            orientation=dict(
                type='rotation_matrix',
                data=np.array([
                     0.0,  0.0,  1.0,
					 1.0,  0.0,  0.0,
					 0.0,  1.0,  0.0 ]) ) )
    ) # typical_poses
    
    transforms = [
        dict(
            f0='awf', f1='rbf',
            pose=dict(
                type='create',
                position=[0.0, 0.0, 0.0],
                orientation=dict(
                    type='quaternion',
                    data=dict( x=0.0, y=0.0, z=0.0, w=1.0 )
                ) ) ),
        dict(
            f0='rbf', f1='rpf',
            pose=dict(
                type='reference',
                key='T_body_panorama') ),
    ] # transforms
    
    for i, camera in enumerate(camera_list):
        name_cbf = f'cbf{i}'
        comment_cbf = f'Camera Body Frame {i}. '
        name_cpf = f'cpf{i}'
        comment_cpf = f'Camera Panorama Frame {i}. '
        name_cif = f'cif{i}'
        comment_cif = f'Camera Image Frame {i}. '
        
        name_list = [name_cbf, name_cpf, name_cif]
        comment_list = [comment_cbf, comment_cpf, comment_cif]
        
        # Update the `frames` variable.
        for name, comment in zip(name_list, comment_list):
            frames.append( dict( name=name, comment=comment ) )
        
        # Update the `transforms` variable.
        q = camera['q'] # Last element is w.
        transforms.append( dict(
            f0='rbf', f1=name_cbf,
            pose=dict(
                type='create',
                position=camera['t'],
                orientation=dict(
                    type='quaternion',
                    data=dict( x=q[0], y=q[1], z=q[2], w=q[3] ) ) ) ) )
        transforms.append( dict(
            f0=name_cbf, f1=name_cpf,
            pose=dict(
                type='reference',
                key='T_body_panorama' ) ) )
        transforms.append( dict(
            f0=name_cbf, f1=name_cif,
            pose=dict(
                type='reference',
                key='T_body_image' ) ) )
        
    return dict( frames=frames, typical_poses=typical_poses, transforms=transforms)

def write_frame_graph(fn: str, camera_list: List[dict]):
    plain_dict = construct_frame_graph_dict(camera_list)
        
    # Construct the PrettyDict.
    pd = PrettyDict()
    pd['frames']        = plain_dict['frames']
    pd['typical_poses'] = plain_dict['typical_poses']
    pd['transforms']    = plain_dict['transforms']
    
    # Update the Printer objects.
    pd.auto_update_printer()
    
    # Change the printer for the orientations represented as rotation_matrices.
    for key, pose in pd['typical_poses'].items():
        if pose['orientation']['type'] == 'rotation_matrix':
            pd.p['typical_poses'][key]['orientation']['data'] = NumPyLineBreakPrinter(shape=(3, 3))
            
    for i, transform in enumerate(pd['transforms']):
        if transform['pose']['type'] == 'create':
            if transform['pose']['orientation']['type'] == 'rotation_matrix':
                pd.p['transforms'][i]['pose']['orientation']['data'] = NumPyLineBreakPrinter(shape=(3, 3))
    
    # Visualize.
    s = pd.make_str()
    print(s)

    # Write.
    with open(fn, 'w') as fp:
        fp.write( s )

if __name__ == '__main__':
    # camera_config_fn = './data/omnimvs/itbt_sample.yaml'
    # out_dir = './output/omnimvs'
    # out_name = 'metadata_itbt_sample.json'
    
    camera_config_fn = './data/omnimvs/sunny_single.yaml'
    out_dir = './output/omnimvs'
    out_metadata_name = 'metadata.json'
    out_frame_graph_name = 'frame_graph.json'
    
    if not os.path.isdir(out_dir):
        os.makedirs(out_dir)

    # Read the cameras.
    cameras = read_camera_config(camera_config_fn)
    for camera in cameras:
        print(camera)
    
    # Convert to NED.
    cameras_ned = convert_to_ned(cameras)
    
    # Write the metadata.
    out_fn = os.path.join(out_dir, out_metadata_name)
    write_metadata(out_fn, cameras_ned)
    
    # Write the frame graph.
    out_fn = os.path.join(out_dir, out_frame_graph_name)
    write_frame_graph(out_fn, cameras_ned)
    
    # Visualize the frames.
    visualize_frames(cameras)
    visualize_frames(cameras_ned)