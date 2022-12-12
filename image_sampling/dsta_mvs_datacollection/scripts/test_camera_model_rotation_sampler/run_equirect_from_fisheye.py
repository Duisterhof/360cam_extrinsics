import os
import sys

# The path of the current Python script.
_CURRENT_PATH       = os.path.dirname(os.path.realpath(__file__))
_TOP_PATH           = os.path.join(_CURRENT_PATH, '..', '..')
sys.path.insert( 0, _TOP_PATH )
for i, p in enumerate(sys.path):
    print(f'{i}: {p}')

import cv2
import numpy as np

from data_collection.mvs_utils.camera_models import (DoubleSphere, Equirectangular)
from data_collection.mvs_utils.shape_struct import ShapeStruct
from data_collection.mvs_utils.ftensor import (FTensor, f_zeros)
from data_collection.image_sampler import IDENTITY_ROT
from data_collection.image_sampler.camera_model_sampler import CameraModelRotation

def read_image(fn):
    img = cv2.imread(fn, cv2.IMREAD_UNCHANGED)
    assert img is not None, f'Read {fn} failed. '
    return img

if __name__ == '__main__':
    out_dir = './output'

    in_fisheye_fn = './data/000000_Fisheye.png'

    if not os.path.isdir(out_dir):
        os.makedirs(out_dir)

    # Create a fisheye camera model as the raw camera model.
    fov = 195
    ss = ShapeStruct( H=1028, W=1224 )
    camera_model_raw = DoubleSphere(
        -0.196, 0.589, 
        232, 232,
        611.5, 513.5, fov, ss, 
        in_to_tensor=True,
        out_to_numpy=True)

    # Create an equirectangular camera model as the target camera model.
    nss = ShapeStruct( H=512, W=2048 )
    camera_equi_target = Equirectangular(
        nss, 
        latitude_span=( -np.pi/2, 0 ),
        open_span=True,
        in_to_tensor=True, 
        out_to_numpy=False)
    # nss = ShapeStruct( H=368, W=2048 )
    # camera_equi_target = Equirectangular(
    #     nss, 
    #     latitude_span=( -np.pi/3, 0 ),
    #     open_span=True,
    #     in_to_tensor=True, 
    #     out_to_numpy=False)

    # Rotation that aligns negative y-axis of the equirectangular (target) camera 
    # to the positive z-axis of the fisheye camera (raw).
    R_raw_target = f_zeros((3, 3), f0=IDENTITY_ROT.f0, f1=IDENTITY_ROT.f1)
    R_raw_target.is_rotation = True
    R_raw_target[0, 0] =  1
    R_raw_target[2, 1] = -1
    R_raw_target[1, 2] =  1

    # Create a sampler.
    sampler = CameraModelRotation(camera_model_raw, camera_equi_target, R_raw_fisheye=R_raw_target)

    # Read the fisheye image.
    fisheye_img = read_image(in_fisheye_fn)

    sampled, invalid_mask = sampler(fisheye_img)

    # Save.
    cv2.imwrite( os.path.join(out_dir, 'resampled_equirect.png'), sampled )