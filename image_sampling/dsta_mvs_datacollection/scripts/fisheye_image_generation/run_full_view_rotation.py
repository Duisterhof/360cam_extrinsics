import os
import sys

# Configure the Python search path.
_TOP_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
sys.path.insert(0, _TOP_DIR)
for i, p in enumerate(sys.path):
    print(f'{i}: {p}')

import argparse
import cv2
import numpy as np

import torch

# Configure matplotlib for Windows WSL.
import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt

# Local.
from data_collection.mvs_utils.shape_struct import ShapeStruct
from data_collection.mvs_utils.camera_models import DoubleSphere
from data_collection.image_sampler.full_view_rotation import FullViewRotation

from utils import get_all_orientations

def read_image(fn):
    img = cv2.imread(fn, cv2.IMREAD_UNCHANGED)
    assert img is not None, f'Read {fn} failed. '
    return img

def circle_mask(shape, c, r):
    '''
    shape (2-element): H, W.
    c (2-element): center coordinate, (x, y)
    r (float): the radius.
    '''

    # Get a meshgrid of pixel coordinates.
    H, W = shape[:2]
    x = np.arange(W, dtype=np.float32)
    y = np.arange(H, dtype=np.float32)
    xx, yy = np.meshgrid(x, y, indexing='xy')

    # Get the distance to the center.
    d = np.sqrt( ( xx - c[0] )**2 + ( yy - c[1] )**2 )

    return d <= r

def handle_args():
    parser = argparse.ArgumentParser(description='Test generating a fisheye image from a Unreal Panorama.')
    
    parser.add_argument('--backend', type=str, choices=['torch', 'opencv'], 
                        help='The backend. ')
    
    return parser.parse_args()

def main():
    def write_heat_map(fn, mat, c, r, title='No title'):
        fig, ax = plt.subplots(1, 1, figsize=(10, 10))
        im = ax.imshow( mat, cmap=plt.get_cmap('jet'), vmin=0.5, vmax=10 )

        # Draw the circle.
        draw_circle = plt.Circle((c[0], c[1]), r, fill=False, color=(1,1,1))
        ax.add_artist(draw_circle)

        fig.colorbar(im)
        ax.set_title(title)
        fig.savefig(fn)

    args = handle_args()
    out_dir_suffix = {
        'torch': 'torch',
        'opencv': 'opencv'
    }
    
    in_panorama_fn = 'data/panorama/000000_CubeScene.png'

    out_dir_base = 'output/full_view'
    out_dir = f'{out_dir_base}_{out_dir_suffix[args.backend]}'

    in_panorama = read_image(in_panorama_fn)

    if not os.path.isdir(out_dir):
        os.makedirs(out_dir)

    fov = 195
    ss = ShapeStruct( H=1028, W=1224 )
    camera_model = DoubleSphere(
        -0.196, 0.589, 
        235, 235,
        611.5, 513.5, fov, ss, 
        in_to_tensor=True,
        out_to_numpy=False) # Was True before using the torch version.

    # If the panorama frame is the original frame where z-axis if pointing backwards,
    # Then all the orientations of the fisheye image are in the opposite direction.
    # This means that the following code will generate the fisheye images in the opposite
    # directions.
    R_raw_fisheye_list, case_names_fisheye = get_all_orientations()

    flag_mask_output_done = False
    for case_name, R_raw_fisheye in zip( case_names_fisheye, R_raw_fisheye_list ):
        print(case_name)
        converter = FullViewRotation(camera_model, R_raw_fisheye)
        if args.backend =='opencv':
            converter.use_ocv = True

        sampled, valid_mask = converter(in_panorama)
        out_fn = os.path.join(out_dir, 'result_full_fisheye_%s.png' % (case_name))
        cv2.imwrite(out_fn, sampled)
        
        if not flag_mask_output_done:
            valid_mask = valid_mask.astype(np.uint8) * 255
            out_fn = os.path.join(out_dir, 'valid_mask.png')
            cv2.imwrite(out_fn, valid_mask)
            print(f'Write mask to {out_fn}. ')
            flag_mask_output_done = True
    
    # Mean sampling difference.
    print('Mean sampling difference. ')
    converter = FullViewRotation(camera_model, R_raw_fisheye_list[0])
    if args.backend =='opencv':
            converter.use_ocv = True
    
    support_shape = in_panorama.shape[:2]
    mean_sampling_diff, valid_mask = converter.compute_mean_samping_diff(support_shape)

    # Get a circle mask.
    circle_mask_r = 350
    circle_mask_c = ( camera_model.cx, camera_model.cy)
    mask = circle_mask( ss.shape, circle_mask_c, circle_mask_r)

    single_mean = mean_sampling_diff[mask].mean()
    print(f'Masked mean sampling diff = {single_mean}')
    out_fn = os.path.join(out_dir, 'mean_diff.png')
    title = f'fisheye {ss.shape[:2]} from panorama support at {support_shape[:2]}, masked mean = {single_mean:.2f}'
    print(title)
    write_heat_map(out_fn, mean_sampling_diff, circle_mask_c, circle_mask_r, title)

if __name__ == '__main__':
    with torch.no_grad():
        main()