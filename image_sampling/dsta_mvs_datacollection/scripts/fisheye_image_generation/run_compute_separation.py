
import os
import sys

# Configure the Python search path.
_TOP_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
sys.path.insert(0, _TOP_DIR)
for i, p in enumerate(sys.path):
    print(f'{i}: {p}')

import numpy as np

import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt

from data_collection.mvs_utils.shape_struct import ShapeStruct
from data_collection.mvs_utils.camera_models import DoubleSphere
from data_collection.image_sampler.six_images import SixPlanarAsBase

def write_heat_map(fn, mat, title='No title'):
    fig, ax = plt.subplots(1, 1, figsize=(10, 10))
    im = ax.imshow( mat, cmap=plt.get_cmap('jet'), vmin=0.5, vmax=10 )
    fig.colorbar(im)
    ax.set_title(title)
    fig.savefig(fn)

if __name__ == '__main__':
    out_dir = 'output/20220618_05_boxes_fov90'
    
    w_span = [ 500, 1200 ]
    
    fov = 195
    # shape = np.array([686,686])
    ss = ShapeStruct(H=1028, W=1224)
    camera_model = DoubleSphere(
        -0.196, 0.589, 
        235, 235,
        612, 514, fov, ss,
        in_to_tensor=True, 
        out_to_numpy=True)

    converter = SixPlanarAsBase(fov, camera_model, np.eye(3, dtype=np.float32))
    converter.enable_cuda()
    
    separations = []
    
    for w in range(*w_span, 20):
        support_shape = [ w, w ]
        mean_sampling_diff, valid_mask = converter.compute_mean_samping_diff(support_shape)
        single_mean = mean_sampling_diff[valid_mask].mean()
        print(f'w = {w}, global mean sampling diff = {single_mean}')
        separations.append(single_mean)
        
    fig, ax = plt.subplots(1, 1, figsize=(10, 10))
    ax.plot(range(*w_span, 20), separations)
    out_fn = os.path.join(out_dir, 'separations.png')
    ax.set_title(f'Mean separation of fisheye {ss.shape[:2]} from 6 supports of width from {w_span[0]} to {w_span[1]}')
    fig.savefig(out_fn)
    
    # Special one.
    w = 640
    mean_sampling_diff, valid_mask = converter.compute_mean_samping_diff([w, w])
    single_mean = mean_sampling_diff[valid_mask].mean()
    print(f'Global mean sampling diff = {single_mean}')
    out_fn = os.path.join(out_dir, f'mean_diff_{w}.png')
    title = f'fisheye {ss.shape[:2]} from 6 support at {support_shape[:2]}, global mean = {single_mean:.2f}'
    print(title)
    write_heat_map(out_fn, mean_sampling_diff, title)
        