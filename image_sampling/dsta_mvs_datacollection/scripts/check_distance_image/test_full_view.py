# Author: Yaoyu Hu <yaoyuh@andrew.cmu.edu>
# Date: 2022-09-09

import os
import sys

# The path of the current Python script.
_CURRENT_PATH       = os.path.dirname(os.path.realpath(__file__))
_TOP_PATH           = os.path.join(_CURRENT_PATH, '..', '..')
_DATA_PIPELINE_PATH = os.path.join(_TOP_PATH, 'data_collection', 'ord_data_pipeline_rework', 'src')
sys.path.insert( 0, _DATA_PIPELINE_PATH )
sys.path.insert( 0, _TOP_PATH )
for i, p in enumerate(sys.path):
    print(f'{i}: {p}')

import colorcet as cc
import cv2
import numpy as np
from plyfile import PlyData, PlyElement
import time

def hex_to_RGB(hex):
    ''' "#FFFFFF" -> [255,255,255] '''
    # Pass 16 to the integer function for change of base
    return [int(hex[i:i+2], 16) for i in range(1,6,2)]

def convert_colorcet_2_array(ccArray):
    cmap = np.zeros((256, 3), dtype=np.uint8)

    for i in range(len(ccArray)):
        rgb = hex_to_RGB( ccArray[i] )
        cmap[i, :] = rgb

    return cmap

def read_image(fn):
    assert( os.path.isfile(fn) ), \
        f'{fn} does not exist. '

    return cv2.imread(fn, cv2.IMREAD_UNCHANGED)

def read_compressed_float(fn):
    assert( os.path.isfile(fn) ), \
        f'{fn} does not exist. '
    
    return np.squeeze( 
        cv2.imread(fn, cv2.IMREAD_UNCHANGED).view('<f4'),
        axis=-1)

def scale_float_image(img, limits=None, valid_mask=None):
    if ( limits is not None ):
        min_v, max_v = limits
    else:
        min_v = img[valid_mask, ...].min() \
            if valid_mask is not None \
            else img.min()

        max_v = img[valid_mask, ...].max() \
            if valid_mask is not None \
            else img.max()
    
    return ( img - min_v ) / ( max_v - min_v )

def clip_float_and_convert_uint(img):
    return (np.clip(img, 0, 1) * 255).astype(np.uint8)

def meshgrid_img(img, dtype=np.int32):
    H, W = img.shape[:2]
    x = np.arange(W, dtype=dtype)
    y = np.arange(H, dtype=dtype)

    return np.meshgrid(x, y)

def panorama_lon_lat(img):
    # The meshgrid.
    xx, yy = meshgrid_img(img, np.float32)

    # The angles.
    H, W = img.shape[:2]
    # NOTE: xx and yy do not contain 2*np.pi!
    return xx / W * 2 * np.pi, yy / H * np.pi

def write_ply_dist_lon_lat(fn, dist, lon, lat, mask, max_v):
    y = -dist * np.cos(lat)
    s =  dist * np.sin(lat)
    x =  s * np.cos( lon - np.pi )
    z = -s * np.sin( lon - np.pi )

    # mask = dist <= max_v
    # mask = mask.reshape((-1,))

    # vertex = np.stack((x, y, z), axis=-1).reshape((-1, 3))
    # vertex = vertex[mask]
    # vertex = vertex.view([('x', 'f4'), ('y', 'f4'), ('z', 'f4')]).reshape((-1,))

    # Colormap.
    # cmap = np.expand_dims(convert_colorcet_2_array(cc.rainbow), axis=0)
    cmap = convert_colorcet_2_array(cc.rainbow).reshape((-1, len(cc.rainbow), 3))

    # New mask.
    mask = np.logical_and( mask, dist <= max_v )

    # Assign color.
    min_v = dist[mask].min()
    d = (dist - min_v) / ( max_v - min_v )
    cx = d * (cmap.shape[1]-1)
    cy = np.zeros_like(cx)

    color = cv2.remap(cmap, cx, cy, interpolation=cv2.INTER_LINEAR)

    x = x[mask]
    y = y[mask]
    z = z[mask]
    color = color[mask, ...]

    vertex = [ ( v[0], v[1], v[2], v[3][0], v[3][1], v[3][2] ) 
                for v in zip( x, y, z, color ) ]

    vertex = np.array( vertex, dtype=[
        ('x', 'f4'), ('y', 'f4'), ('z', 'f4'),
        ('red', 'u1'), ('green', 'u1'), ('blue', 'u1')] )

    # PLY element.
    ele = PlyElement.describe(vertex, 'vertex')

    # Write.
    PlyData([ele], text=False).write(fn)

def write_panorama_dist_as_ply(fn, dist, max_v=50):
    # # Allocate memory.
    # points = np.zeros((dist.size,), 
    #     dtype=[('x', 'f4'), ('y', 'f4'), ('z', 'f4')])

    # Get the longitude and latitude values.
    lon, lat = panorama_lon_lat(dist)

    # Mask.
    mask = dist <= max_v

    # Write as dist-lon-lat data.
    write_ply_dist_lon_lat(fn, dist, lon, lat, mask, max_v)

def _test_dist():
    print('========== Distance image. ==========')

    out_dir = './output'

    if not os.path.isdir(out_dir):
        os.makedirs(out_dir)

    # Load the distance image.
    # imgFn = '/home/yaoyuh/Playground/20220316_MVS/ICRA2022/SampleData_OnlyOne/training/rig/000000_CubeDistance.png'
    imgFn = '/home/yaoyuh/Playground/20220316_MVS/ICRA2022/SampleData_GridSample/training/rig/000000_CubeDistance.png'
    img = read_compressed_float(imgFn)

    # Save a visualization.
    img_scaled = scale_float_image(img, limits=[0, 50])
    cv2.imwrite( os.path.join(out_dir,  'Vis_Dist.png'), clip_float_and_convert_uint(img_scaled) )

    # Save a point cloud.
    write_panorama_dist_as_ply(os.path.join(out_dir, 'Dist.ply'), img)

if __name__ == '__main__':
    print('Hello, %s! ' % ( os.path.basename(__file__) ))

    # _test_rgb()
    _test_dist()
