
import numpy as np

# The following inputs need the main script to properly configure the Python search path.
from data_collection.mvs_utils.ftensor import ( FTensor, f_zeros )
from data_collection.image_sampler.six_images_common import (
    FRONT, BACK, LEFT, RIGHT, TOP, BOTTOM )

def get_all_orientations(only_front=False):
    global FRONT, RIGHT, BOTTOM, LEFT, TOP, BACK
    
    R_raw_fisheye_list = []
    case_names_fisheye = []
    
    # Front.
    # R_raw_fisheye = np.zeros((3, 3), dtype=np.float32)
    R_raw_fisheye = f_zeros((3, 3), f0='raw', f1='fisheye')
    R_raw_fisheye[0, 0] = 1
    R_raw_fisheye[1, 1] = 1
    R_raw_fisheye[2, 2] = 1
    R_raw_fisheye.is_rotation = True
    R_raw_fisheye_list.append(R_raw_fisheye)
    case_names_fisheye.append(FRONT)
    
    if not only_front:
        # Right.
        R_raw_fisheye = f_zeros((3, 3), f0='raw', f1='fisheye')
        R_raw_fisheye[2, 0] = -1 
        R_raw_fisheye[1, 1] =  1 
        R_raw_fisheye[0, 2] =  1
        R_raw_fisheye.is_rotation = True
        R_raw_fisheye_list.append(R_raw_fisheye)
        case_names_fisheye.append(RIGHT)
        
        # Bottom.
        R_raw_fisheye = f_zeros((3, 3), f0='raw', f1='fisheye')
        R_raw_fisheye[0, 0] =  1 
        R_raw_fisheye[2, 1] = -1 
        R_raw_fisheye[1, 2] =  1
        R_raw_fisheye.is_rotation = True
        R_raw_fisheye_list.append(R_raw_fisheye)
        case_names_fisheye.append(BOTTOM)
        
        # Left.
        R_raw_fisheye = f_zeros((3, 3), f0='raw', f1='fisheye')
        R_raw_fisheye[2, 0] =  1 
        R_raw_fisheye[1, 1] =  1 
        R_raw_fisheye[0, 2] = -1
        R_raw_fisheye.is_rotation = True
        R_raw_fisheye_list.append(R_raw_fisheye)
        case_names_fisheye.append(LEFT)
        
        # Top.
        R_raw_fisheye = f_zeros((3, 3), f0='raw', f1='fisheye')
        R_raw_fisheye[0, 0] =  1 
        R_raw_fisheye[2, 1] =  1 
        R_raw_fisheye[1, 2] = -1
        R_raw_fisheye.is_rotation = True
        R_raw_fisheye_list.append(R_raw_fisheye)
        case_names_fisheye.append(TOP)
        
        # Back.
        R_raw_fisheye = f_zeros((3, 3), f0='raw', f1='fisheye')
        R_raw_fisheye[0, 0] = -1 
        R_raw_fisheye[1, 1] =  1 
        R_raw_fisheye[2, 2] = -1
        R_raw_fisheye.is_rotation = True
        R_raw_fisheye_list.append(R_raw_fisheye)
        case_names_fisheye.append(BACK)

    return R_raw_fisheye_list, case_names_fisheye
