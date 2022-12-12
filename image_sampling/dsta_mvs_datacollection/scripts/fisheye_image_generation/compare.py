
import argparse
import cv2
import numpy as np
import os
import shutil

def get_filename_parts(fn):
    s0 = os.path.split(fn)
    s1 = os.path.splitext(s0[1])
    if s0[0] == '':
        s0[0] = '.'
    return s0[0], *s1

def read_image(fn):
    img = cv2.imread(fn, cv2.IMREAD_UNCHANGED)
    assert img is not None, f'Failed to read {fn}. '
    
    if img.ndim == 2:
        img = np.expand_dims(img, axis=2)
    
    return img

def compare_and_write(out_fn_base, vis_scale, imgs, valid_mask=None):
    # Compute the difference.
    diff = imgs[0].astype(np.float32) - imgs[1].astype(np.float32)
    diff_norm = np.linalg.norm(diff, axis=2)
    
    # Print info.
    diff_norm_masked = diff_norm[valid_mask] \
        if valid_mask is not None \
        else diff_norm
    
    print(f'diff_norm_masked.max()  = {diff_norm_masked.max()}')
    print(f'diff_norm_masked.min()  = {diff_norm_masked.min()}')
    print(f'diff_norm_masked.mean() = {diff_norm_masked.mean()}')
    
    # Save a visualization.
    diff_vis = np.clip(diff_norm, 0, 255).astype(np.uint8)
    cv2.imwrite( out_fn_base, diff_vis )
    
    # Figure out the filename parts.
    parts = get_filename_parts(out_fn_base)
    out_fn_vis_scaled = os.path.join( parts[0], f'{parts[1]}_a_scale_by_{vis_scale}{parts[2]}' )
    
    # Save a scaled visualization.
    diff_vis_scaled = np.clip(diff_norm * vis_scale, 0, 255).astype(np.uint8)
    cv2.imwrite(out_fn_vis_scaled, diff_vis_scaled)

def copy_file(ori_fn, out_dir, suffix):
    parts = get_filename_parts(ori_fn)
    target_fn = os.path.join( out_dir, f'{parts[1]}_{suffix}{parts[2]}' )
    shutil.copy( ori_fn, target_fn )
    print(f'{ori_fn} copied to \n{target_fn}')

def handle_args():
    parser = argparse.ArgumentParser(description='Compare the output from the opencv version and the torch version. ')
    parser.add_argument('--dir-opencv', type=str, default='output/20220620_04_w896_opencv',
                        help='The directory of the output of the opencv version. ')
    parser.add_argument('--dir-torch', type=str, default='output/20220620_04_w896_torch',
                        help='The directory of the output of the torch version. ')
    return parser.parse_args()

def run_compare(args, test_info_str, fn_base, vis_fn_base, valid_mask_fn_base=None):
    print(f'{test_info_str}... ')
    
    fn_torch = os.path.join( args.dir_torch, fn_base )
    fn_opencv = os.path.join( args.dir_opencv, fn_base )
    
    # Test if the target file exists.
    if not os.path.isfile( fn_torch ):
        print('Skip the test. ')
        return
    
    # Read the two equirectangluar images.
    img_torch = read_image(fn_torch)
    img_opencv = read_image(fn_opencv)
    
    # The mask.
    if valid_mask_fn_base is not None:
        fn_valid_mask = os.path.join( args.dir_torch, valid_mask_fn_base )
        valid_mask = read_image(fn_valid_mask)
    else:
        valid_mask = None
    
    # The equirectangular images.
    compare_and_write( os.path.join( args.dir_torch, vis_fn_base ),
                      10, 
                      [ img_torch, img_opencv ],
                      valid_mask)
    
    # Copy the image to the torch directory.
    copy_file( fn_opencv, args.dir_torch, 'a_opencv' )

if __name__ == '__main__':
    
    # Handle the arguments.
    args = handle_args()
    
    run_compare( args, 
                'Fisheye iamges', 
                'result_six_fisheye_front.png',
                'diff_vis_fisheye.png',
                'valid_mask.png' )
    
    print('')
    run_compare( args, 
                'Equirectangular images', 
                'result_six_equi_rect.png',
                'diff_vis_equi_rect.png' )
    
    print('')
    run_compare( args, 
                'Equirectangular distance images', 
                'result_six_equi_rect_distance_vis.png',
                'diff_vis_equi_rect_dist_vis.png' )
    
    print('')
    run_compare( args, 
                'Fisheye from UE panorama', 
                'result_full_fisheye_front.png',
                'diff_vis_fisheye.png',
                'valid_mask.png' )
    
    print('Done. ')