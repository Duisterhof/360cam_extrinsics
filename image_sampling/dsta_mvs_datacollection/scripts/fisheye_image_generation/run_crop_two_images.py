
import cv2
import numpy as np
import os

def read_image(fn):
    img = cv2.imread(fn, cv2.IMREAD_UNCHANGED)
    assert img is not None, f'{fn} does not exist. '
    return img

if __name__ == '__main__':
    data_dir = 'output/20220618_02'
    # data_dir = 'output/20220618_03_fov89'
    img_fn_cube = 'result_six_equi_rect_cube.png'
    img_fn_equi = 'result_six_equi_rect.png'
    
    # Crop specification.
    x0, y0 = 1940, 480
    # x0, y0 = 0, 480
    h, w = 100, 100
    s = 10
    x1 = x0 + w
    y1 = y0 + h
    
    
    # Read the two images.
    img_cube = read_image( os.path.join(data_dir, img_fn_cube) )
    img_equi = read_image( os.path.join(data_dir, img_fn_equi) )
    
    # Crop.
    crop_cube = img_cube[ y0:y1, x0:x1, ... ]
    crop_equi = img_equi[ y0:y1, x0:x1, ... ]
    
    # Save.
    cv2.imwrite(os.path.join(data_dir, 'crop_cube.png'), crop_cube)
    cv2.imwrite(os.path.join(data_dir, 'crop_equi.png'), crop_equi)
    
    # Resize.
    crop_resized_cube = cv2.resize( crop_cube, (0, 0), fx=s, fy=s, interpolation=cv2.INTER_LINEAR )
    cv2.imwrite(os.path.join(data_dir, 'crop_resized_cube.png'), crop_resized_cube)
    
    crop_resized_equi = cv2.resize( crop_equi, (0, 0), fx=s, fy=s, interpolation=cv2.INTER_LINEAR )
    cv2.imwrite(os.path.join(data_dir, 'crop_resized_equi.png'), crop_resized_equi)
    
    
    print('Done. ')