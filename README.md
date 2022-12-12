# Extrinsics from correspondences 


To run the code in this repository run:

```bash
 python3 main.py 
 ```

This repo has a number of dependencies, among others :

* Torch, Torchvision
* Plyfile
* Numba
* Kornia
* pytransform3d


# Change the camera intrinsics

To change the intrinsics change the config/calibration.yaml file, currently only ds-none is supported. 

# Add more data

To add more data add more images in data/camX. Make sure the naming convention is consistent and the number of images is equal. 

# Change parameters

Might make an argument parser in the future, but for now you'll have to change the hyper parameters in main.py


# Results

Calibrating two 195 degrees FOV fisheye lenses, we obtain the following result:

<img src="ransac.gif"
     alt="Calib results"
     style="float: left; margin-right: 10px;" />