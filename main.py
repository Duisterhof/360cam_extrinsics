from extract_features import FeatureExtractor
from pixel2ray import Pixel2Ray
from ransac_E import Essential_Optimizer
from pytransform3d.transform_manager import TransformManager
from pytransform3d import transformations as pt
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import numpy as np

cam0_path = 'data/cam0/'
cam1_path = 'data/cam1/'
calib_path = 'config/calibration.yaml'

cams = Pixel2Ray(calib_path)
cams.load_cams()

extractor = FeatureExtractor(cam0_path,cam1_path)
extractor.extract_features()
# extractor.vis_features()

rays0 = cams.cams[0].pixel_2_ray(extractor.cam0_coords)[0].numpy().reshape((3,-1))
rays1 = cams.cams[1].pixel_2_ray(extractor.cam1_coords)[0].numpy().reshape((3,-1))

print(rays0.shape)
print(rays1.shape)

E_Class = Essential_Optimizer(rays0,rays1)
E_Class.compute_system()
E_Class.full_lstsq()
# E_Class.ransac_loop(0.5,5000)
E_Class.full_lstsq()
E_Class.extract_motion()

print(E_Class.E)
print(E_Class.R)
print(E_Class.t)

cam0_2_cam1 = pt.transform_from(E_Class.R,E_Class.t)

tm = TransformManager()
tm.add_transform("Cam0","Cam1",cam0_2_cam1)
fig = plt.figure()
ax = tm.plot_frames_in("Cam0", s=0.5)
# ax.set_xlim((-0.25, 0.75))
# ax.set_ylim((-0.5, 0.5))
# ax.set_zlim((0.0, 1.0))

def rotate(angle):
    ax.view_init(azim=angle)

# print("Making animation")
# rot_animation = animation.FuncAnimation(fig, rotate, frames=np.arange(0, 361, 1), interval=25)
# rot_animation.save('ransac.gif', dpi=80, writer='imagemagick')
plt.show()

