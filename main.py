from extract_features import FeatureExtractor
from pixel2ray import Pixel2Ray

cam0_path = 'data/cam0/'
cam1_path = 'data/cam1/'
calib_path = 'config/calibration.yaml'

cams = Pixel2Ray(calib_path)
cams.load_cams()

extractor = FeatureExtractor(cam0_path,cam1_path)
extractor.extract_features()

rays0 = cams.cams[0].pixel_2_ray(extractor.cam0_coords)[0].numpy().reshape((3,-1))
rays1 = cams.cams[1].pixel_2_ray(extractor.cam1_coords)[0].numpy().reshape((3,-1))

# extractor.vis_features()
