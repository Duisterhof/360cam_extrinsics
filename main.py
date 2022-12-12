from extract_features import FeatureExtractor
from pixel2ray import Pixel2Ray

cam0_path = 'data/cam0/'
cam1_path = 'data/cam1/'
calib_path = 'config/calibration.yaml'

cams = Pixel2Ray(calib_path)
cams.load_cams()

# extractor = FeatureExtractor(cam0_path,cam1_path)
# extractor.extract_features()
# extractor.vis_features()
