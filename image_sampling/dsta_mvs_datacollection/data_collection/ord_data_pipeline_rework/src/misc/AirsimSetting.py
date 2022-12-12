import json
from os.path import expanduser
from settings import get_args

'''
Usage: 
python AirsimSetting.py --data-collection --gamma 1.0 --min-exposure 0.1 --max-exposure 0.5
python AirsimSetting.py --mapping
'''
class AirsimSetting(object):
    def __init__(self):
        home = expanduser("~")
        self.settingfile = home+"/Documents/AirSim/settings.json"
        with open(self.settingfile,'r') as f:
            self.params = json.load(f)
        print self.settingfile
        print self.params
        # import ipdb;ipdb.set_trace()

    def set_brightness(self, gamma, maxExposure=0.7, minExposure=0.3):
        cam0params = self.params['CameraDefaults']['CaptureSettings'][0]
        cam0params['TargetGamma'] = gamma
        cam0params['AutoExposureMaxBrightness'] = maxExposure
        cam0params['AutoExposureMinBrightness'] = minExposure

    def set_resolution(self, width, height):
        cam_params = self.params['CameraDefaults']['CaptureSettings']
        for camparam in cam_params:
            camparam['Width'] = width
            camparam['Height'] = height

    def set_fov(self, fov):
        cam_params = self.params['CameraDefaults']['CaptureSettings']
        for camparam in cam_params:
            camparam['FOV_Degrees'] = fov

    def set_viewmode(self, display):
        if display:
            if self.params.has_key('ViewMode'):
                self.params['ViewMode'] = ''
        else:
            self.params['ViewMode'] = 'NoDisplay'

    def dumpfile(self):
        with open(self.settingfile, 'w') as f:
            json.dump(self.params, f, indent = 4, )
        print('setting.cfg file saved..')

def set_mapping_setting():
    airsimSetting = AirsimSetting()
    airsimSetting.set_resolution(320, 320)
    airsimSetting.set_fov(90)
    airsimSetting.set_viewmode(display=False)
    airsimSetting.dumpfile()
    print('settings.json for mapping..')

def set_data_setting(gamma, maxExposure, minExposure):
    airsimSetting = AirsimSetting()
    airsimSetting.set_resolution(640, 480)
    airsimSetting.set_fov(90)
    airsimSetting.set_viewmode(display=True)
    airsimSetting.set_brightness(gamma, maxExposure, minExposure)
    airsimSetting.dumpfile()
    print('settings.json for data collection, gamma {}, exposure ({}, {})'.format(gamma, maxExposure, minExposure))

if __name__ == '__main__':

    args = get_args()
    if args.mapping:
        set_mapping_setting()
    elif args.data_collection:
        set_data_setting(args.gamma, args.max_exposure, args.min_exposure)

    # airsimSetting = AirsimSetting()
    # airsimSetting.set_brightness(3.7)
    # airsimSetting.dumpfile()