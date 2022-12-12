# Read test_trajs.txt
# Copy the trajectories to the test folder

from os.path import isfile,isdir,join
from os import system, mkdir

rootdir = './'
targetdir = 'tartanair_test'

with open('test_traj.txt','r') as f:
    lines=f.readlines()

trajs = []
for ll in lines:
    ll = ll.strip()
    if len(ll)>0:
        trajs.append(ll.split(','))

folderlist = ['image_left', 'image_right', 'pose_left.txt', 'pose_right.txt']
for ttind, tt in enumerate(trajs):
    envname = tt[0]

    if ttind<10:
        cat = 'Easy'
    else:
        cat = 'Hard'
    
    for ttt, ms in zip([tt[1], tt[2]],['mono', 'stereo']): # Monocular, stereo
        datafolder, trajfolder = ttt.split('_')

        if envname=='house_dist0' and trajfolder=='P008': # hack to fix a problem
            envname='house'

        if datafolder=='Easy' or datafolder=='Hard': # hack again
            rootdir='tartanair'
        elif datafolder=='Data':
            rootdir='tartan_data'

        sourcefolder = join(rootdir, envname, datafolder, trajfolder) 
        print ttind, sourcefolder
        assert(isdir(sourcefolder))

        sourcevideo = join(rootdir, envname, 'video', datafolder+'_'+trajfolder+'.mp4')
        assert(isfile(sourcevideo))

        if envname=='house_dist0': # hack
            envname='house'

        targetname = envname+'_'+trajfolder
        targetfolder = join(targetdir, ms, cat, targetname)
        mkdir(targetfolder)
        print 'mkdir ', targetfolder

        # cpoy video to test folder
        targetvidname = ms + '_' + cat + '_' + envname + '_' + trajfolder+'.mp4'
        cmd = 'cp ' + sourcevideo + ' ' + join(targetdir, 'video', targetvidname)
        print '  exe:',cmd
        system(cmd)
        # copy sub-folders in a trajectory
        for cpc in folderlist:
            if cpc[-4:]=='.txt':
                cmd = 'cp '
                assert(isfile(join(sourcefolder,cpc)))
            else:
                cmd = 'cp -r '
                assert(isdir(join(sourcefolder,cpc)))


            cmd = cmd + join(sourcefolder,cpc) + ' ' + join(targetfolder,cpc)
            print '  exe: ', cmd
            system(cmd)



