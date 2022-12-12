from os.path import isfile, join, dirname, isdir
from os import listdir, system, mkdir
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import sys
import numpy as np

if __name__ == '__main__':

    ClipLen = 200
    ShiftCenter = True

    if len(sys.argv)<3: 
        print("USAGE: python post_process.py INPUT_DIR OUTPUT_DIR")
        sys.exit() 

    inputdir = sys.argv[1] # 
    outputdir = sys.argv[2]

    if not isdir(inputdir):
        print("Cannot find the input directory: {}".format(inputdir))
        sys.exit() 

    if not isdir(outputdir): # assume the parent dir exists!
        suc = mkdir(outputdir)
        if suc>0:
            print("Cannot make output directory: {}".format(outputdir))
            sys.exit()
    else: # not over write the folder
        print("Output directory already exists: {}".format(outputdir))
        sys.exit()
        
    trajfolders = listdir(inputdir)
    trajfolders = [tf for tf in trajfolders if tf[0]=='P']

    trajfolders.sort()

    outcount = 0
    for trajfolder in trajfolders: 
        trajfolderpath = inputdir + '/' + trajfolder

        # input subfolders
        limgfolder = trajfolderpath + '/image_left'
        rimgfolder = trajfolderpath + '/image_right'
        lposefile = trajfolderpath + '/pose_left.txt'
        rposefile = trajfolderpath + '/pose_right.txt'
        assert(isdir(limgfolder))
        assert(isdir(rimgfolder))
        assert(isfile(lposefile))
        assert(isfile(rposefile))

        # read all the data to memory
        limgs = listdir(limgfolder)
        limgs = [li for li in limgs if li[-3:]=='png']
        limgs.sort()
        rimgs = listdir(rimgfolder)
        rimgs = [ri for ri in rimgs if ri[-3:]=='png']
        rimgs.sort()
        lposes = np.loadtxt(lposefile)
        rposes = np.loadtxt(rposefile)
        # with open(lposefile,'r') as lf:
        #     lposes = lf.readlines() 
        # with open(rposefile,'r') as rf:
        #     rposes = rf.readlines()
        datalen = len(limgs)
        assert(len(limgs)==len(rimgs))
        assert(len(rimgs)==len(lposes))
        assert(len(lposes)==len(rposes))
        # print len(limgs), len(rimgs), len(lposes), len(rposes)

        # output subfolders
        dataind = 0
        while dataind + ClipLen <= datalen:
            outtrajdir = outputdir + '/P' + str(outcount).zfill(3)
            mkdir(outtrajdir)
            outlimgfolder = outtrajdir + '/image_left' 
            outrimgfolder = outtrajdir + '/image_right'
            outlposefile = outtrajdir + '/pose_left.txt'
            outrposefile = outtrajdir + '/pose_right.txt'
            mkdir(outlimgfolder)
            mkdir(outrimgfolder)

            for k in range(dataind, dataind + ClipLen):
                limgname = str(k).zfill(6)+'_left.png'
                rimgname = str(k).zfill(6)+'_right.png'
                outlimgname = str(k-dataind).zfill(6)+'_left.png' # indexing from 0
                outrimgname = str(k-dataind).zfill(6)+'_right.png'
                system('cp '+limgfolder+'/'+limgname + ' '+outlimgfolder+'/'+outlimgname)
                system('cp '+rimgfolder+'/'+rimgname + ' '+outrimgfolder+'/'+outrimgname)

            outlposes = lposes[dataind:dataind+ClipLen]
            outrposes = rposes[dataind:dataind+ClipLen]
            if ShiftCenter:
                outlposes[:,0:3] = outlposes[:,0:3] - outlposes[0,0:3]
                outrposes[:,0:3] = outrposes[:,0:3] - outrposes[0,0:3]
            np.savetxt(outlposefile, outlposes)
            np.savetxt(outrposefile, outrposes)

            # plt.figure()
            plt.plot(outlposes[:,0], -outlposes[:,1], '.-')
            plt.savefig(outtrajdir+'/traj.jpg')
            plt.clf()

            dataind += ClipLen
            outcount += 1
            print('Output folder {}, data index {}'.format(outtrajdir, dataind))

    # for filename in filenames:
    #   img = cv2.imread(join(imgdir, filename))
    #   # img = cv2.resize(img, (newwidth, newheight))
    #   fout.write(img)

    # fout.release()
