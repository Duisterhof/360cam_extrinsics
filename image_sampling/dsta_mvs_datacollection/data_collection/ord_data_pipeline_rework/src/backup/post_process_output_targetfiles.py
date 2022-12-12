'''
Add an intermediate output for target files
This has been integrated into post_process_tartan
Fix all kinds of problems: flow file size, copy files as needed etc. 
Copy the flow files on the cluster, for physical continuous
'''

from os.path import isfile, join, dirname, isdir
from os import listdir, system, mkdir
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import sys
import numpy as np
from data_validation import FileLogger

def get_posefile_names(source_start, source_end, target_start):
    '''
    source_start: 'abandonedfactory/Data/P000/image_left/000000'
    source_end: 'abandonedfactory/Data/P000/image_left/002175'
    target_start: 'abandonedfactory/Easy/P000/image_left/002175'
    '''
    sourcedir = source_start.split('image_left')[0]
    source_left_file = sourcedir + 'pose_left.txt'
    source_right_file = sourcedir + 'pose_right.txt'
    startind = int(source_start.split('/')[-1])
    endind = int(source_end.split('/')[-1])
    targetdir = target_start.split('image_left')[0]
    target_left_file = targetdir + 'pose_left.txt'
    target_right_file = targetdir + 'pose_right.txt'

    return source_left_file, source_right_file, startind, endind, target_left_file, target_right_file

def shift_and_save_poselist(poselist, posefile):
    np.savetxt(posefile, poselist)

def left2all(leftfilepath):

    folders = ['depth_left', 
    'depth_right', 
    'image_left', 
    'image_right', 
    'seg_left', 
    'seg_right'] 

    suffixes = ["_left_depth.npy",
    "_right_depth.npy",
    "_left.png",
    "_right.png",
    "_left_seg.npy",
    "_right_seg.npy"] 

    # handle flow files
    flowfolder = 'flow'
    flow_suffix = "_flow.npy"
    flow_mask_suffix = "_mask.npy"

    filelist = []

    for folder, suffix in zip(folders, suffixes): 
        newfile = leftfilepath.replace('image_left', folder) + suffix
        filelist.append(newfile)

    indstr = leftfilepath.split('/')[-1]
    flow_str = indstr + '_' + str(int(indstr)+1).zfill(6)
    flow_prefix = leftfilepath.split('image_left')[0] + flowfolder + '/' + flow_str
    flowfile = flow_prefix + flow_suffix
    maskfile = flow_prefix + flow_mask_suffix
    filelist.append(flowfile)
    filelist.append(maskfile)

    return filelist    


def targetfilelist(linelist, trajfolderpath):

    targetfilelist = []

    flow_count = len(linelist) - 1
    for frame_ind, line in enumerate(linelist):
        frame_str = str(frame_ind).zfill(6) # the new frame id

        targetfilelist.append(join(trajfolderpath, 'image_left', frame_str))


    return targetfilelist
    

def output_new_traj(traj_ind, output_traj_dir, linelist, out_env_dir):
    Data2EasyHard = {'Data':'Easy', 'Data_fast':'Hard'}
    # save a new trajectory
    sss = linelist[-1].split('/')
    datafolder = sss[-4]
    target_datafolder = Data2EasyHard[datafolder]
    trajfolder = 'P'+str(traj_ind).zfill(3)
    trajname = target_datafolder+'_'+trajfolder

    datadir = join(out_env_dir, target_datafolder)
    trajfolderpath = join(datadir, trajfolder)
    envname = out_env_dir.split('/')[-1]

    targetfiles = targetfilelist(linelist, trajfolderpath)
    out_traj_file = join(output_traj_dir, trajname+'_relative_target.txt')
    with open(out_traj_file, 'w') as f:
        for ll in targetfiles:
            ll = envname + ll.split(envname)[-1]
            f.write(ll+'\n')

    out_traj_file = join(output_traj_dir, trajname+'_relative.txt')
    with open(out_traj_file, 'w') as f:
        for ll in linelist:
            ll = envname + ll.split(envname)[-1]
            f.write(ll+'\n')

def cut_env_trajs(good_frame_file, out_env_dir, min_frame_num = 300):
    '''
    Input: a file of list of image sequences
           output directory (env level)
    Output: a number of txt files: Easy_P00X.txt, Hard_P00X.txt
            one statistic file contrains: (traj_file, first_frame, last_frame, frame_num)
    '''
    with open(good_frame_file, 'r') as ff:
        lines = ff.readlines()

    output_traj_dir = join(out_env_dir, 'trajfiles')
    if not isdir(output_traj_dir):
        mkdir(output_traj_dir)

    traj_ind = 0 # current output trajectory index
    frame_ind = 0 # current frame id
    last_frame_ind = -1 # last frame id
    last_datafolder = ''
    linelist = []
    sta_file = join(output_traj_dir, 'traj_info.txt')


    for line in lines:
        line = line.strip()
        strs = line.split('/')
        framestr = strs[-1]
        frame_ind = int(framestr)
        datafolder = strs[-4]

        if frame_ind == last_frame_ind+1:
            # the frame is continuous
            linelist.append(line)

        else:
            # see if the frame_count is enough
            if len(linelist) >= min_frame_num:
                # save a new trajectory
                output_new_traj(traj_ind, output_traj_dir, linelist, out_env_dir)
                traj_ind += 1
                if datafolder!=last_datafolder:
                    traj_ind = 0
            # restart counting
            linelist = [line]

        last_frame_ind = frame_ind
        last_datafolder = datafolder

    if len(linelist) >= min_frame_num: # save last trajectory if long enough
        output_new_traj(traj_ind, output_traj_dir, linelist, out_env_dir)

    

if __name__ == '__main__':

    # if len(sys.argv)<3: 
    #     print("USAGE: python post_process.py INPUT_DIR OUTPUT_DIR")
    #     sys.exit() 

    # inputdir = sys.argv[1] # 
    # outputdir = sys.argv[2]

    # if not isdir(inputdir):
    #     print("Cannot find the input directory: {}".format(inputdir))
    #     sys.exit() 

    # if not isdir(outputdir): # assume the parent dir exists!
    #     suc = mkdir(outputdir)
    #     if suc>0:
    #         print("Cannot make output directory: {}".format(outputdir))
    #         sys.exit()
    # else: # not over write the folder
    #     print("Output directory already exists: {}".format(outputdir))

    from settings import get_args    
    args = get_args()
    inputdir = args.data_root
    outputdir = args.target_root
    if args.env_folders=='': # read all available folders in the data_root_dir
        env_folders = listdir(data_root_dir)    
    else:
        env_folders = args.env_folders.split(',')
    print('Detected envs {}'.format(env_folders))

    # env_folders = ['abandonedfactory','carwelding','gascola','house','neighborhood','office2','seasonsforest','soulcity','abandonedfactory_night','endofworld','hongkongalley','house_dist0','ocean','oldtown','seasonsforest_winter','westerndesert','amusement','flow_error.log','hospital','japanesealley','office','seasidetown','slaughter']
    
    for env_folder in env_folders:
        env_dir = join(inputdir, env_folder)
        print('Env: {}'.format(env_dir))
        out_env_dir = join(outputdir, env_folder)
        if not isdir(out_env_dir):
            mkdir(out_env_dir)
        # ana_dir = join( env_dir, 'analyze')
        # print('Working on env {}'.format(env_dir))
        # good_frame_file = join(ana_dir, env_folder+'_good_frames.txt')

        # out_env_dir = join(outputdir, env_folder)
        # if not isdir(out_env_dir):
        #     mkdir(out_env_dir)

        # cut_env_trajs(good_frame_file, out_env_dir)

        # # copy the trajfiles folder to local machine
        # mkdir('/home/wenshan/tmp/data/tartanair/{}'.format(env_folder))
        # cmd = 'scp -r wenshanw@perceptron.ri.cmu.edu:/data/datasets/wenshanw/tartanair/{}/trajfiles /home/wenshan/tmp/data/tartanair/{}/'.format(env_folder, env_folder)
        # system(cmd)

        # # resave flow files from float64 to float32
        # for datafolder in ['Data', 'Data_fast']:
        #     data_dir = join(env_dir, datafolder)
        #     if not isdir(data_dir):
        #         print '!!data folder missing', data_dir
        #         continue
        #     trajfolders = listdir(data_dir)
        #     trajfolders = [ tf for tf in trajfolders if tf[0]=='P' ]
        #     trajfolders.sort()
        #     print('  Find {} trajectories. '.format(len(trajfolders)))
        #     for trajfolder in trajfolders:
        #         flow_dir = join(data_dir, trajfolder, 'flow')
        #         flowfiles = listdir(flow_dir)
        #         flowfiles = [ ff for ff in flowfiles if ff.find('flow.npy')>=0 ]
        #         flowfiles.sort()

        #         print('    Find {} flow files'.format(len(flowfiles)))
        #         for flowfile in flowfiles:
        #             flowfilename = join(flow_dir, flowfile)
        #             flownp = np.load(flowfilename)
        #             flownp = flownp.astype(np.float32)
        #             np.save(flowfilename, flownp)
        #         print('    Resaved {} flow files'.format(len(flowfiles)))


        # copy paste flow folder to make it stored in continuous space
        for datafolder in ['Data', 'Data_fast']:
            data_dir = join(env_dir, datafolder)
            out_data_dir = join(out_env_dir, datafolder)
            if not isdir(data_dir):
                print '!!data folder missing', data_dir
                continue
            if not isdir(out_data_dir):
                mkdir(out_data_dir)

            trajfolders = listdir(data_dir)
            trajfolders = [ tf for tf in trajfolders if tf[0]=='P' ]
            trajfolders.sort()
            print('  Find {} trajectories. '.format(len(trajfolders)))
            for trajfolder in trajfolders:
                flow_dir = join(data_dir, trajfolder, 'flow')
                out_traj_dir = join(out_data_dir, trajfolder)
                out_flow_dir = join(out_traj_dir, 'flow')
                if not isdir(flow_dir): 
                    print '!!cannot find flow folder', flow_dir
                    continue

                if not isdir(out_traj_dir):
                    mkdir(out_traj_dir)
                if not isdir(out_flow_dir):
                    mkdir(out_flow_dir)
                # cmd = 'mv ' + flow_dir + ' ' + flow_dir+'_bk'
                # print('    cmd: {}'.format(cmd))
                # system(cmd)
                cmd = 'cp -r ' + flow_dir+'/*.npy' + ' ' + out_flow_dir+'/'
                print('    cmd: {}'.format(cmd))
                system(cmd)
                # cmd = 'rm -rf ' + flow_dir+'_bk'
                # print('    cmd: {}'.format(cmd))
                # system(cmd)

    # print get_posefile_names('abandonedfactory/Data/P000/image_left/000000',
    #                         'abandonedfactory/Data/P000/image_left/002175',
    #                         'abandonedfactory/Easy/P000/image_left/002175')
