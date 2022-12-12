'''
1. Cut the trajectory based on EMVMAME_good_frames.txt
2. Rearrange the files by creating symbolic links
3. Shift the posefiles wrt KITTI style
Ex: python post_process_tartan.py /home/wenshan/tmp/data/tartan /home/wenshan/tmp/data/tartan_test_output
'''

from os.path import isfile, join, dirname, isdir
from os import listdir, system, mkdir
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import sys
import numpy as np
from data_validation import FileLogger

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


def gen_symbolic_links(linelist, trajfolderpath, errorlog):
    '''
    linelist: a list of left image path
    trajfolderpath: the trajectory folder P00X for output symbolic files
    Output:
        - depth_left
        - depth_right
        - flow
        - image_left
        - image_right
        - seg_left
        - seg_right
        - pose_left.txt
        - pose_right.txt
    '''

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
    flowfolder = trajfolderpath + '/flow'
    flow_suffix = "_flow.npy"
    flow_mask_suffix = "_mask.npy"

    # create image folders
    for folder in folders:
        outdir = join(trajfolderpath, folder)
        if not isdir(outdir):
            mkdir(outdir)
    # create flow folder
    outdir = join(trajfolderpath, 'flow')
    if not isdir(outdir):
        mkdir(outdir)

    targetfilelist = []

    flow_count = len(linelist) - 1
    for frame_ind, line in enumerate(linelist):
        frame_str = str(frame_ind).zfill(6) # the new frame id
        flow_str = frame_str + '_' + str(frame_ind+1).zfill(6)
        source_indstr = line.split('/')[-1]
        source_ind = int(source_indstr)

        for folder, suffix in zip(folders, suffixes): 
            sourcefile = line.replace('image_left', folder) + suffix
            targetfolder = join(trajfolderpath, folder)
            targetfile = join(targetfolder, frame_str + suffix)
            cmd = 'ln -s '+ sourcefile + ' ' + targetfile
            if not isfile(sourcefile):
                errorlog.logline('Missing file '+ sourcefile)
            system(cmd)

        targetfilelist.append(join(trajfolderpath, 'image_left', frame_str))

        # handle flow files
        if frame_ind < flow_count: # skip last one frame for flow
            source_flow_str = source_indstr + '_' + str(source_ind+1).zfill(6)

            sourcefile = line.split('image_left')[0] + 'flow/' + source_flow_str + '_flow.npy'
            targetfolder = join(trajfolderpath, 'flow')
            targetfile = join(targetfolder, flow_str + '_flow.npy')
            cmd = 'ln -s '+ sourcefile + ' ' + targetfile
            if not isfile(sourcefile):
                errorlog.logline('Missing file '+ sourcefile)
            system(cmd)

            sourcefile = line.split('image_left')[0] + 'flow/' + source_flow_str + '_mask.npy'
            targetfolder = join(trajfolderpath, 'flow')
            targetfile = join(targetfolder, flow_str + '_mask.npy')
            cmd = 'ln -s '+ sourcefile + ' ' + targetfile
            if not isfile(sourcefile):
                errorlog.logline('Missing file '+ sourcefile)
            system(cmd)

    # update the posefile
    lposefile = line.split('image_left')[0] + 'pose_left.txt'
    rposefile = line.split('image_left')[0] + 'pose_right.txt'
    startind = int(linelist[0].split('/')[-1])
    endind = int(linelist[-1].split('/')[-1]) + 1

    if isfile(lposefile):
        lposelist = np.loadtxt(lposefile)
        if len(lposelist) < endind:
            errorlog.logline('!!! Posefile too short '+ lposefile + ', end ind '+ str(endind))
        else:
            new_poselist = lposelist[startind:endind]
            shift_and_save_poselist(new_poselist, join(trajfolderpath, 'pose_left.txt'))
    else:
        errorlog.logline('!!! No posefile '+ lposefile)
    if isfile(rposefile):
        rposelist = np.loadtxt(rposefile)
        if len(rposelist) < endind:
            errorlog.logline('!!! Posefile too short '+ rposefile + ', end ind '+ str(endind))
        else:
            new_poselist = rposelist[startind:endind]
            shift_and_save_poselist(new_poselist, join(trajfolderpath, 'pose_right.txt'))
    else:
        errorlog.logline('!!! No posefile '+ rposefile)


    return targetfilelist


def output_new_traj(traj_ind, output_traj_dir, linelist, staf, out_env_dir, errorlog):
    Data2EasyHard = {'Data':'Easy', 'Data_fast':'Hard'}
    # save a new trajectory
    envname = out_env_dir.split('/')[-1]
    sss = linelist[-1].split('/')
    datafolder = sss[-4]
    target_datafolder = Data2EasyHard[datafolder]
    trajfolder = 'P'+str(traj_ind).zfill(3)
    trajname = target_datafolder+'_'+trajfolder
    # write the source files list
    out_traj_file = join(output_traj_dir, trajname+'.txt')
    with open(out_traj_file, 'w') as f:
        for ll in linelist:
            f.write(ll+'\n')
    # write new info to statistic file
    staf.write(trajname + ' ' + str(len(linelist)) + '\n')
    staf.write('    '+linelist[0]+'\n')
    staf.write('    '+linelist[-1]+'\n')
    staf.write('\n')  

    print('    Create new trajectory: '+ trajname + ' len: '+ str(len(linelist)) )

    datadir = join(out_env_dir, target_datafolder)
    if not isdir(datadir):
        mkdir(datadir)
    trajfolderpath = join(datadir, trajfolder)
    if not isdir(trajfolderpath):
        mkdir(trajfolderpath)
    targetfiles = gen_symbolic_links(linelist, trajfolderpath, errorlog)

    # write the target files list w/ relative path
    out_traj_file = join(output_traj_dir, trajname+'_relative_target.txt')
    with open(out_traj_file, 'w') as f:
        for ll in targetfiles:
            ll = envname + ll.split(envname)[-1]
            f.write(ll+'\n')

    # write the source files list w/ relative path
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
    staf = open(sta_file, 'w')

    errorlog = FileLogger(join(output_traj_dir, 'frame_error.log'))

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
                output_new_traj(traj_ind, output_traj_dir, linelist, staf, out_env_dir, errorlog)
                traj_ind += 1
                if datafolder!=last_datafolder:
                    traj_ind = 0
            # restart counting
            linelist = [line]

        last_frame_ind = frame_ind
        last_datafolder = datafolder

    if len(linelist) >= min_frame_num: # save last trajectory if long enough
        output_new_traj(traj_ind, output_traj_dir, linelist, staf, out_env_dir, errorlog)

    staf.close()
    errorlog.close()
    

if __name__ == '__main__':

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
        
    env_folders = listdir(inputdir)    

    for env_folder in env_folders:
        env_dir = join(inputdir, env_folder)
        ana_dir = join( env_dir, 'analyze')
        print('Working on env {}'.format(env_dir))
        good_frame_file = join(ana_dir, env_folder+'_good_frames.txt')

        out_env_dir = join(outputdir, env_folder)
        if not isdir(out_env_dir):
            mkdir(out_env_dir)

        cut_env_trajs(good_frame_file, out_env_dir)

