# Compress the data to save space

import numpy as np
import cv2
from os.path import join, isdir
from os import mkdir, listdir
import time
from settings import get_args
from os import system

def convert_rgb(filename, output_filename):
    img = cv2.imread(filename)
    cv2.imwrite(output_filename, img) # default compression


def convert_flow(filename, output_filename):
    flow = np.load(filename)   
    mask = np.load(filename.replace('flow.npy', 'mask.npy'))

    flow16 = flow32to16(flow, mask)
    cv2.imwrite(output_filename, flow16)

def convert_depth(filename, output_filename):
    depth = np.load(filename)   
    disp = depth2disp(depth)
    disp16, mask  = disp32to16(disp)
    cv2.imwrite(joutput_filename, disp16)

def convert_depth_lossless(filename, output_filename):
    depth = np.load(filename)   
    depth_uint8  = depth_float32_rgba(depth)
    cv2.imwrite(output_filename, depth_uint8)

def convert_seg(filename, output_filename):
    seg = np.load(filename)
    cv2.imwrite(output_filename, seg ) # default compression

def convert_file(filename, output_filename, filetype):
    if filetype[:5]=='image':
        convert_rgb(filename, output_filename)
    elif filetype[:5]=='depth':
        convert_depth_lossless(filename, output_filename)
    elif filetype[:3]=='seg':
        convert_seg(filename, output_filename)
    elif filetype[:4]=='flow':
        convert_flow(filename, output_filename)
    else:
        print('Unknow file type: {}'.format(filetype))

def convert_trajectory(trajpath, output_path):
    # convert rgb
    print('Working on trajectory: {}'.format(trajpath))
    # floders = {'image_left': 'png', 'image_right': 'png', 
    #             'depth_left': 'npy', 'depth_right': 'npy',
    #             'seg_left': 'npy', 'seg_right': 'npy',
    #             'flow': 'flow.npy', 'flow2': 'flow.npy'
    #             }
    floders = {'flow4': 'flow.npy', 'flow6': 'flow.npy'}
    for ff, suffix in floders.items(): 
        inputfolder = join(trajpath, ff)
        outputfolder = join(output_path, ff)
        if not isdir(inputfolder):
            print('  ==> Could not find {}'.format(inputfolder))
            continue
        if not isdir(outputfolder):
            mkdir(outputfolder)
        filelist = listdir(inputfolder)
        filelist = [f for f in filelist if f.endswith(suffix)]
        filelist.sort()
        start = time.time()
        for fn in filelist:
            imgfile = join(inputfolder, fn)
            outimgfile = join(outputfolder, fn.replace('.npy', '.png'))
            convert_file(imgfile, outimgfile, ff)
        print('  Converted {} images from {}, use {}s.'.format(len(filelist), ff, time.time()-start))

def depth_float32_rgba(depth):
    '''
    depth: float32, h x w
    store depth in uint8 h x w x 4
    and use png compression
    '''
    depth_rgba = depth[...,np.newaxis].view("<u1")
    return depth_rgba

def depth_rgba_float32(depth_rgba):
    depth = depth_rgba.view("<f4")
    return np.squeeze(depth, axis=-1)

# mask = 1  : CROSS_OCC 
#      = 10 : SELF_OCC
#      = 100: OUT_OF_FOV
#      = 200: OVER_THRESHOLD
def flow32to16(flow32, mask8):
    '''
    flow_32b (float32) [-512.0, 511.984375]
    flow_16b (uint16) [0 - 65535]
    flow_16b = (flow_32b * 64) + 32768  
    '''
    # mask flow values that out of the threshold -512.0 ~ 511.984375
    mask1 = flow32 < -512.0
    mask2 = flow32 > 511.984375
    mask = mask1[:,:,0] + mask2[:,:,0] + mask1[:,:,1] + mask2[:,:,1]
    # convert 32bit to 16bit
    h, w, c = flow32.shape
    flow16 = np.zeros((h, w, 3), dtype=np.uint16)
    flow_temp = (flow32 * 64) + 32768
    flow_temp = np.clip(flow_temp, 0, 65535)
    flow_temp = np.round(flow_temp)
    flow16[:,:,:2] = flow_temp.astype(np.uint16)
    mask8[mask] = 200
    flow16[:,:,2] = mask8.astype(np.uint16)

    return flow16

def flow16to32(flow16):
    '''
    flow_32b (float32) [-512.0, 511.984375]
    flow_16b (uint16) [0 - 65535]
    flow_32b = (flow16 -32768) / 64
    '''
    flow32 = flow16[:,:,:2].astype(np.float32)
    flow32 = (flow32 - 32768) / 64.0

    mask8 = flow16[:,:,2].astype(np.uint8)
    return flow32, mask8

def copyposefile(filename, output_filename):
    cmd = 'cp ' + filename + ' ' + output_filename
    system(cmd)


# python compress_convert.py --data-root /data/datasets/wenshanw/tartan_data --data-folders Data,Data_fast --target-root /project/learningvo/tartanair_v1_5 \
#                             --env-folders 
                            # abandonedfactory,amusement,endofworld,
                            # hongkongalley,house,japanesealley,office,
                            # oldtown,seasonsforest,slaughter,westerndesert,
                            # abandonedfactory_night,carwelding,gascola,
                            # hospital,house_dist0,
                            # neighborhood,ocean,office2,
                            # seasidetown,seasonsforest_winter,soulcity
if __name__ == '__main__':
    args = get_args()

    data_root_dir = args.data_root
    data_folders = args.data_folders.split(',')
    out_root_dir = args.target_root

    if args.env_folders=='': # read all available folders in the data_root_dir
        env_folders = listdir(data_root_dir)    
    else:
        env_folders = args.env_folders.split(',')
    print('Detected envs {}'.format(env_folders))

    for env_folder in env_folders:
        env_dir = data_root_dir+'/'+ env_folder
        print('Working on env {}'.format(env_dir))

        out_env_dir = out_root_dir+'/'+env_folder
        if not isdir(out_env_dir):
            mkdir(out_env_dir)

        for data_folder in data_folders:
            datapath = env_dir +'/'+ data_folder
            if not isdir(datapath):
                print('!!data folder missing '+ datapath)
                continue
            print('    Opened data folder {}'.format(datapath))
            
            out_datapath = out_env_dir+'/'+data_folder
            if not isdir(out_datapath):
                mkdir(out_datapath)

            trajfolders = listdir(datapath)
            trajfolders = [ tf for tf in trajfolders if tf[0]=='P' ]
            trajfolders.sort()
            print('    Found {} trajectories'.format(len(trajfolders)))

            for trajfolder in trajfolders:
                trajpath = datapath +'/' +trajfolder

                out_trajpath = out_datapath + '/' + trajfolder
                if not isdir(out_trajpath):
                    mkdir(out_trajpath)

                convert_trajectory(trajpath, out_trajpath)
                # copyposefile(trajpath+'/pose_left.txt', out_trajpath+'/pose_left.txt')
                # copyposefile(trajpath+'/pose_right.txt', out_trajpath+'/pose_right.txt')



# # ========== Test functions ===========
# def test_compare_seg():
#     import time
#     # compare the data
#     imgnum = 734
#     compression = 'rle'
#     start = time.time()
#     for k in range(imgnum):
#         filename = str(k).zfill(6) + '_left_seg.npy'
#         seg1 = np.load(join(trajdir, 'seg_left', filename))
#         # print img1.dtype
#         # cv2.imwrite(join(trajdir,'test'+str(compression),filename), img1, [cv2.IMWRITE_PNG_COMPRESSION, compression] )
#         cv2.imwrite(join(trajdir,'test'+str(compression),filename.replace('.npy','.png')), seg1 )
#     print 'write time:', (time.time()-start)/imgnum * 1000
#     start = time.time()
#     for k in range(imgnum): 
#         filename = str(k).zfill(6) + '_left_seg.png'
#         img1 = cv2.imread(join(trajdir, 'test'+str(compression), filename))
#     print 'read time:', (time.time()-start)/imgnum * 1000
#     # output the difference
#     # for k in range(imgnum): 
#     #     filename = str(k).zfill(6) + '_left_seg.npy'
#     #     img1 = np.load(join(trajdir, 'seg_left', filename))
#     #     img2 = cv2.imread(join(trajdir, 'test'+str(compression), filename.replace('.npy','.png')),cv2.IMREAD_UNCHANGED)
#     #     diff = img1-img2
#     #     print diff.min(),diff.max()

# def test_compare_rgb():
#     import time
#     # compare the data
#     imgnum = 734
#     compression = 'rle'
#     # start = time.time()
#     # for k in range(imgnum):
#     #     filename = str(k).zfill(6) + '_left.png'
#     #     img1 = cv2.imread(join(trajdir, 'image_left', filename))
#     #     # print img1.dtype
#     #     # cv2.imwrite(join(trajdir,'test'+str(compression),filename), img1, [cv2.IMWRITE_PNG_COMPRESSION, compression] )
#     #     cv2.imwrite(join(trajdir,'test'+str(compression),filename), img1 )
#     # print 'write time:', (time.time()-start)/imgnum * 1000
#     #     # img2 = cv2.imread(join(trajdir,'test',filename))
#     # # print img2.dtype
#     # # diff = img1-img2
#     # # print diff.min(),diff.max()
#     # # compare the velocity
#     # start = time.time()
#     # for k in range(imgnum): 
#     #     filename = str(k).zfill(6) + '_left.png'
#     #     img1 = cv2.imread(join(trajdir, 'image_left', filename))
#     # print 'read time:', (time.time()-start)/imgnum * 1000

#     # start = time.time()
#     # for k in range(imgnum): 
#     #     filename = str(k).zfill(6) + '_left.png'
#     #     img1 = cv2.imread(join(trajdir, 'test'+str(compression), filename))
#     # print 'read time2:', (time.time()-start)/imgnum * 1000

#     # output the difference
#     for k in range(imgnum): 
#         filename = str(k).zfill(6) + '_left.png'
#         img1 = cv2.imread(join(trajdir, 'image_left', filename))
#         img2 = cv2.imread(join(trajdir, 'test'+str(compression), filename))
#         diff = img1-img2
#         print diff.min(),diff.max()


# def test_compare_flow():
#     import time
#     # compare the data
#     imgnum = 734
#     compression = 3
#     # start = time.time()
#     for k in range(imgnum-1):
#         filename = str(k).zfill(6) + '_' + str(k+1).zfill(6)  + '_flow.npy'
#         flow1 = np.load(join(trajdir, 'flow', filename))   
#         mask1 = np.load(join(trajdir, 'flow', filename.replace('flow', 'mask')))
#         # print flow1.dtype, mask1.dtype

#         # flow16 = flow32to16(flow1, mask1)
#         # cv2.imwrite(join(trajdir, 'test', filename.replace('npy','png')), flow16)
#         # cv2.imwrite(join(trajdir, 'test', filename.replace('.npy','_3.png')), flow16, [cv2.IMWRITE_PNG_COMPRESSION, 3])
#         # cv2.imwrite(join(trajdir, 'test', filename.replace('.npy','_5.png')), flow16, [cv2.IMWRITE_PNG_COMPRESSION, 5])
#         # cv2.imwrite(join(trajdir, 'test', filename.replace('.npy','_9.png')), flow16, [cv2.IMWRITE_PNG_COMPRESSION, 9])

#         flow16_2 = cv2.imread(join(trajdir, 'test', filename.replace('npy','png')), cv2.IMREAD_UNCHANGED)
#         # print flow16_2.dtype
#         flow2, mask2 = flow16to32(flow16_2)
#         # print flow2.dtype, mask2.dtype

#         diff_flow = flow1 - flow2
#         diff_mask = mask1 - mask2
#         print diff_flow.max(), diff_flow.min(), np.abs(diff_flow).mean()
#         # print diff_mask.max(), diff_mask.min(), np.abs(diff_mask).mean()

#     # # # compare the time
#     # # acc_timenp = 0.
#     # # acc_time0 = 0.
#     # # acc_time1 = 0.
#     # # acc_time2 = 0.
#     # # acc_time3 = 0.
#     # # acc_time5 = 0.
#     # # acc_time7 = 0.
#     # # acc_time9 = 0.
#     # # for k in range(imgnum-1):
#     # #     filename = str(k).zfill(6) + '_' + str(k+1).zfill(6)  + '_flow.npy'
#     # #     flow1 = np.load(join(trajdir, 'flow', filename))   
#     # #     mask1 = np.load(join(trajdir, 'flow', filename.replace('flow', 'mask')))
#     # #     print flow1.dtype, mask1.dtype

#     # #     flow16 = flow32to16(flow1, mask1)

#     # #     start = time.time()
#     # #     np.save(join(trajdir, 'testnp', filename.replace('npy','png')), flow16)
#     # #     acc_timenp += time.time() - start
#     # #     # start = time.time()
#     # #     # cv2.imwrite(join(trajdir, 'test0', filename.replace('npy','png')), flow16, [cv2.IMWRITE_PNG_COMPRESSION, 0])
#     # #     # acc_time0 += time.time() - start
#     # #     # start = time.time()
#     # #     # cv2.imwrite(join(trajdir, 'test1', filename.replace('.npy','_3.png')), flow16, [cv2.IMWRITE_PNG_COMPRESSION, 1])
#     # #     # acc_time1 += time.time() - start
#     # #     # start = time.time()
#     # #     # cv2.imwrite(join(trajdir, 'test2', filename.replace('.npy','_5.png')), flow16, [cv2.IMWRITE_PNG_COMPRESSION, 2])
#     # #     # acc_time2 += time.time() - start

#     # #     start = time.time()
#     # #     cv2.imwrite(join(trajdir, 'test7', filename.replace('.npy','_7.png')), flow16, [cv2.IMWRITE_PNG_COMPRESSION, 7])
#     # #     acc_time7 += time.time() - start
#     # # print acc_timenp/(imgnum-1), acc_time7/(imgnum-1)

#     # ll = ['np']
#     # pp = ['']
#     # for lll,ppp in zip(ll,pp):
#     #     start = time.time()
#     #     for k in range(imgnum-1):
#     #         filename = str(k).zfill(6) + '_' + str(k+1).zfill(6)  + '_flow.npy'
#     #         filenpathname = join(trajdir, 'test'+lll, filename.replace('.npy',ppp+'.png'))
#     #         # img = cv2.imread(filenpathname, cv2.IMREAD_UNCHANGED)
#     #         filenpathname = join(trajdir, 'test'+lll, filename.replace('.npy',ppp+'.png.npy'))
#     #         img = np.load(filenpathname)
#     #         # print img.shape, img.dtype
#     #     print lll, (time.time()-start)/(imgnum-1) * 1000
#     # import ipdb;ipdb.set_trace()


# def test_compare_disp():
#     import time
#     from os.path import isdir
#     from os import mkdir
#     # compare the data
#     imgnum = 734
#     compression = 3
#     # start = time.time()
#     ll = ['', '0', '1', '3','5','9']
#     ll = ['np']
#     ll = ['']
#     for lll in ll:
#         dispdir = join(trajdir, 'test'+lll)
#         if not isdir(dispdir):
#             mkdir(dispdir)

#         writetime = 0.
#         readtime = 0.
#         for k in range(imgnum):
#             filename = str(k).zfill(6) + '_left_depth.npy'
#             depth1 = np.load(join(trajdir, 'depth_left', filename))   
#             disp1 = depth2disp(depth1)

#             disp16, mask  = disp32to16(disp1)
#             # print disp16.dtype, mask.sum()
#             start = time.time()
#             if lll == '':
#                 cv2.imwrite(join(dispdir, filename.replace('npy','png')), disp16)
#             elif lll == 'np':
#                 np.save(join(dispdir, filename), disp16)
#             else:
#                 cv2.imwrite(join(dispdir, filename.replace('.npy','.png')), disp16, [cv2.IMWRITE_PNG_COMPRESSION, int(lll)])
#             writetime += time.time() - start
#             # cv2.imwrite(join(trajdir, 'test', filename.replace('.npy','_5.png')), flow16, [cv2.IMWRITE_PNG_COMPRESSION, 5])
#             # cv2.imwrite(join(trajdir, 'test', filename.replace('.npy','_9.png')), flow16, [cv2.IMWRITE_PNG_COMPRESSION, 9])
#             start = time.time()
#             if lll == 'np':
#                 disp16_2 = np.load(join(dispdir, filename))
#             else:
#                 disp16_2 = cv2.imread(join(dispdir, filename.replace('npy','png')), cv2.IMREAD_UNCHANGED)
#             readtime += time.time() - start
#             # print disp16_2.dtype
#             disp2 = disp16to32(disp16_2)
#             depth2 = disp2depth(disp2)
#             # print flow2.dtype, mask2.dtype

#             diff_disp = disp1 - disp2
#             diff_depth = depth1 - depth2
#             # print diff_disp.max(), diff_disp.min(), np.abs(diff_disp).mean(), diff_depth.max(), diff_depth.min(), np.abs(diff_depth).mean()
        
#         print lll, 'readtime:', readtime/imgnum, 'writetime:', writetime/imgnum
#         print '========='
#         print
#     # # # compare the time
#     # # acc_timenp = 0.
#     # # acc_time0 = 0.
#     # # acc_time1 = 0.
#     # # acc_time2 = 0.
#     # # acc_time3 = 0.
#     # # acc_time5 = 0.
#     # # acc_time7 = 0.
#     # # acc_time9 = 0.
#     # # for k in range(imgnum-1):
#     # #     filename = str(k).zfill(6) + '_' + str(k+1).zfill(6)  + '_flow.npy'
#     # #     flow1 = np.load(join(trajdir, 'flow', filename))   
#     # #     mask1 = np.load(join(trajdir, 'flow', filename.replace('flow', 'mask')))
#     # #     print flow1.dtype, mask1.dtype

#     # #     flow16 = flow32to16(flow1, mask1)

#     # #     start = time.time()
#     # #     np.save(join(trajdir, 'testnp', filename.replace('npy','png')), flow16)
#     # #     acc_timenp += time.time() - start
#     # #     # start = time.time()
#     # #     # cv2.imwrite(join(trajdir, 'test0', filename.replace('npy','png')), flow16, [cv2.IMWRITE_PNG_COMPRESSION, 0])
#     # #     # acc_time0 += time.time() - start
#     # #     # start = time.time()
#     # #     # cv2.imwrite(join(trajdir, 'test1', filename.replace('.npy','_3.png')), flow16, [cv2.IMWRITE_PNG_COMPRESSION, 1])
#     # #     # acc_time1 += time.time() - start
#     # #     # start = time.time()
#     # #     # cv2.imwrite(join(trajdir, 'test2', filename.replace('.npy','_5.png')), flow16, [cv2.IMWRITE_PNG_COMPRESSION, 2])
#     # #     # acc_time2 += time.time() - start

#     # #     start = time.time()
#     # #     cv2.imwrite(join(trajdir, 'test7', filename.replace('.npy','_7.png')), flow16, [cv2.IMWRITE_PNG_COMPRESSION, 7])
#     # #     acc_time7 += time.time() - start
#     # # print acc_timenp/(imgnum-1), acc_time7/(imgnum-1)

#     # ll = ['np']
#     # pp = ['']
#     # for lll,ppp in zip(ll,pp):
#     #     start = time.time()
#     #     for k in range(imgnum-1):
#     #         filename = str(k).zfill(6) + '_' + str(k+1).zfill(6)  + '_flow.npy'
#     #         filenpathname = join(trajdir, 'test'+lll, filename.replace('.npy',ppp+'.png'))
#     #         # img = cv2.imread(filenpathname, cv2.IMREAD_UNCHANGED)
#     #         filenpathname = join(trajdir, 'test'+lll, filename.replace('.npy',ppp+'.png.npy'))
#     #         img = np.load(filenpathname)
#     #         # print img.shape, img.dtype
#     #     print lll, (time.time()-start)/(imgnum-1) * 1000
#     # import ipdb;ipdb.set_trace()

# def test_compare_depth():
#     import time
#     from os.path import isdir
#     from os import mkdir
#     # compare the data
#     imgnum = 734
#     compression = 3
#     # start = time.time()
#     ll = ['']#, '0', '1', '3','5','9']
#     # ll = ['np']
#     # ll = ['']
#     for lll in ll:
#         dispdir = join(trajdir, 'test'+lll)
#         if not isdir(dispdir):
#             mkdir(dispdir)

#         writetime = 0.
#         readtime = 0.
#         for k in range(imgnum):
#             filename = str(k).zfill(6) + '_left_depth.npy'
#             depth1 = np.load(join(trajdir, 'depth_left', filename))   

#             # depth_uint16  = depth_float32_uint8(depth1)
#             depth_uint16  = depth_float32_rgba(depth1)
#             # print disp16.dtype, mask.sum()
#             start = time.time()
#             if lll == '':
#                 cv2.imwrite(join(dispdir, filename.replace('npy','png')), depth_uint16)
#             elif lll == 'np':
#                 np.save(join(dispdir, filename), depth_uint16)
#             else:
#                 cv2.imwrite(join(dispdir, filename.replace('.npy','.png')), depth_uint16, [cv2.IMWRITE_PNG_COMPRESSION, int(lll)])
#             writetime += time.time() - start
#             # cv2.imwrite(join(trajdir, 'test', filename.replace('.npy','_5.png')), flow16, [cv2.IMWRITE_PNG_COMPRESSION, 5])
#             # cv2.imwrite(join(trajdir, 'test', filename.replace('.npy','_9.png')), flow16, [cv2.IMWRITE_PNG_COMPRESSION, 9])
#             start = time.time()
#             if lll == 'np':
#                 depth16_2 = np.load(join(dispdir, filename))
#             else:
#                 depth16_2 = cv2.imread(join(dispdir, filename.replace('npy','png')), cv2.IMREAD_UNCHANGED)
#             readtime += time.time() - start
#             # print depth16_2.dtype
#             # depth2 = depth_uint8_float32(depth16_2)
#             depth2 = depth_rgba_float32(depth16_2)
#             # print flow2.dtype, mask2.dtype

#             diff_depth = depth1 - depth2
#             print diff_depth.max(), diff_depth.min(), np.abs(diff_depth).mean()
        
#         print lll, 'readtime:', readtime/imgnum, 'writetime:', writetime/imgnum
#         print '========='
#         print

# def depth2disp(depth, blxfx = 80):
#     '''
#     Convert depth to disparity: disparity = focal_x * baseline / depth
#     focal_x * baseline = 320 * 0.25 = 80
#     '''
#     return blxfx / depth

# def disp2depth(disp, blxfx = 80):
#     return blxfx / disp

# def depth32float2uint(depth, maxthresh=1024):
#     '''
#     depth: float32, h x w
#     1. mask depth > maxthresh
#     2. depth *= 2**(32-10)
#     '''
#     mask = depth > maxthresh
#     # import ipdb;ipdb.set_trace()
#     h, w = depth.shape
#     depth_uint16 = np.zeros((h, w, 3), dtype=np.uint16)
#     depth_32 = np.round(depth * (2**22))
#     depth_uint16[:,:,0] = depth_32 % (2**16)
#     depth_uint16[:,:,1] = depth_32 / (2**16)
#     depth_uint16[mask,2] = 1

#     return depth_uint16

# def depth32uint2float(depth_uint16):
#     depth = depth_uint16[:,:,1].astype(np.float32) * (2**16)
#     depth = depth_uint16[:,:,0] + depth
#     depth = depth / (2**22)
#     return depth

# def depth_float32_uint8(depth, maxthresh=512):
#     '''
#     depth: float32, h x w
#     1. mask depth > maxthresh
#     2. depth *= 2**(24-9)
#     '''
#     factor = 2**8
#     factor2 = 2**15
#     depth = np.clip(depth, 0, maxthresh-1.0/factor2)
#     # import ipdb;ipdb.set_trace()
#     h, w = depth.shape
#     depth_uint8 = np.zeros((h, w, 3), dtype=np.uint8)
#     depth_32 = np.round(depth * factor2)
#     depth_uint8[:,:,0] = depth_32 % factor
#     depth_32 = depth_32 / factor
#     depth_uint8[:,:,1] = depth_32 % factor
#     depth_uint8[:,:,2] = depth_32 / factor

#     return depth_uint8

# def depth_uint8_float32(depth_uint8):
#     depth = depth_uint8[:,:,2].astype(np.float32) * (2**8)
#     depth = (depth_uint8[:,:,1] + depth) * (2**8)
#     depth = depth_uint8[:,:,0] + depth
#     depth = depth / (2**15)
#     return depth

# def disp32to16(disp32):
#     '''
#     Convert disparity_32b to disparity 16b
#     disp_32b: [-512.0, 511.984375]
#     disp_16b: [0 - 65535]
#     disp_16b = (disp_32b * 64) + 32768  
#     '''
#     mask1 = disp32 < -512.0
#     mask2 = disp32 > 511.984375
#     mask = mask1 + mask2
#     # convert 
#     disp_temp = (disp32 * 64) + 32768
#     disp_temp = np.clip(disp_temp, 0, 65535)
#     disp_temp = np.round(disp_temp)
#     disp16 = disp_temp.astype(np.uint16)

#     return disp16, mask

# def disp16to32(disp16):
#     disp32 = disp16.astype(np.float32)
#     disp32 = (disp32 - 32768) / 64.0
#     return disp32

