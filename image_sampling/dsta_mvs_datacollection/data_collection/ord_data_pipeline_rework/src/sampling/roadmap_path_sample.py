#!/usr/bin/env python
import time
import numpy as np
import matplotlib.pyplot as plt
from os import mkdir, listdir
from os.path import exists, join, isfile, isdir
import pickle

from RandomPoseReSampler import RandomPoseReSampler

from settings import get_args

def node_distance(init, goal):
    diff = np.array([goal.position.x-init.position.x,
                     goal.position.y-init.position.y,
                     goal.position.z-init.position.z])
    return np.linalg.norm(diff, axis=0)


def sample_nodes_edges(pathgraph, pathsampler, outputgraphfilename, Nodenum, 
                        Minx, Maxx, Miny, Maxy, Minz, Maxz, 
                        Edgenum, TrajlenMinThresh, TrajlenMaxThresh, MaxFailureNum):
    ''' 
    Call sample_nodes_service and get the poselist
    Call roadmap_srv to connect nodes with edges
    Save the result to a graph file
    '''
    # if the graph is not empty, load the graph first
    pre_nodenum = pathgraph.graph.number_of_nodes()
    pre_nodelist = []
    pre_nodeslinknum = np.zeros(pre_nodenum, dtype=np.int32)
    for k, node in enumerate(pathgraph.graph.nodes):
        pre_nodelist.append(node)
        edgenum = len(pathgraph.graph.edges(node))
        pre_nodeslinknum[k] = edgenum

    # node sampling through ros service
    nodeposes = pathsampler.sample_nodes(Nodenum, Minx, Maxx, Miny, Maxy, Minz, Maxz) 
    nodeslinknum = np.zeros(Nodenum, dtype=np.int32)

    if nodeposes is None: # the service call has not returned the nodes
        print ('Node sample service failed')
        return

    # combine the pre-loaded nodes and link 
    if pre_nodenum>0:
        nodeposes = nodeposes + pre_nodelist 
        nodeslinknum = np.concatenate((nodeslinknum, pre_nodeslinknum))
        Nodenum = len(nodeposes)
        print('Combine pre-loaded nodes and links {}, total node {}'.format(len(pre_nodelist),Nodenum))

    pathsampler.publish_nodes_marker(nodeposes)
    for k,point in enumerate(nodeposes): # add link to the nodes
        initpose = nodeposes[k]
        randlist_after = np.random.permutation(Nodenum - k -1) + k + 1 # query nodes after the current one first
        randlist_befor = np.random.permutation(k)
        randlist = np.concatenate((randlist_after, randlist_befor))
        ind = 0
        failnum = 0
        while nodeslinknum[k] < Edgenum: # randomly sample from all the nodes
            if ind >= Nodenum-1: # Not enough edges after all the nodes have been sampled
                print('No enough edges '+str(nodeslinknum[k]) + ' sampled')
                break
            randind = randlist[ind]
            ind += 1
            # if nodeslinknum[ind] >= Edgenum: # this node has enough edges already
            #     continue
            goalpose = nodeposes[randind]
            if pathgraph.graph.has_edge(initpose, goalpose):
                print('edge exists ({}, {})'.format(k,randind))
                continue
            nodedist = node_distance(initpose, goalpose)
            if nodedist > TrajlenMinThresh and nodedist < TrajlenMaxThresh: # do not link to nearby node
                path = pathsampler.plan_edge(initpose, goalpose) # call OMPL
                pathsampler.publish_endpoints_marker(initpose, goalpose)
                if (path is not None) and (len(path)>0): 
                    # path = pathsampler.smooth_path(path)
                    pathsampler.publish_path_marker(path)
                    # pathsampler.publish_path_marker(path)
                    nodeslinknum[k] += 1
                    nodeslinknum[randind] += 1
                    print('{} - {}, find {} edges'.format(k, randind, nodeslinknum[k]))
                    pathgraph.graph.add_edge(initpose, goalpose, path=path)
                    # import ipdb;ipdb.set_trace()
                else: # planning failed
                    failnum += 1
                    if failnum>MaxFailureNum:
                        print('Too many failures, got edges of '+str(nodeslinknum[k]))
                        break
            # import ipdb;ipdb.set_trace()

    # import ipdb;ipdb.set_trace()
    # save the graph to disk
    if isfile(outputgraphfilename):
        timestr = time.strftime('%m%d_%H%M%S',time.localtime())
        outputgraphfilename = outputgraphfilename.split('.graph')[0] + '_' + timestr+'.graph'
    pickle.dump(pathgraph.graph, open(outputgraphfilename, 'w'))
    print('Graph file saved '+ outputgraphfilename)

def load_graph_and_sample_loop(pathsampler, pathgraph, outdir, samplemode=0, outfolder_suffix='',selected_path_folder = None, cycle_minnodes=2):


    pathgraph.prune_graph()
    print('nodes {}, edges {}.'.format(pathgraph.graph.number_of_nodes(), pathgraph.graph.number_of_edges()))

    # output dir
    if outfolder_suffix == '':
        timestr = time.strftime('%m%d_%H%M%S',time.localtime())
        dirname = outdir+'/ros_path_'+timestr
    else: 
        dirname = outdir+'/ros_path_'+outfolder_suffix
    if not exists(dirname):
        mkdir(dirname)

    # selected output dir in interactive mode
    if selected_path_folder is not None:
        selecteddir = outdir+'/'+selected_path_folder
        if not exists(selecteddir):
            mkdir(selecteddir)
        existingfiles = listdir(selecteddir)
        existingfiles = [ff for ff in existingfiles if (ff.endswith('txt'))]
        selected_ind = len(existingfiles)

    samplenum = 0
    plt.figure(figsize=(12,6))
    while True:
        pathsampler.vis_graph(pathgraph.graph, visedge=False)
        cycle = pathgraph.sample_cycle(mode=samplemode, minnodes=cycle_minnodes)
        if cycle is None:
            break
        cycle_poses = pathgraph.cycle2poselist(cycle)
        samplenum += 1
        print('{} - cycle nodes: {}, path length: {}'.format(samplenum, len(cycle), len(cycle_poses)))
        # pathsampler.publish_path_marker(cycle_poses,markerid=0) 
        distlist1 = pathgraph.node_dist_statistics(cycle_poses)
        # plt.subplot(121)
        # plt.hist(distlist1, bins=10)

        smooth_cycle_poses = pathsampler.smooth_path(cycle_poses)
        print ('    length after smooth: {}'.format(len(smooth_cycle_poses)))
        pathsampler.publish_path_marker(smooth_cycle_poses,markerid=1) 
        distlist2 = pathgraph.node_dist_statistics(smooth_cycle_poses)
        # plt.subplot(122)
        # plt.hist(distlist2, bins=10)
        # plt.show()

        positionlist = pathgraph.poselist2positionlist(smooth_cycle_poses)
        positions_np = np.array(positionlist, dtype=np.float32)
        pathgraph.visualize_poslist(positions_np, dirname+'/P'+str(samplenum-1).zfill(3)+'.png')
        np.savetxt(dirname+'/P'+str(samplenum-1).zfill(3)+'.txt', positions_np)

        if selected_path_folder is not None: # interaction mode
            selected = raw_input('    ==> Input y to select this path... ')
            if selected == 'y':
                savename = selecteddir+'/P'+str(selected_ind).zfill(3)+'.txt'
                np.savetxt(savename, positions_np)
                selected_ind += 1
                print('One path saved to '+savename)

        pathgraph.delete_path(cycle)
        # import ipdb;ipdb.set_trace()

    # generate visualization figures in the selected folder
    if selected_path_folder is not None:
        files = listdir(selecteddir)
        files = [ff for ff in files if (ff.endswith('txt'))]
        files.sort()
        plt.figure()
        for ff in files:
            poselist = np.loadtxt(selecteddir+'/'+ff)
            plt.subplot(121)
            plt.plot(poselist[:,0], -poselist[:,1],'.-') # flip y axis to make it top-down view
            plt.subplot(122)
            plt.plot(poselist[:,0], -poselist[:,2],'.-') # flip z axis 
            savename = (selecteddir+'/'+ff).replace('txt','png')
            plt.savefig(savename)

    if selected_path_folder is not None: # interaction mode

            savename = selecteddir+'/P'+str(selected_ind).zfill(3)+'.txt'

    if selected_path_folder is not None:
        return selected_path_folder
    return 'ros_path_'+timestr

def rospath2positionlist(posesampler, trajdir, outdir, outfolder_suffix=''):

    trajfiles = listdir(trajdir)
    trajfiles = [ff for ff in trajfiles if ff[-3:]=='txt']
    trajfiles.sort()
    # output dir
    if outfolder_suffix != '': 
        dirname = outdir+'/position_'+outfolder_suffix
    elif isdir(outdir+'/position'):
        timestr = time.strftime('%m%d_%H%M%S',time.localtime())
        dirname = outdir+'/position_'+timestr
    else:
        dirname = outdir+'/position'

    if not exists(dirname):
        mkdir(dirname)

    for k,trajfile in enumerate(trajfiles):
        filename = dirname+'/P'+str(k).zfill(3)+'.txt'
        figname = dirname+'/P'+str(k).zfill(3)+'.png'
        traj_np = np.loadtxt(trajdir + '/' + trajfile)
        positions = posesampler.sample_poses(traj_np.tolist(), visfilename=figname)
        np.savetxt(filename, positions)


def roadmap_path_sample(args):
    # read args
    EnvDir = args.environment_dir
    GraphFilename = args.graph_filename

    SampleGraph = args.sample_graph # nodes and edges
    SamplePath = args.sample_path # loopy path
    SamplePosition = args.sample_position # random linear velociy 
    PruneGraphNodeNum = args.prune_node_num

    Nodenum = args.node_num
    Edgenum = args.edge_num
    TrajlenMinThresh = args.min_dist_thresh
    TrajlenMaxThresh = args.max_dist_thresh
    MaxFailureNum = args.max_failure_num
    MinX = args.node_range_xmin
    MaxX = args.node_range_xmax
    MinY = args.node_range_ymin
    MaxY = args.node_range_ymax
    MinZ = args.node_range_zmin
    MaxZ = args.node_range_zmax

    MaxDist = args.dist_max
    MinDist = args.dist_min
    MaxAcc = args.acc_max
    MaxStep = args.step_max

    pathsampler = None
    graphfilename = join(EnvDir, 'OccMap', GraphFilename)

    if SampleGraph:
        from PathGraph import PathGraph
        if isfile(graphfilename): # load graph file from file
            pathgraph = PathGraph(graphfilename, prunenode=PruneGraphNodeNum) 
        else:
            pathgraph = PathGraph() 

        from NodePathSampler import NodePathSampler
        pathsampler = NodePathSampler()
        outgraphfilename = 'node{}_edge{}_len{}_{}.graph'.format(Nodenum, Edgenum, TrajlenMinThresh, TrajlenMaxThresh)
        outputgraphfilename = join(EnvDir, 'OccMap', outgraphfilename)
        sample_nodes_edges(pathgraph, pathsampler, outputgraphfilename, Nodenum, 
                            MinX, MaxX, MinY, MaxY, MinZ, MaxZ, 
                            Edgenum, TrajlenMinThresh, TrajlenMaxThresh, MaxFailureNum)

    # Pathgraph visualization
    # graphfilename = '/home/wenshan/tmp/maps/carwelding/OccMap/node40_edge10_len20_40.graph'
    # graphfilename = '/home/wenshan/tmp/maps/oldtown_1804/OccMap/node100_edge10_len20_50_sparse.graph'
    # pathgraph = PathGraph(graphfilename) 

    ros_path_folder = args.ros_path_dir
    if SamplePath:
        if not SampleGraph: # separate from the node/edge sampling
            from NodePathSampler import NodePathSampler
            pathsampler = NodePathSampler()
            from PathGraph import PathGraph
            if isfile(graphfilename): # load graph file from file
                pathgraph = PathGraph(graphfilename) 
            else:
                print('SamplePath: No prebuild graph loaded..')

        if args.interactive:
            selected_path_folder = ros_path_folder
            if selected_path_folder=="":
                selected_path_folder = "ros_path_selected"
        else:
            selected_path_folder = None
        SampleCycleMode = args.sample_cycle_mode
        ros_path_folder = load_graph_and_sample_loop(pathsampler, pathgraph, EnvDir, 
                                                    SampleCycleMode, outfolder_suffix = args.ros_path_suffix, 
                                                    selected_path_folder = selected_path_folder, cycle_minnodes=args.cycle_min_nodes)

    if SamplePosition:
        posesampler = RandomPoseReSampler(DistMax = MaxDist, DistMin = MinDist, AccMax = MaxAcc, StepMax = MaxStep)
        rospath2positionlist(posesampler, join(EnvDir, ros_path_folder), EnvDir, outfolder_suffix=args.position_path_suffix)




if __name__ == '__main__':

    args = get_args()

    roadmap_path_sample(args)