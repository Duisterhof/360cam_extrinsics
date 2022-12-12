import numpy as np
import networkx as nx
from networkx.algorithms import *
import pickle
import matplotlib.pyplot as plt

def node_distance(init, goal):
    diff = np.array([goal.position.x-init.position.x,
                     goal.position.y-init.position.y,
                     goal.position.z-init.position.z])
    return np.linalg.norm(diff, axis=0)

class PathGraph(object):
    '''
    A wrapper of networkx
    A list of geometry_msgs.msg.Pose is stored in the edge
    '''
    def __init__(self, graphfilename=None, prunenode=0):
        if graphfilename is None:
            self.graph = nx.Graph()
        else:
            self.graph = pickle.load(open(graphfilename)) 
            if prunenode>0:
                self.prune_graph(MinNodeInGraph=prunenode)

    def edge_statistics(self):
        nodenum = self.graph.number_of_nodes()
        statis = np.zeros(nodenum)
        max_edgenum = 0
        for node in self.graph.nodes:
            edgenum = len(self.graph.edges(node))
            ind = edgenum -1 
            statis[ind] += 1
            if max_edgenum < edgenum:
                max_edgenum = edgenum
        return statis[:max_edgenum]

    def visualize_poslist(self, poselist, savename=''):
        # plt.figure(figsize=(12,6))
        plt.subplot(121)
        plt.plot(poselist[:,0], -poselist[:,1],'.-') # flip y axis to make it top-down view
        plt.subplot(122)
        plt.plot(poselist[:,0], -poselist[:,2],'.-') # flip z axis 

        if savename != '':
            plt.savefig(savename)
        else:
            plt.show()

    def prune_graph(self, MinNodeInGraph=10, MinEdgeEachNode=2):
        '''
        delete sepatated small graph
        delete nodes with less edges
        '''

        compnum = number_connected_components(self.graph)
        complist = list(connected_components(self.graph))
        complens = [len(comp) for comp in complist]
        # accending order w.r.t nodes number
        comp_sorted_ind = sorted(range(compnum), key=lambda k: complens[k])

        print ('{} components, length: {}'.format(compnum, complens))
        print ('edge distribution: {}'.format(self.edge_statistics()))

        for ind in comp_sorted_ind:
            nodenum = len(complist[ind])
            if nodenum < MinNodeInGraph:
                # delete the nodes from the graph
                for node in complist[ind]:
                    self.graph.remove_node(node)
                print 'delete node:', nodenum
            else:
                # delete the nodes with too less edges
                for node in complist[ind]:
                    edgenum = len(self.graph.edges(node))
                    if edgenum < MinEdgeEachNode:
                        self.graph.remove_node(node)
            print ('edge distribution: {}'.format(self.edge_statistics()))

    def delete_path(self, path):
        for node in path:
            self.graph.remove_node(node)

    def sample_cycle(self, mode=0, minnodes=5):
        '''
        mode 0: random
             1: min len
             2: max len
        '''
        cyclelist = cycle_basis(self.graph)
        cyclenum = len(cyclelist)
        print ('    cycle number: {}'.format(cyclenum))
        # filter the cycle length
        if cyclenum == 0:
            return None
        if mode==0:
            cyclelist = [cc for cc in cyclelist if len(cc)>=minnodes]
            cyclenum = len(cyclelist)
            print ('    filtered w/ minnodes {}, cycle number: {}'.format(minnodes, cyclenum))
            if cyclenum == 0:
                return None
            sampleind = np.random.randint(cyclenum)
        else:
            cyclelens = [len(cc) for cc in cyclelist]
            cycle_sorted_ind = sorted(range(cyclenum), key=lambda k: cyclelens[k])
            if mode==1:
                sampleind = cycle_sorted_ind[0]
            else: 
                sampleind = cycle_sorted_ind[-1]
        return cyclelist[sampleind]

    def nodes2poselist(self, node1, node2, thresh = 0.1):
        '''
        Return the list of Poses stored in the edge between node1 and node2
        Check the threshold to make sure the start and end nodes are aligned
        '''
        if self.graph.has_edge(node1, node2):
            path = self.graph.get_edge_data(node1, node2)['path']
            if node_distance(node1, path[0]) < thresh and node_distance(node2, path[-1]) < thresh:
                return path
            elif node_distance(node1, path[-1]) < thresh and node_distance(node2, path[0]) < thresh:
                return path[::-1]
            else:
                print ('Path between nodes are not aligned..')
                return []
        else:
            return []

    def cycle2poselist(self, cycle):
        '''
        Return the Poses stored in the edges of a cycle
        '''
        poselist = []
        cyclelen = len(cycle)
        for k, node1 in enumerate(cycle):
            node2 = cycle[(k+1)%cyclelen]
            path = self.nodes2poselist(node1, node2)
            poselist.extend(path[:-1])
        # add the start pt to the end pt to complete the loop
        poselist.append(poselist[0])
        return poselist

    def node_dist_statistics(self, poses):
        pathlen = len(poses)
        distlist = [0] * (pathlen-1)

        for k in range(pathlen-1):
            distlist[k] = node_distance(poses[k], poses[k+1])
        return distlist

    def poselist2positionlist(self, poselist):
        '''
        convert ros Pose to list [x,y,z]
        '''
        positionlist = []
        for pose in poselist:
            positionlist.append([pose.position.x, pose.position.y, pose.position.z])
        return positionlist

if __name__ == '__main__':
    graphfilename = '/home/wenshan/tmp/maps/carwelding/OccMap/node50_edge8_len10_30.graph'
    PathGraph(graphfilename)