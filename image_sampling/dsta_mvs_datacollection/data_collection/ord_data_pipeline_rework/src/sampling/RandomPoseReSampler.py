import numpy as np
import matplotlib.pyplot as plt
import os
if ( not ( "DISPLAY" in os.environ ) ):
    plt.switch_backend('agg')
    print("Environment variable DISPLAY is not present in the system.")
    print("Switch the backend of matplotlib to agg.")

# TODO: Calculate and verify the threshold values
class RandomPoseReSampler(object):
    '''
    sample poses from path with random distance
    '''
    def __init__(self, DistMax = 0.3, DistMin = 0.0, AccMax = 0.1, StepMax = 20):
        self.DistMax = DistMax
        self.DistMin = DistMin
        self.AccMax = AccMax
        self.StepMax = StepMax

    def pos_distance(self, pos1, pos2):
        diff = np.array([pos1[0]-pos2[0],
                         pos1[1]-pos2[1],
                         pos1[2]-pos2[2]])
        return np.linalg.norm(diff, axis=0)

    def pt_interpolate(self, pt1, pt2, loc):
        '''
        loc -> [0, 1)
        '''
        return [pt1[0] * (1-loc) + pt2[0] * loc,
                pt1[1] * (1-loc) + pt2[1] * loc,
                pt1[2] * (1-loc) + pt2[2] * loc]

    def visualize_poselist(self, path1, path2, savename=''):
        path1_np = np.array(path1)
        path2_np = np.array(path2)

        # mirror the y axis
        plt.subplot(121)
        plt.plot(path1_np[:,0], -path1_np[:,1],'o-')
        plt.plot(path2_np[:,0], -path2_np[:,1],'.-')

        plt.subplot(122)
        plt.plot(path1_np[:,0], -path1_np[:,2],'o-')
        plt.plot(path2_np[:,0], -path2_np[:,2],'.-')

        if savename != '':
            plt.savefig(savename)
        else:
            plt.show()

    # TODO: this sample pattern might be problematic
    def sample_poses(self, path, visulize=False, visfilename=''):
        '''
        path is a list of positions [[x0,y0,z0], [x1,y1,z1],...]
        re-sample the poses along the path with random distance
        '''
        pathlen = len(path)
        poselist = [path[0]]
        location = [0 ,1 ,0.0] # start_postion_ind, end_position_ind, percentage -> [0,1)
        dist = (self.DistMin + self.DistMax)/2.0

        complete = False
        while not complete:
            acc = np.random.uniform(-self.AccMax, self.AccMax)
            step = np.random.randint(self.StepMax) + 1 
            # print 'sample: ',acc, step, dist

            for count in range(step):
                # import ipdb;ipdb.set_trace()
                newdist = dist + acc
                # print count, '-' , newdist
                if newdist > self.DistMax or newdist < self.DistMin:
                    break # end this round
                dist = newdist
                segmentlen = self.pos_distance(path[location[0]], path[location[1]])
                # print '  segmentlen', segmentlen
                remainSeg = segmentlen  * (1-location[2])
                if remainSeg > newdist: # sample on current segment
                    location[2] = location[2] + newdist/segmentlen
                    posenew = self.pt_interpolate(path[location[0]], path[location[1]], location[2])
                    poselist.append(posenew)
                    # print '  sample on this segment', location, posenew                    
                else: # move forward to the next pt on the path
                    while(newdist>=remainSeg): # 
                        if location[1]+1 >= pathlen: # sample complete
                            complete = True
                            poselist.append(path[-1])
                            # print '  hit the end'
                            break
                        else:
                            location[0] += 1
                            location[1] += 1
                            newdist -= remainSeg
                            remainSeg = self.pos_distance(path[location[0]], path[location[1]]) 
                            # print '  segmentlen', remainSeg
                            location[2] = 0
                    if complete:
                        break
                    location[2] = newdist / remainSeg
                    posenew = self.pt_interpolate(path[location[0]], path[location[1]], location[2])
                    poselist.append(posenew)
                    # print '  move to next segment', location, posenew

        if visulize:
            self.visualize_poselist(path[0:location[1]], poselist)
        if visfilename!='': # save a figure to the disk
            self.visualize_poselist(path[0:location[1]], poselist, visfilename)

        print 'path len {}, poselist len {}'.format(len(path), len(poselist))
        return poselist
