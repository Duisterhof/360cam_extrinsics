import numpy as np
import matplotlib.pyplot as plt
import random 
from sympy import *
import cv2 

class Essential_Optimizer:
    def __init__(self,pts1,pts2):

        self.pts1 = pts1.T
        self.pts2 = pts2.T
        self.num_points = self.pts1.shape[0]

    def compute_system(self):
        
        pts1 = self.pts1
        pts2 = self.pts2

        self.A = np.zeros((self.num_points,8))
        self.A[:,0] = np.multiply(pts2[:,0],pts1[:,0])
        self.A[:,1] = np.multiply(pts2[:,0],pts1[:,1])
        self.A[:,2] = pts2[:,0]
        self.A[:,3] = np.multiply(pts2[:,1],pts1[:,0])
        self.A[:,4] = np.multiply(pts2[:,1],pts1[:,1])
        self.A[:,5] = pts2[:,1]
        self.A[:,6] = pts1[:,0]
        self.A[:,7] = pts1[:,1]
        self.b = -np.ones_like(pts2[:,0])
        self.all_idxs = list(range(self.num_points))

    def compute_E_RANSAC_8(self):
        random.shuffle(self.all_idxs)
        rand_idxs = self.all_idxs[:8]
        self.local_A = self.A[rand_idxs,:]
        self.local_b = self.b[rand_idxs]

        solution = np.linalg.lstsq(self.local_A,self.local_b)
        solution_vector = np.append(solution[0],[1.0])
        self.E = solution_vector.reshape((3,3))


    def compute_inliers(self,tol):
        # compute lines for all points using fundamental matrix
        lines = self.E.dot(self.pts1.T)
        distances = []
        for i, line in enumerate(lines.T):
            # print(line.shape)
            # print(self.pts2[i].shape)
            distances.append(np.abs(line.T.dot(self.pts2[i]))/(np.sqrt(line[0]**2+line[1]**2)))
        
        distances = np.array(distances)
        inliers = distances[distances<tol]
        self.latest_inliers = inliers.shape[0]/(self.num_points)
        self.inlier_idxs = np.where(distances<tol)[0]

    def ransac_loop(self,tol,iterations,method='8'):
        self.best_inliers = 0
        self.best_F = np.zeros((3,3))
        self.inliers = []
        for i in range(iterations):
            if method == '8':
                self.compute_E_RANSAC_8()

            self.compute_inliers(tol)
            if (self.latest_inliers > self.best_inliers):
                self.best_inliers = self.latest_inliers
                self.best_E = self.E
        
            self.inliers.append(self.best_inliers)
    
        self.E = self.best_E 

    def full_lstsq(self):
        solution = np.linalg.lstsq(self.A,self.b)
        solution_vector = np.append(solution[0],[1.0])
        self.E = solution_vector.reshape((3,3))
    
    def extract_motion(self):
        w,u, vt = cv2.SVDecomp(self.E)
        if np.linalg.det(u) < 0:
            u *= -1.0
        if np.linalg.det(vt) < 0:
            vt *= -1.0 
        W = np.array([[0,-1,0],[1,0,0],[0,0,1]])
        self.R = u@W@vt 
        self.t = u[:,2]


# train = Get_E('data/q1b/toytrain/toytrain_corresp_raw.npz',['data/q1b/toytrain/image_1.jpg','data/q1b/toytrain/image_2.jpg'],'data/q1b/toytrain/intrinsic_matrices_toytrain.npz')
# train.compute_system()
# train.ransac_loop(1.2,10000,method='7')
# train.show_epipolar()