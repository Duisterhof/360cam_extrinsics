import numpy as np
import matplotlib.pyplot as plt
import random 
from sympy import *

class Problem:
    def __init__(self,path_correnspondences,paths_imgs,K_path):
        self.correspondences = np.load(path_correnspondences)
        self.imgs = []
        for img_path in paths_imgs:
            self.imgs.append(plt.imread(img_path))
        self.K = np.load(K_path)
        self.K1 = self.K['K1']
        self.K2 = self.K['K2']

        self.pts1 = self.correspondences['pts1']
        self.pts2 = self.correspondences['pts2']
        self.pts1 = np.hstack((self.pts1,np.ones_like(self.pts1[:,0]).reshape((-1,1))))
        self.pts2 = np.hstack((self.pts2,np.ones_like(self.pts2[:,0]).reshape((-1,1))))
        self.pts1_raw = self.pts1
        self.pts2_raw = self.pts2

    def compute_system(self,num_points = -1):
        
        pts1 = self.pts1
        pts2 = self.pts2
        self.num_points = pts1.shape[0]
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
    def normalize_points(self):
        self.avg_1 = np.mean(self.pts1,axis=0)
        self.avg_2 = np.mean(self.pts2,axis=0)

        self.all_avg = 0.5*self.avg_1  + 0.5*self.avg_2

        self.dist_mean_1 = np.mean(np.linalg.norm(self.pts1-self.all_avg,axis=1),axis=0)
        self.dist_mean_2 = np.mean(np.linalg.norm(self.pts2-self.all_avg,axis=1),axis=0)
        
        s_1 = np.sqrt(2)/(self.dist_mean_1)
        s_2 = np.sqrt(2)/(self.dist_mean_2)

        #  make points homogeneous
        self.pts1 = np.hstack((self.pts1,np.ones_like(self.pts1[:,0]).reshape((-1,1))))
        self.pts2 = np.hstack((self.pts2,np.ones_like(self.pts2[:,0]).reshape((-1,1))))

        #perform normalizing 
        self.T_1 = np.array([[s_1,0,-s_1*self.all_avg[0]],[0,s_1,-s_1*self.all_avg[1]],[0,0,1.0]])
        self.T_2 = np.array([[s_2,0,-s_2*self.all_avg[0]],[0,s_2,-s_2*self.all_avg[1]],[0,0,1.0]])
    
        # self.pts1 = (self.T_1.dot(self.pts1.T)).T
        # self.pts2 = (self.T_2.dot(self.pts2.T)).T

    def compute_F_RANSAC_8(self):
        random.shuffle(self.all_idxs)
        rand_idxs = self.all_idxs[:8]
        self.local_A = self.A[rand_idxs,:]
        self.local_b = self.b[rand_idxs]

        solution = np.linalg.lstsq(self.local_A,self.local_b)
        solution_vector = np.append(solution[0],[1.0])
        self.F = solution_vector.reshape((3,3))


    def compute_F_RANSAC_7(self):
        random.shuffle(self.all_idxs)
        rand_idxs = self.all_idxs[:7]
        self.local_A = self.A[rand_idxs,:]
        self.local_b = self.b[rand_idxs]

        self.local_A = np.hstack((self.local_A,-self.local_b.reshape((-1,1))))

        u, s, vh = np.linalg.svd(self.local_A)

        f1 = vh[-2]
        f2 = vh[-1]

        F1 = Matrix(f1.reshape((3,3)))
        F2 = Matrix(f2.reshape((3,3)))
        alpha = symbols("alpha")

        F_all = F1*alpha + (1-alpha)*F2
        polynomial = F_all.det()

        # derive the equation from slide 21, L11
        coeff = [polynomial.coeff(alpha,3),polynomial.coeff(alpha,2),polynomial.coeff(alpha,1),polynomial.coeff(alpha)]
        solutions = np.roots(coeff)
        solutions_img = [solution.imag for solution in  solutions ]
        best_solution_idx = np.argmin(solutions_img)
        self.F = solutions[best_solution_idx].real*(f1.reshape((3,3))) + (1-solutions[best_solution_idx].real)*(f2.reshape((3,3)))



    def compute_inliers(self,tol):
        # compute lines for all points using fundamental matrix
        lines = self.F.dot(self.pts1.T)
        distances = []
        for i, line in enumerate(lines.T):
            # print(line.shape)
            # print(self.pts2[i].shape)
            distances.append(np.abs(line.T.dot(self.pts2[i]))/(np.sqrt(line[0]**2+line[1]**2)))
        
        distances = np.array(distances)
        inliers = distances[distances<tol]
        self.latest_inliers = inliers.shape[0]/(self.num_points)


    def ransac_loop(self,tol,iterations,method='8'):
        self.best_inliers = 0
        self.best_F = np.zeros((3,3))
        self.inliers = []
        for i in range(iterations):
            if method == '8':
                self.compute_F_RANSAC_8()
            elif method == '7':
                self.compute_F_RANSAC_7()

            self.compute_inliers(tol)
            if (self.latest_inliers > self.best_inliers):
                self.best_inliers = self.latest_inliers
                self.best_F = self.F
                print(self.best_inliers)
            self.inliers.append(self.best_inliers)

    def compute_E(self):
        self.E = self.K2.T.dot(self.F.dot(self.K1))
        print("E is : ")
        print(self.E)

    def show_epipolar(self):
        f, (ax1,ax2) = plt.subplots(nrows=1,ncols=2)
        ax1.imshow(self.imgs[0])
        
        ax2.imshow(self.imgs[1])
        x_s = np.linspace(0,self.imgs[1].shape[1],100)

        num_samples = 10
        for i in range(num_samples):
            sample_idx = random.randint(0,self.num_points)
            ax1.scatter(self.pts1_raw[sample_idx,0],self.pts1_raw[sample_idx,1],color='red',s=5)

            l = self.best_F.dot(self.pts1[sample_idx,:])
            y_s = (-np.multiply(x_s,l[0]) - np.ones_like(x_s)*l[-1])/l[1]
            ax2.plot(x_s,y_s)

        ax2.set_xlim((0,self.imgs[1].shape[1]))
        ax2.set_ylim((self.imgs[1].shape[0],0))

        # l = self.F.dot(self.pts2[0,:])
        # y_s = -np.multiply(x_s,l[0]) - np.ones_like(x_s)*l[-1]*self.pts2[0,-1]
        # ax2.plot(x_s,y_s)

        plt.show()

        plt.plot(self.inliers)
        plt.xlabel('Iterations')
        plt.ylabel('Inlier Ratio')
        plt.grid()
        plt.show()


train = Problem('data/q1b/toytrain/toytrain_corresp_raw.npz',['data/q1b/toytrain/image_1.jpg','data/q1b/toytrain/image_2.jpg'],'data/q1b/toytrain/intrinsic_matrices_toytrain.npz')
train.compute_system()
train.ransac_loop(1.2,10000,method='7')
train.show_epipolar()