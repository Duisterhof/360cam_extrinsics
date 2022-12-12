import cv2 
import glob
import matplotlib.pyplot as plt 
import numpy as np

class FeatureExtractor:
    def __init__(self,cam0_path,cam1_path):
        self.cam0_imgpaths = glob.glob(cam0_path+'*')
        self.cam1_imgpaths = glob.glob(cam1_path+'*')

        self.cam0_imgpaths.sort()
        self.cam1_imgpaths.sort()

        assert len(self.cam0_imgpaths) == len(self.cam1_imgpaths)

        self.n_imgs = len(self.cam0_imgpaths)

        self.imgscam0 = []
        self.imgscam1 = []

        for i, cam0_path in enumerate(self.cam0_imgpaths):
            self.imgscam0.append(cv2.imread(cam0_path,cv2.IMREAD_GRAYSCALE))
            self.imgscam1.append(cv2.imread(self.cam1_imgpaths[i],cv2.IMREAD_GRAYSCALE))

    def extract_features(self,descriptor = 'SIFT'):

        self.cam0_coords = []
        self.cam1_coords = []

        if (descriptor == 'SIFT'):
            self.imgs3 = []
            for i in range(self.n_imgs):
                img0 = self.imgscam0[i]
                img1 = self.imgscam1[i]

                sift = cv2.SIFT_create()
                kp1, des1 = sift.detectAndCompute(img0,None)
                kp2, des2 = sift.detectAndCompute(img1,None)

                bf = cv2.BFMatcher()
                matches = bf.knnMatch(des1,des2,k=2)

                good = []
                for m,n in matches:
                    if m.distance < 0.75*n.distance:
                        good.append([m])
                        self.cam0_coords.append(list(kp1[m.queryIdx].pt))
                        self.cam1_coords.append(list(kp2[m.trainIdx].pt))

                # cv.drawMatchesKnn expects list of lists as matches.
                img3 = cv2.drawMatchesKnn(img0,kp1,img1,kp2,good,None,flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
                self.imgs3.append(img3)
    
        self.cam0_coords = np.array(self.cam0_coords).T 
        self.cam1_coords = np.array(self.cam1_coords).T

    def vis_features(self):
        for img3 in self.imgs3:
            plt.imshow(img3),plt.show()