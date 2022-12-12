import cv2 
import glob
import matplotlib.pyplot as plt 

cam0_path = 'data/cam0/'
cam1_path = 'data/cam1/'


cam0_imgpaths = glob.glob(cam0_path+'*')
cam1_imgpaths = glob.glob(cam1_path+'*')

cam0_imgpaths.sort()
cam1_imgpaths.sort()

assert len(cam0_imgpaths) == len(cam1_imgpaths)

n_imgs = len(cam0_imgpaths)

imgscam0 = []
imgscam1 = []

for i, cam0_path in enumerate(cam0_imgpaths):
    imgscam0.append(cv2.imread(cam0_path,cv2.IMREAD_GRAYSCALE))
    imgscam1.append(cv2.imread(cam1_imgpaths[i],cv2.IMREAD_GRAYSCALE))

for i in range(n_imgs):
    img0 = imgscam0[i]
    img1 = imgscam1[i]

    sift = cv2.SIFT_create()
    kp1, des1 = sift.detectAndCompute(img0,None)
    kp2, des2 = sift.detectAndCompute(img1,None)

    bf = cv2.BFMatcher()
    matches = bf.knnMatch(des1,des2,k=2)

    good = []
    for m,n in matches:
        if m.distance < 0.75*n.distance:
            good.append([m])
    # cv.drawMatchesKnn expects list of lists as matches.
    img3 = cv2.drawMatchesKnn(img0,kp1,img1,kp2,good,None,flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
    plt.imshow(img3),plt.show()