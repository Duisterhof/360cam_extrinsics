# collect images in cv mode
import numpy as np
from math import cos, sin, tanh, pi, atan2, asin
import time

import sys
import random
import argparse

from pyquaternion import Quaternion as Quaternionpy # quaternion multiplication

from transform import Rotation, RotationSpline # quaternion spline
import numpy as np
# import matplotlib.pyplot as plt

np.set_printoptions(precision=3, suppress=True, threshold=10000)

def get_args():

    parser = argparse.ArgumentParser(description='sample_pipeline')

    # image collection - collect_images
    parser.add_argument('--rand-degree', type=int, default=30,
                        help='random angle added to the position when sampling (default: 15)')

    parser.add_argument('--smooth-count', type=int, default=10,
                        help='lengh of smoothed trajectory (default: 10)')

    parser.add_argument('--max-yaw', type=int, default=360,
                        help='yaw threshold (default: 360)')

    parser.add_argument('--min-yaw', type=int, default=-360,
                        help='yaw threshold (default: -360)')

    parser.add_argument('--max-pitch', type=int, default=20,
                        help='yaw threshold (default: 45)')

    parser.add_argument('--min-pitch', type=int, default=-45,
                        help='yaw threshold (default: -45)')

    parser.add_argument('--max-roll', type=int, default=20,
                        help='yaw threshold (default: 90)')

    parser.add_argument('--min-roll', type=int, default=-20,
                        help='yaw threshold (default: -90)')

    args = parser.parse_args()

    return args

class Quaternionr():
    w_val = 0.0
    x_val = 0.0
    y_val = 0.0
    z_val = 0.0

    def __init__(self, x_val = 0.0, y_val = 0.0, z_val = 0.0, w_val = 1.0):
        self.x_val = x_val
        self.y_val = y_val
        self.z_val = z_val
        self.w_val = w_val

    @staticmethod
    def nanQuaternionr():
        return Quaternionr(np.nan, np.nan, np.nan, np.nan)

    def __add__(self, other):
        if type(self) == type(other):
            return Quaternionr( self.x_val+other.x_val, self.y_val+other.y_val, self.z_val+other.z_val, self.w_val+other.w_val )
        else:
            raise TypeError('unsupported operand type(s) for +: %s and %s' % ( str(type(self)), str(type(other))) )

    def __mul__(self, other):
        if type(self) == type(other):
            t, x, y, z = self.w_val, self.x_val, self.y_val, self.z_val
            a, b, c, d = other.w_val, other.x_val, other.y_val, other.z_val
            return Quaternionr( w_val = a*t - b*x - c*y - d*z,
                                x_val = b*t + a*x + d*y - c*z,
                                y_val = c*t + a*y + b*z - d*x,
                                z_val = d*t + z*a + c*x - b*y)
        else:
            raise TypeError('unsupported operand type(s) for *: %s and %s' % ( str(type(self)), str(type(other))) )

    def __truediv__(self, other): 
        if type(other) == type(self): 
            return self * other.inverse()
        elif type(other) in [int, float] + np.sctypes['int'] + np.sctypes['uint'] + np.sctypes['float']:
            return Quaternionr( self.x_val / other, self.y_val / other, self.z_val / other, self.w_val / other)
        else: 
            raise TypeError('unsupported operand type(s) for /: %s and %s' % ( str(type(self)), str(type(other))) )

    def dot(self, other):
        if type(self) == type(other):
            return self.x_val*other.x_val + self.y_val*other.y_val + self.z_val*other.z_val + self.w_val*other.w_val
        else:
            raise TypeError('unsupported operand type(s) for \'dot\': %s and %s' % ( str(type(self)), str(type(other))) )

    def cross(self, other):
        if type(self) == type(other):
            return (self * other - other * self) / 2
        else:
            raise TypeError('unsupported operand type(s) for \'cross\': %s and %s' % ( str(type(self)), str(type(other))) )

    def outer_product(self, other):
        if type(self) == type(other):
            return ( self.inverse()*other - other.inverse()*self ) / 2
        else:
            raise TypeError('unsupported operand type(s) for \'outer_product\': %s and %s' % ( str(type(self)), str(type(other))) )

    def rotate(self, other):
        if type(self) == type(other):
            if other.get_length() == 1:
                return other * self * other.inverse()
            else:
                raise ValueError('length of the other Quaternionr must be 1')
        else:
            raise TypeError('unsupported operand type(s) for \'rotate\': %s and %s' % ( str(type(self)), str(type(other))) )        

    def conjugate(self):
        return Quaternionr(-self.x_val, -self.y_val, -self.z_val, self.w_val)

    def star(self):
        return self.conjugate()

    def inverse(self):
        return self.star() / self.dot(self)

    def sgn(self):
        return self/self.get_length()

    def get_length(self):
        return ( self.x_val**2 + self.y_val**2 + self.z_val**2 + self.w_val**2 )**0.5

    def to_numpy_array(self):
        return np.array([self.x_val, self.y_val, self.z_val, self.w_val], dtype=np.float32)

def to_eularian_angles(q):
    z = q.z_val
    y = q.y_val
    x = q.x_val
    w = q.w_val
    ysqr = y * y

    # roll (x-axis rotation)
    t0 = +2.0 * (w*x + y*z)
    t1 = +1.0 - 2.0*(x*x + ysqr)
    roll = atan2(t0, t1)

    # pitch (y-axis rotation)
    t2 = +2.0 * (w*y - z*x)
    if (t2 > 1.0):
        t2 = 1
    if (t2 < -1.0):
        t2 = -1.0
    pitch = asin(t2)

    # yaw (z-axis rotation)
    t3 = +2.0 * (w*z + x*y)
    t4 = +1.0 - 2.0 * (ysqr + z*z)
    yaw = atan2(t3, t4)

    return (pitch, roll, yaw)

    
def to_quaternion(pitch, roll, yaw):
    t0 = cos(yaw * 0.5)
    t1 = sin(yaw * 0.5)
    t2 = cos(roll * 0.5)
    t3 = sin(roll * 0.5)
    t4 = cos(pitch * 0.5)
    t5 = sin(pitch * 0.5)

    q = Quaternionr()
    q.w_val = t0 * t2 * t4 + t1 * t3 * t5 #w
    q.x_val = t0 * t3 * t4 - t1 * t2 * t5 #x
    q.y_val = t0 * t2 * t5 + t1 * t3 * t4 #y
    q.z_val = t1 * t2 * t4 - t0 * t3 * t5 #z
    return q

class QuaternionSampler(object):
    '''
    Quaternionpy: qyquternion, multiplication and linear interpolation
    Quaternionr: airsim, eular-quaternion translation
    scipy Rotation: spline smoothing
    '''
    def __init__(self, args):
        self.args = args

        self.MaxRandAngle = np.clip(self.args.rand_degree, 0, 90)
        self.MaxRandRad = self.MaxRandAngle * pi / 180.
        self.SmoothCount =  self.args.smooth_count
        self.MaxYaw = min(self.args.max_yaw, 180) * pi / 180.
        self.MinYaw = max(self.args.min_yaw, -180) * pi / 180.
        self.MaxPitch = min(self.args.max_pitch, 90) * pi / 180.
        self.MinPitch = max(self.args.min_pitch, -90) * pi / 180.
        self.MaxRoll = min(self.args.max_roll, 180) * pi / 180.
        self.MinRoll = max(self.args.min_roll, -180) * pi / 180.


    def next_quaternion(self,idx):
        pass

    def init_random_yaw(self):
        randomyaw = np.random.uniform(self.MinYaw, self.MaxYaw)
        print ('Random yaw {}, angle {}'.format(randomyaw, randomyaw*180.0/pi))
        qtn = to_quaternion(0., 0., randomyaw)
        return Quaternionpy(qtn.w_val, qtn.x_val, qtn.y_val, qtn.z_val), randomyaw

    def random_quaternion(self):
        '''
        return Quaternionpy for multiplication
        '''
        theta =  np.random.random()*self.MaxRandRad*2 - self.MaxRandRad
        axi = np.random.random(3)
        axi = axi/np.linalg.norm(axi)
        return Quaternionpy(axis=axi, angle=theta)    

    def clip_quaternion(self, quatpy):
        '''
        Input quatpy: Quaternionpy
        Return new_ori_clip: Quaternionpy
        '''
        quatr = Quaternionr(quatpy.x, quatpy.y, quatpy.z, quatpy.w)
        (pitch, roll, yaw) = to_eularian_angles(quatr)
        pitch_clip = np.clip(pitch, self.MinPitch, self.MaxPitch)
        roll_clip = np.clip(roll, self.MinRoll, self.MaxRoll)
        yaw_clip = np.clip(yaw, self.MinYaw, self.MaxYaw)
        quatr_clip = to_quaternion(pitch_clip, roll_clip, yaw_clip)
        quatpy_clip = Quaternionpy(quatr_clip.w_val, quatr_clip.x_val, quatr_clip.y_val, quatr_clip.z_val)

        return quatpy_clip, (roll_clip, pitch_clip, yaw_clip)


class RandQuaternionSampler(QuaternionSampler):

    def reset(self, posenum):
        self.orientation, _ = self.init_random_yaw()
        self.orilist = []
        self.oriind = self.SmoothCount


    def next_quaternion(self,idx):
        if self.oriind >= self.SmoothCount: # sample a new orientation
            rand_ori = self.random_quaternion()
            new_ori = rand_ori * self.orientation

            quatpy_clip, (roll, pitch, yaw) = self.clip_quaternion(new_ori)

            qtnlist = Quaternionpy.intermediates(self.orientation, quatpy_clip, self.SmoothCount-1, include_endpoints=True)
            self.orientation = quatpy_clip
            self.orilist = list(qtnlist)
            self.oriind = 1
            # print "sampled new", new_ori, ', after clip', self.orientation #, 'list', self.orilist

        next_qtn = self.orilist[self.oriind]
        self.oriind += 1
        # print "  return next", next_qtn
        return Quaternionr(next_qtn.x, next_qtn.y, next_qtn.z, next_qtn.w)

def quatpy2eular(quatpy):
    qqq=Quaternionr(quatpy.x, quatpy.y, quatpy.z, quatpy.w)
    rrr= to_eularian_angles(qqq)
    return np.array(rrr)*180/pi # pitch, roll, yaw

def rpy_diff(rpy1, rpy2, degree=False):
    rpydiff = np.array(rpy1) - np.array(rpy2)
    rpydiff = np.array(rpy1) - np.array(rpy2)
    if degree:
        thresh = 180
    else:
        thresh = pi
    rpydiff[rpydiff>thresh] = rpydiff[rpydiff>thresh] - 2*thresh
    rpydiff[rpydiff<-thresh] = rpydiff[rpydiff<-thresh] + 2*thresh
    return np.abs(rpydiff)

def quatarray2eular(quatarray):
    qqq=Quaternionr(quatarray[0], quatarray[1], quatarray[2], quatarray[3])
    (pitch, roll, yaw)= to_eularian_angles(qqq)
    return np.array([roll, pitch, yaw])*180/pi # roll, pitch, yaw

def array2quatpy(quatarray):
    return Quaternionpy(quatarray[3], quatarray[0], quatarray[1], quatarray[2])

class RandQuaternionSplineSampler(QuaternionSampler):
    '''
    Calculate and smooth the quaternion in reset function
    Assume the trajectory is not super long: a few thousands is reasonable
    '''
    def reset(self, posenum):
        # timestamp for key frames
        times = range(0, posenum + self.SmoothCount, self.SmoothCount)
        keynum = len(times)

        # generate key frame orientations for the whole sequence
        orientation, yaw = self.init_random_yaw()
        # angles = [[0.,0.,yaw]]
        last_angle = [0.,0.,yaw]
        quats = [[orientation.x, orientation.y, orientation.z, orientation.w]]

        # import ipdb;ipdb.set_trace()
        k = 1
        while k < keynum:
            rand_ori = self.random_quaternion()
            orientation = rand_ori * orientation #orientation
            orientation, (roll, pitch, yaw) = self.clip_quaternion(orientation)
            if np.all(rpy_diff([roll, pitch, yaw], last_angle)<self.MaxRandRad):
                last_angle = [roll, pitch, yaw]
                # angles.append([roll, pitch, yaw])
                quats.append([orientation.x, orientation.y, orientation.z, orientation.w])
                k += 1

        rotations_quats = Rotation.from_quat(np.array(quats), normalized=False)
        spline_quats = RotationSpline(times, rotations_quats)
        self.orilist_quats = spline_quats(np.array(range(posenum))).as_quat()
        self.anglist_quats = spline_quats(np.array(range(posenum))).as_euler('ZYX', degrees=True)

        # self.orilist = self.orilist_quats.copy()
        StepThresh = self.MaxRandAngle/self.SmoothCount
        for k in range(len(self.anglist_quats)-1):
            anglediff = rpy_diff(self.anglist_quats[k], self.anglist_quats[k+1], degree=True)
            if np.any(anglediff>StepThresh*1.5):
                print ('{} angle diff above thresh: {}'.format(k, anglediff))
                # a hacking solution - use linear interpolation instead
                quatsind = int(k/self.SmoothCount)
                quatsind_start = quatsind * self.SmoothCount
                quatsind_end = min(quatsind_start + self.SmoothCount, len(self.orilist_quats))
                qtns = Quaternionpy.intermediates(array2quatpy(quats[quatsind]), array2quatpy(quats[quatsind+1]), self.SmoothCount-1, include_endpoints=True)
                qtnlist = []
                for qtn in qtns:
                    qtnlist.append([qtn.x, qtn.y, qtn.z, qtn.w])
                # check again the angle difference
                # it turns out that the slerp results will also violate the degree threshold
                angles=[]
                inthresh = True
                for qtn in qtnlist:
                    angles.append(quatarray2eular(qtn))
                for w in range(len(angles)-1):
                    anglediff2 = rpy_diff(angles[w], angles[w+1], degree=True)
                    # print '  ',anglediff2
                    if np.any(anglediff2 > StepThresh*1.5):
                        inthresh = False
                self.orilist_quats[quatsind_start:quatsind_end] = qtnlist[0:quatsind_end-quatsind_start]
                if not inthresh:
                    print('slerp interpolation still above thresh..')
                    print(anglediff2)
            # else:
            #     print k, anglediff

        # clip the quaternion


        # print("Posenum {}, key frame number {}, spline num {}".format(posenum, keynum, len(self.orilist)))
        return quats # for debugging angles, 

    def next_quaternion(self, idx):
        assert(idx<len(self.orilist_quats))
        quat = self.orilist_quats[idx]
        quat_py = Quaternionpy(quat[3], quat[0], quat[1], quat[2])
        quat_clip, _ = self.clip_quaternion(quat_py)

        return Quaternionr(quat_clip.x, quat_clip.y, quat_clip.z, quat_clip.w)


if __name__ == '__main__':

    args = get_args()
    import matplotlib.pyplot as plt

    # # test the RandQuaternionSampler
    # randquaternionsampler = RandQuaternionSampler(args)
    randquaternionsampler = RandQuaternionSplineSampler(args)
    anglelist = []
    randquaternionsampler.reset(1000)
    for k in range(1000):
        quatr = randquaternionsampler.next_quaternion(k)
        (pitch, roll, yaw) = to_eularian_angles(quatr)
        anglelist.append([roll,pitch, yaw])

    anglelist = np.array(anglelist) * 180.0/pi
    plt.plot(anglelist[:,0],'.-')
    plt.plot(anglelist[:,1],'.-')
    plt.plot(anglelist[:,2],'.-')     
    plt.legend(['roll', 'pitch', 'yaw']) 
    plt.grid()
    plt.show()  

