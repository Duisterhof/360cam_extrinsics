import numpy as np
import time
from ImageClient import ImageClient
from os import mkdir
from os.path import isdir
from signal import signal, SIGINT
from sys import exit

envname = 'abandonfactory'

subfolder = 'collected'
metafolder = '/home/wenshan/tmp/maps/'
imgclient = ImageClient(['0'], ['scene'])
positionlist = []



def handler(signal_received, frame):
    # Handle any cleanup here
    assert(isdir(metafolder))
    if not isdir(metafolder+envname):
        mkdir(metafolder+envname)
    outfolder = metafolder+envname+'/'+subfolder
    if not isdir(outfolder):
        mkdir(outfolder)

    timestr = time.strftime('%m%d_%H%M%S',time.localtime())
    filename = outfolder+'/'+timestr+'.txt'

    np.savetxt(filename, positionlist)
    print('Saved {} positions, {}'.format(len(positionlist), filename))
    exit(0)

if __name__ == '__main__':
    # Tell Python to run the handler() function when SIGINT is recieved
    signal(SIGINT, handler)

    print('Running. Press CTRL-C to exit.')
    count = 0
    while True:
        pose = imgclient.getpose()
        position = pose.position
        positionlist.append([position.x_val, position.y_val, position.z_val])
        count += 1
        if count % 100==0:
            print ('Collecting {}...'.format(count))

        time.sleep(0.1)
