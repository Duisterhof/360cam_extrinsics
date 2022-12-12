import argparse
from dataclasses import dataclass
from functools import partial
from glob import glob
from multiprocessing import Pool, Queue
import os
import struct
from typing import Any, List, Tuple

import sqlite3

import cv2
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm

@dataclass
class DistanceStats:
    img_id : str
    min_dist : float
    max_dist : float
    mean_dist : float
    std_dist : float
    histogram : List[int]
    bin_edges : List[float]
    __tablename__ = 'depthstats'
    __tablecolumns__ = [
        ('img_id', 'TEXT PRIMARY KEY'),
        ('min_dist', 'REAL'),
        ('max_dist', 'REAL'),
        ('mean_dist', 'REAL'),
        ('std_dist', 'REAL'),
        ('histogram', 'BLOB'),
        ('bin_edges', 'BLOB')
    ]

    def to_db(self) -> Tuple:
        return (
            self.img_id,
            self.min_dist,
            self.max_dist,
            self.mean_dist,
            self.std_dist,
            struct.pack(f'{len(self.histogram)}d', *self.histogram),
            struct.pack(f'{len(self.bin_edges)}d', *self.bin_edges)
        )

    @staticmethod
    def from_db(d:Tuple):
        n_hist = len(d[5])//8 # number of histogram values
        n_bins = len(d[6])//8 # number of bin edge values

        return DistanceStats(
            img_id = d[0],
            min_dist = d[1],
            max_dist = d[2],
            mean_dist = d[3],
            std_dist = d[4],
            histogram = list(struct.unpack(f'{n_hist}d', d[5])),
            bin_edges = list(struct.unpack(f'{n_bins}d', d[6])),
        )

def calc_dist_stats(depth:Any, min_dist:float=0.0, max_dist:float=100.0, n_bins:int=10, img_id:str='') -> DistanceStats:
    '''Calculate depth statistics.

    If image is detected to be invalid, i.e. `min(depth) <= 0` indicating
    camera clips the scene, the returned statistics are dummy values,
    `min_depth = 0`.
    
    Parameters
    ----------
    depth : ndarray, float32
        Single channel depth image.
    min_dist : float
        Minimum distance to clip image.
    max_dist : float
        Maximum distance to clip image.
    n_bins:
        Number of bins for histogram computation.
    img_id : str
        Reference to image.

    Returns
    -------
    stats : DepthStats
        Calculated statistics.
    '''
    clipped = np.clip(depth, min_dist, max_dist)

    min_ = np.min(depth)

    if min_ <= 0.0:
        # image is invalid, camera clips the scene

        # return dummy values
        return DistanceStats(
            img_id=img_id,
            min_dist=0,
            max_dist=0,
            mean_dist=0,
            std_dist=0,
            histogram=[0],
            bin_edges=[0]
        )
    else:
        min_ = np.min(clipped)
        max_ = np.max(clipped)
        mean = np.mean(depth)
        std = np.std(depth)
        inv = np.divide(1.0, clipped)
        inv_min = np.min(inv)
        inv_max = np.max(inv)
        bins = np.linspace(inv_min, inv_max, n_bins, endpoint=True)

        hist, bin_edges = np.histogram(inv, bins, (inv_min, inv_max))

        stats = DistanceStats(
            img_id=img_id,
            min_dist=float(min_),
            max_dist=float(max_),
            mean_dist=float(mean),
            std_dist=float(std),
            histogram=hist.astype(float).tolist(),
            bin_edges=bin_edges.astype(float).tolist()
        )

        return stats

def render_dist_stats(
        fn:str,
        stats:DistanceStats,
        figsize=(6.4, 4.8),
        dpi:float=None,
        backend:str='png',
        transparent:bool=False
    ) -> None:
    ''' Render statistics to image.

    Parameters
    ----------
    fn : str
        Filename to save image to.
    stats: ImageStats
        Statistics dictionary.
    figsize : Tuple[float, float]
        Figure dimension (width, height) in inches.
    dpi : float
        Dots per inch.
    backend : str
        Rendering backend
    transparent : bool
        Render with transparent background.
    '''
    fig = plt.figure(figsize=figsize, dpi=dpi)
    plt.bar(stats.bin_edges[:-1], stats.histogram)
    plt.title('{}\nMin: {:.2f}, Max: {:.2f}, Mean: {:.2f}, Std: {:.2f}'.format(stats.img_id, stats.min_depth, stats.max_depth, stats.mean_depth, stats.std_depth))
    plt.yticks([])
    plt.savefig(fn, dpi=dpi, backend=backend, transparent=transparent)
    plt.close(fig)


@dataclass
class MonoStats:
    img_id : str
    min:float
    max:float
    mean:float
    std:float
    __tablename__='monostats'
    __tablecolumns__ = [
        ('img_id', 'TEXT PRIMARY KEY'),
        ('min', 'REAL'),
        ('max', 'REAL'),
        ('mean', 'REAL'),
        ('std', 'REAL')
    ]

    def to_db(self) -> Tuple:
        return (
            self.img_id,
            self.min,
            self.max,
            self.mean,
            self.std
        )

    def from_db(d:Tuple):
        return MonoStats(
            img_id=d[0],
            min=d[1],
            max=d[2],
            mean=d[3],
            std=d[4]
        )

def calc_mono_stats(img:Any, img_id:str='') -> MonoStats:
    '''Calculate stats on grayscale image.

    Parameters
    ----------
    img : ndarray, unint8
        Single channel grayscale image
    img_id : str
        Reference to image.

    Returns
    -------
    stats : MonoStats
        Calculated statistics.
    '''

    img = img.astype(float)

    return MonoStats(
        img_id=img_id,
        min=float(np.min(img)),
        max=float(np.max(img)),
        mean=float(np.mean(img)),
        std=float(np.std(img)),
    )


class DBManager:
    def __init__(self, db_fn:str):
        self.con = sqlite3.connect(db_fn)

    def register_class(self, cls_):
        self.con.execute('CREATE TABLE IF NOT EXISTS {}({})'.format(
            cls_.__tablename__,
            ', '.join(f'{col_name} {col_type}'for col_name, col_type in cls_.__tablecolumns__)
        ))

    def insert(self, item):
        data = item.to_db()
        try:
            self.con.execute(f'INSERT INTO {item.__tablename__} VALUES({",".join("?"*len(data))})', data)
        except sqlite3.IntegrityError:
            print(f'Data for image {item.img_id} already exists in database. Skipping.')

        self.con.commit()

def id_from_filename(fn:str) -> str:
    '''Create image ID from filename.'''
    d, f = os.path.split(fn)
    d, cam = os.path.split(d)
    d, traj = os.path.split(d)
    d, env = os.path.split(d)
    return '/'.join([env, traj, cam, f])

def dist_stat_task(args:argparse.Namespace, fn:str) -> DistanceStats:
    label = id_from_filename(fn)

    depth = np.squeeze(cv2.imread(fn, cv2.IMREAD_UNCHANGED).view('<f4'), axis=-1)
    return calc_dist_stats(depth, min_dist=args.min_dist, max_dist=args.max_dist, n_bins=args.num_bins, img_id=label)

def mono_stat_task(args:argparse.Namespace, fn:str) -> MonoStats:
    label = id_from_filename(fn)

    img = cv2.imread(fn, cv2.IMREAD_GRAYSCALE)
    return calc_mono_stats(img, img_id=label)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Calulate image statistics.')
    parser.add_argument('-n', dest='num_processes', type=int, required=False, default=4, help='Number of processes to use.')
    parser.add_argument('--bins', dest='num_bins', type=int, required=False, default=10, help='Number of bins for depth histogram.')
    parser.add_argument('--mindist', dest='min_dist', type=float, required=False, default=0.5, help='Min distance for depth clipping.')
    parser.add_argument('--maxdist', dest='max_dist', type=float, required=False, default=100.0, help='Max distance for depth clipping.')
    parser.add_argument('path', type=str, help='Path to dataset directory.')
    parser.add_argument('outdb', type=str, help='Path to output database.')
    args = parser.parse_args()

    worker_pool = Pool(args.num_processes)
    queue = Queue()

    dbman = DBManager(args.outdb)
    dbman.register_class(DistanceStats)
    dbman.register_class(MonoStats)

    # Dataset directory is of the form:
    # {dir_name}/{env_name}/P{run_number}/{rig or cam_number}/{frame_number}_CubeScene.png
    rgb_fns = glob(os.path.join(args.path, '*', '*', '*', '*_CubeScene.png'))
    depth_fns = glob(os.path.join(args.path, '*', '*', '*', '*_CubeDistance.png'))

    print('Processing depth statistics...')
    for stats in tqdm(worker_pool.imap_unordered(partial(dist_stat_task, args), depth_fns, chunksize=1), total=len(depth_fns)):
        dbman.insert(stats)

    print('Processing mono statistics...')
    for stats in tqdm(worker_pool.imap_unordered(partial(mono_stat_task, args), rgb_fns, chunksize=1), total=len(rgb_fns)):
        dbman.insert(stats)

    print('Done.')