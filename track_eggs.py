import os
import sys
import torch
from pathlib import Path
from tqdm import tqdm
import numpy as np
import json
import argparse
import cv2 as cv
import logging
import matplotlib.pyplot as plt
from matplotlib.patches import Circle
from timeit import default_timer as timer

from scipy.spatial.distance import cdist
from scipy.optimize import linear_sum_assignment


from run_params import get_run_params

class egg_track:
    def __init__(self, id, fnum, score, x, y):        
        self.id = id
        self.fnum = [fnum]
        self.score = [score]
        self.x = [x]
        self.y = [y]
    
    def add(self, fnum, score, x, y):
        self.fnum.append(fnum)
        self.score.append(score)
        self.x.append(x)
        self.y.append(y)

def next_frame( egg_detections ):
    if len(egg_detections.indices)==0:
        return None, egg_detections
    else:
        ind = iter(egg_detections.indices)
        fnum = next(ind)
        n=1
        try:
            while frame==next(ind):
                n += 1
        except StopIteration:
            pass
    frame = {'fnum':fnum,
             'score':egg_detections.peak_vals[:n],
             'x':egg_detections.x[:n],
             'y':egg_detections.y[:n]
             }
    egg_detections.peak_vals = egg_detections.peak_vals[n:]
    egg_detections.indices = egg_detections.indices[n:]
    egg_detections.x = egg_detections.x[n:]
    egg_detections.y = egg_detections.y[n:]
    return frame, egg_detections

def kill_old_tracks(tracks_current, fnum, lost_sec):
    keep, done = [], []
    for track in tracks_current:
        if fnum-track.fnum[-1] > lost_sec:
            done.append(track)
        else:
            keep.append(track)
    return keep, done

def track_eggs( eggs_detections, params, big_value=1e10 ):
    tracks_current = []
    tracks_done = []
    id = 0
    while len(eggs_detections.indices):
        # Get next frame with tracks:
        frame, eggs_detections = next_frame( eggs_detections )
        # Now that we know the frame number, remove old tracks:
        tracks_current, old = kill_old_tracks(tracks_current, frame.fnum, params.lost_sec)
        tracks_done = tracks_done + old
        if len(tracks_current):
            # Get coordinates of tracked eggs
            xy = np.array(list(map(lambda tr: [tr.x[-1],tr.y[-1]], tracks_current)))
            # Get coordinates of new detections:
            xy_new = np.array([frame['x'],frame['y']]).T
            all_dists = cdist( xy, xy_new )
            # All dists greater than max should be excluded as matches 
            # Thus make their distances infeasible (otherwise will include these matches)
            all_dists[all_dists>params.radius] = big_value
            row_ind, col_ind = linear_sum_assignment(all_dists)
            match_dists = all_dists[row_ind, col_ind]   
            good = match_dists <= params.radius
            for row, col in zip(row_ind[good], col_ind[good]):
                tracks_current[row].add( frame['fnum'], frame['scores'][col], frame['x'][col], frame['y'][col] )         
            rest = np.arange(len(frame['scores']))[1-good]
            for nt in rest:
                tracks_current.append( egg_track(id,frame['fnum'],frame['scores'][nt], frame['x'][nt], frame['y'][nt]))
                id += 1
        else:
            for nt in range(len(frame['scores'])):
                tracks_current.append( egg_track(id,frame['fnum'],frame['scores'][nt], frame['x'][nt], frame['y'][nt]))
                id += 1
    return tracks_current, tracks_done

class track_params:
    def __init__(self,
                  radius = 20,   # radius to match
                  lost_sec = 20,     # Time to lose a track
                 ):
        self.radius = radius
        self.lost_sec = lost_sec


def track_detections(args, prefix):

    logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    params = get_run_params(args.run)
    assert params.target_downscale==4, f'Assumes target_downscale is 4'
    # Load best checkpoint for current run:
    params.load_opt = 'best'
    params.load_run = None

    run_dir = os.path.join(os.path.dirname(__file__), params.output_dir, f'{params.run:03d}')
    vid_dir = os.path.join(run_dir,'video')

    search = prefix + '*.json'
    detections = list(Path(vid_dir).rglob(search))
    detections.sort()

    print(f'Found {len(detections)} files of type: {search}')

    for path in detections:
        with open(str(path),'r') as f:
            eggs_detections = json.load(f)  # These are defined in run_video
            #vid_detect = {  'video': self.reader.name,
            #            'samples_per_sec': self.reader.samples_per_sec,
            #            'nskip': self.reader.skip,
            #            'peak_vals': peak_vals,
            #            'indices': indices,
            #            'x': x,
            #            'y': y,
            #          }
        track_eggs(eggs_detections)

if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Train the UNet on images and target masks')
    parser.add_argument('run', type=int, help='Run')
    parser.add_argument('--folder', type=str, default='/mnt/home/dmorris/Data/hens/Hens_2021',  help='Folder for videos')    
    parser.add_argument('--prefix', type=str, nargs='+', default='',  help='search prefix')     
    args = parser.parse_args()

    for prefix in args.prefix:
        track_detections(args, prefix)

