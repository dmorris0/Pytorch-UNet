''' Track eggs on video detections (made by run_video.py)

    Example usage on videos from camera 4 from 07/28/2020:
      python track_eggs.py 54 --prefix ch4_0728 --minseq 5 --minlen 2

    Note: the video writing option uses OpenCV. This will need to be replaced with 
    another video writer (similar to run_video.py)
    
    Note: to write videos requires PyAV.  Install it with:
      conda install av -c conda-forge
'''
import os
import sys
from pathlib import Path
from tqdm import tqdm
import numpy as np
import json
import argparse
import matplotlib.pyplot as plt
from matplotlib.patches import Circle
from timeit import default_timer as timer
from time import sleep

from scipy.spatial.distance import cdist
from scipy.optimize import linear_sum_assignment

from run_video import VideoReader
from run_params import get_run_params

import torch
import torchvision
torchvision.disable_beta_transforms_warning()
from torchvision.io import write_video


class egg_track:
    def __init__(self, id, fnum, score, x, y, minseq=5):        
        self.id = id
        self.fnum = [fnum]
        self.score = [score]
        self.x = [x]
        self.y = [y]
        self.minseq = minseq
        self.nseq = 1
        self.valid_start = self.nseq >= self.minseq 
            
    def add(self, fnum, score, x, y):
        if fnum == self.fnum[-1]+1:
            self.nseq += 1
        else:
            self.nseq = 1
        self.fnum.append(fnum)
        self.score.append(score)
        self.x.append(x)
        self.y.append(y)
        if not self.valid_start:
            self.valid_start = self.nseq >= self.minseq
 

    def plot(self, ax, vis_color, nonvis_color, radius=None, linewidth=2, frame_no=None):
        ind = 0
        vis = True
        if frame_no is None:        
            ax.plot(self.x, self.y, linestyle='-',marker='.',color=vis_color)
        else:
            if frame_no in self.fnum:
                ind = self.fnum.index(frame_no)
            else:
                ind = np.max(np.array(self.fnum)<frame_no)
                vis = False
                print('Non-vis')                
        color = vis_color if vis else nonvis_color                
        if not radius is None:
            circ = Circle( [self.x[ind],self.y[ind]], radius, color=color,  linewidth=linewidth, fill=False)
            ax.add_patch(circ)
        # ax.text(self.x[ind]+10, self.y[ind], str(len(self.score)), color=color, fontsize=12)

def plot_all_tracks(tracks, annotations, title=None, radius=None):
    fig = plt.figure(num='Egg Tracks', figsize=(8,4) )
    ax = fig.add_subplot(1,1,1)
    for track in tracks:
        color = next(ax._get_lines.prop_cycler)['color']
        track.plot(ax, vis_color=color, nonvis_color=color, radius=radius)
    ax.set_xlabel(r'$x$', labelpad=6)            
    ax.set_ylabel(r'$y$', labelpad=6)            
    ax.invert_yaxis()
    ax.axis('equal')
    ax.set_title(title)
    ax.set_xlim( (0,1920))
    ax.set_ylim( (1080,0))
    for anno in annotations:
        for t in anno['targets']:
            ax.plot(t[0],t[1],marker='o',markersize=20,color='k',fillstyle='none')

class my_event:
    ''' This is an alternative to mouse input '''
    def __init__(self):
        self.button = 3

sim_mouse = my_event()

class PlotTracksOnVideo:

    def __init__(self, 
                 reader, 
                 tracks, 
                 annotations=[], 
                 show_annotations = False,
                 title=None, 
                 radius=12, 
                 store_frames = False,
                 start_frame = 0):
        self.reader = reader
        self.tracks = tracks
        self.fill_annotation_list(annotations)
        self.show_annotations = show_annotations
        self.title = title
        self.radius = radius
        self.show_all_frames = False
        self.frames_since_lost = 0
        self.frame_no = start_frame-1
        self.store_frames = store_frames
        if self.store_frames:
            self.vidlist = []
        else:
            self.vidlist = None
        self.fig = plt.figure(num='Egg Tracks', figsize=(6,3.38) )
        if self.store_frames is None:
            self.ax = self.fig.add_subplot(1,1,1)
        else:
            self.ax = self.fig.add_axes([0,0,1,1])  # Cover figure with axes
        self.fig.canvas.mpl_connect('button_press_event',self.plot_next)
        self.fig.canvas.mpl_connect('key_press_event', self.key_press)
        self.done = False
        self.finished = False
        self.plot_next( None )
        self.play_video()        
        plt.show()

    def fill_annotation_list(self, annotations):
        anno_list = []
        for anno in annotations:
            for t in anno['targets']:
                anno_list.append(t)
        self.anno_list = anno_list
        print(f'Annotations for video: {self.anno_list}')

    def is_annotated(self, xy, radius):
        match = [np.sqrt((xy[0]-t[0])**2 + (xy[1]-t[1])**2) < radius for t in self.anno_list]
        ret = len(match)>0 and True in match
        return ret

    def key_press(self, event):
        if event.key == "q":
            self.done = True
            self.finished = True
            plt.close(self.fig)            

    def play_video(self):
        while self.plot_next( sim_mouse ) and not self.done:
            plt.show(block=False)

    def show_image(self):
        frame,_ = self.reader.get_nth(self.frame_no)
        if frame is None:
            self.done = True
            plt.close(self.fig)
            return False
        self.ax.cla()
        self.ax.imshow(frame.permute(1,2,0).numpy())
        self.ax.axis('off')
        return True

    def plot_next(self, event):
        if not event is None and event.button == 1:
            return True  # Don't do anything with left button
        n = 0
        while n==0 and not self.done:
            show = False
            self.frame_no += 1
            if self.show_all_frames:
                show = self.show_image()
            while len(self.tracks) and self.tracks[0].fnum[-1] < self.frame_no:
                self.tracks.pop(0)  # Delete tracks that are passed
            if len(self.tracks)==0:
                self.done = True
                plt.close(self.fig)
                return False
            n=0
            for track in self.tracks:
                if track.fnum[0] > self.frame_no:
                    break
                if track.fnum[-1] >= self.frame_no:
                    if n==0 and not self.show_all_frames:
                        show = self.show_image()      
                    n += 1
                    #color = next(self.ax._get_lines.prop_cycler)['color']
                    # Yellow if detection, orange if not
                    if self.is_annotated([track.x[0],track.y[0]],50):
                        vis_color = (1,1,0.1) 
                        nonvis_color = (1,0.5,0.1)
                    else:
                        vis_color = (1,0,0)
                        nonvis_color = (1,0,0)
                    track.plot(self.ax, vis_color=vis_color, nonvis_color=nonvis_color, radius=self.radius, linewidth=2, frame_no=self.frame_no)
            if n>0:
                self.frames_since_lost = 0
                if self.show_annotations:
                    for t in self.anno_list:
                        self.ax.plot(t[0],t[1],marker='o',markersize=20,color='k',fillstyle='none')
            else:
                if self.frames_since_lost < 10:
                    show = self.show_image()   
                self.frames_since_lost += 1                
            self.ax.set_title(f'{self.title} Frame: {self.frame_no}, Tracks: {n}')
            #print(f'Frame {self.frame_no}, Tracks: {n}, Show: {show}, NF {self.frames_since_lost}')
            self.fig.canvas.draw()
            self.fig.canvas.flush_events()            
            if show and self.store_frames:
                mat = torch.tensor(np.array(self.fig.canvas.renderer._renderer)[:,:,:3])
                self.vidlist.append(mat)
        return True

def next_frame( egg_detections ):
    if len(egg_detections['indices'])==0:
        return None, egg_detections
    else:
        ind = iter(egg_detections['indices'])
        fnum = next(ind)
        n=1
        try:
            while fnum==next(ind):
                n += 1
        except StopIteration:
            pass
    frame = {'fnum':fnum,
             'scores':egg_detections['peak_vals'][:n],
             'x':egg_detections['x'][:n],
             'y':egg_detections['y'][:n]
             }
    egg_detections['peak_vals'] = egg_detections['peak_vals'][n:]
    egg_detections['indices'] = egg_detections['indices'][n:]
    egg_detections['x'] = egg_detections['x'][n:]
    egg_detections['y'] = egg_detections['y'][n:]
    return frame, egg_detections

def kill_old_tracks(tracks_current, fnum, lost_sec):
    keep, done = [], []
    for track in tracks_current:
        time_for_lost = lost_sec if track.valid_start else 0
        if fnum-track.fnum[-1] > time_for_lost + 1:
            done.append(track)
        else:
            keep.append(track)
    return keep, done

def track_eggs( eggs_detections, params, big_value=1e10 ):
    tracks_current = []
    tracks_done = []
    id = 0
    while len(eggs_detections['indices']):
        # Get next frame with tracks:
        frame, eggs_detections = next_frame( eggs_detections )
        # Now that we know the frame number, remove old tracks:
        tracks_current, old = kill_old_tracks(tracks_current, frame['fnum'], params.lost_sec)
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
            # Get all detections that don't have good associations to tracks:       
            rest = [ele for ele in list(range(len(frame['scores']))) if ele not in set(col_ind[good])]
            for nt in rest:
                # Only use a detection to start a track if score > 0
                if frame['scores'][nt] >= 0:
                    tracks_current.append( egg_track(id,frame['fnum'],frame['scores'][nt], frame['x'][nt], frame['y'][nt], params.minseq))
                id += 1
        else:
            for nt in range(len(frame['scores'])):
                tracks_current.append( egg_track(id,frame['fnum'],frame['scores'][nt], frame['x'][nt], frame['y'][nt], params.minseq))
                id += 1
    return tracks_done, tracks_current

class track_params:
    def __init__(self,
                 run,
                 radius = 50,   # radius to match
                 lost_sec = 20,     # Time to lose a track
                 minseq = 5,
                 ):
        self.run = run
        self.radius = radius
        self.lost_sec = lost_sec
        self.minseq = minseq

def load_annotations(video_name, image_folder):
    annotations = []
    for anno in Path(image_folder).glob(video_name.stem + '*.json'):
        with open(str(anno),'r') as f:
            annotations.append( json.load(f) )
    return annotations

def track_detections(args, prefix):

    params = track_params(args.run, 
                          radius = args.radius, 
                          lost_sec = args.lost, 
                          minseq = args.minseq)

    run_dir = os.path.join(os.path.dirname(__file__), 'out_eggs', f'{params.run:03d}')
    vid_dir = os.path.join(run_dir,'video')
    if args.vidout:
        vid_out_dir = args.outfolder
        os.makedirs(vid_out_dir, exist_ok=True)
    else:
        vid_out_dir = None

    search = prefix + '*.json'
    detections = list(Path(vid_dir).rglob(search))
    detections.sort()
    vsearch = prefix + '*.avi'
    videos = list(Path(args.folder).rglob(vsearch))
    videos.sort()
    video_names = [x.name for x in videos]

    print(f'Found {len(detections)} files of type: {search}')


    for path, nextpath in zip(detections, detections[1:]+detections[-1:]):
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
        tracks_d, tracks_c = track_eggs(eggs_detections, params)

        # keep tracks of minimum length:
        tracks = [x for x in tracks_d + tracks_c if len(x.score)>= args.minlen]

        annotations = load_annotations(path, args.images)
        nextannotations = load_annotations(nextpath, args.images)
        print(f'Loaded {len(annotations)} annotations for {path.name}')
        if len(annotations)==0 or sum([len(x['targets']) for x in annotations])==0:
            #If no annotations in current frame, check if there are annotations in next video:
            if len(nextannotations)==0 or sum([len(x['targets']) for x in nextannotations])==0:
                #If none in next too, then skip tracking
                continue

        if args.onvideo:
            video = videos[video_names.index(path.name.replace('json','avi'))]
            reader = VideoReader(str(video), 1)
            print(f'Video: {video.name}')
            pt = PlotTracksOnVideo(reader, 
                                   tracks, 
                                   annotations, 
                                   show_annotations = args.showanno,
                                   title = f'Run: {params.run}, Radius: {params.radius}, Lost {params.lost_sec} (sec)',
                                   store_frames = args.vidout,
                                   start_frame = args.start)     
            if args.vidout:
                vid_file = str(Path(vid_out_dir) / video.stem) + f'_{args.minlen}_{args.minseq}.avi'
                print('Writing:',vid_file)
                write_video(vid_file, torch.stack(pt.vidlist), fps=args.outfps )

            if pt.finished:
                break     
        else:
            plot_all_tracks(tracks, annotations, f'Tracks run: {params.run}, Radius: {params.radius}, Lost {params.lost_sec} (sec)')

            plt.show()
            #plt.savefig('temp.png')
            #print('Done')

if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Plot tracks on video')
    parser.add_argument('run', type=int, help='Run')
    parser.add_argument('--videodir',  type=str, default='/mnt/research/3D_Vision_Lab/Hens/Hens_2021_sec',  help='Folder for videos')    
    parser.add_argument('--detectdir', type=str, default=None,  help='Folder for detections (output of run_video.py)')    
    parser.add_argument('--images', type=str, default='/mnt/research/3D_Vision_Lab/Hens/ImagesJPG',  help='Folder for images with annotations')        
    parser.add_argument('--prefix', type=str, nargs='+', default=[''],  help='search prefix')     
    parser.add_argument('--onvideo', action='store_true',  help='Plot on tracks on video')    
    parser.add_argument('--minlen', type=int, default=1, help='Minimum length of track (in observations)')
    parser.add_argument('--radius', type=float, default=50, help='Association radius')
    parser.add_argument('--lost', type=int, default=0, help='Seconds lost but still continue track')
    parser.add_argument('--minseq', type=int, default=5, help='Minimum sequential seconds for valid_start')
    parser.add_argument('--outfps', type=float, default=5., help='How fast to play video in fps')
    parser.add_argument('--vidout', action='store_true',  help='Store video')    
    parser.add_argument('--outfolder', type=str, default='/mnt/scratch/'+os.environ["USER"]+"/trackvid",  help='Output folder for video tracks')    
    parser.add_argument('--showanno', action='store_true',  help='Plot annotations')    
    parser.add_argument('--start', type=int, default=0, help='Start frame')
    
    args = parser.parse_args()


    for prefix in args.prefix:
        track_detections(args, prefix)


    # python track_eggs.py 53 --prefix ch4_0729 --minlen 4 --playrate 10 --onvideo --vidout
