''' Track eggs on video detections (made by run_video.py)

    Example usage on video detections on camera 4 from 07/30/2020:
      python track_eggs.py 54 --prefix ch4_0730 --minseq 5 --minlen 2 \
          --videodir /mnt/research/3D_Vision_Lab/Hens/Hens_2021_sec \
          --imagedir /mnt/research/3D_Vision_Lab/Hens/ImagesJPG \
          --detectdir /mnt/scratch/dmorris/Hens_Detections_054 \
          --onvideo

    The --onvideo option will plot tracks on video.  Without this, will 
    plot all tracks for whole video on the first frame.  

    To save output as a video include the option (requires --onvideo option):
          --vidtrackdir /mnt/scratch/dmorris/trackvid

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

from scipy.spatial.distance import cdist
from scipy.optimize import linear_sum_assignment

import torch
import torchvision
torchvision.disable_beta_transforms_warning()
from torchvision.io import write_video

image_path = str( Path(__file__).parents[1] / 'imagefunctions' ) 
sys.path.append(image_path)
from hens.VideoIOtorch import VideoReader

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
 
    # TODO: https://iquilezles.org/articles/palettes/ cosine color palette for IDs
    # TODO: For non-vis, how do we want to draw it?
    def get_color(self):
        return

    def plot(self, ax, vis_color, nonvis_color, radius=None, linewidth=2, frame_no=None):
        ind = 0
        vis = True
        # Plot total path on single frame
        if frame_no is None:        
            ax.plot(self.x, self.y, linestyle='-',marker='.',color=vis_color)
        else:
            # If track is in frame, then index is set to frame
            if frame_no in self.fnum:
                ind = self.fnum.index(frame_no)
            # If track not in frame, then index 
            else:
                ind = np.count_nonzero(np.array(self.fnum)<frame_no) - 1  # last frame where we saw track
                vis = False
                print(f'Non-vis (cur frame {frame_no} | using loc from {self.fnum[ind]})')                
        color = vis_color if vis else nonvis_color                
        if not radius is None:
            circ = Circle( [self.x[ind],self.y[ind]], radius, color=color,  linewidth=linewidth, fill=False)
            ax.add_patch(circ)
        # ax.text(self.x[ind]+10, self.y[ind], str(len(self.score)), color=color, fontsize=12)

def plot_all_tracks(tracks, annotations, frame=None, title=None, radius=None):
    fig = plt.figure(num='Egg Tracks', figsize=(8,4) )
    ax = fig.add_subplot(1,1,1)
    if not frame is None:
        ax.imshow(frame.permute(1,2,0).numpy())
    # TODO: Implement consistent coloring based on ID then color them based on that
    for track in tracks:
        color = next(ax._get_lines.prop_cycler)['color']
        track.plot(ax, vis_color=color, nonvis_color=color, radius=radius)
    ax.set_xlabel(r'$x$', labelpad=6)            
    ax.set_ylabel(r'$y$', labelpad=6)            
    ax.axis('equal')
    ax.set_title(title)
    if frame is None:
        ax.invert_yaxis()
        ax.set_xlim( (0,1920))
        ax.set_ylim( (1080,0))
    for anno in annotations:
        # These are manual annotations just to give insight into which tracks are probably real
        for t in anno['targets']:
            # Draw big white circle around them
            ax.plot(t[0],t[1],marker='o',markersize=20,color='w',fillstyle='none')
    ax.axis('tight')

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
        print('** Press "q" to quit video  **')   
        self.plot_next( None )
        self.play_video()     
        plt.show()

    def fill_annotation_list(self, annotations):
        anno_list = []
        for anno in annotations:
            for t in anno['targets']:
                anno_list.append(t)
        self.anno_list = anno_list
        #print(f'Loaded {len(self.anno_list)} egg annotations for video')
        #print(f'Annotations for video: {self.anno_list}')

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
                    if self.is_annotated([track.x[-1],track.y[-1]],50):
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
                        self.ax.plot(t[0],t[1],marker='o',markersize=20,color='w',fillstyle='none')
            else:
                if self.frames_since_lost < 10:
                    show = self.show_image()   
                self.frames_since_lost += 1                
            self.ax.set_title(f'{self.title} Frame: {self.frame_no}, Tracks: {n}')
            #print(f'Frame {self.frame_no}, Tracks: {n}, Show: {show}, NF {self.frames_since_lost}')
            self.ax.axis('off')
            self.fig.canvas.draw()
            self.fig.canvas.flush_events()            
            if show and self.store_frames:
                mat = torch.tensor(np.array(self.fig.canvas.renderer._renderer)[:,:,:3])
                self.vidlist.append(mat)
        return True

def next_frame( egg_detections ):
    # The index is the frame number / time in seconds of the detections
    # Here, collect all detections having index equal to the first index in egg_detections
    # This is the next frame.
    # Then remove these detections from egg_detections
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

def track_eggs( eggs_detections, start_id, tracks_c, params, big_value=1e10 ):
    # Here is the main tracking loop
    tracks_current = tracks_c
    tracks_done = []
    id = start_id

    while len(eggs_detections['indices']):
        # Get next frame with tracks:
        frame, eggs_detections = next_frame( eggs_detections )
        # Now that we know the frame number, remove old tracks:
        tracks_current, old = kill_old_tracks(tracks_current, frame['fnum'], params.lost_sec)
        tracks_done = tracks_done + old
        if len(tracks_current):  # If we have current tracks, then find associations with detections
            # Get coordinates of tracked eggs
            xy = np.array(list(map(lambda tr: [tr.x[-1],tr.y[-1]], tracks_current)))
            # Get coordinates of new detections:
            xy_new = np.array([frame['x'],frame['y']]).T
            # We're doing association between current tracks and new detections using 
            # the Hungarian algorithm -- this finds best pairwise association excluding double assignments
            # First find all the pairwise distances between tracks and detections:
            all_dists = cdist( xy, xy_new )
            # All dists greater than max should be excluded as matches 
            # Thus make their distances infeasible (otherwise will include these matches)
            all_dists[all_dists>params.radius] = big_value
            # Now find the best pairwise association:
            row_ind, col_ind = linear_sum_assignment(all_dists)
            match_dists = all_dists[row_ind, col_ind]   
            good = match_dists <= params.radius  # Keep associations with distance apart <= params.radius
            for row, col in zip(row_ind[good], col_ind[good]):
                # Update each track using the associated detection:
                tracks_current[row].add( frame['fnum'], frame['scores'][col], frame['x'][col], frame['y'][col] )  
            # Get all detections that don't have good associations to tracks:       
            rest = [ele for ele in list(range(len(frame['scores']))) if ele not in set(col_ind[good])]
        else:
            # If no current tracks then start new tracks with potentially all detections
            rest = range(len(frame['scores']))

        # now start new tracks:            
        for nt in rest:
            # Only use a detection to start a track if score > 0
            if frame['scores'][nt] >= 0:
                tracks_current.append( egg_track(id,frame['fnum'],frame['scores'][nt], frame['x'][nt], frame['y'][nt], params.minseq))
                id += 1
                
    return tracks_done, tracks_current, id

class track_params:
    def __init__(self,
                 run,
                 radius = 50,       # Radius to match
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
    """General function to track detections in videos"""
    params = track_params(args.run, 
                          radius = args.radius, 
                          lost_sec = args.lost, 
                          minseq = args.minseq)

    # find all detections from run_video.py:
    search = prefix + '*.json'
    detections = list(Path(args.detectdir).rglob(search))
    detections.sort()
    vsearch = prefix + '*.avi'
    videos = list(Path(args.videodir).rglob(vsearch))
    videos.sort()
    video_names = [x.name for x in videos]

    print(f'Found {len(detections)} files of type: {search}')

    if args.vidtrackdir:
        # Create output folder:
        os.makedirs(args.vidtrackdir, exist_ok=True)

    tracks_c = tracks_d = []
    start_id = 0

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
        tracks_d, tracks_c, start_id = track_eggs(eggs_detections, start_id, tracks_c, params)
        start_id += 1   # iterate ID so that it is prepared to start new track
        # keep tracks of minimum length:
        tracks = [x for x in tracks_d + tracks_c if len(x.score)>= args.minlen]
        annotations = load_annotations(path, args.imagedir)
        print(f'Loaded {len(annotations)} annotations for {path.name}')
        #nextannotations = load_annotations(nextpath, args.imagedir)
        #if len(annotations)==0 or sum([len(x['targets']) for x in annotations])==0:
        #    #If no annotations in current frame, check if there are annotations in next video:
        #    if len(nextannotations)==0 or sum([len(x['targets']) for x in nextannotations])==0:
        #        #If none in next too, then skip tracking
        #        continue

        video = videos[video_names.index(path.name.replace('json','avi'))]
        reader = VideoReader(str(video), 1)
        #print(f'Video for tracking: {video.name}')
            
        if args.onvideo: 
            pt = PlotTracksOnVideo(reader, 
                                   tracks, 
                                   annotations, 
                                   show_annotations = not args.hideanno,
                                   title = f'{video.name}, Radius: {params.radius}, Lost {params.lost_sec} (sec)',
                                   store_frames = args.vidtrackdir,
                                   start_frame = args.start)     
            if args.vidtrackdir:
                vid_file = str(Path(args.vidtrackdir) / video.stem) + f'_{args.minlen}_{args.minseq}.avi'
                print('Writing:',vid_file)
                write_video(vid_file, torch.stack(pt.vidlist), fps=args.outfps )

            if pt.finished:
                break     
        else:
            # Plot tracks on first frame of video:
            frame, _ = reader.get_next()
            plot_all_tracks(tracks, annotations, frame,
                            f'Tracks run: {params.run}, Radius: {params.radius}, Lost {params.lost_sec} (sec)', )

            plt.show()
            #plt.savefig('temp.png')
            #print('Done')

if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Plot tracks on video')
    parser.add_argument('run', type=int, help='Run')
    parser.add_argument('--videodir',  type=str, default='/mnt/research/3D_Vision_Lab/Hens/Hens_2021_sec',  help='Folder for videos')    
    parser.add_argument('--detectdir', type=str, default=None,  help='Folder for detections (output of run_video.py)')    
    parser.add_argument('--imagedir', type=str, default='/mnt/research/3D_Vision_Lab/Hens/ImagesJPG',  help='Folder for images with annotations')        
    parser.add_argument('--prefix', type=str, nargs='+', default=[''],  help='search prefix')     
    parser.add_argument('--minlen', type=int, default=1, help='Minimum length of track (in observations)')
    parser.add_argument('--radius', type=float, default=50, help='Association radius')
    parser.add_argument('--lost', type=int, default=0, help='Seconds lost but still continue track')
    parser.add_argument('--minseq', type=int, default=5, help='Minimum sequential seconds for valid_start')
    parser.add_argument('--outfps', type=float, default=5., help='How fast to play video in fps')
    parser.add_argument('--onvideo', action='store_true',  help='Plot on tracks on video')    
    parser.add_argument('--vidtrackdir', type=str, default=None,  help='Save tracks on videos')    
    parser.add_argument('--hideanno', action='store_true',  help='Hide image annotations')    
    parser.add_argument('--start', type=int, default=0, help='Start frame')
    
    args = parser.parse_args()


    for prefix in args.prefix:
        track_detections(args, prefix)

