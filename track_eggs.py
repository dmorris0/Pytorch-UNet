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

class egg_track(object):
    def __init__(self, id, vid_i, fnum, score, x, y, minseq=5):        
        self.id = id
        self.fnum = {vid_i: [fnum]}
        self.score = {vid_i: [score]}
        self.x = {vid_i: [x]}
        self.y = {vid_i: [y]}
        self.minseq = minseq
        self.nseq = 1
        self.valid_start = self.nseq >= self.minseq 

    def add(self, vid_i, fnum, score, x, y):
        # Last video this track was seen in
        last_vid = next(iter(reversed(self.fnum.keys())))
        # Check if last seen is in current video
        if last_vid == vid_i:
            if fnum == self.fnum[vid_i][-1]+1:
                self.nseq += 1
        # If track was last seen on last frame in last video, then continue sequence
        elif last_vid == vid_i - 1 and fnum == 1 and self.fnum[last_vid][-1] == 1799:
            self.nseq += 1
        else:
            self.nseq = 1

        last_vid = next(iter(reversed(self.fnum.keys())))
        # If the last frame seen was the current video
        if last_vid == vid_i:
            self.fnum[vid_i].append(fnum)
            self.score[vid_i].append(score)
            self.x[vid_i].append(x)
            self.y[vid_i].append(y)
        # If the last frame seen was in the last video then create new entry for current video
        elif last_vid == vid_i - 1:
            self.fnum[vid_i] = [fnum]
            self.score[vid_i] = [score]
            self.x[vid_i] = [x]
            self.y[vid_i] = [y]
        # Lost time should not be >30 minutes, so throw an error
        else:
            raise ValueError("Lost time allowed > 30 minutes so track last seen over a video ago")

        if not self.valid_start:
            self.valid_start = self.nseq >= self.minseq
 
    # https://iquilezles.org/articles/palettes/ cosine color palette for IDs
    def get_color(self, vis=True):
        if not vis:
            return (0, 0, 0)
        
        a = np.array([0.5, 0.5, 0.5])
        b = np.array([0.5, 0.5, 0.5])
        c = np.array([1.0, 1.0, 1.0])
        d = np.array([0.00, 0.33, 0.67])
        # This is how often colors will repeat, lower the cycle period, the more variation between tracks
        CYCLE_PERIOD = 5.0

        return tuple(a + b * np.cos(2*np.pi*(c*self.id/CYCLE_PERIOD+d)))

    def plot(self, ax, vid_i, radius=None, linewidth=2, frame_no=None):
        ind = 0
        vis = True
        # Plot total path on single frame
        if frame_no is None:        
            ax.plot(self.x[vid_i], self.y[vid_i], linestyle='-',marker='.',color=self.get_color(vis))
        else:
            loc = []
            # If track is in frame, then index is set to frame
            if frame_no in self.fnum[vid_i]:
                ind = self.fnum[vid_i].index(frame_no)
                loc = [self.x[vid_i][ind], self.y[vid_i][ind]]
            # If track not in frame, then index is last frame where we saw
            else:
                # If track hasn't been seen in this video, take last frame from last video
                vis = False
                if frame_no < self.fnum[vid_i][0]:
                    loc = [self.x[vid_i - 1][-1], self.y[vid_i - 1][-1]]
                else:
                    ind = np.count_nonzero(np.array(self.fnum[vid_i])<frame_no) - 1  # last frame where we saw track
                    loc = [self.x[vid_i][ind], self.y[vid_i][ind]]
                # print(f'Non-vis (cur frame {frame_no} | using loc from {self.fnum[ind]})')                
        if not radius is None:
            circ = Circle(loc, radius, color=self.get_color(vis),  linewidth=linewidth, fill=False)
            # Add ID label to video
            plt.text(loc[0] + 10, loc[1] - 10, self.id, color=self.get_color(vis))
            ax.add_patch(circ)

class TrackEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, egg_track):
            return obj.__dict__
        return json.JSONEncoder.default(self, obj)

def plot_all_tracks(tracks, annotations, vid_i, frame=None, title=None, radius=None):
    fig = plt.figure(num='Egg Tracks', figsize=(8,4) )
    ax = fig.add_subplot(1,1,1)
    if not frame is None:
        ax.imshow(frame.permute(1,2,0).numpy())
    # Only take tracks which have frames in this video!
    valid_tracks = [track for track in tracks if vid_i in track.x]
    for track in valid_tracks:
        track.plot(ax, vid_i, radius=radius)
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
                 vid_i, 
                 tracks, 
                 annotations=[], 
                 show_annotations = False,
                 title=None, 
                 radius=12, 
                 store_frames = False,
                 start_frame = 0):
        self.reader = reader
        self.vid_i = vid_i
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
            # Can no longer pop because this will mess up tracks
            # while len(self.tracks) and self.tracks[0].fnum[self.vid_i][-1] < self.frame_no:
            #     self.tracks.pop(0)  # Delete tracks that are passed

            # TODO: Find some sort of starting index sliding window method to be more efficient
            # if len(self.tracks)==0:
            # Stop video if you've finished plotting last track or there are no tracks to plot
            if len(self.tracks) == 0 or self.tracks[-1].fnum[self.vid_i][-1] == self.frame_no:
                self.done = True
                plt.close(self.fig)
                return False
            n=0
            
            for track in self.tracks:
                # Don't print routes that aren't in current video
                if self.vid_i not in track.fnum:
                    continue

                # Don't show routes that have not started yet
                # Since they're sorted by finishing frame, there might be valid ones after
                # If this is the first video the track is in, then don't show if it starts later
                if list(track.fnum.keys())[0] == self.vid_i:
                    if track.fnum[self.vid_i][0] > self.frame_no:
                        continue
                
                # Ensure that the track ends on or after the current frame
                # NOTE: With how this currently works, a track might extend into future videos and disappear for a few frames
                # NOTE: Videos are processed in order, so can't check if this track will continue to the future
                if track.fnum[self.vid_i][-1] >= self.frame_no:
                    if n==0 and not self.show_all_frames:
                        show = self.show_image()      
                    n += 1

                    track.plot(self.ax, self.vid_i, radius=self.radius, linewidth=2, frame_no=self.frame_no)
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

def mask(xy, ch):
    """
    Returns true if a point falls under a certain mask for a channel
    """

    # For channel 4, do simple square masking for now unless it doesn't work well
    if ch == 4:
        return xy[0] < 1500 and xy[1] < 930

def mask_filter(frame, ch_num):
    """
    Throws out all detections that are in invalid areas.

    Unique mask per channel and will raise exception if mask hasn't been created for channel yet
    """
    val_channels = [4]
    if ch_num not in val_channels:
        raise ValueError(f"Channel {ch_num} has not had a mask created or is invalid.")
    
    # pull all valid indices
    valid_ind = [i for i in range(len(frame['scores'])) if not mask((frame['x'][i], frame['x'][i], ch_num))]
    filtered_frame = {'fnum': frame['fnum'], 
                      'scores': [],
                      'x': [],
                      'y': []}
    for i in valid_ind:
        filtered_frame['scores'].append(frame['scores'][i])
        filtered_frame['x'].append(frame['x'][i])
        filtered_frame['y'].append(frame['y'][i])

    return filtered_frame


def kill_old_tracks(tracks_current, vid_i, fnum, lost_sec):
    keep, done = [], []
    for track in tracks_current:
        time_for_lost = lost_sec if track.valid_start else 0
        last_vid = next(iter(reversed(track.fnum.keys())))
        # If last frame seen in was in current video, then normal calculation
        if last_vid == vid_i:
            if fnum-track.fnum[vid_i][-1] > time_for_lost + 1:
                done.append(track)
            else:
                keep.append(track)
        # If last frame was seen in last video
        elif last_vid == vid_i - 1:
            frames_from_last = 1799 - track.fnum[last_vid][-1]
            # Frames lost in last video and frames lost in this one so far
            if fnum + frames_from_last > time_for_lost + 1:
                done.append(track)
            else:
                keep.append(track)
        # Last frame seen was in a video before the last, must be done
        else:
            done.append(track)

    return keep, done

def track_eggs(vid_i, ch_num, eggs_detections, tracks_c, params, big_value=1e10):
    # Here is the main tracking loop
    tracks_current = tracks_c
    tracks_done = []
    id = int(1e5) # set to large number to distinguish between real IDs and FAKE

    while len(eggs_detections['indices']):
        # Get next frame with tracks:
        frame, eggs_detections = next_frame( eggs_detections )

        # Filter out eggs which are in separate pens or occluded by pen
        frame = mask_filter(frame, ch_num)

        # Now that we know the frame number, remove old tracks:
        tracks_current, old = kill_old_tracks(tracks_current, vid_i, frame['fnum'], params.lost_sec)
        tracks_done = tracks_done + old


        if len(tracks_current):  # If we have current tracks, then find associations with detections
            # Get coordinates of tracked eggs
            # Get last xy coordinates, could be in previous videos
            xy = np.array(list(map(lambda tr: [tr.x[list(tr.x.keys())[-1]][-1],tr.y[list(tr.y.keys())[-1]][-1]], tracks_current)))
            # Get coordinates of new detections:
            # Set threshold to only take probable points 0.14 (53.5% conf)
            # These were determined from plotting true_pos vs false_pos
            indices = np.where(np.array(frame['scores']) >= 0.14)
            filtered_x = np.array(frame['x'])[indices]
            filtered_y = np.array(frame['y'])[indices]
            xy_new = np.array([filtered_x, filtered_y]).T
            # We're doing association between current tracks and new detections using 
            # the Hungarian algorithm -- this finds best pairwise association excluding double assignments
            # First find all the pairwise distances between tracks and detections:
            all_dists = cdist( xy, xy_new )
            # All dists greater than max should be excluded as matches 
            # Thus make their distances infeasible (otherwise will include these matches)

            # Now find the best pairwise association:
            row_ind, col_ind = linear_sum_assignment(all_dists)
            match_dists = all_dists[row_ind, col_ind]   

            all_dists[all_dists>params.radius] = big_value

            good = match_dists <= params.radius  # Keep associations with distance apart <= params.radius
            for row, col in zip(row_ind[good], col_ind[good]):
                # Update each track using the associated detection:
                tracks_current[row].add(vid_i, frame['fnum'], frame['scores'][col], frame['x'][col], frame['y'][col] )  
            # Get all detections that don't have good associations to tracks:       
            rest = [ele for ele in list(range(len(frame['scores']))) if ele not in set(col_ind[good])]
        else:
            # If no current tracks then start new tracks with potentially all detections
            rest = range(len(frame['scores']))

        # now start new tracks:            
        for nt in rest:
            # Only use a detection to start a track if score > 0
            # 0.282 (57% conf) is max diff between true pos and false pos
            if frame['scores'][nt] >= 0.282:
                tracks_current.append( egg_track(id, vid_i, frame['fnum'],frame['scores'][nt], frame['x'][nt], frame['y'][nt], params.minseq))
                id += 1
                
    return tracks_done, tracks_current

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

def track_length(track):
    """General function to track how long a track is through all videos"""
    frames_present = 0
    for frames in track.fnum.values():
        frames_present += len(frames)
    return frames_present

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

    tracks = []
    tracks_c = []
    tracks_d = []
    vid_indexing = {}
    start_id = 0

    for vid_i, path in enumerate(detections):
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
        
        # Pull channel number
        file = os.path.basename(path)
        f_name = file.split('.')[0]
        c_name = f_name.split('_')[0]
        c_num = int(c_name[2:])
        
        # map file basename to index of file for JSON later
        vid_indexing[vid_i] = f_name
        
        # extract filename from path, used as key for tracks
        tracks_d, tracks_c = track_eggs(vid_i, c_num, eggs_detections, tracks_c, params)

        # keep tracks of minimum length:
        # only plot tracks that meet length requirement and have frames in this video

        tracks_to_plot = [t for t in tracks_d + tracks_c if (track_length(t) >= args.minlen and vid_i in t.fnum)]

        # store all finished tracks
        # if we're on last video, then also store current tracks that aren't "done" to all tracks
        if vid_i < len(detections) - 1:
            tracks.extend(t for t in tracks_d if track_length(t) >= args.minlen)
        else:
            tracks.extend(t for t in tracks_d + tracks_c if track_length(t) >= args.minlen)

        # sort tracks to add by ending frame so that you're able to tell when to end
        tracks_to_plot = sorted(tracks_to_plot, key=lambda track: track.fnum[vid_i][-1])            

        # get list of tracks whose ID need to be updated (just got found this video)
        tracks_to_update = [t for t in tracks_to_plot if t.id >= 1e5]
        # sort list of tracks so the first ones to appear have lower IDs
        tracks_to_update = sorted(tracks_to_update, key=lambda track: track.fnum[vid_i][0])

        if len(tracks_to_update) > 0:
            for i, track in enumerate(tracks_to_update):
                track.id = start_id + i

            start_id += i + 1   # iterate ID so that it is prepared to start new track

        # or 1799 - x.fnum[vid_i][-1] <= params.lost_sec
        # Using this to keep tracks leads to a lot of garbage at the end, so just kill dead tracks

        print(f"{len(tracks_to_plot)} tracks found")
        print(f"Start id is {start_id}")
        # recode track ids to not jump over numbers

        print("----------- TRACKS PLOTTED -------------")
        for track in tracks_to_plot:
            print("-----------------")
            print(f"Track {track.id}, {track.fnum[vid_i][0]} - {track.fnum[vid_i][-1]}")
            print(f"x: {track.x[vid_i][0]}")
            print(f"y: {track.y[vid_i][0]}")
            print(f"len: {track_length(track)}")

        annotations = load_annotations(path, args.imagedir)
        print(f'Loaded {len(annotations)} annotations for {path.name}')

        video = videos[video_names.index(path.name.replace('json','avi'))]
        reader = VideoReader(str(video), 1)
        #print(f'Video for tracking: {video.name}')
            
        if args.onvideo: 
            pt = PlotTracksOnVideo(reader, 
                                   vid_i,
                                   tracks_to_plot, 
                                   annotations, 
                                   show_annotations = not args.hideanno,
                                   title = f'{video.name}, Radius: {params.radius}, Lost {params.lost_sec} (sec)',
                                   store_frames = args.vidtrackdir,
                                   start_frame = args.start)  
            # Ensure that video has frames to plot   
            if args.vidtrackdir and len(pt.vidlist) > 0:
                vid_file = str(Path(args.vidtrackdir) / video.stem) + f'_{args.minlen}_{args.minseq}.avi'
                print('Writing:',vid_file)
                write_video(vid_file, torch.stack(pt.vidlist), fps=args.outfps )

            if pt.finished:
                break     
        else:
            # Plot tracks on first frame of video:
            frame, _ = reader.get_next()
            plot_all_tracks(tracks_to_plot, annotations, vid_i, frame,
                            f'Tracks run: {params.run}, Radius: {params.radius}, Lost {params.lost_sec} (sec)', )

            plt.show()
            #plt.savefig('temp.png')
            #print('Done')

    # write all the tracks down to JSON file
    # store the first frame and video where we find the egg 
    if args.jsondir:
        tracks_json = []
        tracks = sorted(tracks, key=lambda track: track.id)
        for t in tracks:
            first_vid = list(t.fnum.keys())[0]
            tracks_json.append({'id': t.id, 'vid': str(vid_indexing[first_vid]), 'frame': t.fnum[first_vid][0]})

        with open(prefix + '_tracks.json', "w") as f:
            json.dump(tracks_json, f, indent=4)

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
    parser.add_argument('--jsondir', type=str, default=None, help='Save tracks found to JSON file')    
    parser.add_argument('--hideanno', action='store_true',  help='Hide image annotations')    
    parser.add_argument('--start', type=int, default=0, help='Start frame')
    
    args = parser.parse_args()


    for prefix in args.prefix:
        track_detections(args, prefix)

