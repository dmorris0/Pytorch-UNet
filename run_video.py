''' Run detector on videos.

    To run trained model 54 on camera ch3 and plot results (without saving).
      python run_video.py 54 \
          --prefix ch3 \
          --loadmodel /mnt/research/3D_Vision_Lab/Hens/models/054_UNetQuarter.pth \
          --inputdir /mnt/research/3D_Vision_Lab/Hens/Hens_2021_sec

    It will run recursively on inputdir finding all videos from these cameras.  
    Running and plotting is slower than below with just saving results:
          
    To run trained model 54 on camera ch4 videos from 23/07/30 and save results to detectdir:
      python run_video.py 54 \
          --prefix ch4_0730 \
          --loadmodel /mnt/research/3D_Vision_Lab/Hens/models/054_UNetQuarter.pth \
          --inputdir /mnt/research/3D_Vision_Lab/Hens/Hens_2021_sec \
          --detectdir /mnt/scratch/dmorris/Hens_Detections_054

    --minval <val> is the threshold on whether a peak is returned as a detection.  A detection with value 0
        has probability of sigmoid(0) = 0.5  This ia a good value for a confident detection.    
        The minval of -0.5 is the default, meaning we'll also save very low confidence detections (<0).  This is
        mostly for tracking eggs that are already confidently detected, similar to canny edge detection. 

    Note: run on GPU.  On a single amd20 GPU, each 30 minute video stored at 1fps using MJPG will 
    process in about 1 minute.  (Far better than on a CPU)
    
    After this is run on videos, tracks can be made with: track_eggs.py

    Daniel Morris, 2023

'''

import os
import sys
import torch
from pathlib import Path
from tqdm import tqdm
import numpy as np
import json
import argparse
import logging
import matplotlib.pyplot as plt
from matplotlib.patches import Circle
import threading
from timeit import default_timer as timer
from filelock import Timeout, SoftFileLock
from sys import setrecursionlimit
import time
import torch
import torchvision
torchvision.disable_beta_transforms_warning()
torchvision.set_video_backend("video_reader")
import torchvision.transforms.v2 as transforms

from run_params import get_run_params, init_model

image_path = str( Path(__file__).parents[1] / 'imagefunctions' ) 
sys.path.append(image_path)
from hens.heatmap_score import Peaks, MatchScore
from hens.image_fun import up_scale_coords
from hens.VideoIOtorch import VideoReader


to_float = transforms.Compose(
            [
                transforms.ConvertImageDtype(torch.float32),
            ]
)

class PlotVideo:

    def __init__(self, 
                 reader, 
                 model, 
                 device, 
                 peaks, 
                 out_name=None, # If None, then waits for mouse click to go to next frame
                 doplot=True, 
                 radius=12, 
                 figname='Camera', 
                 nms_proximity=0):

        self.reader = reader
        self.model = model
        self.device = device
        self.peaks = peaks
        self.proximity = nms_proximity
        self.matches = MatchScore(max_distance = radius)

        self.name = Path(reader.name).name
        self.figname = figname
        self.fig = None
        self.radius = radius
        self.inc = -1
        self.next_video = True
        self.annotations = None

        self.out_name = out_name
        self.doplot = doplot
        if self.out_name is None:
            print(f'''
    Will stop for detections >= minval: {self.peaks.min_val}
    Orange circles:  peaks >= 0
    Blue circles:    peaks <  0
    Space:           Continue
    q:               Quit video
                  ''')
            #self.inc = 50
            #self.reader.get_nth(50)
            setrecursionlimit(len(self.reader)+1)  # Plotting uses recursion for each frame
            self.plot_all()
        else:
            self.run_all()

    def plot_all(self):
        self.get_next()
        self.plot()
        if self.frame is None or self.next_video == False:
            return
        if self.peak_vals.size:
            plt.show(block=True)
        else:
            plt.show(block=False)
            self.plot_all()

    def run_all(self):
        peak_vals, indices, x, y = [], [], [], []

        start_time = timer()
        while True:
            self.get_next()
            if self.frame is None:
                break
            if self.peak_vals.size:
                peak_vals = peak_vals + self.peak_vals.tolist()
                indices = indices + [self.inc]*self.peak_vals.size
                x = x + self.imcoords[:,0].tolist()
                y = y + self.imcoords[:,1].tolist()            
            if self.doplot:
                self.plot()
                plt.show(block=False)
        total_time = timer() - start_time

        vid_detect = {  'video': self.reader.name,
                        'sample_time_secs': self.reader.sample_time_secs,
                        'nskip': self.reader.skip,
                        'peak_vals': peak_vals,
                        'indices': indices,
                        'x': x,
                        'y': y,
                      }
        print(f'  Detections: {(np.array(peak_vals)>=0).sum():4d} >= 0, {(np.array(peak_vals)<0).sum():4d} < 0 --> {Path(self.out_name).name}.  Done in {total_time/60:.1f} min')        
        #print(f'{len(peak_vals):5d} detections in {len(self.reader)} frames in {total_time/60:.2} min from {Path(self.out_name).name}')
        with open(self.out_name, 'w') as f:
            json.dump(vid_detect, f, indent=2)
            

    def get_next(self):
        self.inc += 1
        self.frame, frame_no = self.reader.get_next()
        if self.frame is None:
            self.peak_vals = None
            self.imcoords = None
        else:
            img = to_float(self.frame[None,...]).to(device=self.device, memory_format=torch.channels_last)   
            
            heatmap = self.model(img)
            scale = img.shape[-1]/heatmap.shape[-1]

            detections = self.peaks.peak_coords( heatmap )
            peak_vals = self.peaks.heatmap_vals( heatmap, detections )
            if self.proximity > 0:
                detections, peak_vals = self.peaks.nms(detections, peak_vals, self.proximity / scale, to_torch=True)
            self.peak_vals = peak_vals[0][0].cpu().numpy()
            self.imcoords = up_scale_coords( detections[0][0].cpu().numpy(), scale )      

    def plot(self):        
        if self.frame is None:
            plt.close(self.fig)
            print('Done video')
        else:
            self.plot_detections() 
        
    def init_fig(self, figsize):
        if self.fig is None:
            self.fig = plt.figure(num=self.figname, figsize=figsize )
            if self.out_name is None:
                self.fig.canvas.mpl_connect('key_press_event', self.key_press)

    def key_press(self, event):
        if event.key == " ": 
            self.plot_all()  # Go to next
        elif event.key == "q":
            self.next_video = False
            plt.close(self.fig)
            print('Done')            
            
    def draw_targets(self, ax, targets, radius, color ):
        ax.set_xlim(*ax.get_xlim())
        ax.set_ylim(*ax.get_ylim())
        if radius==0:
            ax.plot(xy[:,0], xy[:,1],'+',color=color, markeredgewidth=2., markersize=10.)
        else:
            for xy in targets:
                circ = Circle( xy, radius, color=color,  linewidth=2., fill=False)
                ax.add_patch(circ)

    def plot_detections(self):
        #, img, targets, peak_vals, annotations=None, name=None):
        #self.frame, self.imcoords, self.peak_vals, self.annotations, name=self.name)
        self.init_fig( (6,6) )
        self.fig.clf()
        ax = self.fig.add_subplot(1, 1, 1)
        ax.imshow(self.frame.permute(1,2,0).numpy())
        if not self.annotations is None:
            self.draw_targets(ax, self.annotations["targets"], self.radius*2, color=(1,1,1))
        if len(self.peak_vals)>0:
            self.draw_targets(ax, self.imcoords[self.peak_vals<0], self.radius, color=(.2,.5,1.))
            self.draw_targets(ax, self.imcoords[self.peak_vals>=0], self.radius, color=(1.,.5,.2))
        ax.set_title(f'{self.name}: {self.inc}, N det: {self.imcoords.shape[0]}')
        self.fig.canvas.draw()
        self.fig.canvas.flush_events()

def are_we_already_done(args, out_dir, prefix):
    ''' Return True if completed all videos '''
    if not args.detectdir:
        return False
    search = prefix + '*.' + args.suffix
    for path in Path(args.inputdir).rglob(search):
        out_name = os.path.join(out_dir, path.name.replace('.'+args.suffix,'.json') )
        if not os.path.exists(out_name):
            return False
    return True

@torch.inference_mode()
def run_vid(args, prefix, delete_old_locks_min=10):

    logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    params = get_run_params(args.run)
    min_val = args.minval

    if args.detectdir:
        os.makedirs(args.detectdir, exist_ok=True)

        if are_we_already_done(args, args.detectdir, prefix):
            print(f'Already completed all videos of type: {prefix}*.{args.suffix}, so quitting')
            return True

    # If we don't specify a loadmodel, then want to always load the best checkpoint (rather than last checkpoint)
    params.load_opt = 'best'
    model, _ = init_model(params, device, args.loadmodel)
            
    model.eval()
    peaks = Peaks(1, device, min_val=min_val)

    search = prefix + '*.' + args.suffix
    videos = list(Path(args.inputdir).rglob(search))
    videos.sort()

    print(f'Device:                 {device}')
    print(f'Input Video folder:     {args.inputdir}')
    print(f'Number of videos:       {len(videos)} of type: {search}')
    print(f'Min peaks for detecion: {min_val}')    

    if args.detectdir:
        print(f'Storing detections in:  {args.detectdir}')
    nskip=0
    first = True
    for path in videos:

        out_name = os.path.join(args.detectdir, path.name.replace('.'+args.suffix,'.json') )
        if os.path.exists(out_name):
            nskip += 1
            continue                
        if nskip:
            print(f'Skipping {nskip} completed videos')
            nskip=0

        if not args.detectdir:                 
            # Only plotting
            reader = VideoReader(str(path), sample_time_secs=1)
            pv = PlotVideo(reader, model, device, peaks, out_name=None, 
                           doplot=args.detectdir==None, 
                           figname=f'Cam {prefix}', 
                           nms_proximity=args.nms)
            if not pv.next_video:
                break       
            else:
                continue
                   
        lock_name = out_name.replace('.json','.lock')
        # Delete old locks:
        if os.path.exists(lock_name) and (time.time() - os.stat(lock_name).st_mtime) / 60 > delete_old_locks_min:
            os.remove(lock_name) 
        lock = SoftFileLock(lock_name, timeout=0.1)
        try:
            with lock:
                reader = VideoReader(str(path), sample_time_secs=1)
                if first:
                    print(f'Starting with {len(reader)} frames from {reader.name}')
                    first = False

                pv = PlotVideo(reader, model, device, peaks, out_name=out_name, 
                               doplot=args.detectdir==None, 
                               figname=f'Cam {prefix}', 
                               nms_proximity=args.nms)
                if not pv.next_video:
                    break
        except Timeout:
            pass
            # print(f'Skipping {out_name} since locked')



if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Train the UNet on images and target masks')
    parser.add_argument('run', type=int, help='Run: loads this model and outputs in folder for this')    
    parser.add_argument('--loadmodel', type=str, default=None,  help='Load a specified model name in models folder -- overrides model specified in params')
    parser.add_argument('--suffix', type=str, default='avi',    help='Video suffix')
    parser.add_argument('--minval', type=float, default=-0.5,  help='Minimum peak value for detection')      
    parser.add_argument('--prefix', type=str, nargs='+', default='',  help='Select camera(s) with this, ex: ch1 ch2 will do cams 1 and 2')     
    parser.add_argument('--nms', type=float, default=12.,  help='Do NMS for 12 pixels')      
    parser.add_argument('--inputdir', type=str, default="/mnt/research/3D_Vision-Lab/Hens/Hens_2021_sec",  help='Input folder')
    parser.add_argument('--detectdir', type=str, default=None,  help='Output folder.  If None, then does not save detections')
    args = parser.parse_args()

    # ** Multiple threads do not work right now.  Just use a single --prefix **

    if len(args.prefix)>1 and args.detectdir:
        print(f'Running separate threads for prefixes: {args.prefix}')
        # do multiple threads when not plotting
        threads = []  # 1 thread per prefix
        for prefix in args.prefix:
            threads.append( threading.Thread(target=run_vid, args=(args,prefix)) )
        
        for t in threads:
            t.start()
        
        for t in threads:
            t.join()  #wait for all threads to complete
    else:
        if len(args.prefix):
            for prefix in args.prefix:
                run_vid(args,prefix)
        else:
            run_vid(args,'')
