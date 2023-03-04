import os
import sys
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms
import torchvision.transforms.functional as TF
import torchvision.ops
from pathlib import Path
from torch import optim
from torch.utils.data import DataLoader, random_split
from tqdm import tqdm
import numpy as np
import platform
import json
import argparse
import cv2 as cv
import logging
from timeit import default_timer as timer
import matplotlib.pyplot as plt
from matplotlib.patches import Circle

from unet import UNetBlocks

dirname = os.path.dirname(__file__)
dataset_path = os.path.join( os.path.dirname(dirname), 'cvdemos', 'image')
sys.path.append(dataset_path)
from heatmap_score import Peaks
from synth_data import DrawData
from image_dataset import up_scale_coords

from run_params import get_run_params, find_checkpoint

class PlotVideo:

    def __init__(self, filename, model, peaks, samples_per_sec = 1, radius=12):

        self.cap = cv.VideoCapture(filename)
        self.model = model
        self.peaks = peaks
        self.samples_per_sec = samples_per_sec        
        self.fps = int(self.cap.get(cv.CAP_PROP_FPS))
        self.skip = self.fps // self.samples_per_sec
        self.nsamp = int(self.cap.get(cv.CAP_PROP_FRAME_COUNT)) // self.skip        
        print(f'Plotting {self.nsamp} frames from {filename}')

        self.name = Path(filename).name
        self.fig = None
        self.radius = radius
        self.xout = [None, None]
        self.inc = -1
        self.next_video = True
        self.do_next()
        plt.show()

    def __del__(self):
        if not self.cap is None:
            self.cap.release()

    def do_next(self):
        ret, frame = self.cap.read()
        self.inc += 1
        if ret:
            imsize = frame.shape
            if imsize[0] > 1200:
                imsize = (imsize[0]//2, imsize[1]//2, imsize[2])
                frame = cv.resize(frame, (imsize[1],imsize[0]), interpolation=cv.INTER_LINEAR)
            frame = cv.cvtColor(frame, cv.COLOR_BGR2RGB)  # Trained on RGB images
            img = torch.from_numpy(frame.astype(np.float32).transpose(2,0,1)/255)[None,...]
            img = img.to(device=device, dtype=torch.float32, memory_format=torch.channels_last)            
            self.xout = model([img, self.xout[1]])
            heatmap = self.xout[0]
            detections = self.peaks.peak_coords( heatmap )
            peak_vals = self.peaks.heatmap_vals( heatmap, detections )[0][0].cpu().numpy()
            imcoords = up_scale_coords( detections[0][0].cpu().numpy(), img.shape[1]//heatmap.shape[1] )
            self.plot_detections(frame, imcoords, peak_vals)
            for _ in range(self.skip-1):
                _ = self.cap.read()
        else:
            plt.close(self.fig)
            print('Done video')
        
    def init_fig(self, figsize):
        if self.fig is None:
            self.fig = plt.figure(num='Image', figsize=figsize )
            self.fig.canvas.mpl_connect('button_press_event',self.onclick)

    def onclick(self, event):
        ''' Left click: got to next image
            Right click: got to next group (ex from train to validation)
        '''
        # item = "image" if event.button==1 else "group"
        if event.button==1:
            self.do_next( )
        else:
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

    def plot_detections(self, img, targets, peak_vals):
        self.init_fig( (6,6) )
        self.fig.clf()
        ax = self.fig.add_subplot(1, 1, 1)
        ax.imshow(img, vmin=0, vmax=255, cmap="gray")
        if len(peak_vals)>0:
            self.draw_targets(ax, targets[peak_vals<0], self.radius, color=(.2,.5,1.))
            self.draw_targets(ax, targets[peak_vals>=0], self.radius, color=(1.,.5,.2))
        ax.set_title(f'{self.name}: {self.inc}, N det: {targets.shape[0]}')
        self.fig.canvas.draw()
        self.fig.canvas.flush_events()

def run_video(filename, out_name, model, peaks):
        cap = cv.VideoCapture(filename)
        samples_per_sec = 1
        fps = cap.get(cv.CAP_PROP_FPS)
        nskip = fps // samples_per_sec
        vid_detect = {  'video': filename,
                        'samples_per_sec': samples_per_sec,
                        'nskip': nskip,                      
                      }
        inc = 0
        xout = [None, None]
        all_scores=[]
        while True:
            #start = timer()
            for _ in range(nskip):
                ret, frame = cap.read()
            if not ret:
                print('')
                break
            #read_time = timer() - start
            imsize = frame.shape
            if imsize[0] > 1200:
                imsize = (imsize[0]//2, imsize[1]//2, imsize[2])
                frame = cv.resize(frame, (imsize[1],imsize[0]), interpolation=cv.INTER_LINEAR)
            frame = cv.cvtColor(frame, cv.COLOR_BGR2RGB)  # Trained on RGB images
            img = torch.from_numpy(frame.astype(np.float32).transpose(2,0,1)/255)[None,...]
            img = img.to(device=device, dtype=torch.float32, memory_format=torch.channels_last)            
            #start_model = timer()
            xout = model([img, xout[1]])
            #model_time = timer()-start_model
            #start_detections = timer()
            heatmap = xout[0]
            detections = peaks.peak_coords( heatmap )
            #start_scores = timer()            
            peak_vals = peaks.heatmap_vals( heatmap, detections )[0][0].cpu().tolist()
            #detections_time = start_scores-start_detections
            #scores_time = timer()-start_scores
            imcoords = up_scale_coords( detections[0][0], params.target_downscale )
            vid_detect[str(inc)] = {"detections":imcoords.cpu().tolist(),
                                    "peak_vals":peak_vals} 
            #plot_detections(frame, imcoords, 12, str(inc) )
            all_scores += peak_vals
            inc += 1
            nd = len(peak_vals)
            out = str(min(nd,9)) if nd > 0 else '.'
            print(f'{out} {inc}') if inc%100==0 else print(out,end='')
        vid_detect['all_scores'] = all_scores
        with open(out_name, 'w') as f:
            json.dump(vid_detect, f, indent=2)
        print(f'===== Wrote: {Path(out_name).name}')

@torch.inference_mode()
def run_model(
        model,
        device,
        params,
        vfolder,
        search='*.mp4',
        min_val=-0.5,
        do_plot=True):

    model.eval()
    peaks = Peaks(1, device, min_val=min_val)  # Do peak finding on CPU
    print(f'Device: {device}')

    run_dir = os.path.join(os.path.dirname(__file__), params.output_dir, f'{params.run:03d}')
    out_dir = os.path.join(run_dir,'video')
    os.makedirs(out_dir, exist_ok=True)

    videos = list(Path(vfolder).rglob(search))
    videos.sort()

    print(f'Found {len(videos)} files of type: {search}')

    if do_plot:
        for path in videos:
            pv = PlotVideo(str(path), model, peaks)
            if not pv.next_video:
                break
    else:
        print(f'Storing detections in: {out_dir}')
        for path in videos:
            out_name = os.path.join(out_dir, path.name.replace('.mp4','.json') )
            if os.path.exists(out_name):
                print(f'Already completed {Path(out_name).name}')
                continue
            else:
                print(80*"=")
            print(f'Processing: {path.name}')
            run_video(str(path), out_name, model, peaks)

if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Train the UNet on images and target masks')
    parser.add_argument('run', type=int, help='Run')
    #parser.add_argument('--vfolder', type=str, default='/mnt/home/dmorris/Data/hens/videos',  help='Folder for videos')    
    parser.add_argument('--vfolder', type=str, default='/mnt/home/dmorris/Data/hens/Hens_2021',  help='Folder for videos')    
    parser.add_argument('--prefix', type=str, default='',  help='search prefix')        
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    params = get_run_params(args.run)
    assert params.target_downscale==4, f'Assumes target_downscale is 4'
    # Load best checkpoint for current run:
    params.load_opt = 'best'
    params.load_run = None
    

    n_channels = (params.n_previous_images + 1) * 3
    assert params.target_downscale==4, f'Assumes downscaling by 4'
    model = UNetBlocks(n_channels=3, n_classes=params.classes, max_chans=params.max_chans,
                        pre_merge = params.pre_merge, post_merge = params.post_merge)            

    model = model.to(memory_format=torch.channels_last)

    cpoint, epoch = find_checkpoint(params)
    if cpoint:
        state_dict = torch.load(cpoint, map_location=device)
        model.load_state_dict(state_dict)
        logging.info(f'Model loaded: {cpoint}')
    else:
        raise Exception('No model checkpoint')

    model.to(device=device)

    run_model(
            model=model,
            device=device,
            params=params,
            vfolder=args.vfolder,
            search=args.prefix+'*.mp4')
        