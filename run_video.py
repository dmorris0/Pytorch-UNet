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
import time

from unet import UNetBlocks

dirname = os.path.dirname(__file__)
dataset_path = os.path.join( os.path.dirname(dirname), 'cvdemos', 'image')
sys.path.append(dataset_path)
from heatmap_score import Peaks
from synth_data import DrawData
from image_dataset import up_scale_coords

from run_params import get_run_params, find_checkpoint

class VideoReader:
    def __init__(self, filename, samples_per_sec):
        self.name = filename
        self.cap = cv.VideoCapture(filename)
        self.samples_per_sec = samples_per_sec   
        self.fps = int(self.cap.get(cv.CAP_PROP_FPS))
        self.skip = self.fps // self.samples_per_sec
        self.nsamp = int(self.cap.get(cv.CAP_PROP_FRAME_COUNT)) // self.skip        
        self.index = -1

    def get_next(self):
        ret, frame = self.cap.read()
        self.index += 1
        frame_no = self.index * 30
        if ret:
            imsize = frame.shape
            if imsize[0] > 1200:
                imsize = (imsize[0]//2, imsize[1]//2, imsize[2])
                frame = cv.resize(frame, (imsize[1],imsize[0]), interpolation=cv.INTER_LINEAR)
            for _ in range(self.skip-1):
                _ = self.cap.read()
            self.index += self.skip
        return ret, frame, frame_no

    def __del__(self):
        if not self.cap is None:
            self.cap.release()

def get_image_name(im_folder, json_name):
    name = os.path.join(im_folder, json_name).replace('.json','.jpg')
    if os.path.exists(name):
        return name
    elif os.path.exists(name.replace('.jpg','.png')):
        return name.replace('.jpg','.png')
    else:
        print(f'Error: missing image for: {json_name} in {im_folder}')
        return None

class ImageReader:
    def __init__(self, im_folder, search='*.json'):
        self.im_list = [get_image_name(im_folder, x.name) for x in Path(im_folder).glob(search)]
        self.im_list.sort()
        self.nsamp = len(self.im_list)
        self.index = -1
        self.name = im_folder

    def get_next(self):
        self.index += 1
        if self.index < self.nsamp:
            frame = cv.imread(self.im_list[self.index])
            return True, frame, Path(self.im_list[self.index]).name
        else:
            return False, None, None

class PlotVideo:

    def __init__(self, reader, model, peaks, out_name=None, noplot=False, radius=12):

        self.reader = reader
        self.model = model
        self.peaks = peaks
        print(f'Running on {self.reader.nsamp} frames from {reader.name}')

        self.name = Path(reader.name).name
        self.fig = None
        self.radius = radius
        self.xout = [None, None]
        self.inc = -1
        self.next_video = True

        self.out_name = out_name
        self.noplot = noplot
        if self.out_name is None:
            self.get_next()
            self.plot()
            plt.show()
        else:
            self.run_all()

    def run_all(self):
        self.xout = [None, None]
        peak_vals, indices, x, y = [], [], [], []

        while True:
            self.get_next()
            if self.frame is None:
                break
            if self.peak_vals.size:
                peak_vals = peak_vals + self.peak_vals.tolist()
                indices = indices + [self.inc]*self.peak_vals.size
                x = x + self.imcoords[:,0].tolist()
                y = y + self.imcoords[:,1].tolist()            
            #if self.noplot:
            #    nd = self.peak_vals.size
            #    out = str(min(nd,9)) if nd > 0 else '.'
            #    print(f'{out} {self.inc+1}') if (self.inc+1)%100==0 else print(out,end='')
            if not self.noplot:
                self.plot()
                plt.show(block=False)

        vid_detect = {  'video': self.reader.name,
                        'samples_per_sec': self.reader.samples_per_sec,
                        'nskip': self.reader.skip,
                        'peak_vals': peak_vals,
                        'indices': indices,
                        'x': x,
                        'y': y,
                      }
        print(f'  Detections: {(np.array(peak_vals)>0).sum()} > 0, {(np.array(peak_vals)<0).sum()} <= 0', end='')        
        with open(self.out_name, 'w') as f:
            json.dump(vid_detect, f, indent=2)
        print(f'  Wrote: {Path(self.out_name).name}')
            

    def get_next(self):
        ret, frame, _ = self.reader.get_next()
        self.inc += 1
        if ret:
            self.frame = cv.cvtColor(frame, cv.COLOR_BGR2RGB)  # Trained on RGB images
            img = torch.from_numpy(self.frame.astype(np.float32).transpose(2,0,1)/255)[None,...]
            img = img.to(device=device, dtype=torch.float32, memory_format=torch.channels_last)            
            self.xout = model([img, self.xout[1]])
            heatmap = self.xout[0]            
            detections = self.peaks.peak_coords( heatmap )
            self.peak_vals = self.peaks.heatmap_vals( heatmap, detections )[0][0].cpu().numpy()
            self.imcoords = up_scale_coords( detections[0][0].cpu().numpy(), frame.shape[0]/heatmap.shape[2] )            
        else:
            self.frame = None
            self.peak_vals = None
            self.imcoords = None

    def plot(self):        
        if self.frame is None:
            plt.close(self.fig)
            print('Done video')
        else:
            self.plot_detections(self.frame, self.imcoords, self.peak_vals, name=self.name)
        
    def init_fig(self, figsize):
        if self.fig is None:
            self.fig = plt.figure(num='Image', figsize=figsize )
            if self.out_name is None:
                self.fig.canvas.mpl_connect('button_press_event',self.onclick)

    def onclick(self, event):
        ''' Left click: got to next image
            Right click: got to next group (ex from train to validation)
        '''
        # item = "image" if event.button==1 else "group"
        if event.button==1:
            self.get_next()
            self.plot()
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

    def plot_detections(self, img, targets, peak_vals, name=None):
        self.init_fig( (6,6) )
        self.fig.clf()
        ax = self.fig.add_subplot(1, 1, 1)
        ax.imshow(img, vmin=0, vmax=255, cmap="gray")
        if len(peak_vals)>0:
            self.draw_targets(ax, targets[peak_vals<0], self.radius, color=(.2,.5,1.))
            self.draw_targets(ax, targets[peak_vals>=0], self.radius, color=(1.,.5,.2))
        name = self.name if name is None else name
        ax.set_title(f'{name}: {self.inc}, N det: {targets.shape[0]}')
        self.fig.canvas.draw()
        self.fig.canvas.flush_events()

@torch.inference_mode()
def run_model(
        model,
        device,
        params,
        folder,
        prefix,
        do_png,
        min_val=-0.5,
        save=False,
        noplot=False):

    model.eval()
    peaks = Peaks(1, device, min_val=min_val)  # Do peak finding on CPU
    print(f'Device: {device}')

    run_dir = os.path.join(os.path.dirname(__file__), params.output_dir, f'{params.run:03d}')
    out_dir = os.path.join(run_dir,'video')
    os.makedirs(out_dir, exist_ok=True)

    if do_png:
        reader = ImageReader(folder, search=prefix+'*.json')
        pv = PlotVideo(reader, model, peaks)            
    else:
        search = prefix + '*.mp4'
        videos = list(Path(folder).rglob(search))
        videos.sort()

        print(f'Found {len(videos)} files of type: {search}')

        out_name = None
        if save:
            print(f'Storing detections in: {out_dir}')
        for path in videos:
            if save:
                out_name = os.path.join(out_dir, path.name.replace('.mp4','.json') )
                if os.path.exists(out_name):
                    print(f'Already completed {Path(out_name).name}')
                    continue
            reader = VideoReader(str(path), samples_per_sec=1)
            pv = PlotVideo(reader, model, peaks, out_name, noplot)
            if not pv.next_video:
                break
        #else:
        #    for path in videos:
        #        reader = VideoReader(str(path), samples_per_sec=1)
        #        pv = PlotVideo(reader, model, peaks)
        #        if not pv.next_video:
        #            break

if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Train the UNet on images and target masks')
    parser.add_argument('run', type=int, help='Run')
    #parser.add_argument('--vfolder', type=str, default='/mnt/home/dmorris/Data/hens/videos',  help='Folder for videos')    
    parser.add_argument('--folder', type=str, default='/mnt/home/dmorris/Data/hens/Hens_2021',  help='Folder for videos')    
    parser.add_argument('--prefix', type=str, default='',  help='search prefix')     
    parser.add_argument('--png', action='store_true',  help='Read PNGs instead')            
    parser.add_argument('--save', action='store_true',  help='Save detections')            
    parser.add_argument('--noplot', action='store_true',  help='No plotting -- only if saving detections')            
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
            folder=args.folder,
            prefix=args.prefix,
            do_png=args.png,
            save=args.save,
            noplot=args.noplot)
        