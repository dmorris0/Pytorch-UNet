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
import threading
from timeit import default_timer as timer
import time

from unet import UNetBlocks, UNetTrack

dirname = os.path.dirname(__file__)
dataset_path = os.path.join( os.path.dirname(dirname), 'cvdemos', 'image')
sys.path.append(dataset_path)
from heatmap_score import Peaks, MatchScore
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
        return ret, frame, frame_no, None

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
        self.json_list = list(Path(im_folder).glob(search))
        self.json_list.sort()
        self.im_list = [get_image_name(im_folder, x.name) for x in self.json_list]
        self.nsamp = len(self.im_list)
        self.index = -1
        self.name = im_folder

    def get_next(self):
        self.index += 1
        if self.index < self.nsamp:
            frame = cv.imread(self.im_list[self.index])
            with open(self.json_list[self.index]) as f:
                annotation = json.load(f)
            return True, frame, Path(self.im_list[self.index]).name, annotation
        else:
            return False, None, None, None

class RunDetections:
    
    def __init__(self,
                 reader, 
                 model, 
                 device, 
                 peaks, 
                 nms_proximity,
                 radius=12,
                 out_name=None ):
        self.reader = reader
        self.model = model
        self.device = device
        self.peaks = peaks
        self.proximity = nms_proximity
        self.matches = MatchScore(max_distance = radius)
        self.out_name = out_name
        self.nmisses = 0
        self.n = 0
        self.all_annotations = {'images':{},
                                'nmisses':[]}
        self.run()

    def run(self): 
        print(f'Running on {len(self.reader.im_list)} annotated images')
        while self.next():
            pass
        print(f'Found {self.nmisses} misses')
        print(f'Saving to: {self.out_name}')
        with open(self.out_name,'w') as f:
            json.dump(self.all_annotations,f, indent=2)

    def next(self): 
        ret, frame, name, annotations = self.reader.get_next()
        if ret:
            self.frame = cv.cvtColor(frame, cv.COLOR_BGR2RGB)  # Trained on RGB images
            img = torch.from_numpy(self.frame.astype(np.float32).transpose(2,0,1)/255)[None,...]
            img = img.to(device=self.device, dtype=torch.float32, memory_format=torch.channels_last)   
            
            xout = self.model.apply_to_stack(img, Nmax=1)
            heatmap = xout[0]            
            scale = img.shape[-1]/heatmap.shape[-1]

            detections = self.peaks.peak_coords( heatmap )
            peak_vals = self.peaks.heatmap_vals( heatmap, detections )
            if self.proximity > 0:
                detections, peak_vals = self.peaks.nms(detections, peak_vals, self.proximity / scale, to_torch=True)
            peak_vals = peak_vals[0][0].cpu().numpy()
            imcoords = up_scale_coords( detections[0][0].cpu().numpy(), scale )      

            if imcoords.size==0:
                tp_inds = np.array([])
            else:
                _,tp_inds,_ = self.matches.calc_match_scores( imcoords, np.array(annotations['targets']))
            if tp_inds.size:
                tp = np.zeros( (imcoords.shape[0],), dtype=bool)
                tp[tp_inds]=True
                misses = np.logical_not(tp).astype(np.int32).tolist()
                nmisses = sum(misses)
            else:
                misses = []
                nmisses = 0
            self.all_annotations['images'][name]={"targets":imcoords.tolist(), "misses":misses}
            self.all_annotations['nmisses'].append(nmisses)
            self.nmisses += nmisses   
            nsofar = len(self.all_annotations['nmisses'])
            if nsofar % 10 == 0:
                print('.',end='')
                if nsofar % 800 == 0:
                    print(f'{nsofar}')
        return ret

class PlotVideo:

    def __init__(self, 
                 reader, 
                 model, 
                 device, 
                 peaks, 
                 out_name=None, 
                 noplot=False, 
                 radius=12, 
                 figname='Camera', 
                 compare_name=None,
                 nms_proximity=0):

        self.reader = reader
        self.model = model
        self.device = device
        self.peaks = peaks
        self.proximity = nms_proximity
        self.matches = MatchScore(max_distance = radius)
        print(f'Running on {self.reader.nsamp} frames from {reader.name}')

        self.name = Path(reader.name).name
        self.figname = figname
        self.fig = None
        self.radius = radius
        self.xout = [None, None]
        self.inc = -1
        self.next_video = True
        self.annotations = None

        if compare_name is None:
            self.compare_annotations = None
        else:
            print(f'Comparing with: {compare_name}')
            with open(compare_name, 'r') as f:
                self.compare_annotations = json.load(f)
                self.compare_annotations['peak_vals'] = np.array(self.compare_annotations['peak_vals'])
                self.compare_annotations['indices'] = np.array(self.compare_annotations['indices'])
                self.compare_annotations['x'] = np.array(self.compare_annotations['x'])
                self.compare_annotations['y'] = np.array(self.compare_annotations['y'])

        self.out_name = out_name
        self.noplot = noplot
        if self.out_name is None:
            self.get_next()
            self.compare()
            self.plot()
            plt.show()
        else:
            self.run_all()

    def run_all(self):
        self.xout = [None, None]
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
            #if self.noplot:
            #    nd = self.peak_vals.size
            #    out = str(min(nd,9)) if nd > 0 else '.'
            #    print(f'{out} {self.inc+1}') if (self.inc+1)%100==0 else print(out,end='')
            if not self.noplot:
                self.plot()
                plt.show(block=False)
        total_time = timer() - start_time

        vid_detect = {  'video': self.reader.name,
                        'samples_per_sec': self.reader.samples_per_sec,
                        'nskip': self.reader.skip,
                        'peak_vals': peak_vals,
                        'indices': indices,
                        'x': x,
                        'y': y,
                      }
        print(f'  Detections: {(np.array(peak_vals)>0).sum()} > 0, {(np.array(peak_vals)<0).sum()} <= 0 --> {Path(self.out_name).name}.  Done in {total_time/60:.1} min')        
        with open(self.out_name, 'w') as f:
            json.dump(vid_detect, f, indent=2)
            

    def get_next(self):
        ret, frame, _, self.annotations = self.reader.get_next()
        self.inc += 1
        if ret:
            self.frame = cv.cvtColor(frame, cv.COLOR_BGR2RGB)  # Trained on RGB images
            img = torch.from_numpy(self.frame.astype(np.float32).transpose(2,0,1)/255)[None,...]
            img = img.to(device=self.device, dtype=torch.float32, memory_format=torch.channels_last)   
            
            self.xout = self.model.apply_to_stack(img, Nmax=1)
            scale = img.shape[-1]/self.xout.shape[-1]

            heatmap = self.xout[0]            
            detections = self.peaks.peak_coords( heatmap )
            peak_vals = self.peaks.heatmap_vals( heatmap, detections )
            if self.proximity > 0:
                detections, peak_vals = self.peaks.nms(detections, peak_vals, self.proximity / scale, to_torch=True)
            self.peak_vals = peak_vals[0][0].cpu().numpy()
            self.imcoords = up_scale_coords( detections[0][0].cpu().numpy(), scale )      

            if self.annotations is None:
                self.tp_inds = None
            else:
                if self.imcoords.size==0:
                    self.tp_inds = np.array([])
                else:
                    _,self.tp_inds,_ = self.matches.calc_match_scores( self.imcoords, np.array(self.annotations['targets']))

        else:
            self.frame = None
            self.peak_vals = None
            self.imcoords = None

    def compare(self):
        if not self.compare_annotations is None:
            prev_scores = self.compare_annotations['peak_vals'][np.nonzero(self.compare_annotations['indices']==self.inc)[0]]
            if prev_scores.size:
                prev_scores.sort()

            new_scores = self.peak_vals.copy()
            new_scores.sort()            
            if len(prev_scores)!=len(new_scores):
                print(f'Prev scores vs new: {prev_scores} vs. {new_scores}')
            elif len(prev_scores)>0:
                diff = np.abs(np.array(prev_scores) - new_scores).sum()
                if diff>0.01:
                    print(f'Prev scores vs new: {prev_scores} vs. {new_scores}')
                else:
                    print(f'{self.inc}: {len(prev_scores)} detections agree with loaded values')


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
                self.fig.canvas.mpl_connect('button_press_event',self.onclick)

    def onclick(self, event):
        ''' Left click: got to next image
            Right click: got to next group (ex from train to validation)
        '''
        # item = "image" if event.button==1 else "group"
        if event.button==1:
            self.get_next()
            self.compare()
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

    def plot_detections(self):
        #, img, targets, peak_vals, annotations=None, name=None):
        #self.frame, self.imcoords, self.peak_vals, self.annotations, name=self.name)
        self.init_fig( (6,6) )
        self.fig.clf()
        ax = self.fig.add_subplot(1, 1, 1)
        ax.imshow(self.frame)
        if not self.annotations is None:
            self.draw_targets(ax, self.annotations["targets"], self.radius*2, color=(1,1,1))
        if len(self.peak_vals)>0:
            tp = np.zeros( (self.imcoords.shape[0],), dtype=bool)
            tp[self.tp_inds]=True
            self.draw_targets(ax, self.imcoords[tp], self.radius, color=(.2,.5,1.))
            self.draw_targets(ax, self.imcoords[np.logical_not(tp)], self.radius, color=(1.,.5,.2))
            #self.draw_targets(ax, self.imcoords[self.peak_vals<0], self.radius, color=(.2,.5,1.))
            #self.draw_targets(ax, self.imcoords[self.peak_vals>=0], self.radius, color=(1.,.5,.2))
        ax.set_title(f'{self.name}: {self.inc}, N det: {self.imcoords.shape[0]}')
        self.fig.canvas.draw()
        self.fig.canvas.flush_events()

def are_we_already_done(args, out_dir, prefix):
    ''' Return True if completed all videos '''
    if args.compare or args.png or not args.save:
        return False
    search = prefix + '*.mp4'
    for path in Path(args.folder).rglob(search):
        out_name = os.path.join(out_dir, path.name.replace('.mp4','.json') )
        if not os.path.exists(out_name):
            return False
    return True

@torch.inference_mode()
def run_vid(args, prefix):

    logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    params = get_run_params(args.run)
    assert params.target_downscale==4, f'Assumes target_downscale is 4'
    # Load best checkpoint for current run:
    params.load_opt = 'best'
    params.load_run = None
    min_val = 0 # -0.5

    run_dir = os.path.join(os.path.dirname(__file__), params.output_dir, f'{params.run:03d}')
    out_dir = os.path.join(run_dir,'video')
    os.makedirs(out_dir, exist_ok=True)

    if are_we_already_done(args, out_dir, prefix):
        print(f'Already completed all videos of type: {prefix}*.mp4, so quitting')
        return True

    if params.model_name=='UNetBlocks':
        model = UNetBlocks(n_channels=3, n_classes=params.classes, max_chans=params.max_chans,
                           pre_merge = params.pre_merge, post_merge = params.post_merge)            
    elif params.model_name=='UNetTrack':
        model = UNetTrack(add_prev_im=params.add_prev_im, add_prev_out=params.add_prev_out,
                          n_classes=params.classes, max_chans=params.max_chans)

    model = model.to(memory_format=torch.channels_last)

    cpoint, epoch = find_checkpoint(params)
    if cpoint:
        state_dict = torch.load(cpoint, map_location=device)
        model.load_state_dict(state_dict)
        logging.info(f'Model loaded: {cpoint}')
    else:
        raise Exception('No model checkpoint')

    model.to(device=device)

    model.eval()
    peaks = Peaks(1, device, min_val=min_val)  # Do peak finding on CPU
    print(f'Device: {device}')


    if args.compare:
        reader = VideoReader(args.compare, samples_per_sec=1)
        compare_name = os.path.join(out_dir, Path(args.compare).name.replace('.mp4','.json') )
        pv = PlotVideo(reader, model, device, peaks, out_name=None, noplot=args.noplot, figname=f'Cam {prefix}', compare_name=compare_name)
        return

    if args.png:
        reader = ImageReader(args.folder, search=prefix+'*.json')
        if args.save:
            pv = RunDetections(reader, model, device, peaks, nms_proximity=args.nms, out_name = os.path.join(run_dir, "image_detections.json") )
        else:
            pv = PlotVideo(reader, model, device, peaks, nms_proximity=args.nms)            
    else:
        search = prefix + '*.mp4'
        videos = list(Path(args.folder).rglob(search))
        videos.sort()

        print(f'Found {len(videos)} files of type: {search}')

        out_name = None
        if args.save:
            print(f'Storing detections in: {out_dir}')
        nskip=0
        for path in videos:
            if args.save:
                out_name = os.path.join(out_dir, path.name.replace('.mp4','.json') )
                if os.path.exists(out_name):
                    nskip += 1
                    continue                
            if nskip:
                print(f'Skipping {nskip} completed videos')
                nskip=0
            reader = VideoReader(str(path), samples_per_sec=1)

            pv = PlotVideo(reader, model, device, peaks, out_name, args.noplot, figname=f'Cam {prefix}', nms_proximity=args.nms)
            if not pv.next_video:
                break


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Train the UNet on images and target masks')
    parser.add_argument('run', type=int, help='Run')    
    parser.add_argument('--folder', type=str, default='/mnt/research/3D_Vision_Lab/Hens/ImagesJPG',  help='Folder for videos')    
    #parser.add_argument('--folder', type=str, default='/mnt/home/dmorris/Data/hens/Hens_2021',  help='Folder for videos')    
    parser.add_argument('--prefix', type=str, nargs='+', default='',  help='search prefix')     
    parser.add_argument('--png', action='store_true',  help='Read JPGs or PNGs instead')            
    parser.add_argument('--savefile', type=str, default='',  help='Write detections')    
    parser.add_argument('--noplot', action='store_true',  help='No plotting -- only if saving detections')            
    parser.add_argument('--compare', type=str, default='',  help='MP4 file for comparing to previous run')    
    parser.add_argument('--nms', type=float, default=12.,  help='Do NMS for 12 pixels')      
    parser.add_argument('--save', action='store_true',  help='Save detections')            
    args = parser.parse_args()

    # OpenCV maximizes number of threads it uses
    # For multiple threaded operation avoid using too many:
    #nt = cv.getNumThreads()
    #cv.setNumThreads(nt//len(args.prefix))

    if args.noplot:
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
