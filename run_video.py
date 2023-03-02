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

from unet import UNetBlocks
from plot_data import plot_detections

dirname = os.path.dirname(__file__)
dataset_path = os.path.join( os.path.dirname(dirname), 'cvdemos', 'image')
sys.path.append(dataset_path)
from heatmap_score import Peaks
from synth_data import DrawData
from image_dataset import up_scale_coords

from run_params import get_run_params, find_checkpoint

def is_30fps(name):
    return name[:3] == 'ch1' or name[:3] == 'ch2'

@torch.inference_mode()
def run_model(
        model,
        device,
        params,
        vfolder,
        nskip30fps=30,
        min_val=0):

    model.eval()
    peaks = Peaks(1, device, min_val=min_val)  # Do peak finding on CPU
    print(f'Device: {device}')

    run_dir = os.path.join(os.path.dirname(__file__), params.output_dir, f'{params.run:03d}')
    out_dir = os.path.join(run_dir,'video')
    os.makedirs(out_dir, exist_ok=True)

    videos = list(Path(vfolder).rglob('*.mp4'))
    print(f'Found {len(videos)} mp4 files')
    print(f'Storing detections in: {out_dir}')

    for path in videos[:3]:
        out_name = os.path.join(out_dir, path.name.replace('.mp4','.json') )
        if os.path.exists(out_name):
            print(f'Already completed {Path(out_name).name}')
            continue
        else:
            print(80*"=")
        nskip = nskip30fps if is_30fps(path.name) else nskip30fps // 2
        vid_detect = {  'video': str(path),
                        'nskip': nskip,                      
                      }
        all_scores=[]
        print(f'Processing: {path.name}')
        cap = cv.VideoCapture(str(path))
        inc = 0
        xout = [None, None]
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
            img = torch.from_numpy(frame.astype(np.float32).transpose(2,0,1)/255)[None,...]
            img = img.to(device=device, dtype=torch.float32, memory_format=torch.channels_last)            
            #start_model = timer()
            xout = model([img, xout[1]])
            #model_time = timer()-start_model
            #start_detections = timer()
            heatmap = xout[0]
            detections = peaks.peak_coords( heatmap )
            #start_scores = timer()            
            scores = peaks.heatmap_vals( heatmap, detections )[0][0].cpu().tolist()
            #detections_time = start_scores-start_detections
            #scores_time = timer()-start_scores
            imcoords = up_scale_coords( detections[0][0], params.target_downscale )
            vid_detect[str(inc)] = {"detections":imcoords.cpu().tolist(),
                                    "scores":scores} 
            #plot_detections(frame, imcoords, 12, str(inc) )
            all_scores += scores
            inc += 1
            nd = len(scores)
            out = str(min(nd,9)) if nd > 0 else '.'
            print(f'{out} {inc}') if inc%100==0 else print(out,end='')
            #frame_time = timer()-start
            #print(f'Total: {frame_time:.3f}, Read: {read_time:.3f}, Model: {model_time:.3f}, Detect: {detections_time:.3f}, Scores: {scores_time:.3f}')
        vid_detect['all_scores'] = all_scores
        with open(out_name, 'w') as f:
            json.dump(vid_detect, f, indent=2)
        print(f'===== Wrote: {Path(out_name).name}')

if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Train the UNet on images and target masks')
    parser.add_argument('run', type=int, help='Run')
    #parser.add_argument('--vfolder', type=str, default='/mnt/home/dmorris/Data/hens/videos',  help='Folder for videos')    
    parser.add_argument('--vfolder', type=str, default='/mnt/home/dmorris/Data/hens/Hens_2021',  help='Folder for videos')    
    args = parser.parse_args()

    params = get_run_params(args.run)

    logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

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
        logging.info(f'Model loaded from {cpoint}')
    else:
        raise Exception('No model checkpoint')

    model.to(device=device)

    run_model(
            model=model,
            device=device,
            params=params,
            vfolder=args.vfolder)
        