''' Run trained model on test data
    Usage example
      python run.py 54 \
                --outfrac 5 \
                --loadmodel /mnt/research/3D_Vision_Lab/Hens/models/054_UNetQuarter.pth \
                --inputfile /mnt/research/3D_Vision_Lab/Hens/eggs/Eggs_ch1_23-06-04.h5 \
                --runoutdir /mnt/scratch/dmorris/testruns/Eggs_ch1_23-06-04
    Gets parameters from run 54 using get_run_params()
    Loads model: 054_UNetQuarter.pth, which must match parameters in run 54
    if <num> in --outfrac <num> is > 0 (default), then saves images and heatmaps in output folder.
    Saves 1/<num> of the input images.  Ex. 10 will save 1/10 of input images in runoutdir.

    Afterwards, use plot_data.py to plot the output detections and heatmaps (if --outfrac is set)
    
    Daniel Morris, 2023
'''
import os
import sys
import torch
import numpy as np
import argparse
from pathlib import Path

import torchvision
torchvision.disable_beta_transforms_warning()
from torch.utils.data import DataLoader

from unet import UNetQuarter

image_path = str( Path(__file__).parents[1] / 'imagefunctions') 
sys.path.append(image_path)
from hens.synth_data import plot_simple_heatmap
from hens.VideoIOtorch import VideoReader
import matplotlib.pyplot as plt

if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Run the trained model')
    parser.add_argument('loadmodel', type=str, help='Load a trained model (.pth file)')
    parser.add_argument('videoname', type=str, help='Video file name')    
    parser.add_argument('--peakthresh', type=float, default=0, help='Peak value threshold in range -inf to inf')    
    parser.add_argument('--nth', type=int, default=15, help='Sample every nth image')    
    parser.add_argument('--outputdir', type=str, default=None, help='If provided then saves annotated images to this')
    
    args = parser.parse_args()

    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    print(f'Loading model: {args.loadmodel}')
    model = UNetQuarter(n_channels=3, n_classes=1, max_chans=96)            

    state_dict = torch.load(args.loadmodel, map_location=device)
    model.load_state_dict(state_dict)
    print(f'Model loaded: {args.loadmodel}')
    if args.outputdir:
        os.makedirs(args.outputdir, exist_ok=True)
    else:
        basename = None

    model = model.to(memory_format=torch.channels_last, device=device)
            
    vr = VideoReader(args.videoname, args.nth )        

    inum=0
    while True:
        img, _ = vr.get_next()
        if img is None:
            break    
        image = img[None,...].to(device=device, dtype=torch.float32, memory_format=torch.channels_last) / 255
        
        heatmap = model(image)
        
        if args.outputdir:
            basename = str( Path(args.outputdir) / (Path(args.videoname).stem + f'_{inum:03d}') )

        plot_simple_heatmap(image[0].cpu().permute( (1,2,0)).numpy(), 
                            heatmap[0,0].cpu().detach().numpy(), 
                            min_peak_val=args.peakthresh, 
                            basename = basename )
        inum += 1        
        plt.show()
                    
