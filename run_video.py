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

from unet import UNet, UNetSmall, UNetSmallQuarter
from plot_data import save_scores, plot_scores

dirname = os.path.dirname(__file__)
dataset_path = os.path.join( os.path.dirname(dirname), 'cvdemos', 'image')
sys.path.append(dataset_path)
from heatmap_score import Peaks
from synth_data import DrawData
from image_dataset import up_scale_coords

from run_params import get_run_params

def is_30fps(name):
    return name[:3] == 'ch1' or name[:3] == 'ch2'

@torch.inference_mode()
def run_model(
        model,
        device,
        params,
        vfolder,
        nskip30fps=30,
        min_val=-0.5):

    model.eval()
    peaks = Peaks(1, device)

    run_dir = os.path.join(os.path.dirname(__file__), params.output_dir, f'{params.run:03d}')
    out_dir = os.path.join(run_dir,'video')
    os.makedirs(out_dir, exist_ok=True)

    for path in Path(vfolder).rglob('*.mp4'):
        out_name = os.path.join(out_dir, path.name.replace('.mp4','.json') )
        nskip = nskip30fps if is_30fps(path.name) else nskip30fps // 2
        vid_detect = {  'video': str(path),
                        'nskip': nskip,                      
                      }
        print(f'Processing: {path.name}')
        cap = cv.VideoCapture(str(path))
        inc = 0
        xout = [None, None]
        while True:
            cap.set(cv.CAP_PROP_POS_FRAMES, inc * nskip)
            ret, frame = cap.read()
            if not ret:
                break
            imsize = frame.shape
            if imsize[0] > 1200:
                imsize = (imsize[0]//2, imsize[1]//2, imsize[2])
                frame = cv.resize(frame, (imsize[1],imsize[0]), interpolation=cv.INTER_LINEAR)
            img = torch.from_numpy(frame.astype(np.float32).transpose(2,0,1)/255)[None,...]
            xout = model([img, xout[1]])
            detections = peaks.peak_coords( xout[0], min_val=min_val )
            scores = peaks.heatmap_vals( xout[0], detections, to_numpy=True)
            vid_detect[str(inc)] = {"detections":detections,
                                    "scores":scores} 


    test_loss, scores, test_dice, test_pr, test_re = evaluate_bce(
                    model, test_loader, device, criterion, params.amp, 
                    params.target_downscale, params.max_distance, 0,
                    outname )

    print(f'Dice: {test_dice:.3}, Precision: {test_pr:.3}, Recall: {test_re:.3}')

    testscores = np.concatenate( ((0,test_loss,),scores) ) 

    save_scores(os.path.join(run_dir, "test_scores.csv"), testscores)

    print(f'To see plots again run: python plot_data.py {params.run} --test')
    dd = DrawData(outname)
    dd.plot()

if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Train the UNet on images and target masks')
    parser.add_argument('run', type=int, help='Run')
    parser.add_argument('--vfolder', type=str, default='/mnt/home/dmorris/Data/hens',  help='Folder for videos')    
    args = parser.parse_args()

    params = get_run_params(run)

    print(80*"=")
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    assert params.n_previous_images==0, f'Only works with 0 previous images'
    assert params.target_downscale==4, f'Assumes target_downscale is 4'

    n_channels = (params.n_previous_images + 1) * 3
    model = UNetSmallQuarter(n_channels=n_channels, n_classes=params.classes, max_chans=params.max_chans)
    model = model.to(memory_format=torch.channels_last)

    run_dir = os.path.join(os.path.dirname(__file__), params.output_dir, f'{run:03d}')
    dir_checkpoint = Path(os.path.join(run_dir,'checkpoints'))
    files = [str(x) for x in list(dir_checkpoint.glob('*.pth'))]
    files.sort()
    checkpoint_file = files[-1]
    state_dict = torch.load(checkpoint_file, map_location=device)
    model.load_state_dict(state_dict)
    model.to(device=device)

    run_model(
            model=model,
            device=device,
            params=params,
            vfolder=args.vfolder)
        