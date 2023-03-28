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

from evaluate_bce import evaluate_bce
from unet import UNetBlocks, UNetTrack
from plot_data import save_scores, plot_scores

dirname = os.path.dirname(__file__)
dataset_path = os.path.join( os.path.dirname(dirname), 'cvdemos', 'image')
sys.path.append(dataset_path)
from image_dataset import ImageData
from synth_data import DrawData

from run_params import get_run_params, find_checkpoint
from train import get_criterion

def run_model(
        model,
        device,
        params):

    test_set = ImageData(os.path.join(params.data_dir, params.data_test),'test', 
                            radius=params.dilate, target_downscale=params.target_downscale, rand_flip=False,
                            n_previous_images = max(params.testrepeat, params.n_previous_images) )
    if len(test_set)== 0:
        test_set = ImageData(os.path.join(params.data_dir, params.data_test),'validation', 
                            radius=params.dilate, target_downscale=params.target_downscale, rand_flip=False,
                            n_previous_images = max(params.testrepeat, params.n_previous_images) )
    n_test = len(test_set)
    if n_test==0:
        raise Exception('No data in test set')
    data_pos_weight = test_set.pos_weight
    print(f'Running on {n_test} test images')

    # 3. Create data loaders
    num_workers = np.minimum(8,os.cpu_count()) if params.num_workers is None else params.num_workers
    loader_args = dict(batch_size=1, num_workers=num_workers, pin_memory=True)
    test_loader = DataLoader(test_set, shuffle=False, drop_last=False, **loader_args)

    dir_run = os.path.join(os.path.dirname(__file__), params.output_dir, f'{params.run:03d}')
    os.makedirs(os.path.join(dir_run,'test'), exist_ok=True)
    outname = os.path.join(dir_run,'test',f'output.h5')

    # Find positive weights for single-pixel positives:
    pos_weight = torch.Tensor([data_pos_weight/params.target_downscale**2]).to(device)
    criterion = get_criterion(params, pos_weight)

    test_loss, scores, test_dice, test_pr, test_re = evaluate_bce(
                    model, test_loader, device, criterion, params,
                    0, 0, outname )

    print(f'Dice: {test_dice:.3}, Precision: {test_pr:.3}, Recall: {test_re:.3}')

    testscores = np.concatenate( ((0,test_loss,),scores) ) 

    save_scores(os.path.join(dir_run, "test_scores.csv"), testscores)

    print(f'To see plots again run: python plot_data.py {params.run} --test')
    if params.testoutfrac:
        dd = DrawData(outname, recalc_scores=True, do_nms = params.do_nms)
        dd.plot()

if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Train the UNet on images and target masks')
    parser.add_argument('runlist', type=int, nargs='+',  help='List of runs')
    args = parser.parse_args()

    for run in args.runlist:

        params = get_run_params(run)
        params.load_opt = 'best'
        params.load_run = None
        print(80*"=")
        if os.name == 'nt':        
            device = torch.device('cpu')  # My windows GPU is very slow
        else:
            device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

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
            print(f'Model loaded: {cpoint}')
        else:
            raise Exception('No model checkpoint')

        model.to(device=device)

        run_model(
            model=model,
            device=device,
            params=params)
        