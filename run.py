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
from unet import UNet, UNetSmall, UNetSmallQuarter
from plot_data import save_scores, plot_scores

dirname = os.path.dirname(__file__)
dataset_path = os.path.join( os.path.dirname(dirname), 'cvdemos', 'image')
sys.path.append(dataset_path)
from image_dataset import ImageData
from synth_data import DrawData

from train import get_run_params, get_criterion


def run_model(
        model,
        device,
        params):

    test_set = ImageData(os.path.join(params.data_dir, params.data_test),'test', 
                            radius=params.dilate, target_downscale=params.target_downscale, rand_flip=False)
    if len(test_set)== 0:
        test_set = ImageData(os.path.join(params.data_dir, params.data_test),'validation', 
                            radius=params.dilate, target_downscale=params.target_downscale, rand_flip=False)
    n_test = len(test_set)
    if n_test==0:
        raise Exception('No data in test set')
    data_pos_weight = test_set.pos_weight
    print(f'Running on {n_test} test images')

    # 3. Create data loaders
    num_workers = np.minimum(8,os.cpu_count()) if params.num_workers is None else params.num_workers
    loader_args = dict(batch_size=1, num_workers=num_workers, pin_memory=True)
    test_loader = DataLoader(test_set, shuffle=False, drop_last=False, **loader_args)

    run_dir = os.path.join(os.path.dirname(__file__), params.output_dir, f'{params.run:03d}')
    os.makedirs(os.path.join(run_dir,'test'), exist_ok=True)
    outname = os.path.join(run_dir,'test',f'output.h5')

    # Find positive weights for single-pixel positives:
    pos_weight = torch.Tensor([data_pos_weight/params.target_downscale**2]).to(device)
    criterion = get_criterion(params, pos_weight)

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
    parser.add_argument('runlist', type=int, nargs='+',  help='List of runs')
    args = parser.parse_args()

    for run in args.runlist:

        params = get_run_params(run)

        print(80*"=")
        if os.name == 'nt':        
            device = torch.device('cpu')  # My windows GPU is very slow
        else:
            device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

        n_channels = (params.n_previous_images + 1) * 3
        if params.target_downscale==1:
            model = UNetSmall(n_channels=n_channels, n_classes=params.classes, max_chans=params.max_chans)
        elif params.target_downscale==4:
            model = UNetSmallQuarter(n_channels=n_channels, n_classes=params.classes, max_chans=params.max_chans)
        else:
            raise Exception(f'Invalid target_downscale: {params.target_downscale}')

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
            params=params)
        