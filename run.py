''' Run trained model on test data
    Can use plot_data.py to plot the output detections and heatmaps (use --outfrac to output results to .h5 file)
'''
import os
import sys
import torch
from torch.utils.data import DataLoader, random_split
import numpy as np
import platform
import json
import argparse

from evaluate_bce import evaluate_bce
from unet import UNetQuarter
from plot_data import save_scores, plot_scores

dirname = os.path.dirname(__file__)
dataset_path = os.path.join( os.path.dirname(dirname), 'cvdemos', 'image')
sys.path.append(dataset_path)
from image_dataset2 import ImageData
from synth_data import DrawData

from run_params import get_run_params, find_checkpoint
from train import get_criterion

def run_model(
        model,
        device,
        params):

    test_set = ImageData(os.path.join(params.data_dir, params.data_test),'test', 
                            radius=params.dilate, target_downscale=params.target_downscale,
                            transform = 'no_rot')
    if len(test_set)== 0:
        test_set = ImageData(os.path.join(params.data_dir, params.data_test),'validation', 
                            radius=params.dilate, target_downscale=params.target_downscale,
                            transform = 'no_rot')
    n_test = len(test_set)
    if n_test==0:
        raise Exception('No data in test set')
    data_pos_weight = test_set.pos_weight
    print(f'Running on {n_test} test images')

    # 3. Create data loaders
    num_workers = np.minimum(8,os.cpu_count()) if params.num_workers is None else params.num_workers
    loader_args = dict(batch_size=1, num_workers=num_workers, pin_memory=True)
    test_loader = DataLoader(test_set, shuffle=False, drop_last=False,
                             collate_fn=lambda batch: tuple(zip(*batch)), 
                             **loader_args)

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
    parser.add_argument('--outfrac', type=float, default=None,  help='Fraction of images to save, None: use input params')
    
    args = parser.parse_args()

    for run in args.runlist:

        params = get_run_params(run)       
        # Override load options: 
        params.load_opt = 'best'
        params.load_run = None
        if not args.outfrac is None:
            params.testoutfrac = args.outfrac
        print(80*"=")
        device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

        if params.model_name=='UNetQuarter':
            model = UNetQuarter(n_channels=3, n_classes=params.classes, max_chans=params.max_chans)            
        else:
            print(f'Error, unknown model {params.model_name}')

        cpoint, epoch = find_checkpoint(params)
        if cpoint:
            state_dict = torch.load(cpoint, map_location=device)
            model.load_state_dict(state_dict)
            print(f'Model loaded: {cpoint}')
        else:
            raise Exception('No model checkpoint')

        model = model.to(memory_format=torch.channels_last, device=device)

        run_model(
            model=model,
            device=device,
            params=params)
        