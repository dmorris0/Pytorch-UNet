''' Run trained model on test data
    Usage example
      python run.py 54 \
                --outfrac 5 \
                --loadmodel /mnt/research/3D_Vision_Lab/Hens/models/054_UNetQuarter.pth \
                --inputfile /mnt/research/3D_Vision_Lab/Hens/eggs/Eggs_ch1_23-06-04.h5 \
                --outputdir /mnt/scratch/dmorris/testruns/Eggs_ch1_23-06-04
    Gets parameters from run 54 using get_run_params()
    Loads model: 054_UNetQuarter.pth, which must match parameters in run 54
    if <num> in --outfrac <num> is > 0 (default), then saves images and heatmaps in output folder.
    Saves 1/<num> of the input images.  Ex. 10 will save 1/10 of input images in outputdir.

    Afterwards, use plot_data.py to plot the output detections and heatmaps (if --outfrac is set)
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


from evaluate_bce import evaluate_bce
from plot_data import save_scores, plot_scores

dirname = os.path.dirname(__file__)
dataset_path = os.path.join( os.path.dirname(dirname), 'cvdemos', 'image')
sys.path.append(dataset_path)
from image_dataset2 import ImageData
from synth_data import DrawData

from run_params import get_run_params, init_model
from train import get_criterion

def run_model(
        model,
        device,
        params,
        outfrac,
        inputfile,
        outputdir):

    if inputfile is None:
        inputfile = os.path.join(params.data_dir, params.data_test)

    test_set = ImageData( inputfile, radius=params.dilate, 
                          target_downscale=params.target_downscale,
                          transform = 'none')
    n_test = len(test_set)
    if n_test==0:
        raise Exception('No data in test set')
    print(f'Running on {n_test} images in: {inputfile}')

    # 3. Create data loaders
    num_workers = np.minimum(8,os.cpu_count()) if params.num_workers is None else params.num_workers
    loader_args = dict(batch_size=1, num_workers=num_workers, pin_memory=True)
    test_loader = DataLoader(test_set, shuffle=False, drop_last=False,
                             collate_fn=lambda batch: tuple(zip(*batch)), 
                             **loader_args)

    if outputdir is None:
        outputdir = os.path.join(os.path.dirname(__file__), params.output_dir, f'{params.run:03d}',Path(inputfile).stem)
    os.makedirs(outputdir, exist_ok=True)
    outname = os.path.join(outputdir,f'images.h5')

    # Find positive weights for single-pixel positives:
    criterion = get_criterion(params)

    test_loss, scores, test_dice, test_pr, test_re = evaluate_bce(
                    model, test_loader, device, criterion, params,
                    0, 0, outname, outfrac )

    print(f'Dice: {test_dice:.3}, Precision: {test_pr:.3}, Recall: {test_re:.3}')

    testscores = np.concatenate( ((0,test_loss,),scores) ) 

    save_scores(os.path.join(outputdir, "test_scores.csv"), testscores)

    print(f'To see plots again run: python plot_data.py {params.run} --test')
    if params.testoutfrac:
        dd = DrawData(outname, recalc_scores=True, do_nms = params.do_nms)
        dd.plot()

if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Run the trained model')
    parser.add_argument('runlist', type=int, nargs='+',  help='List of runs')
    parser.add_argument('--inputfile', type=str, default=None,  help='Input .h5 data filename including full path. (Assumes .json too).  In None, then finds it from params')
    parser.add_argument('--outputdir', type=str, default=None,  help='Output folder.  In None, then finds name from params')
    parser.add_argument('--outfrac', type=int, default=0,    help='Saves 1/outfrac of the image results (ex: 10 will save 1/10 images), 0 means none saved')
    parser.add_argument('--loadmodel', type=str, default=None,  help='Load a specified model name -- overrides model specified in params')
    
    args = parser.parse_args()

    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    for run in args.runlist:

        print(80*"=")

        params = get_run_params(run)       
        if not args.outfrac is None:
            params.testoutfrac = args.outfrac

        # If we don't specify a loadmodel, then want to always load the best checkpoint (rather than last checkpoint)
        params.load_opt = 'best'
        model, _ = init_model(params, device, args.loadmodel)
            
        run_model(
            model     = model,
            device    = device,
            params    = params,
            outfrac   = args.outfrac,
            inputfile = args.inputfile,
            outputdir = args.outputdir,
            )
        