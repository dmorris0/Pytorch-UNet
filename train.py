''' Train UNet model

    To run, first create a new run number in run_params.py and set your parameters
    there.  For example, I set parameters for run number 54.  Then to train do:
      python train.py 54

    If load_opt='last' in run_params.py, then running the above will load the last checkpoint
    and continue training.
    
    This does training and validation.  Plots and checkpoints are saved in: out_eggs/ and Plots/
    To run a trained model on test data use: run.py
    To plot performance as well as test results use: plot_data.py

    Daniel Morris, 2023

'''

import logging
import os
import sys
import torch
import torch.nn as nn
import torchvision
torchvision.disable_beta_transforms_warning()
import torchvision.ops
from pathlib import Path
from torch import optim
from torch.utils.data import DataLoader, random_split
from tqdm import tqdm
import numpy as np
import platform
import json
import argparse
import shutil

from evaluate_bce import evaluate_bce
from unet import UNetQuarter
from plot_data import save_scores, read_scores, plot_scores
from run_params import get_run_params, get_run_dirs, find_checkpoint, init_model

dirname = os.path.dirname(__file__)
dataset_path = os.path.join( os.path.dirname(dirname), 'cvdemos', 'image')
sys.path.append(dataset_path)
from image_dataset2 import ImageData
from image_fun import boxes_to_centers, down_scale_coords
from heatmap_score import Peaks, MatchScore


def get_criterion(params, pos_weight = torch.Tensor([1.])):
    if params.focal_loss_ag is None:
        criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    else:
        criterion = lambda inputs, targets: torchvision.ops.sigmoid_focal_loss(
            inputs, targets, alpha=params.focal_loss_ag[0], gamma=params.focal_loss_ag[1], reduction='mean')
    return criterion

def train_model( model, device, params, epoch):

    # 1. Create dataset
    if params.data_validation is None:
        dataset = ImageData(os.path.join(params.data_dir, params.data_train),'train', 
                            radius=params.dilate, target_downscale=params.target_downscale) #, 
                            #rand_flip=True, n_previous_images = params.n_previous_images)
        data_pos_weight = dataset.pos_weight
        n_val = int(len(dataset) * 0.12)
        n_train = len(dataset) - n_val
        train_set, val_set = random_split(dataset, [n_train, n_val], generator=torch.Generator().manual_seed(0))
    else:
        train_set = ImageData(os.path.join(params.data_dir, params.data_train),'train',     
                              radius=params.dilate, target_downscale=params.target_downscale) #, 
                            #rand_flip=True, n_previous_images = params.n_previous_images)
        val_set   = ImageData(os.path.join(params.data_dir, params.data_validation),'validation',
                              radius=params.dilate, target_downscale=params.target_downscale) #, 
                            #rand_flip=False, n_previous_images = params.n_previous_images)
        n_train, n_val = len(train_set), len(val_set)
        data_pos_weight = train_set.pos_weight


    # 3. Create data loaders
    if True:
        num_workers = np.minimum(8,os.cpu_count()) if params.num_workers is None else params.num_workers
    else:
        num_workers = 0
        print('Zero workers')
    loader_args = dict(batch_size=params.batch_size, num_workers=num_workers, pin_memory=True)
    # Add collate to generate lists instead of stacks to handle varying number of bounding boxes:
    train_loader = DataLoader(train_set, shuffle=True, 
                              collate_fn=lambda batch: tuple(zip(*batch)),  
                              **loader_args)
    val_loader = DataLoader(val_set, shuffle=False, drop_last=True, 
                            collate_fn=lambda batch: tuple(zip(*batch)),
                            **loader_args)

    dir_run, dir_checkpoint = get_run_dirs(params.output_dir, params.run)
    Path(dir_checkpoint).mkdir(parents=True, exist_ok=True)
    os.makedirs( os.path.join(dir_run,'val'), exist_ok=True)
    os.makedirs( os.path.join(dir_run,'train'), exist_ok=True)
    dir_plot = os.path.join(params.output_dir,'Plots')
    os.makedirs( dir_plot, exist_ok=True)

    with open(os.path.join(dir_run, 'params.json'),'w') as f:
        json.dump(vars(params), f, indent=4)    # Use vars() to convert it to a dict

    # Check if we are resuming or starting from scratch:
    if epoch>0:
        etscores = read_scores(os.path.join(dir_run, "train_scores.csv")).tolist()
        bscores = read_scores(os.path.join(dir_run, "val_scores.csv")).tolist()
        with open(os.path.join(dir_run, 'train', f'log_{epoch:03d}.json'),'r') as f:
            logparams = json.load(f)
        global_step, best_val_dice, best_epoch = logparams['step'], logparams['best_val_dice'], logparams['best_epoch']
        lr = logparams['lr']
        val_step = logparams['val_step'] if 'val_step' in logparams else logparams['epoch']
    else:
        lr = params.lr
        global_step, val_step, best_val_dice, best_epoch = 0, 0, 0, 0
        etscores, bscores = [], []

    logging.info(f'''Starting training:
        Run:              {params.run}
        UNet:             {params.model_name}
        Comment:          {params.comment}
        Start epoch:      {epoch+1}
        Load-opt:         {params.load_opt}
        Load-run:         {params.load_run}
        Epochs:           {params.epochs}     
        Max Channels:     {params.max_chans}   
        Batch size:       {params.batch_size}
        Learning rate:    {params.lr}
        Device:           {device.type}
        Focal Loss:       {params.focal_loss_ag}
    ''')

    # 4. Set up the optimizer, the loss, the learning rate scheduler and the loss scaling for AMP
    optimizer = optim.RMSprop(model.parameters(),
                              lr=lr, weight_decay=params.weight_decay, momentum=params.momentum, foreach=True)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'max', patience=5)  # goal: maximize Dice score
    grad_scaler = torch.cuda.amp.GradScaler(enabled=params.amp)

    # Find positive weights for single-pixel positives:
    pos_weight = torch.Tensor([data_pos_weight/params.target_downscale**2]).to(device)
    criterion = get_criterion(params, pos_weight)  # pos_weight is only used if not doing focal loss

    peaks = Peaks(1, device, min_val=0.)
    matches = MatchScore(max_distance = params.max_distance/params.target_downscale)

    # 5. Begin training
    for epoch in range(epoch+1, params.epochs + 1):
        model.train()
        epoch_loss = 0
        totscores = np.zeros( (5,))
        with tqdm(total=n_train, desc=f'Epoch {epoch}/{params.epochs}', unit='img') as pbar:

            for nb, (images, data) in enumerate(train_loader):  # This is image_dataset2

                images = torch.stack(images).to(device=device, dtype=torch.float32, memory_format=torch.channels_last)
                masks_true = torch.stack([x['mask'] for x in data]).to(device=device, dtype=torch.float32, memory_format=torch.channels_last)  # BCE requires float

                masks_pred = model(images)
                loss = criterion(masks_pred, masks_true)

                optimizer.zero_grad(set_to_none=True)
                grad_scaler.scale(loss).backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), params.gradient_clipping)
                grad_scaler.step(optimizer)
                grad_scaler.update()

                if epoch % params.dice_every_nth == 0:
                    bc = [[down_scale_coords(boxes_to_centers(x["boxes"].detach()), params.target_downscale)] for x in data]
                    detections = peaks.peak_coords( masks_pred.detach() )                        
                    iscores,_,_ = matches.calc_match_scores( detections, bc )  #/params.target_downscale )
                    scores = iscores.sum(axis=0)
                else:
                    scores = np.nan * np.ones((3,))
                totscores += np.concatenate( ((1,loss.item(),),scores) )
                
                pbar.update(images.shape[0])
                global_step += 1
                epoch_loss += loss.item()
                
                pbar.set_postfix(**{'loss ': epoch_loss/(nb+1)})

        val_loss, scores, val_dice, val_pr, val_re = evaluate_bce(
                    model, val_loader, device, criterion, params, epoch, global_step )
        
        bscores.append( np.concatenate( ((global_step,val_loss,),scores) ) )
        val_step += 1
        scheduler.step(val_dice)
        if not np.isnan(val_dice) and val_dice > best_val_dice:
            best_epoch = epoch
            best_val_dice = val_dice

        logging.info(f'Validation Loss {val_loss:.3}, Dice {val_dice:.3f}, Pr {val_pr:.3f}, Re {val_re:.3f}')
        
        with open(os.path.join(dir_run, 'train', f'log_{epoch:03d}.json'),'w') as f:
            json.dump({
                'epoch': epoch,
                'step': global_step, 
                'lr': optimizer.param_groups[0]['lr'],
                'val_dice': val_dice,
                'best_val_dice': best_val_dice,
                'best_epoch': best_epoch,
                'val_step': val_step,
                }, f, indent=4)    # Use vars() to convert it to a dict        

        etscores.append(np.array([global_step, totscores[1]/totscores[0], *totscores[2:]]))
        save_scores(os.path.join(dir_run, "train_scores.csv"), etscores)
        save_scores(os.path.join(dir_run, "val_scores.csv"), bscores)
        filename = os.path.join(dir_run,f"scores_{params.run:03d}.png")
        plot_scores(etscores, bscores, params.run, 
                    filename = filename, comment = params.comment)
        shutil.copy2(filename, dir_plot)

        if params.save_checkpoint:
            state_dict = model.state_dict()
            # state_dict['mask_values'] = dataset.mask_values
            checkpoint_name = os.path.join(dir_checkpoint,f'checkpoint_epoch{epoch:03d}.pth')
            torch.save(state_dict, checkpoint_name)
            # logging.info(f'Checkpoint {epoch} saved!')
            if epoch==best_epoch:
                shutil.copy2(checkpoint_name, os.path.join(dir_checkpoint,'best_checkpoint.pth'))

if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Train the UNet on images and target masks')
    parser.add_argument('runlist', type=int, nargs='+',  help='List of runs')
    args = parser.parse_args()

    for run in args.runlist:

        params = get_run_params(run)

        print(80*"=")
        logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
        device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

        model, epoch = init_model(params, device)

        train_model(model=model, device=device, params=params, epoch=epoch)
        