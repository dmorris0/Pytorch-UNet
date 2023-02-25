import logging
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
import shutil

import wandb
from evaluate_bce import evaluate_bce
from unet import UNet, UNetSmall, UNetSmallQuarter,  UNetBlocks
from plot_data import save_scores, plot_scores
from run_params import get_run_params, get_run_dirs, find_last_checkpoints

dirname = os.path.dirname(__file__)
dataset_path = os.path.join( os.path.dirname(dirname), 'cvdemos', 'image')
sys.path.append(dataset_path)
from image_dataset import ImageData
from heatmap_score import Peaks, MatchScore


def get_criterion(params, pos_weight = torch.Tensor([1.])):
    if params.focal_loss_ag is None:
        criterion = nn.CrossEntropyLoss() if model.n_classes > 1 else nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    else:
        criterion = lambda inputs, targets: torchvision.ops.sigmoid_focal_loss(
            inputs, targets, alpha=params.focal_loss_ag[0], gamma=params.focal_loss_ag[1], reduction='mean')
    return criterion

def train_model(
        model,
        device,
        params):

    # 1. Create dataset
    if params.data_validation is None:
        dataset = ImageData(os.path.join(params.data_dir, params.data_train),'train', 
                            radius=params.dilate, target_downscale=params.target_downscale, 
                            rand_flip=True, n_previous_images = params.n_previous_images)
        data_pos_weight = dataset.pos_weight
        n_val = int(len(dataset) * 0.12)
        n_train = len(dataset) - n_val
        train_set, val_set = random_split(dataset, [n_train, n_val], generator=torch.Generator().manual_seed(0))
    else:
        train_set = ImageData(os.path.join(params.data_dir, params.data_train),'train',     
                              radius=params.dilate, target_downscale=params.target_downscale, 
                            rand_flip=True, n_previous_images = params.n_previous_images)
        val_set   = ImageData(os.path.join(params.data_dir, params.data_validation),'validation',
                              radius=params.dilate, target_downscale=params.target_downscale, 
                            rand_flip=False, n_previous_images = params.n_previous_images)
        n_train, n_val = len(train_set), len(val_set)
        data_pos_weight = train_set.pos_weight


    # 3. Create data loaders
    num_workers = np.minimum(8,os.cpu_count()) if params.num_workers is None else params.num_workers
    loader_args = dict(batch_size=params.batch_size, num_workers=num_workers, pin_memory=True)
    train_loader = DataLoader(train_set, shuffle=True, **loader_args)
    val_loader = DataLoader(val_set, shuffle=False, drop_last=True, **loader_args)

    dir_run, dir_checkpoint = get_run_dirs(params.output_dir, params.run)
    Path(dir_checkpoint).mkdir(parents=True, exist_ok=True)
    os.makedirs( os.path.join(dir_run,'val'), exist_ok=True)
    os.makedirs( os.path.join(dir_run,'train'), exist_ok=True)
    dir_plot = os.path.join(params.output_dir,'Plots')
    os.makedirs( dir_plot, exist_ok=True)

    # (Initialize logging)
    experiment = wandb.init(project='U-Net', resume='allow', anonymous='must')
    wandb.run.name = f'run-{params.run}-{wandb.run.name}'
    experiment.config.update(
        dict(epochs=params.epochs, batch_size=params.batch_size, learning_rate=params.lr,
             save_checkpoint=params.save_checkpoint, img_scale=params.scale, amp=params.amp)
    )
    with open(os.path.join(dir_run,'wandb-run-info.txt'),'w') as f:
        print(f'{wandb.run.name}',file=f)
        print(f'{wandb.run.get_url()}',file=f)
    with open(os.path.join(dir_run, 'params.json'),'w') as f:
        json.dump(vars(params), f, indent=4)    # Use vars() to convert it to a dict

    logging.info(f'''Starting training:
        Run:              {params.run}
        Output:           {params.output_dir}
        Epochs:           {params.epochs}        
        Batch size:       {params.batch_size}
        Learning rate:    {params.lr}
        Training size:    {n_train}
        Validation size:  {n_val}
        Checkpoints:      {params.save_checkpoint}
        Device:           {device.type}
        Images scaling:   {params.scale}
        Mixed Precision:  {params.amp}
        Focal Loss:       {params.focal_loss_ag}
        Dilate:           {params.dilate}
        Target Downscale: {params.target_downscale}
        Max Distance:     {params.max_distance}
    ''')

    # 4. Set up the optimizer, the loss, the learning rate scheduler and the loss scaling for AMP
    optimizer = optim.RMSprop(model.parameters(),
                              lr=params.lr, weight_decay=params.weight_decay, momentum=params.momentum, foreach=True)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'max', patience=5)  # goal: maximize Dice score
    grad_scaler = torch.cuda.amp.GradScaler(enabled=params.amp)

    # Find positive weights for single-pixel positives:
    pos_weight = torch.Tensor([data_pos_weight/params.target_downscale**2]).to(device)
    criterion = get_criterion(params, pos_weight)  # pos_weight is only used if not doing focal loss

    global_step = 0
    val_step = 0

    peaks = Peaks(1, device)
    matches = MatchScore(max_distance = params.max_distance/params.target_downscale)
    etscores = []
    bscores = []
    best_val_dice = 0
    best_epoch = 0

    # 5. Begin training
    for epoch in range(1, params.epochs + 1):
        model.train()
        epoch_loss = 0
        totscores = np.zeros( (5,))
        with tqdm(total=n_train, desc=f'Epoch {epoch}/{params.epochs}', unit='img') as pbar:
            for nb, batch in enumerate(train_loader):
                images, true_masks, centers, ncen = batch['image'], batch['targets'], batch['centers'], batch['ncen']

                images = images.to(device=device, dtype=torch.float32, memory_format=torch.channels_last)
                true_masks = true_masks.to(device=device, dtype=float)  # BCE requires float
                n_max = np.random.randint(params.n_previous_images+1) + 1 if params.n_previous_images and params.rand_previous else None

                with torch.autocast(device.type if device.type != 'mps' else 'cpu', enabled=params.amp):
                    masks_pred = model.apply_to_stack(images, n_max)
                    loss = criterion(masks_pred, true_masks)
                    detections = peaks.peak_coords( masks_pred.detach(), min_val=0.)                        
                    iscores,_,_ = matches.calc_match_scores( detections, centers.detach()/params.target_downscale, ncen.detach() )
                    totscores += np.concatenate( ((1,loss.item(),),iscores.sum(axis=0)) )


                optimizer.zero_grad(set_to_none=True)
                grad_scaler.scale(loss).backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), params.gradient_clipping)
                grad_scaler.step(optimizer)
                grad_scaler.update()

                pbar.update(images.shape[0])
                global_step += 1
                epoch_loss += loss.item()
                experiment.log({
                    'train loss': loss.item(),
                    'step': global_step,
                    'epoch': epoch
                })
                pbar.set_postfix(**{'loss ': epoch_loss/(nb+1)})

                # Evaluation round
                #division_step = (n_train // (10 * batch_size))
                #division_step = len(train_loader) 
                #if division_step > 0:
                #    if global_step % division_step == 0:
                # The above allows validation to run during a training epoch.  I commented it out and shifted the below
                # left 4 tabs so that it runs at the end of an epoch 
        histograms = {}
        for tag, value in model.named_parameters():
            tag = tag.replace('/', '.')
            if not torch.isinf(value).any():
                histograms['Weights/' + tag] = wandb.Histogram(value.data.cpu())
            if not torch.isinf(value.grad).any():
                histograms['Gradients/' + tag] = wandb.Histogram(value.grad.data.cpu())

        val_loss, scores, val_dice, val_pr, val_re = evaluate_bce(
                    model, val_loader, device, criterion, params.amp, 
                    params.target_downscale, params.max_distance, global_step,
                    os.path.join(dir_run,'val',f'step_{val_step:03d}.h5') )
        if val_step>0:
            os.remove(os.path.join(dir_run,'val',f'step_{val_step-1:03d}.h5'))  # delete previous file since pretty big
        bscores.append( np.concatenate( ((global_step,val_loss,),scores) ) )
        val_step += 1
        scheduler.step(val_dice)
        if val_dice > best_val_dice:
            best_epoch = epoch
            best_val_dice = val_dice

        logging.info(f'Validation Loss {val_loss:.3}, Dice {val_dice:.3f}, Pr {val_pr:.3f}, Re {val_re:.3f}')
        try:
            experiment.log({
                                'learning rate': optimizer.param_groups[0]['lr'],
                                'validation loss': val_loss,
                                'validation dice': val_dice,
                                'best_val_dice': best_val_dice,
                                'images': wandb.Image(images[0].cpu()),
                                'masks': {
                                    'true': wandb.Image(true_masks[0].float().cpu()),
                                    'pred': wandb.Image(masks_pred.argmax(dim=1)[0].float().cpu()),
                                },
                                'step': global_step,
                                'epoch': epoch,
                                'best_epoch': best_epoch,
                                **histograms
                            })
        except:
            pass
        
        with open(os.path.join(dir_run, 'train', f'log_{epoch:03d}.json'),'w') as f:
            json.dump({
                'epoch': epoch,
                'step': global_step, 
                'lr': optimizer.param_groups[0]['lr'],
                'val_dice': val_dice,
                'best_val_dice': best_val_dice,
                'best_epoch': best_epoch,
                }, f, indent=4)    # Use vars() to convert it to a dict        

        etscores.append(
            np.array([global_step, totscores[1]/totscores[0], *totscores[2:]]))
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
        if os.name == 'nt':        
            device = torch.device('cpu')  # My windows GPU is very slow
        else:
            device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        logging.info(f'Using device {device}')

        n_channels = (params.n_previous_images + 1) * 3
        if params.target_downscale==1:
            model = UNetSmall(n_channels=n_channels, n_classes=params.classes, max_chans=params.max_chans)
        elif params.target_downscale==4:
            #model = UNetSmallQuarter(n_channels=n_channels, n_classes=params.classes, max_chans=params.max_chans)
            model = UNetBlocks(n_channels=3, n_classes=params.classes, max_chans=params.max_chans,
                               pre_merge = params.pre_merge, post_merge = params.post_merge)            
        else:
            raise Exception(f'Invalid target_downscale: {params.target_downscale}')

        model = model.to(memory_format=torch.channels_last)

        logging.info(f'Network:\n'
                    f'\t{model.n_channels} input channels\n'
                    f'\t{model.n_classes} output channels (classes)\n'
                    f'\t{model.max_chans} max channels')

        if not params.load_opt is None:
            cpts = find_last_checkpoints(params.output_dir, params.load_run)
            state_dict = torch.load(cpts[0], map_location=device)
            # del state_dict['mask_values']
            model.load_state_dict(state_dict)
            logging.info(f'Model loaded from {cpts[0]}')

        model.to(device=device)

        train_model(
            model=model,
            device=device,
            params=params)
        