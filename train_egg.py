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

import wandb
from evaluate_bce import evaluate_bce
from unet import UNet, UNetSmall, UNetSmallQuarter
from plot_data import save_scores, plot_scores

dirname = os.path.dirname(__file__)
dataset_path = os.path.join( os.path.dirname(dirname), 'cvdemos', 'image')

sys.path.append(dataset_path)
from image_dataset import ImageData
from synth_data import DrawData
from heatmap_score import Peaks, MatchScore

if not os.name =="nt":
    global_data_dir = '/mnt/home/dmorris/Data/eggs'
elif platform.node()=='DM-O':
    global_data_dir = 'D:/Data/Eggs/data'
else:
    raise Exception(f"Unknown platform: {platform.node()}")


def train_model(
        model,
        device,
        args):

    # 1. Create dataset
    if args.data_validation is None:
        dataset = ImageData(os.path.join(args.data_dir, args.data_train),'train', 
                            radius=args.dilate, target_downscale=args.target_downscale, rand_flip=True)
        data_pos_weight = dataset.pos_weight
        n_val = int(len(dataset) * 0.12)
        n_train = len(dataset) - n_val
        train_set, val_set = random_split(dataset, [n_train, n_val], generator=torch.Generator().manual_seed(0))
    else:
        train_set = ImageData(os.path.join(args.data_dir, args.data_train),'train',     
                              radius=args.dilate, target_downscale=args.target_downscale, rand_flip=True)
        val_set   = ImageData(os.path.join(args.data_dir, args.data_validation),'validation',
                              radius=args.dilate, target_downscale=args.target_downscale, rand_flip=False)
        n_train, n_val = len(train_set), len(val_set)
        data_pos_weight = train_set.pos_weight


    # 3. Create data loaders
    num_workers = np.minimum(8,os.cpu_count()) if args.num_workers is None else args.num_workers
    loader_args = dict(batch_size=args.batch_size, num_workers=num_workers, pin_memory=True)
    train_loader = DataLoader(train_set, shuffle=True, **loader_args)
    val_loader = DataLoader(val_set, shuffle=False, drop_last=True, **loader_args)

    run_dir = os.path.join(os.path.dirname(__file__), args.output_dir, f'{run:03d}')
    os.makedirs( os.path.join(run_dir,'val'), exist_ok=True)
    os.makedirs( os.path.join(run_dir,'train'), exist_ok=True)
    dir_checkpoint = Path(os.path.join(run_dir,'checkpoints'))
    Path(dir_checkpoint).mkdir(parents=True, exist_ok=True)

    # (Initialize logging)
    experiment = wandb.init(project='U-Net', resume='allow', anonymous='must')
    wandb.run.name = f'run-{args.run}-{wandb.run.name}'
    experiment.config.update(
        dict(epochs=args.epochs, batch_size=args.batch_size, learning_rate=args.lr,
             save_checkpoint=args.save_checkpoint, img_scale=args.scale, amp=args.amp)
    )
    with open(os.path.join(run_dir,'wandb-run-info.txt'),'w') as f:
        print(f'{wandb.run.name}',file=f)
        print(f'{wandb.run.get_url()}',file=f)

    logging.info(f'''Starting training:
        Run:              {args.run}
        Output:           {args.output_dir}
        Epochs:           {args.epochs}        
        Batch size:       {args.batch_size}
        Learning rate:    {args.lr}
        Training size:    {n_train}
        Validation size:  {n_val}
        Checkpoints:      {args.save_checkpoint}
        Device:           {device.type}
        Images scaling:   {args.scale}
        Mixed Precision:  {args.amp}
        Focal Loss:       {args.focal_loss_ag}
        Dilate:           {args.dilate}
        Target Downscale: {args.target_downscale}
        Max Distance:     {args.max_distance}
    ''')

    # 4. Set up the optimizer, the loss, the learning rate scheduler and the loss scaling for AMP
    optimizer = optim.RMSprop(model.parameters(),
                              lr=args.lr, weight_decay=args.weight_decay, momentum=args.momentum, foreach=True)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'max', patience=5)  # goal: maximize Dice score
    grad_scaler = torch.cuda.amp.GradScaler(enabled=args.amp)

    # Find positive weights for single-pixel positives:
    pos_weight = torch.Tensor([data_pos_weight/args.target_downscale**2]).to(device)
    print('Pos Weight:', pos_weight[0].item())
    if args.focal_loss_ag is None:
        criterion = nn.CrossEntropyLoss() if model.n_classes > 1 else nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    else:
        criterion = lambda inputs, targets: torchvision.ops.sigmoid_focal_loss(
            inputs, targets, alpha=args.focal_loss_ag[0], gamma=args.focal_loss_ag[1], reduction='mean')

    global_step = 0
    val_step = 0

    peaks = Peaks(1, device)
    matches = MatchScore(max_distance = args.max_distance/args.target_downscale)
    #tscores = []
    etscores = []
    bscores = []

    # 5. Begin training
    for epoch in range(1, args.epochs + 1):
        model.train()
        epoch_loss = 0
        totscores = np.zeros( (5,))
        with tqdm(total=n_train, desc=f'Epoch {epoch}/{args.epochs}', unit='img') as pbar:
            for nb, batch in enumerate(train_loader):
                images, true_masks, centers, ncen = batch['image'], batch['targets'], batch['centers'], batch['ncen']

                assert images.shape[1] == model.n_channels, \
                    f'Network has been defined with {model.n_channels} input channels, ' \
                    f'but loaded images have {images.shape[1]} channels. Please check that ' \
                    'the images are loaded correctly.'

                images = images.to(device=device, dtype=torch.float32, memory_format=torch.channels_last)
                # true_masks = true_masks.to(device=device, dtype=torch.long)
                true_masks = true_masks.to(device=device, dtype=float)  # BCE requires float

                with torch.autocast(device.type if device.type != 'mps' else 'cpu', enabled=args.amp):
                    masks_pred = model(images)
                    if model.n_classes == 1:
                        loss = criterion(masks_pred, true_masks)
                        detections = peaks.peak_coords( masks_pred.detach(), min_val=0.)                        
                        iscores = matches.calc_match_scores( detections, centers.detach()/args.target_downscale, ncen.detach() )
                        #tscores.append( np.concatenate( ((global_step,loss.item(),),iscores.sum(axis=0)) ) )
                        totscores += np.concatenate( ((1,loss.item(),),iscores.sum(axis=0)) )

                        #loss += dice_loss(F.sigmoid(masks_pred.squeeze(1)), true_masks.float(), multiclass=False)
                    else:
                        loss = criterion(masks_pred, true_masks)
                        #loss += dice_loss(
                        #    F.softmax(masks_pred, dim=1).float(),
                        #    F.one_hot(true_masks, model.n_classes).permute(0, 3, 1, 2).float(),
                        #    multiclass=True
                        #)

                optimizer.zero_grad(set_to_none=True)
                grad_scaler.scale(loss).backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.gradient_clipping)
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
                    model, val_loader, device, criterion, args.amp, 
                    args.target_downscale, args.max_distance, global_step,
                    os.path.join(run_dir,'val',f'step_{val_step:03d}.h5') )
        if val_step>0 and (val_step-1)%10:
            os.remove(os.path.join(run_dir,'val',f'step_{val_step:03d}.h5'))  # delete previous file since pretty big
        bscores.append( np.concatenate( ((global_step,val_loss,),scores) ) )
        val_step += 1
        scheduler.step(val_dice)

        logging.info(f'Validation Loss {val_loss:.3}, Dice {val_dice:.3f}, Pr {val_pr:.3f}, Re {val_re:.3f}')
        try:
            experiment.log({
                                'learning rate': optimizer.param_groups[0]['lr'],
                                'validation loss': val_loss,
                                'validation dice': val_dice,
                                'images': wandb.Image(images[0].cpu()),
                                'masks': {
                                    'true': wandb.Image(true_masks[0].float().cpu()),
                                    'pred': wandb.Image(masks_pred.argmax(dim=1)[0].float().cpu()),
                                },
                                'step': global_step,
                                'epoch': epoch,
                                **histograms
                            })
        except:
            pass

        etscores.append( np.array([global_step, totscores[1]/totscores[0], *totscores[2:]]))
        save_scores(os.path.join(run_dir,"train_scores.csv"), etscores)
        save_scores(os.path.join(run_dir,"val_scores.csv"), bscores)
        plot_scores(etscores, bscores, args.run, os.path.join(run_dir,"scores.png"))


        if args.save_checkpoint:
            state_dict = model.state_dict()
            # state_dict['mask_values'] = dataset.mask_values
            torch.save(state_dict, os.path.join(dir_checkpoint,f'checkpoint_epoch{epoch:03d}.pth'))
            # logging.info(f'Checkpoint {epoch} saved!')

class Args():
    def __init__(self,
                 run: int,
                 data_dir: str = global_data_dir,
                 data_train: str = 'Eggs_train.h5',
                 data_validation: str = 'Eggs_validation.h5',
                 output_dir: str = 'out_eggs',
                 epochs: int = 10,
                 batch_size: int = 4,
                 lr: float = 1e-6,
                 load: str = False,     # load model from a .pth file
                 scale: float = 0.5,    # Downscaling factor of the images
                 amp: bool = False,     # Use mixed precision
                 classes: int = 1,      # Number of classes
                 focal_loss_ag: tuple = (0.25, 2.0),  # None for no focal loss
                 dilate: float = 0.,
                 target_downscale: int = 1,  # Set to 4 to 1/4 size
                 max_distance: int = 12,
                 save_checkpoint: bool = True,
                 weight_decay: float = 1e-8,
                 momentum: float = 0.999,
                 gradient_clipping: float = 1.0,
                 num_workers: int = None,
                 max_chans: int = 64,
                 ):
        self.run = run
        self.data_dir = data_dir
        self.data_train = data_train
        self.data_validation = data_validation
        self.output_dir = output_dir
        self.epochs = epochs
        self.batch_size = batch_size
        self.lr = lr
        self.load = load
        self.scale = scale
        self.amp = amp
        self.classes = classes
        self.focal_loss_ag = focal_loss_ag
        self.dilate = dilate
        self.target_downscale = target_downscale
        self.max_distance = max_distance
        self.save_checkpoint = save_checkpoint
        self.weight_decay = weight_decay
        self.momentum = momentum
        self.gradient_clipping = gradient_clipping
        self.num_workers = num_workers
        self.max_chans = max_chans


if __name__ == '__main__':

    runlist = [16]
    for run in runlist:
        if run==1:
            args = Args(run, epochs = 1,
                        data_train='Eggs_train_small.h5', data_validation=None, 
                        focal_loss_ag=None,      
                        dilate=0.,  
                        target_downscale=4,
                        num_workers=0,
                        max_chans=96)
        elif run==2:
            args = Args(run, epochs = 80,
                        data_train='Eggs_train.h5', data_validation='Eggs_validation.h5', 
                        focal_loss_ag=(0.9,2.0),  
                        dilate=0.,  
                        target_downscale=4)
        elif run==3:
            args = Args(run, epochs = 1,
                        data_train='Eggs_train_small.h5', data_validation=None, 
                        focal_loss_ag=None,      
                        dilate=0.,  
                        target_downscale=4,
                        num_workers=0,
                        load=r"C:\Users\morri\Source\Repos\Pytorch-UNet\out_eggs\001\checkpoints\checkpoint_epoch010.pth")
        elif run==4:
            # Focal loss with alpha = 0.25 fails to detect eggs
            args = Args(run, epochs = 100,
                        data_train='Eggs_train.h5', data_validation='Eggs_validation.h5', 
                        focal_loss_ag=(0.25,2.0),  
                        dilate=0.,  
                        target_downscale=4)
        elif run==5:
            # Alpha 0.99
            args = Args(run, epochs = 100,
                        data_train='Eggs_train.h5', data_validation='Eggs_validation.h5', 
                        focal_loss_ag=(0.99,2.0),                          
                        dilate=0.,  
                        target_downscale=4)
        elif run==6:
            # Batch size: 8
            args = Args(run, epochs = 120,
                        data_train='Eggs_train.h5', data_validation='Eggs_validation.h5', 
                        focal_loss_ag=(0.9,2.0),                          
                        batch_size=8,
                        dilate=0.,  
                        target_downscale=4)
        elif run==7:
            # same as 2, but gamma of 3
            # Best so far from 6 - 10
            args = Args(run, epochs = 120,
                        data_train='Eggs_train.h5', data_validation='Eggs_validation.h5', 
                        focal_loss_ag=(0.9,3.0),                          
                        batch_size=4,
                        dilate=0.,  
                        target_downscale=4,
                        max_chans=64)
        elif run==8:
            # same as 2, but gamma of 4
            args = Args(run, epochs = 100,
                        data_train='Eggs_train.h5', data_validation='Eggs_validation.h5', 
                        focal_loss_ag=(0.9,4.0),                          
                        batch_size=4,
                        dilate=0.,  
                        target_downscale=4,
                        max_chans=64)
        elif run==9:
            # same as 2, except max_chans = 128
            args = Args(run, epochs = 120,
                        data_train='Eggs_train.h5', data_validation='Eggs_validation.h5', 
                        focal_loss_ag=(0.9,2.0),                          
                        batch_size=4,
                        dilate=0.,  
                        target_downscale=4,
                        max_chans=128)
        elif run==10:
            # same as 2, except max_chans = 128
            args = Args(run, epochs = 100,
                        data_train='Eggs_train.h5', data_validation='Eggs_validation.h5', 
                        focal_loss_ag=(0.9,3.0),                          
                        batch_size=4,
                        dilate=0.,  
                        target_downscale=4,
                        max_chans=128)
        elif run==11:
            # same as 7, with gamma of 3, but updated training data and validation
            args = Args(run, epochs = 100,
                        data_train='Eggs_train_23-02-12.h5', data_validation='Eggs_validation_23-02-12.h5', 
                        focal_loss_ag=(0.9,3.0),                          
                        batch_size=4,
                        dilate=0.,  
                        target_downscale=4,
                        max_chans=64)
        elif run==12:
            # same as 11, with gamma of 3, but training data excludes small dataset (our extra self-placed eggs)
            args = Args(run, epochs = 100,
                        data_train='Eggs_train_no_small_23-02-12.h5', data_validation='Eggs_validation_23-02-12.h5', 
                        focal_loss_ag=(0.9,3.0),                          
                        batch_size=4,
                        dilate=0.,  
                        target_downscale=4,
                        max_chans=64)
        elif run==13:
            # Repeat of 7, with latest data and tiles in validation
            args = Args(run, epochs = 120,
                        data_train='Eggs_train_23-02-15.h5', data_validation='Eggs_validation_tile_23-02-15.h5', 
                        focal_loss_ag=(0.9,3.0),                          
                        batch_size=4,
                        dilate=0.,  
                        target_downscale=4,
                        max_chans=64)
        elif run==14:
            # Repeat of 7, with latest data and full images in validation
            args = Args(run, epochs = 120,
                        data_train='Eggs_train_23-02-15.h5', data_validation='Eggs_validation_large_23-02-15.h5', 
                        focal_loss_ag=(0.9,3.0),                          
                        batch_size=4,
                        dilate=0.,  
                        target_downscale=4,
                        max_chans=64)
        elif run==15:
            # Run 13, but no small dataset in training
            args = Args(run, epochs = 120,
                        data_train='Eggs_train_no_small_23-02-15.h5', data_validation='Eggs_validation_tile_23-02-15.h5', 
                        focal_loss_ag=(0.9,3.0),                          
                        batch_size=4,
                        dilate=0.,  
                        target_downscale=4,
                        max_chans=64)
        elif run==16:
            # Repeat 13 but 96 channels
            args = Args(run, epochs = 120,
                        data_train='Eggs_train_23-02-15.h5', data_validation='Eggs_validation_tile_23-02-15.h5', 
                        focal_loss_ag=(0.9,3.0),                          
                        batch_size=4,
                        dilate=0.,  
                        target_downscale=4,
                        max_chans=96)
            
        print(80*"=")
        logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
        if os.name == 'nt':        
            device = torch.device('cpu')  # My windows GPU is very slow
        else:
            device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        logging.info(f'Using device {device}')

        if args.target_downscale==1:
            model = UNetSmall(n_channels=3, n_classes=args.classes, max_chans=args.max_chans)
        elif args.target_downscale==4:
            model = UNetSmallQuarter(n_channels=3, n_classes=args.classes, max_chans=args.max_chans)
        else:
            raise Exception(f'Invalid target_downscale: {args.target_downscale}')

        model = model.to(memory_format=torch.channels_last)

        logging.info(f'Network:\n'
                    f'\t{model.n_channels} input channels\n'
                    f'\t{model.n_classes} output channels (classes)\n'
                    f'\t{model.max_chans} max channels')

        if args.load:
            state_dict = torch.load(args.load, map_location=device)
            # del state_dict['mask_values']
            model.load_state_dict(state_dict)
            logging.info(f'Model loaded from {args.load}')

        model.to(device=device)

        train_model(
            model=model,
            device=device,
            args=args)
        