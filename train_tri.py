import argparse
import logging
import os
import random
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

import wandb
from evaluate_bce import evaluate_bce
from unet import UNet, UNetSmall, UNetSmallQuarter
#from utils.data_loading import BasicDataset, CarvanaDataset
#from utils.dice_score import dice_loss

dirname = os.path.dirname(__file__)
dataset_path = os.path.join( os.path.dirname(dirname), 'cvdemos', 'image')

sys.path.append(dataset_path)
from image_dataset import ImageData
from synth_data import DrawData

if os.name == 'nt':
    datadir = 'D:/Data/Triangles'
else:
    datadir = '/mnt/home/dmorris/Data/Triangles'

datafile = os.path.join(datadir, 'set10.h5')
#datafile = os.path.join(datadir, 'set2000.h5')

def train_model(
        model,
        run,
        datafile,
        dir_output,
        device,
        epochs: int = 5,
        batch_size: int = 1,
        learning_rate: float = 1e-5,
        #val_percent: float = 0.1,
        save_checkpoint: bool = True,
        img_scale: float = 0.5,
        amp: bool = False,
        weight_decay: float = 1e-8,
        momentum: float = 0.999,
        gradient_clipping: float = 1.0,
        focal_loss_ag: tuple = (0.25, 2.0), # None,  # or tuple = (0.25, 2.0),
        dilate: float = 0.,
        target_downscale: int = 1,
        max_distance: int = 12,
):

    # 1. Create dataset
    train_set = ImageData(datafile,'train',     radius=dilate, target_downscale=target_downscale)
    val_set   = ImageData(datafile,'validation',radius=dilate, target_downscale=target_downscale)

    n_train, n_val = len(train_set), len(val_set)

    # 3. Create data loaders
    loader_args = dict(batch_size=batch_size, num_workers=np.minimum(8,os.cpu_count()), pin_memory=True)
    train_loader = DataLoader(train_set, shuffle=True, **loader_args)
    val_loader = DataLoader(val_set, shuffle=False, drop_last=True, **loader_args)

    os.makedirs(dir_output,exist_ok=True)
    dir_val_output = os.path.join(dir_output,'val')
    os.makedirs(dir_val_output,exist_ok=True)

    # (Initialize logging)
    experiment = wandb.init(project='U-Net', resume='allow', anonymous='must')
    experiment.config.update(
        dict(epochs=epochs, batch_size=batch_size, learning_rate=learning_rate,
             save_checkpoint=save_checkpoint, img_scale=img_scale, amp=amp)
    )

    logging.info(f'''Starting training:
        Run:              {run}
        Output:           {dir_output}
        Epochs:           {epochs}        
        Batch size:       {batch_size}
        Learning rate:    {learning_rate}
        Training size:    {n_train}
        Validation size:  {n_val}
        Checkpoints:      {save_checkpoint}
        Device:           {device.type}
        Images scaling:   {img_scale}
        Mixed Precision:  {amp}
        Focal Loss:       {focal_loss_ag}
        Dilate:           {dilate}
        Target Downscale: {target_downscale}
        Max Distance:     {max_distance}
    ''')

    # 4. Set up the optimizer, the loss, the learning rate scheduler and the loss scaling for AMP
    optimizer = optim.RMSprop(model.parameters(),
                              lr=learning_rate, weight_decay=weight_decay, momentum=momentum, foreach=True)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'max', patience=5)  # goal: maximize Dice score
    grad_scaler = torch.cuda.amp.GradScaler(enabled=amp)

    # Find positive weights for single-pixel positives:
    pos_weight = torch.Tensor([train_set.pos_weight/target_downscale**2]).to(device)
    print('Pos Weight:', pos_weight[0].item())
    if focal_loss_ag is None:
        criterion = nn.CrossEntropyLoss() if model.n_classes > 1 else nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    else:
        criterion = lambda inputs, targets: torchvision.ops.sigmoid_focal_loss(
            inputs, targets, alpha=focal_loss_ag[0], gamma=focal_loss_ag[1], reduction='mean')

    global_step = 0

    #for epoch in range(1, 1 + 1):
    #    with tqdm(total=n_train, desc=f'Epoch {epoch}/{epochs}', unit='img') as pbar:
    #        for batch in train_loader:
    #            images, true_masks = batch['image'], batch['targets']


    # 5. Begin training
    for epoch in range(1, epochs + 1):
        model.train()
        epoch_loss = 0
        with tqdm(total=n_train, desc=f'Epoch {epoch}/{epochs}', unit='img') as pbar:
            for batch in train_loader:
                images, true_masks = batch['image'], batch['targets']

                assert images.shape[1] == model.n_channels, \
                    f'Network has been defined with {model.n_channels} input channels, ' \
                    f'but loaded images have {images.shape[1]} channels. Please check that ' \
                    'the images are loaded correctly.'

                images = images.to(device=device, dtype=torch.float32, memory_format=torch.channels_last)
                # true_masks = true_masks.to(device=device, dtype=torch.long)
                true_masks = true_masks.to(device=device, dtype=float)  # BCE requires float

                with torch.autocast(device.type if device.type != 'mps' else 'cpu', enabled=amp):
                    masks_pred = model(images)
                    if model.n_classes == 1:
                        loss = criterion(masks_pred, true_masks)
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
                torch.nn.utils.clip_grad_norm_(model.parameters(), gradient_clipping)
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
                pbar.set_postfix(**{'loss (batch)': loss.item()})

                # Evaluation round
                #division_step = (n_train // (10 * batch_size))
                division_step = len(train_loader)//2
                if division_step > 0:
                    if global_step % division_step == 0:
                        histograms = {}
                        for tag, value in model.named_parameters():
                            tag = tag.replace('/', '.')
                            if not torch.isinf(value).any():
                                histograms['Weights/' + tag] = wandb.Histogram(value.data.cpu())
                            if not torch.isinf(value.grad).any():
                                histograms['Gradients/' + tag] = wandb.Histogram(value.grad.data.cpu())

                        val_score, val_dice, val_pr, val_re = evaluate_bce(
                            model, val_loader, device, criterion, amp, target_downscale, max_distance,
                            os.path.join(dir_val_output,f'val_step_{global_step:03d}.h5') )
                        scheduler.step(val_dice)

                        logging.info(f'Validation Loss {val_score:.3f}, Dice {val_dice:.3f}, Pr {val_pr:.3f}, Re {val_re:.3f}')
                        try:
                            experiment.log({
                                'learning rate': optimizer.param_groups[0]['lr'],
                                'validation loss': val_score,
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


        if save_checkpoint:
            dir_checkpoint = Path(os.path.join(dir_output,'checkpoints'))
            Path(dir_checkpoint).mkdir(parents=True, exist_ok=True)
            state_dict = model.state_dict()
            # state_dict['mask_values'] = dataset.mask_values
            torch.save(state_dict, str(dir_checkpoint / 'checkpoint_epoch{}.pth'.format(epoch)))
            logging.info(f'Checkpoint {epoch} saved!')


def get_args():
    parser = argparse.ArgumentParser(description='Train the UNet on images and target masks')
    parser.add_argument('--epochs', '-e', metavar='E', type=int, default=5, help='Number of epochs')
    parser.add_argument('--batch-size', '-b', dest='batch_size', metavar='B', type=int, default=4, help='Batch size')
    parser.add_argument('--learning-rate', '-l', metavar='LR', type=float, default=1e-5,
                        help='Learning rate', dest='lr')
    parser.add_argument('--load', '-f', type=str, default=False, help='Load model from a .pth file')
    parser.add_argument('--scale', '-s', type=float, default=0.5, help='Downscaling factor of the images')
    parser.add_argument('--validation', '-v', dest='val', type=float, default=10.0,
                        help='Percent of the data that is used as validation (0-100)')
    parser.add_argument('--amp', action='store_true', default=False, help='Use mixed precision')
    parser.add_argument('--convtrans', action='store_true', default=False, help='Use transpose convolution for upsampling')
    parser.add_argument('--classes', '-c', type=int, default=1, help='Number of classes')

    return parser.parse_args()

class Args():
    def __init__(self,
                 run: int = -1,
                 input_data: str = 'set10.h5',
                 epochs: int = 10,
                 batch_size: int = 4,
                 lr: float = 1e-6,
                 load: str = False,     # load model from a .pth file
                 scale: float = 0.5,    # Downscaling factor of the images
                 amp: bool = False,     # Use mixed precision
                 convtrans: bool = False,  #use transpose convolution instead of bilinear upsampling
                 classes: int = 1,      # Number of classes
                 focal_loss_ag: tuple = (0.25, 2.0),  # None for no focal loss
                 dilate: float = 0.,
                 target_downscale: int = 1,  # Set to 4 to 1/4 size
                 max_distance: int = 12,
                 ):
        self.run = run
        self.input_data = input_data
        self.epochs = epochs
        self.batch_size = batch_size
        self.lr = lr
        self.load = load
        self.scale = scale
        self.amp = amp
        self.convtrans = convtrans
        self.classes = classes
        self.focal_loss_ag = focal_loss_ag
        self.dilate = dilate
        self.target_downscale = target_downscale
        self.max_distance = max_distance


if __name__ == '__main__':
    # args = get_args()

    start = 8
    end = 8
    for run in range(start, end+1):
        if run==1:
            args = Args(input_data='set10.h5', run=run, epochs=4, focal_loss_ag=None)
        elif run==2:
            args = Args(input_data='set10.h5', run=run, epochs=5, focal_loss_ag=None, dilate=2.5, target_downscale=4, load='C:/Users/morri/Source/Repos/Pytorch-UNet/checkpoints/checkpoint_epoch5.pth')
        elif run==3:
            args = Args(input_data='set2000.h5',  run=run, focal_loss_ag=(0.25,2.0))
        elif run==4:
            args = Args(input_data='set2000.h5', focal_loss_ag=None,       dilate=0.)
        elif run==5:
            args = Args(input_data='set2000.h5', focal_loss_ag=(0.99,2.0), dilate=0.)
        elif run==6:
            args = Args(input_data='set2000.h5', focal_loss_ag=(0.99,4.0), dilate=0.)
        elif run==7:
            args = Args(input_data='set2000.h5', focal_loss_ag=(0.9,4.0),  dilate=0.)
        elif run==8:
            args = Args(input_data='set2000.h5', focal_loss_ag=None,  dilate=2.5, target_downscale=4)
        elif run==9:
            args = Args(input_data='set2000.h5', focal_loss_ag=(0.9,2.0),  dilate=2.5, target_downscale=4)


        logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
        #device = torch.device('cpu')
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        logging.info(f'Using device {device}')

        # Change here to adapt to your data
        # n_channels=3 for RGB images
        # n_classes is the number of probabilities you want to get per pixel
        if args.target_downscale==1:
            model = UNetSmall(n_channels=1, n_classes=args.classes, bilinear=not args.convtrans)
        elif args.target_downscale==4:
            model = UNetSmallQuarter(n_channels=1, n_classes=args.classes, bilinear=not args.convtrans)
        else:
            raise Exception(f'Invalid target_downscale: {args.target_downscale}')

        # model = UNet(n_channels=3, n_classes=args.classes, bilinear=not args.convtrans)
        model = model.to(memory_format=torch.channels_last)

        logging.info(f'Network:\n'
                    f'\t{model.n_channels} input channels\n'
                    f'\t{model.n_classes} output channels (classes)\n'
                    f'\t{"Bilinear" if model.bilinear else "Transposed conv"} upscaling')

        if args.load:
            state_dict = torch.load(args.load, map_location=device)
            # del state_dict['mask_values']
            model.load_state_dict(state_dict)
            logging.info(f'Model loaded from {args.load}')

        model.to(device=device)
        try:
            train_model(
                model = model,
                run = args.run,
                datafile = os.path.join(datadir, args.input_data),
                dir_output = os.path.join(dirname,'output',f'run_{run:03d}'),
                epochs=args.epochs,
                batch_size=args.batch_size,
                learning_rate=args.lr,
                device=device,
                img_scale=args.scale,
                amp=args.amp,
                focal_loss_ag=args.focal_loss_ag,
                dilate=args.dilate,
                target_downscale=args.target_downscale,
                max_distance=args.max_distance,            
            )
        except torch.cuda.OutOfMemoryError:
            logging.error('Detected OutOfMemoryError! '
                        'Enabling checkpointing to reduce memory usage, but this slows down training. '
                        'Consider enabling AMP (--amp) for fast and memory efficient training')
            torch.cuda.empty_cache()
            model.use_checkpointing()
            train_model(
                model=model,
                epochs=args.epochs,
                batch_size=args.batch_size,
                learning_rate=args.lr,
                device=device,
                img_scale=args.scale,
                val_percent=args.val / 100,
                amp=args.amp
            )