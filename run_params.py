''' run_params.py
    Parameters for training and running
'''
import os
from pathlib import Path
import platform
import torch
from unet import UNetQuarter

def get_run_dirs(output_dir, run):
    dir_run = os.path.join(os.path.dirname(__file__), output_dir, f'{run:03d}')
    dir_checkpoint = os.path.join(dir_run,'checkpoints')    
    return dir_run, dir_checkpoint

def find_checkpoint(params):
    ''' Returns cpoint, epoch
        cpoint: name of checkpoint file
        epoch:  0 unless loading last checkpoint of current run
    '''
    cpoint, epoch = None, 0
    if params.load_opt is None:
        return cpoint, epoch
    run = params.run if params.load_run is None else params.load_run
    _,dir_checkpoint = get_run_dirs(params.output_dir, run)
    if params.load_opt=='best':
        cpoint = os.path.join(dir_checkpoint,'best_checkpoint.pth')
        cpoint = cpoint if os.path.exists(cpoint) else None
    elif params.load_opt=='last':
        if os.path.exists(dir_checkpoint):
            checkpoints = [str(x) for x in list( Path(dir_checkpoint).glob('check*.pth'))]
            if len(checkpoints):
                checkpoints.sort()
                cpoint = checkpoints[-1]
                if run==params.run:
                    epoch = int(cpoint[-7:-4])
    else:
        raise Exception(f'Invalid params.load_opt: {params.load_opt}')
    return cpoint, epoch


def init_model(params, device, loadmodel=None):
    if params.model_name=='UNetQuarter':
        model = UNetQuarter(n_channels=3, n_classes=params.classes, max_chans=params.max_chans)            
        assert params.target_downscale==4, f'Assumes downscaling by 4'
    else:
        print(f'Error, unknown model {params.model_name}')

    if loadmodel is None:            
        cpoint, epoch = find_checkpoint(params)
    else:
        # Load specified model
        cpoint = loadmodel
        if not os.path.exists(str(cpoint)):
            raise Exception(f'Cannot find model {cpoint}')
        epoch = 0
            
    if cpoint:
        state_dict = torch.load(cpoint, map_location=device)
        model.load_state_dict(state_dict)
        print(f'Model loaded: {cpoint}')

    model = model.to(memory_format=torch.channels_last, device=device)

    return model, epoch
    


class Params():
    def __init__(self,
                 run: int,
                 data_dir: str = '',
                 data_train: str = 'Eggs_train.h5',
                 data_validation: str = 'Eggs_validation.h5',
                 data_test: str = '',
                 output_dir: str = 'out_eggs',
                 model_name: str = 'UNetBlocks',
                 do_nms: bool = False,
                 testrepeat: int = 0,
                 testoutfrac: int = 0,
                 add_prev_im: bool = False,
                 add_prev_out: bool = False,
                 n_previous_images: int = 0,
                 n_previous_min: int = 0,
                 rand_previous: bool = False,  # If true, randomly select n_previous_min to n_previous_image for each batch
                 epochs: int = 10,
                 dice_every_nth: int = 1,
                 batch_size: int = 4,
                 lr: float = 1e-6,
                 load_opt: str = 'last',       # 'last': loads latest if available, 'best' loads best, None: does not load previous
                 load_run: int = None,         # if None then current run, else choose a run
                 scale: float = 0.5,    # Downscaling factor of the images
                 amp: bool = False,     # Use mixed precision
                 classes: int = 1,      # Number of classes
                 focal_loss_ag: tuple = (0.25, 2.0),  # None for no focal loss
                 dilate: float = 0.,
                 target_downscale: int = 4,  # Set to 4 to 1/4 size
                 max_distance: int = 12,
                 save_checkpoint: bool = True,
                 weight_decay: float = 1e-8,
                 momentum: float = 0.999,
                 gradient_clipping: float = 1.0,
                 num_workers: int = None,
                 max_chans: int = 64,
                 comment: str = '',
                 pre_merge: bool = False,
                 post_merge: bool = False,
                 do_wandb: bool = False,
                 ):
        self.run = run
        self.data_dir = data_dir
        self.data_train = data_train
        self.data_validation = data_validation
        self.data_test = data_test
        self.output_dir = output_dir
        self.model_name = model_name
        self.do_nms = do_nms
        self.testrepeat = testrepeat
        self.testoutfrac = testoutfrac
        self.add_prev_im = add_prev_im
        self.add_prev_out = add_prev_out
        self.n_previous_images = n_previous_images
        self.n_previous_min = n_previous_min
        self.rand_previous = rand_previous
        self.epochs = epochs
        self.dice_every_nth = dice_every_nth
        self.batch_size = batch_size
        self.lr = lr
        self.load_opt = load_opt
        self.load_run = load_run
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
        self.comment = comment
        self.pre_merge = pre_merge
        self.post_merge = post_merge
        self.do_wandb = do_wandb

        if not self.data_dir:
            if not os.name =="nt":
                self.data_dir = '/mnt/research/3D_Vision_Lab/Hens/eggs'
            elif platform.node()=='DM-O':
                self.data_dir = 'D:/Data/Eggs/data'
            elif platform.node()=="BAE003":
                self.data_dir = 'D:/Morris/Eggs'
            else:
                raise Exception(f"Unknown platform: {platform.node()}")


def get_run_params(run):
        if run==0:
            params = Params(run, epochs = 1,
                        data_train='Eggs_train_small.h5', 
                        data_validation=None,
                        load_opt=None,
                        target_downscale=4,      
                        model_name='UNetQuarter',
                        focal_loss_ag=None,
                        batch_size=1,
                        max_chans=8)
        elif run==1:
            params = Params(run, epochs = 4,
                        comment = 'test train',
                        data_train='Eggs_train_23-02-15.h5', 
                        data_validation='Eggs_validation_tile_23-02-15.h5', 
                        focal_loss_ag=None, #(0.85,4.0),
                        target_downscale=4,      
                        load_opt=None,                    
                        model_name='UNetQuarter',                        
                        batch_size=1,
                        max_chans=8)
        elif run==2:
            params = Params(run, epochs = 80,
                        data_train='Eggs_train.h5', data_validation='Eggs_validation.h5', 
                        focal_loss_ag=(0.9,2.0),  
                        dilate=0.,  
                        target_downscale=4)
        elif run==3:
            params = Params(run, epochs = 1,
                        data_train='Eggs_train_small.h5', data_validation=None, 
                        focal_loss_ag=None,      
                        dilate=0.,  
                        target_downscale=4,
                        num_workers=0,
                        load=r"C:\Users\morri\Source\Repos\Pytorch-UNet\out_eggs\001\checkpoints\checkpoint_epoch010.pth")
        elif run==4:
            # Focal loss with alpha = 0.25 fails to detect eggs
            params = Params(run, epochs = 100,
                        data_train='Eggs_train.h5', data_validation='Eggs_validation.h5', 
                        focal_loss_ag=(0.25,2.0),  
                        dilate=0.,  
                        target_downscale=4)
        elif run==5:
            # Alpha 0.99
            params = Params(run, epochs = 100,
                        data_train='Eggs_train.h5', data_validation='Eggs_validation.h5', 
                        focal_loss_ag=(0.99,2.0),                          
                        dilate=0.,  
                        target_downscale=4)
        elif run==6:
            # Batch size: 8
            params = Params(run, epochs = 120,
                        data_train='Eggs_train.h5', data_validation='Eggs_validation.h5', 
                        focal_loss_ag=(0.9,2.0),                          
                        batch_size=8,
                        dilate=0.,  
                        target_downscale=4)
        elif run==7:
            # same as 2, but gamma of 3
            # Best so far from 6 - 10
            params = Params(run, epochs = 120,
                        data_train='Eggs_train.h5', data_validation='Eggs_validation.h5', 
                        focal_loss_ag=(0.9,3.0),                          
                        batch_size=4,
                        dilate=0.,  
                        target_downscale=4,
                        max_chans=64)
        elif run==8:
            # same as 2, but gamma of 4
            params = Params(run, epochs = 100,
                        data_train='Eggs_train.h5', data_validation='Eggs_validation.h5', 
                        focal_loss_ag=(0.9,4.0),                          
                        batch_size=4,
                        dilate=0.,  
                        target_downscale=4,
                        max_chans=64)
        elif run==9:
            # same as 2, except max_chans = 128
            params = Params(run, epochs = 120,
                        data_train='Eggs_train.h5', data_validation='Eggs_validation.h5', 
                        focal_loss_ag=(0.9,2.0),                          
                        batch_size=4,
                        dilate=0.,  
                        target_downscale=4,
                        max_chans=128)
        elif run==10:
            # same as 2, except max_chans = 128
            params = Params(run, epochs = 100,
                        data_train='Eggs_train.h5', data_validation='Eggs_validation.h5', 
                        focal_loss_ag=(0.9,3.0),                          
                        batch_size=4,
                        dilate=0.,  
                        target_downscale=4,
                        max_chans=128)
        elif run==11:
            # same as 7, with gamma of 3, but updated training data and validation
            params = Params(run, epochs = 100,
                        data_train='Eggs_train_23-02-12.h5', data_validation='Eggs_validation_23-02-12.h5', 
                        focal_loss_ag=(0.9,3.0),                          
                        batch_size=4,
                        dilate=0.,  
                        target_downscale=4,
                        max_chans=64)
        elif run==12:
            # same as 11, with gamma of 3, but training data excludes small dataset (our extra self-placed eggs)
            params = Params(run, epochs = 100,
                        data_train='Eggs_train_no_small_23-02-12.h5', data_validation='Eggs_validation_23-02-12.h5', 
                        focal_loss_ag=(0.9,3.0),                          
                        batch_size=4,
                        dilate=0.,  
                        target_downscale=4,
                        max_chans=64)
        elif run==13:
            # Repeat of 7, with latest data and tiles in validation
            params = Params(run, epochs = 120,
                        data_train='Eggs_train_23-02-15.h5', data_validation='Eggs_validation_tile_23-02-15.h5', 
                        focal_loss_ag=(0.9,3.0),                          
                        batch_size=4,
                        dilate=0.,  
                        target_downscale=4,
                        max_chans=64)
        elif run==14:
            # Repeat of 7, with latest data and full images in validation
            params = Params(run, epochs = 120,
                        data_train='Eggs_train_23-02-15.h5', data_validation='Eggs_validation_large_23-02-15.h5', 
                        focal_loss_ag=(0.9,3.0),                          
                        batch_size=4,
                        dilate=0.,  
                        target_downscale=4,
                        max_chans=64)
        elif run==15:
            # Run 13, but no small dataset in training
            # Surprisingly this does better than with the small dataset.  
            # I will stop using the small dataset
            # This is good -- best solution for 64 channels
            params = Params(run, epochs = 120,
                        data_train='Eggs_train_no_small_23-02-15.h5', data_validation='Eggs_validation_tile_23-02-15.h5', 
                        focal_loss_ag=(0.9,3.0),                          
                        batch_size=4,
                        dilate=0.,  
                        target_downscale=4,
                        max_chans=64)
        elif run==16:
            # Repeat 13 but 96 channels
            # suprisingly worse!  I will change the maxchannels to apply only on the UNet portion
            # not on the pre-downsampling portion, see run 17
            params = Params(run, epochs = 120,
                        data_train='Eggs_train_23-02-15.h5', data_validation='Eggs_validation_tile_23-02-15.h5', 
                        focal_loss_ag=(0.9,3.0),                          
                        batch_size=4,
                        dilate=0.,  
                        target_downscale=4,
                        max_chans=96)
        elif run==17:
            # Compare to 16, where now I changed the maxchannels to apply only on the UNet portion
            # not on the pre-downsampling portion, see run 17
            # Also, excludes the small dataset, which should make a small improvement, see run 15
            params = Params(run, epochs = 80,
                        data_train='Eggs_train_no_small_23-02-15.h5', data_validation='Eggs_validation_tile_23-02-15.h5', 
                        focal_loss_ag=(0.9,3.0),                          
                        batch_size=4,
                        dilate=0.,  
                        target_downscale=4,
                        max_chans=96)
        elif run==18:
            params = Params(run, epochs = 80,
                        comment = 'Like 17 (96 channels) but adjust focal loss to improve precision',
                        data_train='Eggs_train_no_small_23-02-15.h5', data_validation='Eggs_validation_tile_23-02-15.h5', 
                        focal_loss_ag=(0.8,3.0),                          
                        batch_size=4,
                        dilate=0.,  
                        target_downscale=4,
                        max_chans=96)
        elif run==19:
            # This is pretty good for 96 channels
            # Alpha larger -> higher recall, alpha smaller -> higher precision
            params = Params(run, epochs = 100,
                        comment = 'Like 17, 18 (96 channels) but adjust focal loss to improve precision',
                        data_train='Eggs_train_no_small_23-02-15.h5', data_validation='Eggs_validation_tile_23-02-15.h5', 
                        focal_loss_ag=(0.85,3.0),                          
                        batch_size=4,
                        dilate=0.,  
                        target_downscale=4,
                        max_chans=96)
        elif run==20:
            params = Params(run, epochs = 100,
                        comment = 'Similar to 19, but gamma 4',
                        data_train='Eggs_train_no_small_23-02-15.h5', data_validation='Eggs_validation_tile_23-02-15.h5', 
                        focal_loss_ag=(0.85,4.0),                          
                        batch_size=4,
                        dilate=0.,  
                        target_downscale=4,
                        max_chans=96)
        elif run==21:
            # Building on run 15, but with 23-02-16 dataset now adds empty images in training
            params = Params(run, epochs = 120,
                        comment='Building on run 15, but with 23-02-16 dataset now adds empty images in training',
                        data_train='Eggs_train_23-02-16.h5', data_validation='Eggs_validation_tile_23-02-16.h5', 
                        data_test='Eggs_validation_large_23-02-16.h5',
                        focal_loss_ag=(0.9,3.0),                          
                        batch_size=4,
                        dilate=0.,  
                        target_downscale=4,
                        max_chans=64)
        elif run==22:
            # Building on run 20, but with 23-02-16 dataset now adds empty images in training
            # Runs 22 and 21 are similar on tile validations, but 22 is better on full image validation
            # I need to see if this is because of focal loss or because of max_chans ...
            params = Params(run, epochs = 100,
                        comment = 'Building on run 20, but with 23-02-16 dataset now adds empty images in training',
                        data_train='Eggs_train_23-02-16.h5', 
                        data_validation='Eggs_validation_tile_23-02-16.h5', 
                        data_test='Eggs_validation_large_23-02-16.h5',
                        focal_loss_ag=(0.85,4.0),                          
                        batch_size=4,
                        dilate=0.,  
                        target_downscale=4,
                        max_chans=96)
        elif run==23:
            params = Params(run, epochs = 100,
                        comment = 'Same as 22 but with 5 previous images',
                        data_train='Eggs_train_23-02-18.h5', 
                        data_validation='Eggs_validation_tile_23-02-18.h5', 
                        data_test='Eggs_validation_large_23-02-18.h5',
                        n_previous_images=5,
                        focal_loss_ag=(0.85,4.0),                          
                        batch_size=4,
                        dilate=0.,  
                        target_downscale=4,
                        max_chans=96)
        elif run==24:
            params = Params(run, epochs = 100,
                        comment = '0 previous images, train 02-24, No merge',
                        data_train='Eggs_train_23-02-24.h5', 
                        data_validation='Eggs_validation_tile_23-02-18.h5', 
                        data_test='Eggs_validation_large_23-02-18.h5',
                        n_previous_images=0,
                        pre_merge = False,
                        post_merge = False,
                        focal_loss_ag=(0.85,4.0),                          
                        batch_size=6,
                        dilate=0.,  
                        target_downscale=4,
                        max_chans=96)
        elif run==25:
            params = Params(run, epochs = 100,
                        comment = '2 previous images, train 02-24, Pre and Post merge',
                        data_train='Eggs_train_23-02-24.h5', 
                        data_validation='Eggs_validation_tile_23-02-18.h5', 
                        data_test='Eggs_validation_large_23-02-18.h5',
                        n_previous_images=2,
                        pre_merge = True,
                        post_merge = True,
                        focal_loss_ag=(0.85,4.0),                          
                        batch_size=6,
                        dilate=0.,  
                        target_downscale=4,
                        max_chans=96)
        elif run==26:
            params = Params(run, epochs = 100,
                        comment = '3 previous images, train 02-24, Post-merge',
                        data_train='Eggs_train_23-02-24.h5', 
                        data_validation='Eggs_validation_tile_23-02-18.h5', 
                        data_test='Eggs_validation_large_23-02-18.h5',
                        n_previous_images=2,
                        pre_merge = False,
                        post_merge = True,
                        focal_loss_ag=(0.85,4.0),                          
                        batch_size=6,
                        dilate=0.,  
                        target_downscale=4,
                        max_chans=96)
        elif run==27:
            params = Params(run, epochs = 100,
                        comment = '5 previous images, Pre-merge',
                        data_train='Eggs_train_23-02-24.h5', 
                        data_validation='Eggs_validation_tile_23-02-18.h5', 
                        data_test='Eggs_validation_large_23-02-18.h5',
                        n_previous_images=2,
                        pre_merge = True,
                        post_merge = False,
                        focal_loss_ag=(0.85,4.0),                          
                        batch_size=6,
                        dilate=0.,  
                        target_downscale=4,
                        max_chans=96)
        elif run==28:
            params = Params(run, epochs = 100,
                        comment = 'Continue 25: 2 previous images, train 02-24, Pre and Post merge',
                        data_train='Eggs_train_23-02-24.h5', 
                        data_validation='Eggs_validation_tile_23-02-18.h5', 
                        data_test='Eggs_validation_large_23-02-18.h5',
                        load_opt='last',
                        load_run=25,
                        n_previous_images=2,
                        pre_merge = True,
                        post_merge = True,
                        focal_loss_ag=(0.85,4.0),                          
                        batch_size=6,
                        dilate=0.,  
                        target_downscale=4,
                        max_chans=96)
        elif run==29:
            params = Params(run, epochs = 100,
                        comment = 'Continue 26: 3 previous images, train 02-24, Post-merge',
                        data_train='Eggs_train_23-02-24.h5', 
                        data_validation='Eggs_validation_tile_23-02-18.h5', 
                        data_test='Eggs_validation_large_23-02-18.h5',
                        load_opt='last',
                        load_run=26,
                        n_previous_images=2,
                        pre_merge = False,
                        post_merge = True,
                        focal_loss_ag=(0.85,4.0),                          
                        batch_size=6,
                        dilate=0.,  
                        target_downscale=4,
                        max_chans=96)
        elif run==30:
            params = Params(run, epochs = 100,
                        comment = 'Continue 27: 3 previous images, Pre-merge',
                        data_train='Eggs_train_23-02-24.h5', 
                        data_validation='Eggs_validation_tile_23-02-18.h5', 
                        data_test='Eggs_validation_large_23-02-18.h5',
                        load_opt='last',
                        load_run=27,
                        n_previous_images=2,
                        pre_merge = True,
                        post_merge = False,
                        focal_loss_ag=(0.85,4.0),                          
                        batch_size=6,
                        dilate=0.,  
                        target_downscale=4,
                        max_chans=96)
        elif run==31:
            params = Params(run, epochs = 60,
                        comment = '0 previous, max_chans to apply all levels',
                        data_train='Eggs_train_23-02-24.h5', 
                        data_validation='Eggs_validation_tile_23-02-18.h5', 
                        data_test='Eggs_validation_large_23-02-18.h5',
                        load_opt=None,
                        load_run=None,
                        n_previous_images=0,
                        pre_merge = False,
                        post_merge = False,
                        focal_loss_ag=(0.85,4.0),                          
                        batch_size=6,
                        max_chans=96)
        elif run==32:
            params = Params(run, epochs = 60,
                        comment = '0 to 4 previous, post_merge',
                        data_train='Eggs_train_23-02-24.h5', 
                        data_validation='Eggs_validation_tile_23-02-18.h5', 
                        data_test='Eggs_validation_large_23-02-18.h5',
                        load_opt=None,
                        load_run=None,
                        n_previous_images=4,
                        rand_previous=True,
                        pre_merge = False,
                        post_merge = True,
                        focal_loss_ag=(0.85,4.0),                          
                        batch_size=6,
                        max_chans=96)
        elif run==33:
            params = Params(run, epochs = 60,
                        comment = '0 to 4 previous, pre_merge',
                        data_train='Eggs_train_23-02-24.h5', 
                        data_validation='Eggs_validation_tile_23-02-18.h5', 
                        data_test='Eggs_validation_large_23-02-18.h5',
                        load_opt=None,
                        load_run=None,
                        n_previous_images=4,
                        rand_previous=True,
                        pre_merge = True,
                        post_merge = False,
                        focal_loss_ag=(0.85,4.0),                          
                        batch_size=6,
                        max_chans=96)
        elif run==34:
            params = Params(run, epochs = 50,
                        comment = '0 previous, continues 31',
                        data_train='Eggs_train_23-02-25.h5', 
                        data_validation='Eggs_validation_tile_23-02-25.h5', 
                        data_test='Eggs_validation_large_23-02-25.h5',
                        load_opt='last',
                        load_run=31,
                        n_previous_images=0,
                        pre_merge = False,
                        post_merge = False,
                        focal_loss_ag=(0.85,4.0),                          
                        batch_size=8,
                        max_chans=96)
        elif run==35:
            params = Params(run, epochs = 50,
                        comment = '0 to 4 previous, post_merge, continues 32',
                        data_train='Eggs_train_23-02-25.h5', 
                        data_validation='Eggs_validation_tile_23-02-25.h5', 
                        data_test='Eggs_validation_large_23-02-25.h5',
                        load_opt='last',
                        load_run=32,
                        n_previous_images=4,
                        rand_previous=True,
                        pre_merge = False,
                        post_merge = True,
                        focal_loss_ag=(0.85,4.0),                          
                        batch_size=8,
                        max_chans=96)
        elif run==36:
            params = Params(run, epochs = 50,
                        comment = '0 to 4 previous, pre_merge, continues 33',
                        data_train='Eggs_train_23-02-25.h5', 
                        data_validation='Eggs_validation_tile_23-02-25.h5', 
                        data_test='Eggs_validation_large_23-02-25.h5',
                        load_opt='last',
                        load_run=33,
                        n_previous_images=4,
                        rand_previous=True,
                        pre_merge = True,
                        post_merge = False,
                        focal_loss_ag=(0.85,4.0),                          
                        batch_size=8,
                        max_chans=96)
        elif run==37:
            params = Params(run, epochs = 50,
                        comment = '0 to 4 previous, pre_merge and post-merge',
                        data_train='Eggs_train_23-02-25.h5', 
                        data_validation='Eggs_validation_tile_23-02-25.h5', 
                        data_test='Eggs_validation_large_23-02-25.h5',
                        load_opt=None,
                        load_run=None,
                        n_previous_images=4,
                        rand_previous=True,
                        pre_merge = True,
                        post_merge = True,
                        focal_loss_ag=(0.85,4.0),                          
                        batch_size=8,
                        max_chans=96)
        elif run==38:
            params = Params(run, epochs = 120,
                        comment = '0 previous, 23-02-26 training (more negatives)',
                        data_train='Eggs_train_23-02-26.h5', 
                        data_validation='Eggs_validation_tile_23-02-26.h5', 
                        data_test='Eggs_validation_large_23-02-25.h5',
                        load_opt='last',
                        load_run=None,
                        n_previous_images=0,
                        pre_merge = False,
                        post_merge = False,
                        focal_loss_ag=(0.85,4.0),                          
                        batch_size=8,
                        max_chans=96)
        elif run==39:
            params = Params(run, epochs = 50,
                        comment = '0 to 4 previous, pre_merge and post-merge',
                        data_train='Eggs_train_23-02-26.h5', 
                        data_validation='Eggs_validation_tile_23-02-26.h5', 
                        data_test='Eggs_validation_large_23-02-25.h5',
                        load_opt='last',
                        load_run=37,
                        n_previous_images=4,
                        rand_previous=True,
                        pre_merge = True,
                        post_merge = True,
                        focal_loss_ag=(0.85,4.0),                          
                        batch_size=8,
                        max_chans=96)            
        elif run==40:
            params = Params(run, epochs = 250,
                        comment = '0 previous, 23-02-27 training',
                        data_train='Eggs_train_23-02-27.h5', 
                        data_validation='Eggs_validation_tile_23-02-26.h5', 
                        data_test='Eggs_validation_large_23-02-25.h5',
                        load_opt='last',
                        load_run=None,
                        n_previous_images=0,
                        pre_merge = False,
                        post_merge = False,
                        focal_loss_ag=(0.75,4.0),                # resume at epoch 120         
                        #focal_loss_ag=(0.7,4.0),                # resume at epoch 83          
                        #focal_loss_ag=(0.85,4.0),                          
                        dice_every_nth=1,
                        batch_size=8,
                        max_chans=96)
        elif run==41:
            params = Params(run, epochs = 150,
                        comment = '0 to 4 previous, pre_merge',
                        data_train='Eggs_train_23-02-27.h5', 
                        data_validation='Eggs_validation_tile_23-02-26.h5', 
                        data_test='Eggs_validation_large_23-02-25.h5',
                        load_opt='last',
                        load_run=None,
                        n_previous_images=4,
                        rand_previous=True,
                        pre_merge = True,
                        post_merge = False,
                        focal_loss_ag=(0.75,4.0),          # (initial: (.85, 4.0))                
                        dice_every_nth=1,
                        batch_size=8,
                        max_chans=96)            
        elif run==42:
            params = Params(run, epochs = 150,
                        comment = '0 to 4 previous, post_merge',
                        data_train='Eggs_train_23-02-27.h5', 
                        data_validation='Eggs_validation_tile_23-02-26.h5', 
                        data_test='Eggs_validation_large_23-02-25.h5',
                        load_opt='last',
                        load_run=None,
                        n_previous_images=4,
                        rand_previous=True,
                        pre_merge = False,
                        post_merge = True,
                        focal_loss_ag=(0.75,4.0),             # (initial: (.85, 4.0))                     
                        batch_size=8,
                        max_chans=96)            
        elif run==43:
            params = Params(run, epochs = 150,
                        comment = '0 to 4 previous, pre_merge and post_merge',
                        data_train='Eggs_train_23-02-27.h5', 
                        data_validation='Eggs_validation_tile_23-02-26.h5', 
                        data_test='Eggs_validation_large_23-02-25.h5',
                        load_opt='last',
                        load_run=None,
                        n_previous_images=4,
                        rand_previous=True,
                        pre_merge = True,
                        post_merge = True,
                        focal_loss_ag=(0.75,4.0),            # (initial: (.85, 4.0))                      
                        batch_size=8,
                        max_chans=96)            

        elif run==44:
            params = Params(run, epochs = 40,
                        comment = '0 previous, 23-02-27 training',
                        data_train='Eggs_train_23-02-27.h5', 
                        data_validation='Eggs_validation_tile_23-02-26.h5', 
                        data_test='Eggs_validation_large_23-02-25.h5',
                        load_opt='last',
                        load_run=40,
                        n_previous_images=0,
                        pre_merge = False,
                        post_merge = False,
                        focal_loss_ag=(0.5,4.0),                # load epoch 83          
                        #focal_loss_ag=(0.85,4.0),                          
                        dice_every_nth=1,
                        batch_size=8,
                        max_chans=96)            
        elif run==45:
            params = Params(run, epochs = 120,
                        comment = '0 previous, 23-02-28 training',
                        data_train='Eggs_train_23-02-28.h5', 
                        data_validation='Eggs_validation_tile_23-02-26.h5', 
                        data_test='Eggs_validation_large_23-02-25.h5',
                        load_opt='last',
                        load_run=None,
                        n_previous_images=0,
                        pre_merge = False,
                        post_merge = False,
                        focal_loss_ag=(0.8,4.0),                # first 74 frames: (0.75, 4.)       
                        dice_every_nth=1,
                        batch_size=8,
                        max_chans=96)
        elif run==46:
            params = Params(run, epochs = 80,
                        comment = 'Masked images',
                        data_train='Eggs_train_mask_23-03-19.h5', 
                        data_validation='Eggs_validation_mask_tile_23-03-19.h5', 
                        data_test='Eggs_validation_mask_large_23-03-19.h5',
                        load_opt='last',
                        load_run=None,
                        n_previous_images=0,
                        pre_merge = False,
                        post_merge = False,
                        focal_loss_ag=(0.8,4.0),                # first 74 frames: (0.75, 4.)       
                        dice_every_nth=1,
                        batch_size=8,
                        max_chans=96)
        elif run==47:
            params = Params(run, epochs = 120,
                        comment = 'Prev image + Prev output',
                        data_train='Eggs_train_23-02-28.h5', 
                        data_validation='Eggs_validation_tile_23-02-26.h5', 
                        data_test='Eggs_validation_large_23-02-25.h5',
                        load_opt='last',
                        load_run=None,
                        model_name='UNetTrack',
                        add_prev_im=True,
                        add_prev_out=True,
                        n_previous_images=3,
                        n_previous_min=0,
                        rand_previous=True,
                        pre_merge = False,
                        post_merge = False,
                        focal_loss_ag=(0.8,4.0),                # first 74 frames: (0.75, 4.)       
                        dice_every_nth=1,
                        batch_size=8,
                        max_chans=96)
        elif run==48:
            params = Params(run, epochs = 120,
                        comment = 'Prev output',
                        data_train='Eggs_train_23-02-28.h5', 
                        data_validation='Eggs_validation_tile_23-02-26.h5', 
                        data_test='Eggs_validation_large_23-02-25.h5',
                        load_opt='last',
                        load_run=None,
                        model_name='UNetTrack',
                        add_prev_im=False,
                        add_prev_out=True,
                        n_previous_images=2,
                        n_previous_min=0,
                        rand_previous=True,
                        pre_merge = False,
                        post_merge = False,
                        focal_loss_ag=(0.8,4.0),                # first 74 frames: (0.75, 4.)       
                        dice_every_nth=1,
                        batch_size=8,
                        max_chans=96)
        elif run==49:
            params = Params(run, epochs = 120,
                        comment = 'Prev Image',
                        data_train='Eggs_train_23-02-28.h5', 
                        data_validation='Eggs_validation_tile_23-02-26.h5', 
                        data_test='Eggs_validation_large_23-02-25.h5',
                        load_opt='last',
                        load_run=None,
                        model_name='UNetTrack',
                        add_prev_im=True,
                        add_prev_out=False,
                        n_previous_images=1,
                        n_previous_min=0,
                        rand_previous=False,
                        pre_merge = False,
                        post_merge = False,
                        focal_loss_ag=(0.8,4.0),                # first 74 frames: (0.75, 4.)       
                        dice_every_nth=1,
                        batch_size=8,
                        max_chans=96)
        elif run==50:
            params = Params(run, epochs = 120,
                        comment = 'Single Image Detector',
                        data_train='Eggs_train_23-03-24.h5', 
                        data_validation='Eggs_validation_tile_23-03-24.h5', 
                        data_test='Eggs_test_large_23-03-24.h5',
                        load_opt='last',
                        load_run=None,
                        model_name='UNetTrack',
                        do_nms = True,
                        testrepeat=0,
                        testoutfrac=0,
                        add_prev_im=False,
                        add_prev_out=False,
                        n_previous_images=0,
                        n_previous_min=0,
                        rand_previous=False,
                        pre_merge = False,
                        post_merge = False,
                        focal_loss_ag=(0.8,4.0),                # first 74 frames: (0.75, 4.)       
                        dice_every_nth=1,
                        batch_size=12,
                        max_chans=96)
        elif run==51:
            params = Params(run, epochs = 120,
                        comment = 'Repeat run 24 with training data 02-24',
                        data_train='Eggs_train_23-02-24.h5', 
                        data_validation='Eggs_validation_tile_23-03-24.h5', 
                        data_test='Eggs_test_large_23-03-24.h5',
                        load_opt='last',
                        load_run=None,
                        model_name='UNetTrack',
                        do_nms = True,
                        testrepeat=0,
                        testoutfrac=1,
                        add_prev_im=False,
                        add_prev_out=False,
                        n_previous_images=0,
                        n_previous_min=0,
                        rand_previous=False,
                        pre_merge = False,
                        post_merge = False,
                        focal_loss_ag=(0.85,4.0),                # first 74 frames: (0.75, 4.)       
                        dice_every_nth=1,
                        batch_size=12,
                        max_chans=96)
        elif run==52:
            params = Params(run, epochs = 120,
                        comment = 'Train with ch 5 - 8',
                        data_train='Eggs_train_ch_5678_23-03-28.h5', 
                        data_validation='Eggs_validation_ch_1_23-03-28.h5', 
                        data_test='Eggs_test_large_23-03-24.h5',
                        load_opt='last',
                        load_run=None,
                        model_name='UNetTrack',
                        do_nms = True,
                        testrepeat=0,
                        testoutfrac=0,
                        add_prev_im=False,
                        add_prev_out=False,
                        n_previous_images=0,
                        n_previous_min=0,
                        rand_previous=False,
                        pre_merge = False,
                        post_merge = False,
                        focal_loss_ag=(0.85,4.0),                # first 74 frames: (0.75, 4.)       
                        dice_every_nth=1,
                        batch_size=12,
                        max_chans=96)
        elif run==53:
            params = Params(run, epochs = 120,
                        comment = 'Train 1,2,3,5,6,8, Val 7, Test 4, alpha: .8, gamma: 4',
                        data_train='Eggs_train_23-03-28.h5', 
                        data_validation='Eggs_validation_23-03-28.h5', 
                        data_test='Eggs_test_large_23-03-28.h5',
                        load_opt='last',
                        load_run=None,
                        model_name='UNetTrack',
                        do_nms = True,
                        testrepeat=0,
                        testoutfrac=0,
                        add_prev_im=False,
                        add_prev_out=False,
                        n_previous_images=0,
                        n_previous_min=0,
                        rand_previous=False,
                        pre_merge = False,
                        post_merge = False,
                        focal_loss_ag=(0.8,4.0),                # first 74 frames: (0.75, 4.)       
                        dice_every_nth=1,
                        batch_size=12,
                        max_chans=96)
        elif run==54:
            params = Params(run, epochs = 100,
                        comment = 'New augmentation',
                        data_train='Eggs_train_23-03-28.h5', 
                        data_validation='Eggs_validation_23-03-28.h5', 
                        data_test='Eggs_test_large_23-03-28.h5',
                        load_opt='last',
                        load_run=None,
                        model_name='UNetQuarter',
                        do_nms = True,
                        testrepeat=0,
                        focal_loss_ag=(0.8,4.0),                # first 74 frames: (0.75, 4.)       
                        dice_every_nth=1,
                        batch_size=12,
                        max_chans=96)
        elif run==55:
            params = Params(run, epochs = 100,
                        comment = 'Test on Train -- to check train data',
                        data_train='Eggs_train_23-03-28.h5', 
                        data_validation='Eggs_validation_23-03-28.h5', 
                        data_test='Eggs_test_large_23-03-28.h5',
                        load_opt='last',
                        load_run=None,
                        model_name='UNetQuarter',
                        do_nms = True,
                        testrepeat=0,
                        focal_loss_ag=(0.8,4.0),                # first 74 frames: (0.75, 4.)       
                        dice_every_nth=1,
                        batch_size=12,
                        max_chans=96)
        elif run==56:
            params = Params(run, epochs = 100,
                        comment = 'Added data from new units',
                        data_train='Eggs_train_23-07-15.h5', 
                        data_validation='Eggs_validation_23-07-15.h5', 
                        data_test='Eggs_test_large_23-07-15.h5',
                        load_opt='last',
                        load_run=None,
                        model_name='UNetQuarter',
                        do_nms = True,
                        testrepeat=0,
                        focal_loss_ag=(0.8,4.0),              
                        dice_every_nth=1,
                        batch_size=12,
                        max_chans=96)
        else:
            raise Exception(f'Undefined run: {run}')
        return params
