''' evaluate_bce: Does binary cross entropy evaluation
'''
import torch
import torch.nn.functional as F
from tqdm import tqdm
import numpy as np
import h5py, json
import sys
sys.path.append('../cvdemos/image')
from heatmap_score import Peaks, MatchScore

# from utils.dice_score import multiclass_dice_coeff, dice_coeff

class SaveResults:

    def __init__(self, h5filename, batch, Nb, name="validation"):

        if h5filename:
            self.hf = h5py.File(h5filename, 'w')
        else:
            self.hf = None
            return
        self.json_filename = h5filename.replace('.h5','.json')
        group = self.hf.create_group(name)
        N = Nb * batch['image'].shape[0]  # Total number of images
        imshape = (N, batch['image'].shape[2], batch['image'].shape[3], batch['image'].shape[1])
        centershape = (N, batch['centers'].shape[2], 2)
        nshape = (N,)
        heatshape = (N, batch['targets'].shape[2], batch['targets'].shape[3])
        scoreshape = (N, 3)        

        group.create_dataset('images', shape=imshape, dtype='u1', compression='lzf')
        group.create_dataset('centers', shape=centershape, dtype='f4' )
        group.create_dataset('nobj', shape=nshape, dtype='i8' )
        group.create_dataset('heatmap',shape=heatshape, dtype='f4')               
        group.create_dataset('scores',shape=scoreshape, dtype='i8' )             
        group.create_dataset('params',shape=(2,), dtype='f4')

        self.group = group

        self.name = name
        self.data_annotations = {"image_file":h5filename, 
                                 name: {"targets": {},
                                        "scores": {},
                                        "max_targets": batch['centers'].shape[2],
                                        "params": []
                                        }
                                 }
        self.index=0    
    
    def add(self, images, centers, ncens, mask_preds, scores, min_val, max_distance):
        if not self.hf is None:
            for img, cens, ncen, preds, iscores in zip(images.cpu().numpy(), centers.cpu().numpy(), ncens.cpu().numpy(), mask_preds.cpu().numpy(), scores):
                self.group['images'][self.index] = (img.transpose((1,2,0))*255).astype(np.uint8)
                self.group['heatmap'][self.index] = preds[0].astype(np.float32)

                self.data_annotations[self.name]["targets"][str(self.index)] = cens[0][:ncens[0],:].tolist()
                self.data_annotations[self.name]["scores"][str(self.index)] = iscores.tolist()
                #self.group['centers'][self.index] = cens[0]
                #self.group['nobj'][self.index] = ncen[0]
                #self.group['scores'][self.index] = iscores
                self.index += 1        
            #self.group['params'][:] = np.array( [min_val, max_distance]).astype(np.float32)
            self.data_annotations[self.name]["params"] = [min_val, max_distance]

    def write_annotations(self):
        with open(self.json_filename,'w') as f:
            json.dump(self.data_annotations, f)

    def __del__(self):
        ''' Close the file when delete the class '''
        if not self.json_filename is None:
            self.write_annotations()
        if not self.hf is None:
            self.hf.close()

@torch.inference_mode()
def evaluate_bce(net, dataloader, device, criterion, amp, target_downscale, max_distance, h5filename=None):
    net.eval()
    num_val_batches = len(dataloader)
    bce = 0
    scores = np.zeros( (3,) )
    peaks = Peaks(1, device)
    min_val = 0.
    down_max_distance = max_distance / target_downscale
    matches = MatchScore(max_distance = down_max_distance)
    save = None    
    Nb = len(dataloader)

    # iterate over the validation set
    with torch.autocast(device.type if device.type != 'mps' else 'cpu', enabled=amp):
        for batch in tqdm(dataloader, total=num_val_batches, desc='Validation round', unit='batch', leave=False):
            image, mask_true, centers, ncen = batch['image'], batch['targets'], batch['centers'], batch['ncen']

            # move images and labels to correct device and type
            image = image.to(device=device, dtype=torch.float32, memory_format=torch.channels_last)
            # mask_true = mask_true.to(device=device, dtype=torch.long)
            mask_true = mask_true.to(device=device, dtype=float)

            # predict the mask
            mask_pred = net(image)

            detections = peaks.peak_coords( mask_pred, min_val=0.)
            bscores = matches.calc_match_scores( detections, centers/target_downscale, ncen )

            if save is None:
                save = SaveResults(h5filename=h5filename, batch=batch, Nb=Nb)
            save.add( image, centers, ncen, torch.sigmoid(mask_pred), bscores, torch.sigmoid(torch.Tensor([min_val])).item(), down_max_distance )

            scores += bscores.sum(axis=0)

            if net.n_classes == 1:
                assert mask_true.min() >= 0 and mask_true.max() <= 1, 'True mask indices should be in [0, 1]'

                bce += criterion(mask_pred, mask_true)

                # mask_pred = (F.sigmoid(mask_pred) > 0.5).float()
                # compute the Dice score
                #dice_score += dice_coeff(mask_pred, mask_true, reduce_batch_first=False)
            else:
                assert mask_true.min() >= 0 and mask_true.max() < net.n_classes, 'True mask indices should be in [0, n_classes['
                # convert to one-hot format
                mask_true = F.one_hot(mask_true, net.n_classes).permute(0, 3, 1, 2).float()
                mask_pred = F.one_hot(mask_pred.argmax(dim=1), net.n_classes).permute(0, 3, 1, 2).float()
                # compute the Dice score, ignoring background
                #dice_score += multiclass_dice_coeff(mask_pred[:, 1:], mask_true[:, 1:], reduce_batch_first=False)

    net.train()

    dice = 2*scores[0] / (scores[0]+scores.sum()+1e-3)
    precision = scores[0]/ (scores[0]+scores[1]+1e-3)
    recall = scores[0]/ (scores[0]+scores[2]+1e-3)

    return bce / max(num_val_batches, 1), dice, precision, recall
