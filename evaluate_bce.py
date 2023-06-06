''' evaluate_bce: Does binary cross entropy evaluation
'''
import torch
import torch.nn.functional as F
from tqdm import tqdm
import numpy as np
import h5py, json
from timeit import default_timer as timer
from pathlib import Path
import sys, os
image_path = str( Path(__file__).parents[1] / 'imagefunctions' / 'hens') 
sys.path.append(image_path)
from heatmap_score import Peaks, MatchScore
from image_fun import boxes_to_centers, down_scale_coords

class SaveResults:

    def __init__(self, h5filename, images, heatmap, Nb, step, name="validation"):

        if h5filename:
            self.hf = h5py.File(h5filename, 'w')
        else:
            self.hf = None
            return
        self.json_filename = h5filename.replace('.h5','.json')
        group = self.hf.create_group(name)
        N = Nb * images.shape[0]  # Total number of images
        imshape = (N, images.shape[2], images.shape[3], images.shape[1])
        heatshape = (N, heatmap.shape[2], heatmap.shape[3])
        scoreshape = (N, 3)        

        group.create_dataset('images', shape=imshape, dtype='u1', compression='lzf')
        group.create_dataset('heatmap',shape=heatshape, dtype='f4')               
        group.create_dataset('scores',shape=scoreshape, dtype='i8' )             
        group.create_dataset('params',shape=(2,), dtype='f4')

        self.group = group

        self.name = name
        self.data_annotations = {"image_file":h5filename, 
                                 name: {"targets": {},
                                        "scores": {},
                                        "max_targets": 0,
                                        "params": [],
                                        "step": step,
                                        }
                                 }
        self.index=0    
    
    def add(self, images, centers, mask_preds, scores, min_val, max_distance, target_downscale):
        if not self.hf is None:
            for img, cens, preds, iscores in zip(images.cpu().numpy(), centers, mask_preds.cpu().numpy(), scores):
                self.group['images'][self.index] = (img.transpose((1,2,0))*255).astype(np.uint8)
                self.group['heatmap'][self.index] = preds[0].astype(np.float32)
                self.data_annotations[self.name]["targets"][str(self.index)] = cens[0]
                self.data_annotations[self.name]["scores"][str(self.index)] = iscores.tolist()
                self.index += 1        
            self.data_annotations[self.name]["params"] = [min_val, max_distance, target_downscale]

    def write_annotations(self):
        with open(self.json_filename,'w') as f:
            json.dump(self.data_annotations, f, indent=2)

    def __del__(self):
        ''' Close the file when delete the class '''
        if not self.json_filename is None:
            self.write_annotations()
        if not self.hf is None:
            self.hf.close()

@torch.inference_mode()
def evaluate_bce(net, dataloader, device, criterion, params, epoch, step, h5filename=None, outfrac=0):
    net.eval()
    num_val_batches = len(dataloader)
    bce = 0
    scores = np.zeros( (3,) )
    peaks = Peaks(1, device, min_val=0.)
    min_val = 0.
    matches = MatchScore(max_distance = params.max_distance)
    save = None    
    Nb = np.ceil( len(dataloader) / max(1, params.testoutfrac) ).astype(int)

    if h5filename and outfrac:
        print(f'Saving 1/{outfrac} images to {h5filename}')        

    model_time = 0
    nruns = 0
    # iterate over the validation set
    for i, (images, data) in enumerate(tqdm(dataloader, total=num_val_batches, desc='Validation round', unit='batch', leave=False)):

        images = torch.stack(images).to(device=device, dtype=torch.float32, memory_format=torch.channels_last)
        masks_true = torch.stack([x['mask'] for x in data]).to(device=device, dtype=torch.float32, memory_format=torch.channels_last)  # BCE requires float
        
        start_time = timer()
        masks_pred = net(images)
        assert masks_true.min() >= 0 and masks_true.max() <= 1, 'True mask indices should be in [0, 1]'
        bce += criterion(masks_pred, masks_true)

        model_time += timer() - start_time
        nruns += 1

        if epoch % params.dice_every_nth == 0:
            bc = [[down_scale_coords(boxes_to_centers(x["boxes"].detach()), params.target_downscale)] for x in data]            
            detections = peaks.peak_coords( masks_pred.detach() )                        
            if params.do_nms:
                detect_vals = peaks.heatmap_vals( masks_pred, detections )
                detections, _ = peaks.nms( detections, detect_vals, params.max_distance / params.target_downscale, to_torch=True )
            bscores,_,_ = matches.calc_match_scores( detections, bc )        
        else:
            bscores = np.nan*np.ones((params.batch_size,3))

        if h5filename:
            if outfrac and i % outfrac==0:
                if save is None:
                    save = SaveResults(h5filename=h5filename, images=images, heatmap=masks_true, Nb=Nb, step=step)
                centers = [[boxes_to_centers(x["boxes"].detach()).tolist()] for x in data] 
                save.add( images, centers, masks_pred, bscores, min_val, params.max_distance, params.target_downscale )

        scores += bscores.sum(axis=0)


    print(f'Time per batch: {model_time / nruns:.4f} sec')

    net.train()

    dice = 2*scores[0] / (scores[0]+scores.sum()+1e-3)
    precision = scores[0]/ (scores[0]+scores[1]+1e-3)
    recall = scores[0]/ (scores[0]+scores[2]+1e-3)

    return bce.item() / max(num_val_batches, 1), scores, dice, precision, recall
