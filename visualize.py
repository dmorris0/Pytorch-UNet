import os
import sys
from pathlib import Path
from tqdm import tqdm
import numpy as np
import json
import argparse
import cv2 as cv
import matplotlib.pyplot as plt
from matplotlib.patches import Circle
from timeit import default_timer as timer

sys.path.append(str( Path(__file__).parents[1] / 'pypackages' / 'imfolder' ) )
from vidWrite import vidWrite
from VideoReader import VideoReader

def combine_videos(vidfolder, outfolder, date_time):
    format='XVID'
    outname = os.path.join(outfolder,f'{date_time}_{format}.avi')
    print(f'Saving to: {outname}')
    os.makedirs(outfolder, exist_ok=True)
    videos = {}
    names = [f'ch{x+1}' for x in range(8)]
    for name in names:
        search = f'{name}_{date_time}*.mp4'
        videos[name] = [str(x) for x in list(Path(vidfolder).rglob(search))]
        videos[name].sort()

    ind = 0
    readers = [VideoReader(videos[name][ind],1) for name in names]
    vid = vidWrite(outname, fps=10,format=format)
    newsize = None
    done = False    
    nf = 0
    while not done:
        images = []
        for reader in readers:
            ret, img, _, _ = reader.get_next()
            if not ret:
                done=True
                break            
            if newsize is None:
                newsize = (img.shape[1]//4, img.shape[0]//4)
            images.append(cv.resize(img, newsize, interpolation=cv.INTER_AREA))
        if not done:
            nf += 1
            nimg = np.concatenate( 
                ( np.concatenate(images[0:4],axis=1),
                np.concatenate(images[4:8],axis=1) ), axis=0
            )
            cv.imshow("All",nimg)
            key = cv.waitKey(1)
            if key==ord('q'):
                return
            vid.add(nimg)
            if nf >= 100:
                done = True
                return
            
            

if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Train the UNet on images and target masks')
    parser.add_argument('run', type=int, help='Run')
    parser.add_argument('--folder', type=str, default='/mnt/home/dmorris/Data/Hens/Hens_2021',  help='Folder for videos')    
    parser.add_argument('--images', type=str, default='/mnt/home/dmorris/Data/Hens/ImagesJPG',  help='Folder for images')        
    parser.add_argument('--output', type=str, default='out_vis',  help='Folder for output')        
            
    args = parser.parse_args()

    if False:
        combine_videos(vidfolder=args.folder, 
                    outfolder=str(Path(__file__).parent / args.output),
                    date_time='0728_0900')
    if False:
        combine_videos(vidfolder=args.folder, 
                    outfolder=str(Path(__file__).parent / args.output),
                    date_time='0726_0900')
        




    # To get OpenCV working (as conda version does not work):
    # module load GCCcore/11.3.0
    # module load Python/3.10.4
    # act cv
    