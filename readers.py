''' Video and Image readers

    Note: OpenCV is not compatible with torchvision 0.15, so this code needs
    to be modified to exclude OpenCV.  There is a PyTorch video reader described
    here: https://pytorch.org/vision/main/auto_examples/plot_video_api.html
    

'''

import cv2 as cv
import matplotlib.pyplot as plt
from pathlib import Path
import os
import json
import torch
import torchvision
torchvision.set_video_backend("video_reader")
from timeit import default_timer as timer
import numpy as np

class VideoReader:
    def __init__(self, filename, sample_time_secs):
        self.name = filename
        self.cap = cv.VideoCapture(filename)
        self.sample_time_secs = sample_time_secs   # if this is 2, then next frame is every 1/2 second
        self.fps = int(self.cap.get(cv.CAP_PROP_FPS))
        self.skip = int(round(self.fps * self.sample_time_secs))
        self.nsamp = int(self.cap.get(cv.CAP_PROP_FRAME_COUNT)) // self.skip        
        self.index = -1

    def get_next(self):
        ret, frame = self.cap.read()
        self.index += 1
        frame_no = self.index * self.skip
        if ret:
            frame = cv.cvtColor(frame, cv.COLOR_BGR2RGB)  # Trained on RGB images
            imsize = frame.shape
            if imsize[0] > 1200:
                imsize = (imsize[0]//2, imsize[1]//2, imsize[2])
                frame = cv.resize(frame, (imsize[1],imsize[0]), interpolation=cv.INTER_LINEAR)
            for _ in range(self.skip-1):
                _ = self.cap.read()
        return ret, frame, frame_no, None

    def get_nth(self, nth):
        self.index = nth-1
        self.cap.set(cv.CAP_PROP_POS_FRAMES,nth*self.skip)
        return self.get_next()
    
    def __len__(self):
        return self.nsamp
    
    def __del__(self):
        if not self.cap is None:
            self.cap.release()

class AltVideoReader:
    def __init__(self, filename, sample_time_secs):
        self.name = filename
        self.cap = torchvision.io.VideoReader(filename, "video")
        info = self.cap.get_metadata()
        print(info)        
        self.sample_time_secs = sample_time_secs   # if this is 2, then next frame is every 1/2 second
        self.fps = info['video']['fps'][0]
        self.skip = int(round(self.fps * self.sample_time_secs))
        self.nsamp = 1 + int((info['video']['duration'][0]-1/self.fps)/self.sample_time_secs)     
        self.index = -1

    def get_next(self):        
        try:
            if self.index >= 0:
                for _ in range(self.skip-1):
                    next(self.cap)
            data = next(self.cap)
            self.index += 1
            return data['data'], int(round(data['pts']*self.fps))
        except StopIteration:
            return None, None

    def get_nth(self, nth):
        try:
            self.cap.seek(nth*self.sample_time_secs)
            data = next(self.cap)
            self.index = nth
            return data['data'], int(round(data['pts']*self.fps))        
        except StopIteration:
            return None, None
        
    def __len__(self):
        return self.nsamp



def get_image_name(im_folder, json_name):
    name = os.path.join(im_folder, json_name).replace('.json','.jpg')
    if os.path.exists(name):
        return name
    elif os.path.exists(name.replace('.jpg','.png')):
        return name.replace('.jpg','.png')
    else:
        print(f'Error: missing image for: {json_name} in {im_folder}')
        return None

class ImageReader:
    def __init__(self, im_folder, search='*.json'):
        self.json_list = list(Path(im_folder).glob(search))
        self.json_list.sort()
        self.im_list = [get_image_name(im_folder, x.name) for x in self.json_list]
        self.nsamp = len(self.im_list)
        self.index = -1
        self.name = im_folder

    def __len__(self):
        return self.nsamp

    def get_next(self):
        self.index += 1
        if self.index < self.nsamp:
            frame = cv.imread(self.im_list[self.index])
            frame = cv.cvtColor(frame, cv.COLOR_BGR2RGB)  # Trained on RGB images
            with open(self.json_list[self.index]) as f:
                annotation = json.load(f)
            return True, frame, Path(self.im_list[self.index]).name, annotation
        else:
            return False, None, None, None


if __name__ == '__main__':

    # Run this to sanity check the code

    #vidfile = '/mnt/research/3D_Vision_Lab/Hens/Hens_2021/07-28/ch4/ch4_0728_1000.mp4'
    vidfile = '/mnt/scratch/dmorris/Hens_2021_sec/07-28/ch3/ch3_0728_0500.avi'
        
    N = 20
    sample_time_sec = 5

    if True:
        avr = AltVideoReader(vidfile, sample_time_sec)
        numlist = []
        start = timer()
        for i in range(N):
            frame, frame_no = avr.get_next()        
            numlist.append(frame_no)    

        print('next Time:',timer()-start)
        print('next numlist', numlist)

        avr = AltVideoReader(vidfile, sample_time_sec)
        numlist = []
        start = timer()
        for i in range(N):
            frame, frame_no = avr.get_nth(i)        
            numlist.append(frame_no)    

        print('nth Time:',timer()-start)
        print('nth numlist', numlist)
            
        vr = VideoReader(vidfile, sample_time_sec)
        framelist = []
        numlist = []
        indexlist = []
        start = timer()
        for i in range(N):
            ret, frame, frame_no, _ = vr.get_next()        
            numlist.append(frame_no)    

        print('CV Time:',timer()-start)
        print('CV numlist',numlist)
        

    avr = AltVideoReader(vidfile, sample_time_sec)
    aframelist = []
    start = timer()
    for i in range(3):
        frame, frame_no = avr.get_next()        
        aframelist.append(frame)
    frame, frame_no = avr.get_nth(2)        
    aframelist.append(frame)
    
    print('AD1',torch.abs(aframelist[2]-aframelist[3]).sum())

    vr = VideoReader(vidfile, sample_time_sec)
    framelist = []
    start = timer()
    for i in range(3):
        ret, frame, frame_no, _ = vr.get_next()        
        framelist.append(frame)

    ret, frame, frame_no, _ = vr.get_nth(2)
    framelist.append(frame)

    print('CD2',np.abs(framelist[2]-framelist[3]).sum())    

    for f1, f2 in zip(aframelist, framelist):
        print('Diff:', np.abs(f1.permute(1,2,0).numpy()-f2).sum())


    fig = plt.figure()
    ax = fig.subplots(2,2).ravel() 
    for i in range(4):
        ax[i].imshow(framelist[i])
    plt.show()
    