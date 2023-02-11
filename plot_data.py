''' Plot results
'''
import argparse
import os, platform
import csv
import matplotlib.pyplot as plt
from pathlib import Path
import numpy as np
import sys
sys.path.append('../cvdemos/image')
from synth_data import DrawData

def save_scores(filename, scores):
    with open(filename, 'w', newline='') as f:        
        for row in scores:
            print(','.join(map(str, row)), file=f)

def read_scores(filename):
    #scores = []
    with open(filename, 'r', newline='') as f:  
        reader = csv.reader(f, delimiter=",",quoting=csv.QUOTE_NONNUMERIC)
        scores = np.array(list(reader))
        #for row in reader:
        #    scores.append(row)
        #scores = f.readlines()
    return scores # np.array(scores)


def get_measures(scores):
    dice = 2*scores[:,0] / (scores[:,0]*2+scores[:,1]+scores[:,2]+1e-3)
    precision = scores[:,0]/ (scores[:,0]+scores[:,1]+1e-3)
    recall = scores[:,0]/ (scores[:,0]+scores[:,2]+1e-3)
    return dice, precision, recall

def plot_scores(train_scores, val_scores, run, filename=None):
    fig = plt.figure(num='Scores', figsize=(10,12))
    fig.clf()
    ax = fig.add_subplot(3,1,1)
    train_scores, val_scores = np.array(train_scores), np.array(val_scores)
    ax.plot( train_scores[:,0], train_scores[:,1],'-',label='train')
    ax.plot( val_scores[:,0], val_scores[:,1],'-',label='val')
    ax.grid()
    ax.legend()
    ax.set_title(f'Run {run} Loss, Min Train / Val: {train_scores[:,1].min():.3}  /  {val_scores[:,1].min():.3}')
    ax.set_yscale('log')
    ax = fig.add_subplot(3,1,2)
    dice, precision, recall = get_measures(train_scores[:,2:])
    ax.plot(train_scores[:,0], dice,'-',label='Dice')
    ax.plot(train_scores[:,0], precision,'-',label='Precision')
    ax.plot(train_scores[:,0], recall,'-',label='Recall')
    ax.set_title(f'Train, Max Dice: {dice.max():.3}')
    ax.set_ylim( 0, 1)
    ax.grid()
    ax.legend()
    ax = fig.add_subplot(3,1,3)
    dice, precision, recall = get_measures(val_scores[:,2:])
    ax.plot(val_scores[:,0], dice,'-',label='Dice')
    ax.plot(val_scores[:,0], precision,'-',label='Precision')
    ax.plot(val_scores[:,0], recall,'-',label='Recall')
    ax.set_title(f'Validation, Max Dice: {dice.max():.3}')
    ax.set_ylim( 0, 1)
    ax.grid()
    ax.legend()
    plt.show(block=False)
    if not filename is None:
        plt.savefig(filename)


if __name__=="__main__":

    parser = argparse.ArgumentParser(description='Train the UNet on images and target masks')
    parser.add_argument('run', type=int,  help='Run number')

    args = parser.parse_args()

    overwrite_png = True

    if not os.name =="nt":
        global_data_dir = '/mnt/home/dmorris/Data/eggs'
    elif platform.node()=='DM-O':
        global_data_dir = 'D:/Data/Eggs/data'
    else:
        raise Exception(f"Unknown platform: {platform.node()}")


    output_dir ='out_eggs'
    run_dir = os.path.join(os.path.dirname(__file__), output_dir, f'{args.run:03d}')

    #etscores = read_scores(os.path.join(run_dir,"train_epoch_scores.csv"))
    tscores = read_scores(os.path.join(run_dir,"train_scores.csv"))
    vscores = read_scores(os.path.join(run_dir,"val_scores.csv"))

    outpng = os.path.join(run_dir,"scores.png") if overwrite_png else None

    plot_scores(tscores, vscores, args.run, outpng)    

    files = [str(x) for x in list(Path(os.path.join(run_dir,'val')).glob('*.h5'))]
    files.sort()
    filename = files[-1]

    dd = DrawData(filename)
    dd.plot()
    
