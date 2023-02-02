''' Plot results
'''

import os
import sys
sys.path.append('../cvdemos/image')
from synth_data import DrawData

if __name__=="__main__":

    run = 10

    dirname = os.path.dirname(__file__)
    dir_output = os.path.join(dirname,'output',f'run_{run:03d}','val',)

    filename = os.path.join(dir_output, 'step_021.h5')

    dd = DrawData(filename)
    dd.plot()
    
