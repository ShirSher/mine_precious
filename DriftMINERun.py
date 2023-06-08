import torch
import pandas as pd
import numpy as np
import MINE
from random import shuffle, choice

import FullDriftDataset
import utils
from datetime import datetime


_datestr = datetime.now().strftime("%Y_%m_%d_%I_%M_%S_%p")

'''
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
 ^^ ?????????????????????????
 ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

TODO mod input params
int(sys.argv[i]) ...
TODO mod test (train=False) for net
if int(sys.argv[5]) == 1:
    train = True
else:
    train = False

? optimization. commented out, throws ERROR: Unexpected bus error encountered in worker. This might be caused by insufficient shared memory (shm).
torch.backends.cudnn.benchmark = True
'''

# ================
# MINE!
# ================
gamma = 0
optimizer = 2 # Adam
lr = 0.00001
# arbitrary
# on local running out of memory if bs is too large
batch_size = 16
epochs = 60 # for testing
# train = True
# hard code. only combined makes sense
net_num = 1
nobs = 300
nChannels = len(FullDriftDataset._gaze_cols) + len(FullDriftDataset._head_cols)

mine = MINE.MINE(train = True,
                 batch = batch_size,
                 lr = lr,
                 gamma = gamma,
                 optimizer = optimizer,
                 traject_stride = [3,1],
                 traject_kernel = 5,
                 traject_padding = 0,
                 traject_pooling = [1,2],
                 traject_input_dim = [nChannels, nobs])

device = utils._device
print(device)

mine.net = mine.net.to(device)
print(mine.net)


# ================
# Data Loader
# ================
train_selection, val_selection = utils.get_selection()
params = {
            'batch_size': batch_size,
            'shuffle': False
        }

train_dataset = FullDriftDataset.FullDataset(ix_dict=train_selection)
print('len train dataset',len(train_dataset))
train_generator = torch.utils.data.DataLoader(train_dataset, **params)

val_dataset = FullDriftDataset.FullDataset(ix_dict=val_selection)
print('len val dataset',len(val_dataset))
val_generator = torch.utils.data.DataLoader(val_dataset, **params)


# opening MINE.train() which calls MINE.epoch() which calls MINE.learn_mine()
train_results = []
train_losses = []
val_results = []
val_losses = []
max_val_result = 0
min_val_loss = np.inf
safety = 0

for epoch in range(epochs):

    print('\n========================')
    print('Epoch', epoch)

    utils.run('Train', mine, train_generator, train_dataset, train_results, train_losses)
 
    # ^^ if nan across all iterations - in what cases? what indeed
    if len(train_results) == 0:
        if safety > 5:
            # print('Nans for 5 epochs. Halting Training')
            break
        safety += 1
        mine.restart_network()

    # versus mine.net.eval() ie the statistical_estimator which inherits from nn
    with torch.no_grad():

        utils.run('Validation', mine, val_generator, val_dataset, val_results, val_losses)
        
        # ^^ what exactly do you do?
        if (val_results[-1] >= max_val_result) & (val_losses[-1] <= min_val_loss):
            max_val_result = val_results[-1]
            min_val_loss = val_losses[-1]
            torch.save(mine.net.state_dict(), utils.epochs_path+'epoch_{1}_dt_{0}'.format(_datestr, epoch))
            print('model saved with val MI:',max_val_result,', val Loss:',min_val_loss)

    print('========================\n')

pd.DataFrame(train_results).to_pickle(utils.output_path+'train_results')
pd.DataFrame(train_losses).to_pickle(utils.output_path+'train_losses')
pd.DataFrame(val_results).to_pickle(utils.output_path+'val_results')
pd.DataFrame(val_losses).to_pickle(utils.output_path+'val_losses')
