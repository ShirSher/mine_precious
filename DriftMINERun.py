import torch
import torch.optim as optim
import pandas as pd
import numpy as np
import MINE
import sys
import os
import logging
from random import shuffle, choice
import pickle

import FullDriftDataset
import utils
from datetime import datetime
_datestr = datetime.now().strftime("%Y_%m_%d_%I_%M_%S_%p")


data_path = FullDriftDataset.BASE_PATH
participants_set = set([_dir for _dir in os.listdir(data_path) if not _dir.startswith(".")])
exclude_set = set(['DL','OL','SM'])
participants_list = list(participants_set - exclude_set)
participants_list = [p for p in participants_list if 'try' not in p.lower()]
''' ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
 for testing  - rm for train.
    ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ '''
participants_list = participants_list[3:6]
print('Testing on participants,',participants_list)


''' ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
 ^^ aks what are those numbers?
    ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ '''

''' 300 observations per participant - 3sec '''
nStimuli = 100
nobs = 300
print(nStimuli * len(participants_list),'total observations')

gaze_cols = ['norm_pos_x',
             'norm_pos_y']
head_cols = ['head rot x',
             'head rot y',
             'head rot z',
             'head_dir_x',
             'head_dir_y',
             'head_dir_z',
             'head_right_x',
             'head_right_y',
             'head_right_z',
             'head_up_x',
             'head_up_y',
             'head_up_z']
nChannels = len(gaze_cols) + len(head_cols)


selection = {participant:np.array(range(0,nStimuli)) for participant in participants_list}
for k in selection.keys():
    ''' shuffle each individually '''
    ''' inplace function. returns None '''
    shuffle(selection[k])
''' train set '''
''' 240 random observations of each participant '''
cut1 = int(nStimuli * .80)
train_selection = {k:selection[k][:cut1] for k in selection.keys()}

''' validation set '''
''' 30 random observations of each participant '''
cut2 = int(nStimuli * .90)
val_selection = {k:selection[k][cut1:cut2] for k in selection.keys()}

''' test set '''
''' 30 random observations of each participant '''
test_selection = {k:selection[k][cut2:] for k in selection.keys()}

''' save selections for future runs '''
with open('train_selection.pickle', 'wb') as handle:
    pickle.dump(train_selection, handle, protocol=pickle.HIGHEST_PROTOCOL)

with open('val_selection.pickle', 'wb') as handle:
    pickle.dump(val_selection, handle, protocol=pickle.HIGHEST_PROTOCOL)

with open('test_selection.pickle','wb') as handle:
    pickle.dump(test_selection, handle, protocol=pickle.HIGHEST_PROTOCOL)


''' ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
 ^^ ?????????????????????????
 ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ '''
'''
TODO mod input params
int(sys.argv[i]) ...
TODO mod test (train=False) for net
if int(sys.argv[5]) == 1:
    train = True
else:
    train = False
'''
gamma = 0
optimizer = 2 # Adam
# lr = 0.0003
lr = 0.00001
# arbitrary
# on local running out of memory if bs is too large
batch_size = 16
epochs = 60 # for testing
# train = True
# hard code. only combined makes sense
net_num = 1

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


safety = 0

device = utils.device
print(device)

mine.net = mine.net.to(device)
print(mine.net)

# ? optimization. commented out, throws ERROR: Unexpected bus error encountered in worker. This might be caused by insufficient shared memory (shm).
# torch.backends.cudnn.benchmark = True

dataset = FullDriftDataset.FullDataset(ix_dict=train_selection)
print('len train dataset',len(dataset))
# num_workers maybe comment out
params = {'batch_size': batch_size,
          'shuffle': False,
          'num_workers': 16}
traj_generator = torch.utils.data.DataLoader(dataset, **params)

val_dataset = FullDriftDataset.FullDataset(ix_dict=val_selection)
print('len val dataset',len(val_dataset))
val_generator = torch.utils.data.DataLoader(val_dataset, **params)



''' opening MINE.train() which calls MINE.epoch() which calls MINE.learn_mine() '''
train_results = []
train_losses = []
val_results = []
val_losses = []
batchwise_res = {"train":{i:[] for i in range(epochs)},"validate":{i:[] for i in range(epochs)}}
batchwise_loss = {"train":{i:[] for i in range(epochs)},"validate":{i:[] for i in range(epochs)}}
model_state = None
val_temp = 0
loss_temp = np.inf
dirty_run = False
for epoch in range(epochs):

    '''
    Train
    '''
    print('========================')
    print('Training epoch',epoch)
    print('========================')
    epoch_results = []
    epoch_losses = []
    for i, sample in enumerate(traj_generator):
        trajectory, joint, marginal = sample
        if (dirty_run) :
            print('Sample (pre-mut)')
            print('trajectory:', trajectory.shape, type(trajectory))
            print('joint:     ', joint.shape, type(joint))
            print('marg:      ', marginal.shape, type(marginal))

        traj_inp = trajectory.permute(0,2,1).float()
        joint_inp = joint.permute(0,3,1,2).float()
        marg_inp = marginal.permute(0,3,1,2).float()
        if (dirty_run) :
            print('Sample post-mut')
            print('trajectory: ', traj_inp.shape, type(traj_inp))
            print('joint:      ', joint_inp.shape, type(joint_inp))
            print('marginal:   ', marg_inp.shape, type(marg_inp))

        traj_inp = traj_inp.to(device, non_blocking=True)
        joint_inp = joint_inp.to(device, non_blocking=True)
        marg_inp = marg_inp.to(device, non_blocking=True)
        if (dirty_run) :
            print('Trajectory mounted: ', traj_inp.shape, type(traj_inp))
            print('Joint mounted:      ', joint_inp.shape, type(joint_inp))
            print('Marginal mounted:   ', marg_inp.shape, type(marg_inp))

        ''' where is loss recorded, managed '''
        NIM, loss = mine.learn_mine((traj_inp, joint_inp, marg_inp))
        if (dirty_run) :
            print('MI',NIM.detach())
            print('loss',loss.detach())

        if torch.isnan(NIM.detach()):
            ix = batch_size * i
            ''' which samples  '''
            print('NaN epoch {0}:: samples {1}'.format(epoch, dataset.ix_list[ix:ix+batch_size]))
            continue
        else:
            epoch_results.append(NIM.detach())
            epoch_losses.append(loss.detach())
    batchwise_res['train'][epoch] = epoch_results
    batchwise_loss['train'][epoch] = epoch_losses
    train_results.append(np.mean(epoch_results))
    train_losses.append(np.mean(epoch_losses))
    ''' if nan across all iterations - in what cases? '''
    if len(train_results) == 0:
        if safety > 5:
            print('Nans for 5 epochs. Halting Training')
            break
        safety += 1
        mine.restart_network()

    val_epoch_results = []
    val_epoch_losses = []

    '''
    Validate
    '''

    # versus mine.net.eval() ie the statistical_estimator which inherits from nn
    with torch.no_grad():
        print('========================')
        print('Validation epoch',epoch)
        print('========================')
        for val_i, val_sample in enumerate(val_generator):
            trajectory,joint,marginal = sample
            traj_inp = trajectory.permute(0,2,1).float()
            joint_inp = joint.permute(0,3,1,2).float()
            marg_inp = marginal.permute(0,3,1,2).float()

            traj_inp, joint_inp, marg_inp = traj_inp.to(device), joint_inp.to(device), marg_inp.to(device)

            # where is loss recorded, managed
            val_NIM, val_loss = mine.validate_mine((traj_inp, joint_inp, marg_inp))
            print('val_MI',val_NIM.detach())
            print('val_loss',val_loss.detach())
            if torch.isnan(val_NIM):
                ix = batch_size * i
                # which samples
                print('NaN epoch {0}:: samples {1}'.format(epoch, dataset.ix_list[ix:ix+batch_size]))
                continue
            else:
                val_epoch_results.append(val_NIM.detach())
                val_epoch_losses.append(val_loss.detach())

        batchwise_res['validate'][epoch] = epoch_results
        batchwise_loss['validate'][epoch] = epoch_losses
        val_results.append(np.mean(val_epoch_results))
        val_losses.append(np.mean(val_epoch_losses))
        if (np.mean(val_epoch_results) >= val_temp) & (np.mean(val_epoch_losses) <= loss_temp):
            val_temp = np.mean(val_epoch_results)
            loss_temp = np.mean(val_epoch_losses)
            torch.save(mine.net.state_dict(), 'net_epch_{1}_dt_{0}'.format(_datestr,epoch))
            print('model saved with val MI:',val_temp,', val Loss:',loss_temp)

        # where is loss recorded, managed
        # NIM, loss = mine.learn_mine((traj_inp,joint_inp,marg_inp))
        # if torch.isnan(NIM):
        #     ix = batch_size * i
        #     # which samples
        #     print('NaN epoch {0}:: samples {1}'.format(epoch, dataset.ix_list[ix:ix+batch_size]))
        #     continue
        # else:
        #     epoch_results.append(NIM.detach())
        #     epoch_losses.append(loss.detach())

# mine.train(epochs = epochs)
# while len(mine.results) == 0:
#     mine.restart_network()
#     check_trial = mine.train(epochs = epochs)
#     safty+=1
#     if safty>5:
#         break
# safty = 0
# results = pd.DataFrame(mine.results)


train_results = pd.DataFrame(train_results)
val_results = pd.DataFrame(val_results)
# nm = 'results_{0}_{1}_{2}_{3}_{4}_{5}_{6}_{7}_{8}_{9}_{10}'.format(sys.argv[1],int(float(sys.argv[2])*100000),sys.argv[3],sys.argv[4],sys.argv[5],sys.argv[6],sys.argv[7],sys.argv[8],sys.argv[9],sys.argv[10],dataset_status)
res_name = 'results_train'
val_name = 'results_val'
train_results.to_pickle(res_name.format(_datestr))
val_results.to_pickle(val_name.format(_datestr))

loss_name = 'losses_train'
val_name = 'losses_val'
train_losses = pd.DataFrame(train_losses)
val_losses = pd.DataFrame(val_losses)

train_losses.to_pickle(loss_name.format(_datestr))
val_losses.to_pickle(val_name.format(_datestr))
#results.to_csv('results_{0}_{1}_{2}_{3}_{4}.csv'.format(sys.argv[1],int(float(sys.argv[2])*10000),sys.argv[3],sys.argv[4],sys.argv[5],sys.argv[6],sys.argv[7],sys.argv[8]))
# torch.save(mine.net.state_dict, 'net_{0}_{1}_{2}_{3}_{4}_{5}_{6}_{7}_{8}_{9}_{10}'.format(sys.argv[1],int(float(sys.argv[2])*100000),sys.argv[3],sys.argv[4],sys.argv[5],sys.argv[6],sys.argv[7],sys.argv[8],sys.argv[9],sys.argv[10],dataset_status))

# test

## random shuffling should get nothing