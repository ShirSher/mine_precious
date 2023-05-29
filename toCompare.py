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

from datetime import datetime
_datestr = datetime.now().strftime("%Y_%m_%d_%I_%M_%S_%p")
# README.md
# An example of a run:​

# bsub -q gpu-short -app nvidia-gpu -env LSB_CONTAINER_IMAGE=nvcr.io/nvidia/pytorch:19.07-py3 -gpu num=4:j_exclusive=no -R "select[mem>8000] rusage[mem=8000]" -o out.%J -e err.%J python3 \
# Adam, {lr}, batch_size, 
# mine_run.py 2 0.0003 500 200 1 2 combined 3 2 2 512 6 same
# ---

# sys.argv gets 4 values:
# [1] - the optimizer - one of three (1) - SGD,(2) - Adam, (3) - RMSprop
# [2] - lr
# [3] - batch_size
# [4] - epochs
# [5] - train true/false 1/0
# [6] - Net num - 1-3 #set equal 3 Ignore
# [7] - traject/MNIST/combined - What form of mine to compute # combined Ignore
# [8] - number_descending_blocks - i.e. how many steps should the fc networks take,
#                               starting at 2048 nodes and every step divide the size 
#                                by 2. max number is 7                                
# [9] - number_repeating_blocks - the number of times to repeat a particular layer
# [10] - repeating_blockd_size - the size of nodes of the layer to repeat
# [11] - traject_max_depths
# [12] - traject_num_layers
# [13] - same/different minist/trajectory - to use the same image that the 
#        trajectory ran on as the joint distribution. # Ignore? 

BASE_PATH = '../VRDrift/'
part_set = set([_dir for _dir in os.listdir(BASE_PATH) if not _dir.startswith(".")])
exclude_set = set(['DL','OL','SM'])
PART_LIST = list(part_set - exclude_set)
PART_LIST = [p for p in PART_LIST if 'try' not in p.lower()]
# for testing  - rm for train
# PART_LIST = PART_LIST[3:6]
print('Testing on participants,',PART_LIST)

# 300 observations per participant - 3sec
NSTIMULI = 100
NOBS = 300 
print(NSTIMULI * len(PART_LIST),'total observations')

GAZE_COLS = ['norm_pos_x','norm_pos_y']
HEAD_COLS = ['head rot x','head rot y','head rot z','head_dir_x','head_dir_y','head_dir_z','head_right_x','head_right_y','head_right_z','head_up_x','head_up_y','head_up_z']
NCHANNELS = len(GAZE_COLS) + len(HEAD_COLS)

# selection = {i:np.array(range(0,NSTIMULI)) for i in range(len(PART_LIST))}


selection = {participant:np.array(range(0,NSTIMULI)) for participant in PART_LIST}
for k in selection.keys():
    # shuffle each individually 
    # inplace function. returns None
    shuffle(selection[k])
# train set
# 240 random observations of each participant
cut1 = int(NSTIMULI * .80)
train_selection = {k:selection[k][:cut1] for k in selection.keys()}
# validation set
# 30 random observations of each participant
cut2 = int(NSTIMULI * .90)
val_selection = {k:selection[k][cut1:cut2] for k in selection.keys()}
# test set
# 30 random observations of each participant
test_selection = {k:selection[k][cut2:] for k in selection.keys()}

# save selections for future runs
with open('train_selection.pickle', 'wb') as handle:
    # print('saving train selection. n:',len(train_selection))
    pickle.dump(train_selection, handle, protocol=pickle.HIGHEST_PROTOCOL)
    

with open('val_selection.pickle', 'wb') as handle:
    # print('saving val selection. n:',len(val_selection))
    pickle.dump(val_selection, handle, protocol=pickle.HIGHEST_PROTOCOL)
    

with open('test_selection.pickle','wb') as handle:
    # print('saving test selection. n:',len(test_selection))
    pickle.dump(test_selection, handle, protocol=pickle.HIGHEST_PROTOCOL)

# with open(r'train_selection.pickle', 'rb') as handle:
#     train_selection = pickle.load(handle)

# with open(r'val_selection.pickle', 'rb') as handle:
#     val_selection = pickle.load(handle)

# with open(r'test_selection.pickle','rb') as handle:
#     test_selection = pickle.load(handle)


# TODO mod input params 
gamma = 0
# optimizer = int(sys.argv[1])
optimizer = 2 # Adam
# lr = float(sys.argv[2])
# lr = 0.0003 
lr = 0.00001
# batch_size = int(sys.argv[3])
# arbitrary 
# on local running out of memory if bs is too large
batch_size = 16
# epochs = int(sys.argv[4])
epochs = 60 # for testing
# if int(sys.argv[5]) == 1:
#     train = True
# else:
#     train = False
# TODO mod test (train=False) for net
# train = True
# hard code. only combined makes sense
net_num = 1
# traject = str(sys.argv[7])
# number_descending_blocks = int(sys.argv[8])
# number_repeating_blocks = int(sys.argv[9])
# repeating_blockd_size = int(sys.argv[10])
# traject_max_depth = int(sys.argv[11])
# traject_max_depth = 128
# traject_num_layers = int(sys.argv[12])
# traject_num_layers = 3
# first_depth = 32
# dataset_status = str(sys.argv[13])
mine = MINE.MINE(train = True, 
                 batch = batch_size, 
                 lr = lr,
                 gamma = gamma, 
                 optimizer = optimizer,
                #  traject_max_depth = traject_max_depth, 
                #  traject_num_layers = traject_num_layers, 
                 traject_stride = [3,1],
                 traject_kernel = 5, 
                 traject_padding = 0,
                 traject_pooling = [1,2], 
                #  number_descending_blocks = number_descending_blocks, 
                #  number_repeating_blocks=number_repeating_blocks,  
                #  repeating_blockd_size = repeating_blockd_size,
                #  dataset_status = dataset_status,
                 traject_input_dim = [NCHANNELS,NOBS])

# print(mine.net)

safety = 0
# print('Epochs: {}'.format(int(sys.argv[4])))

# CUDA for PyTorch
CUDA_AVAIL = torch.cuda.is_available()
_device = 'cuda:3' if CUDA_AVAIL else 'cpu'

mine.net = mine.net.to(_device)
print(mine.net)
# device = torch.device("cuda:3" if cuda_avail else "cpu")
# print('using device',device)
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



# opening MINE.train() which calls MINE.epoch() which calls MINE.learn_mine()
train_results = []
train_losses = []
val_results = []
val_losses = []
batchwise_res = {"train":{i:[] for i in range(epochs)},"validate":{i:[] for i in range(epochs)}}
batchwise_loss = {"train":{i:[] for i in range(epochs)},"validate":{i:[] for i in range(epochs)}}
model_state = None
val_temp = 0
loss_temp = np.inf 

for epoch in range(epochs):
# Train
    print('Training epoch',epoch)
    print('========================')
    epoch_results = []
    epoch_losses = []
    for i, sample in enumerate(traj_generator):
        trajectory,joint,marginal = sample
        print('Sample (pre-mut) traj,joint,marg',trajectory.shape, type(trajectory),joint.shape, type(joint),marginal.shape, type(marginal))
        traj_inp = trajectory.permute(0,2,1).float()
        joint_inp = joint.permute(0,3,1,2).float()
        marg_inp = marginal.permute(0,3,1,2).float()
        print('Sample post-mut',traj_inp.shape, type(traj_inp), 'joint: ',joint_inp.shape, type(joint_inp), 'marginal:', marg_inp.shape, type(marg_inp))
        traj_inp = traj_inp.to(_device, non_blocking=True)
        print('Trajectory mounted:',traj_inp.shape,type(traj_inp))
        joint_inp = joint_inp.to(_device, non_blocking=True)
        print('Joint mounted:',joint_inp.shape,type(joint_inp))
        marg_inp = marg_inp.to(_device, non_blocking=True)
        print('Marginal mounted:',marg_inp.shape,type(marg_inp))

        # where is loss recorded, managed
        NIM, loss = mine.learn_mine((traj_inp,joint_inp,marg_inp))
        print('MI',NIM.detach())
        print('loss',loss.detach())
        if torch.isnan(NIM.detach()):
            ix = batch_size * i
            # which samples 
            print('NaN epoch {0}:: samples {1}'.format(epoch, dataset.ix_list[ix:ix+batch_size]))
            continue
        else:
            epoch_results.append(NIM.detach())
            epoch_losses.append(loss.detach())
    batchwise_res['train'][epoch] = epoch_results
    batchwise_loss['train'][epoch] = epoch_losses
    train_results.append(np.mean(epoch_results))
    train_losses.append(np.mean(epoch_losses))
    # if nan across all iterations - in what cases?
    if len(train_results) == 0:
        if safety > 5: 
            print('Nans for 5 epochs. Halting Training')
            break
        safety += 1
        mine.restart_network()

    val_epoch_results = []
    val_epoch_losses = []
# Validate 
# versus mine.net.eval() ie the statistical_estimator which inherits from nn
    with torch.no_grad():
        print('-----------------------')
        print('Validation epoch',epoch)
        print('========================')
        for val_i, val_sample in enumerate(val_generator):
            trajectory,joint,marginal = sample
            traj_inp = trajectory.permute(0,2,1).float()
            joint_inp = joint.permute(0,3,1,2).float()
            marg_inp = marginal.permute(0,3,1,2).float()
            
            traj_inp, joint_inp, marg_inp = traj_inp.to(device), joint_inp.to(device), marg_inp.to(device)

            # where is loss recorded, managed
            val_NIM, val_loss = mine.validate_mine((traj_inp,joint_inp,marg_inp))
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
