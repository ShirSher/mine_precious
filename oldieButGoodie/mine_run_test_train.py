#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Feb  2 14:44:27 2020

@author: michael
"""
import torch
import torch.optim as optim
import pandas as pd
import MINE
import sys
import os
import matplotlib.pyplot as plt
# sys.argv gets 4 values:
# [1] - the optimizer - one of three (1) - SGD,(2) - Adam, (3) - RMSprop
# [2] - lr
# [3] - batch_size
# [4] - epochs
# [5] - train true/false 1/0
# [6] - Net num - 1-3
# [7] - traject/MNIST/combined - What form of mine to compute
# [8] - number_descending_blocks - i.e. how many steps should the fc networks take,
#                               starting at 2048 nodes and every step divide the size 
#                                by 2. max number is 7                                
# [9] - number_repeating_blocks - the number of times to repeat a particular layer
# [10] - repeating_blockd_size - the size of nodes of the layer to repeat
# [11] - traject_max_depth
# [12] - traject_num_layers
# [13] - same/different mnist/trajectory - to use the same image that the 
#        trajectory ran on as the joint distribution.


gamma = 0
optimizer = int(sys.argv[1])
lr = float(sys.argv[2])
batch_size = int(sys.argv[3])
epochs = int(sys.argv[4])
if int(sys.argv[5]) == 1:
    train = True
else:
    train = False
net_num = int(sys.argv[6])
traject = str(sys.argv[7])
number_descending_blocks = int(sys.argv[8])
number_repeating_blocks = int(sys.argv[9])
repeating_blockd_size = int(sys.argv[10])
traject_max_depth = int(sys.argv[11])
traject_num_layers = int(sys.argv[12])
dataset_status = str(sys.argv[13])
mine = MINE.MINE(train = train, 
                 traject = traject, 
                 batch = batch_size, 
                 lr = lr,
                 gamma = gamma, 
                 optimizer = optimizer,
                 net_num = net_num, 
                 traject_max_depth = traject_max_depth, 
                 traject_num_layers = traject_num_layers, 
                 traject_stride = [3,1],
                 traject_kernel = 5, 
                 traject_padding = 0,
                 traject_pooling = [1,2], 
                 number_descending_blocks = number_descending_blocks, 
                 number_repeating_blocks=number_repeating_blocks,  
                 repeating_blockd_size = repeating_blockd_size,
                 dataset_status = dataset_status)

print(mine.net)
safty = 0
print('Epochs: {}'.format(int(sys.argv[4])))
mine.train(epochs = epochs)
while len(mine.results) == 0:
    mine.restart_network()
    check_trial = mine.train(epochs = epochs)
    safty+=1
    if safty>5:
        break
safty = 0
results = pd.DataFrame(mine.results)
nm = 'results_{}_{:.0e}_{}_{}_{}_{}_{}_{}_{}_{}_{}'.format(sys.argv[1],float(sys.argv[2]),sys.argv[3],sys.argv[4],sys.argv[5],sys.argv[6],sys.argv[7],sys.argv[8],sys.argv[9],sys.argv[10],dataset_status)
results.to_pickle(nm)
test_nm = 'test_results_{}_{:.0e}_{}_{}_{}_{}_{}_{}_{}_{}_{}'.format(sys.argv[1],float(sys.argv[2]),sys.argv[3],sys.argv[4],sys.argv[5],sys.argv[6],sys.argv[7],sys.argv[8],sys.argv[9],sys.argv[10],dataset_status)
test_results = pd.DataFrame(mine.test_results)
test_results.to_pickle(test_nm)

plt.plot(results.T, label = 'training set')
plt.plot(test_results.T, label = 'test set')
plt.xlabel('epoch')
plt.ylabel('nats')
plt.title('Plot Compering MI in Training and Test Sets\
          {}'.format(nm))
fig_nm = 'fig_{}_{:.0e}_{}_{}_{}_{}_{}_{}_{}_{}_{}'.format(sys.argv[1],float(sys.argv[2]),sys.argv[3],sys.argv[4],sys.argv[5],sys.argv[6],sys.argv[7],sys.argv[8],sys.argv[9],sys.argv[10],dataset_status)
plt.savefig(fig_nm)
#results.to_csv('results_{0}_{1}_{2}_{3}_{4}.csv'.format(sys.argv[1],int(float(sys.argv[2])*10000),sys.argv[3],sys.argv[4],sys.argv[5],sys.argv[6],sys.argv[7],sys.argv[8]))
torch.save(mine.net.state_dict, 'net_{}_{:.0e}_{}_{}_{}_{}_{}_{}_{}_{}_{}'.format(sys.argv[1],float(sys.argv[2]),sys.argv[3],sys.argv[4],sys.argv[5],sys.argv[6],sys.argv[7],sys.argv[8],sys.argv[9],sys.argv[10],dataset_status))

