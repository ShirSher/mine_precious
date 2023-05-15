


import torch
import torch.optim as optim

import matplotlib.pyplot as plt
import numpy as np

import pandas as pd
import os
import pickle


#filles that run on gpu prior to fixing data writing issue:
#2 0.003 500 150 0 0.5 0.1 11 [3,1,1,1,1] [1,2,2,2,2] 256 5 4 0 0> max test - 84 max train - 94
#3 0.003 500 150 0 0.5 0.1 5 [2,1,1,1,1] [1,2,2,2,2] 256 5 4 0 0> max test - 81 max train - 97
#2 0.003 500 150 0 0.5 0.1 5 [1,1,1,1,1] [2,2,2,2,2] 128 3 4 0 0> max test - 71 max train - 99
#2 0.003 500 150 0 0.5 0.2 5 [3,1] [1,2] 512 5 4 0 0            > max test - 84 max train - 86
#2 0.003 500 150 1 0.5 0.1 11 [1,1] [2,2] 128 5 4 0 [5,7,11]    > max test - 84 max train - 90
#2 0.003 500 150 1 0.5 0.1 11 [6,1] [1,2] 128 5 4 0 [5,7,11]    > max test - 81 max train - 86
#
directory = '/home/michael/Documents/Scxript/Syclop-MINE-master/classification_results/test/gpu'
os.chdir(directory)

results = np.ones((60,1))
results = pd.DataFrame(results)
import os
for filename in os.listdir(directory):
    #if len(pd.read_pickle(filename))>1:
    temp_file = pd.Series(torch.load(filename, map_location = torch.device('cpu'))).astype('float')
    results[filename] = temp_file
    #else:
        #print(len(pd.read_pickle(filename)))

#read cpu results
directory = '/home/michael/Documents/Scxript/Syclop-MINE-master/classification_results/test/cpu'
os.chdir(directory)


import os
for filename in os.listdir(directory):
    #if len(pd.read_pickle(filename))>1:
    temp_file = pickle.load(open(filename,'rb'))
    temp_file = pd.Series(temp_file).astype('float')
    results[filename] = temp_file
    #else:
        #print(len(pd.read_pickle(filename)))  

results = results[results.columns[1:]] 
num_to_opt  = {1:'SGD', 2:'Adam', 3:'RMSprop'}  
# Getting the learning rate and batch size from the name:
lr_by_idx = []
name_dict = dict()
for name in results.columns:
    values = [s for s in name.split('_')]
    values = values[2:]
    lr = int(values[1])*1e-4
    batch_size = values[2]
    optimizer = values[0]
    epochs = values[3]
    net_num = values[4]
    drop_fc = int(values[5])*1e-1
    drop_conv = int(values[6])*1e-1
    kernel = values[7]
    stride = values[8]
    pooling = values[9]
    max_depth = values[10]
    num_layers = values[11]
    repeating_block = values[13]
    

    name_dict[name] = [lr,batch_size, optimizer, epochs, net_num, drop_fc, 
             drop_conv, kernel, stride, pooling, max_depth, num_layers, repeating_block]
'''
max_value = 88
max_length = 30
lr_to_see = 0.003
fig = plt.figure(figsize=[17,15])
for key in name_dict:
    #plt.plot(np.linspace(0,29,30)*5,results[key])
    #if int(name_dict[key][2]) == 1:
    plt.plot(np.linspace(0,59,60)*5,results[key],label=[name_dict[key][0],name_dict[key][1],name_dict[key][2],name_dict[key][3],name_dict[key][5],name_dict[key][6],name_dict[key][7],name_dict[key][8],name_dict[key][10],name_dict[key][11]])
    if results[key].max()+0.5 > max_value:
        max_value = results[key].max()+0.5
    if results[key].shape[0]>max_length:
        max_length = results[key].shape[0]
       
plt.axis([0, 250, 0, max_value])
plt.rcParams.update({'font.size': 12})
plt.legend(loc = 'lower right')
plt.title('Classification of Trajectory Data - Test Results')
plt.xlabel('Epochs')
plt.ylabel('nats')
'''
#Best Runs
top_5 = pd.DataFrame(results.max().nlargest(5))
top_5 = results[top_5.index]
max_value = 88
min_value = 75
max_length = 30
lr_to_see = 0.003
fig = plt.figure(figsize=[8,6])
for key in top_5.columns:
    #plt.plot(np.linspace(0,29,30)*5,results[key])
    plt.plot(np.linspace(0,59,60)*5,results[key],label=[name_dict[key][0],name_dict[key][1],name_dict[key][2],name_dict[key][3],name_dict[key][5],name_dict[key][6],name_dict[key][7],name_dict[key][8],name_dict[key][10],name_dict[key][11]])
    if results[key].max()+0.5 > max_value:
        max_value = results[key].max()+0.5
    if results[key].shape[0]>max_length:
        max_length = results[key].shape[0]
plt.plot(np.arange(0,250),np.ones(250)*85.7)
plt.axis([0, 250, 0, max_value])
plt.rcParams.update({'font.size': 12})
plt.legend(loc = 'lower right')
plt.title('Classification of Trajectory Data - Top 5 Test Results')
plt.xlabel('Epochs')
plt.ylabel('nats')

'''
max_value = 2
max_length = 100
lr_to_see = 0.0001
fig = plt.figure(figsize=[12,8])
for key in name_dict:
    if name_dict[key][0] == lr_to_see:
        plt.plot(results[key],label=[name_dict[key][1],num_to_opt[name_dict[key][2]]])
        if results[key].max()+0.5 > max_value:
            max_value = results[key].max()+0.5
        if results[key].shape[0]>max_length:
            max_length = results[key].shape[0]
        
plt.axis([0, 900, 0, max_value])
plt.legend(loc = 'lower right')
plt.title('Mutual Information Neural Estimator MNIST - MNIST lr = {0}'.format(lr_to_see))
plt.xlabel('Epochs')
plt.ylabel('nats')

max_value = 2
max_length = 100
lr_to_see = 0.001
fig = plt.figure(figsize=[12,8])
for key in name_dict:
    if name_dict[key][0] == lr_to_see:
        plt.plot(results[key],label=[name_dict[key][1],num_to_opt[name_dict[key][2]]])
        if results[key].max()+0.5 > max_value:
            max_value = results[key].max()+0.5
        if results[key].shape[0]>max_length:
            max_length = results[key].shape[0]
        
plt.axis([0, 900, 0, max_value])
plt.legend(loc = 'lower right')
plt.title('Mutual Information Neural Estimator MNIST - MNIST lr = {0}'.format(lr_to_see))
plt.xlabel('Epochs')
plt.ylabel('nats')

max_value = 2
max_length = 100
lr_to_see = 0.003
fig = plt.figure(figsize=[12,8])
for key in name_dict:
    if name_dict[key][0] == lr_to_see:
        plt.plot(results[key],label=[name_dict[key][1],num_to_opt[name_dict[key][2]]])
        if results[key].max()+0.5 > max_value:
            max_value = results[key].max()+0.5
        if results[key].shape[0]>max_length:
            max_length = results[key].shape[0]
        
plt.axis([0, 900, 0, max_value])
plt.legend(loc = 'lower right')
plt.title('Mutual Information Neural Estimator MNIST - MNIST lr = {0}'.format(lr_to_see))
plt.xlabel('Epochs')
plt.ylabel('nats')

max_value = 2
max_length = 100
lr_to_see = 0.01
fig = plt.figure(figsize=[12,8])
for key in name_dict:
    if name_dict[key][0] == lr_to_see:
        plt.plot(results[key],label=[name_dict[key][1],num_to_opt[name_dict[key][2]]])
        if results[key].max()+0.5 > max_value:
            max_value = results[key].max()+0.5
        if results[key].shape[0]>max_length:
            max_length = results[key].shape[0]
        
plt.axis([0, 900, 0, max_value])
plt.legend(loc = 'lower right')
plt.title('Mutual Information Neural Estimator MNIST - MNIST lr = {0}'.format(lr_to_see))
plt.xlabel('Epochs')
plt.ylabel('nats')

'''
