#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Feb  2 14:44:27 2020

@author: michael
"""


import sys
import os
import numpy as np
import torch
import torch.optim as optim
import torch.nn as nn
import matplotlib.pyplot as plt
import numpy as np
import pickle

#import pyprind
import networks
import utils


# sys.argv gets 15 values:
# [1] - the optimizer - one of three Adam, SGD, RMSprop
# [2] - lr
# [3] - batch_size
# [4] - epochs
# [5] - Net num - 1-3
# [6] - Dropout fc
# [7] - Dropout conv
# [8] - kernel
# [9] - stride
# [10] - pooling
# [11] - max_depth
# [12] - num_layers                           
# [13] - reapeating_block_depth - what number in num_layers should we repeat
# [14] - repeating_block_number - how many times to repeat
# [15] - data_path - if the data is stored outside the working directory. If not 
#                    sys.argv[0] should be 0


def check_accuracy(dataset):
    net.eval()
    test_loss = 0 
    correct = 0
    test_losses = []
    with torch.no_grad():
        for data, target in dataset:
            data = data.to('cuda', non_blocking=True)
            target = target.to('cuda', non_blocking=True)
            output = net(data)
            test_loss += loss_func(output, target).item()
            pred = output.data.max(1, keepdim = True)[1]
            correct += pred.eq(target.data.view_as(pred)).sum()
        test_loss /= len(test_loader.dataset)
        test_losses.append(test_loss)
        test_accuracy = 100.*correct/len(dataset.dataset)
        #print("test loss = {} , percentage of correct answers = {}".format(test_loss,100.*correct/len(dataset.dataset)))
    return test_accuracy, np.mean(test_losses)


#Transforming the sys.argv to training veriables
optimizer = int(sys.argv[1])
lr = float(sys.argv[2])
batch_size = int(sys.argv[3])
epochs = int(sys.argv[4])
net_num = int(sys.argv[5])
p_fc = float(sys.argv[6])
p_conv = float(sys.argv[7])
kernel = int(sys.argv[8])
stride = list(map(int,list(sys.argv[9][1:-1].split(','))))
pooling = list(map(int,list(sys.argv[10][1:-1].split(','))))
max_depth = int(sys.argv[11])
num_layers = int(sys.argv[12])
repeating_block_depth = int(sys.argv[13])
repeating_block_size = int(sys.argv[14])
head_kernel = list(map(int,list(sys.argv[15][1:-1].split(','))))
#if data_path is 0 we use the cwd:
#if len(data_path) == 1:
#    data_path = os.getcwd()
data_path = os.getcwd()
if os.path.exists('train.pickle'):
    if os.path.exists('test.pickle'):  
        print('found train.pickle')
        train = pickle.load(open('train.pickle', 'rb'))
        test = pickle.load(open('test.pickle', 'rb'))
    else:
        train, test = utils.create_train_test(dr = data_path)
        with open('train.pickle', 'wb') as f:
            pickle.dump(train, f)
        with open('test.pickle', 'wb') as f:
            pickle.dump(test,f)
else:
    print('didnt find train.pickle')
    train, test = utils.create_train_test(dr = data_path)
    with open('train.pickle', 'wb') as f:
        pickle.dump(train, f)
    
    with open('test.pickle', 'wb') as f:
        pickle.dump(test,f)

if net_num == 1:
    net = networks.conv1d_classifier_(input_dim = [9,1000,0], output_size = 10, p_conv=p_conv, p_fc = p_fc, 
                     max_depth = max_depth, num_layers = num_layers, repeating_block_depth = repeating_block_depth, 
                     repeating_block_size = repeating_block_size,stride = stride, kernel = kernel, padding = 0,
                     pooling = pooling)
elif net_num == 2:
    net = networks.conv1d_classifier_multi_head(kernel = head_kernel)
print(net)
print('cuda availble?:',torch.cuda.is_available())
#Defining the optimizer
if type(optimizer) != int: 
    optimizer = int(optimizer)
if optimizer == 1:
    optimizer = optim.SGD(net.parameters(), lr = lr)
    print('')
    print('Optimizer: SGD')
elif optimizer == 2:
    optimizer = optim.Adam(net.parameters(), lr = lr)
    print('')
    print('Optimizer: Adam')
else:
    optimizer = optim.RMSprop(net.parameters(), lr = lr)
    print('')
    print('Optimizer: RMS')

print('defined network')
#optimizer = optim.Adam(net.parameters(), lr=3e-3)
loss_func = nn.CrossEntropyLoss()

if torch.cuda.is_available():
    print('cuda availble')
    net.cuda()
#bar = pyprind.ProgBar(len(train)/batch_size*epochs, monitor = True)
dataloader = torch.utils.data.DataLoader(train,  batch_size = batch_size, shuffle = True)
train_loss = []
test_loader = torch.utils.data.DataLoader(test,  batch_size = len(test), shuffle = True)
test_accuracies = []
train_accuracies = []
for i in range(epochs):
    #print('Starting Epoch Number {}'.format(i+1))
    batch_loss = []
    correct = 0
    for batch_idx, (data,target) in enumerate(dataloader):
        data = data.to('cuda', non_blocking=True)
        target = target.to('cuda', non_blocking=True)
        optimizer.zero_grad()
        output = net(data)
        loss = loss_func(output, target)
        loss.backward()
        optimizer.step()
        #bar.update()
        batch_loss.append(loss.item())
        pred = output.data.max(1, keepdim = True)[1]
        correct += pred.eq(target.data.view_as(pred)).sum()
    
    if i%5==0:
        print('')
        print('Epoch number {}'.format(i))
        print('Loss is {}'.format(np.mean(batch_loss)))
        train_accur, _ = check_accuracy(dataloader)
        train_accuracies.append(train_accur)
        test_accur, _ = check_accuracy(test_loader)
        test_accuracies.append(test_accur)
        net.train()
        print('Training Accuracy is: {}'.format(train_accur))
        print('Test Accuracy is: {}'.format(test_accur))

    train_loss.append(np.mean(batch_loss))

# sys.argv gets 4 values:
# [1] - the optimizer - one of three Adam, SGD, RMSprop
# [2] - lr
# [3] - batch_size
# [4] - epochs
# [5] - Net num - 1-3
# [6] - Dropout fc
# [7] - Dropout conv
# [8] - kernel
# [9] - stride
# [10] - pooling
# [11] - max_depth
# [12] - num_layers                           
# [13] - reapeating_block_depth - what number in num_layers should we repeat
# [14] - repeating_block_number - how many times to repeat
# [15] - data_path - if the data is stored outside the working directory. If not 
#                    sys.argv[0] should be 0

name_test= 'test_accuracies_{0}_{1}_{2}_{3}_{4}_{5}_{6}_{7}_{8}_{9}_{10}_{11}_{12}_{13}'.format(sys.argv[1],
                            int(float(sys.argv[2])*10000),sys.argv[3],
                            sys.argv[4],sys.argv[5],int(float(sys.argv[6])*10),
                            int(float(sys.argv[7])*10),sys.argv[8],sys.argv[9],sys.argv[10],
                                sys.argv[11],sys.argv[12],sys.argv[13],sys.argv[14])
name_train= 'train_accuracies_{0}_{1}_{2}_{3}_{4}_{5}_{6}_{7}_{8}_{9}_{10}_{11}_{12}_{13}'.format(sys.argv[1],
                            int(float(sys.argv[2])*10000),sys.argv[3],
                            sys.argv[4],sys.argv[5],int(float(sys.argv[6])*10),
                            int(float(sys.argv[7])*10),sys.argv[8],sys.argv[9],sys.argv[10],
                                sys.argv[11],sys.argv[12],sys.argv[13],sys.argv[14])
name_train_loss= 'train_loss_{0}_{1}_{2}_{3}_{4}_{5}_{6}_{7}_{8}_{9}_{10}_{11}_{12}_{13}'.format(sys.argv[1],
                            int(float(sys.argv[2])*10000),sys.argv[3],
                            sys.argv[4],sys.argv[5],int(float(sys.argv[6])*10),
                            int(float(sys.argv[7])*10),sys.argv[8],sys.argv[9],sys.argv[10],
                                sys.argv[11],sys.argv[12],sys.argv[13],sys.argv[14])
torch.save( test_accuracies, name_test)
torch.save( train_accuracies, name_train )
torch.save( train_loss, name_train_loss)
#results.to_csv('results_{0}_{1}_{2}_{3}_{4}.csv'.format(sys.argv[1],int(float(sys.argv[2])*10000),sys.argv[3],sys.argv[4],sys.argv[5],sys.argv[6],sys.argv[7],sys.argv[8]))
torch.save(net.state_dict, 'class_net_{0}_{1}_{2}_{3}_{4}_{5}_{6}_{7}_{8}_{9}_{10}_{11}_{12}_{13}'.format(sys.argv[1],
                            int(float(sys.argv[2])*10000),sys.argv[3],
                            sys.argv[4],sys.argv[5],int(float(sys.argv[6])*10),
                            int(float(sys.argv[7]))*10,sys.argv[8],sys.argv[9],sys.argv[10],
                                sys.argv[11],sys.argv[12],sys.argv[13],sys.argv[14]))

