#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jan 20 10:00:35 2020

@author: michael
"""
import torch
import torch.optim as optim
import torch.nn as nn
import torchvision
import matplotlib.pyplot as plt
import os
import numpy as np
import pickle
import pyprind


import networks
import utils

def check_accuracy(dataset):
    net.eval()
    test_loss = 0 
    correct = 0
    test_losses = []
    with torch.no_grad():
        for data, target in dataset:
            output = net(data)
            test_loss += loss_func(output, target).item()
            pred = output.data.max(1, keepdim = True)[1]
            correct += pred.eq(target.data.view_as(pred)).sum()
        test_loss /= len(test_loader.dataset)
        test_losses.append(test_loss)
        test_accuracy = 100.*correct/len(dataset.dataset)
        #print("test loss = {} , percentage of correct answers = {}".format(test_loss,100.*correct/len(dataset.dataset)))
    return test_accuracy, np.mean(test_losses)

max_depth = 256
num_layers = 5
stride = [1,1]
kernel = 5
pooling = [2,2]
'''
net = networks.conv1d_classifier_(input_dim = [9,1000,0], output_size = 10, p_conv=0.1, p_fc = 0.5, 
                 max_depth = max_depth, num_layers = num_layers, repeating_block_depth = 3, 
                 repeating_block_size = 3,stride = stride, kernel = kernel, padding = 0,
                 pooling = pooling)
'''
net = networks.conv1d_classifier_multi_head()
#Best 0.1 0.5 256 5 0 0 [2,1,1] pooling -[1,2,2] kernel - 5 
#overfit early:  0.1 0.5 256 5 0 0 stride - [1,1,1] pooling -[2,2,2] kernel - 3 
#overfit, reaches 84% 0.1 0.5 256 5 0 0 stride - [1,1,1] pooling -[2,2,2] kernel - 5
#8 layers not better
#kernel 3 stride 5 not better
#
#512?

#net = networks.trajectory_classifier_dropout2(max_depth = 256)
print(net)
print('defined network')
optimizer = optim.RMSprop(net.parameters(), lr=3e-3)
loss_func = nn.CrossEntropyLoss()
epochs = 50
batch_size = 500
bar = pyprind.ProgBar(len(train)/batch_size*epochs, monitor = True)
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
        optimizer.zero_grad()
        output = net(data)
        loss = loss_func(output, target)
        loss.backward()
        optimizer.step()
        bar.update()
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

print('Training Accuracy is:')
train_accur, _ = check_accuracy(dataloader)
print(train_accur)
plt.figure(figsize = [12,8])
plt.plot(train_accuracies)
plt.title('Train accuracies Epochs')
plt.xlabel('Epochs')
plt.ylabel('Accuracies %')

plt.figure(figsize = [12,8])
plt.plot(test_accuracies)
plt.title('Test Accuracies Over Epochs')
plt.xlabel('Epochs')
plt.ylabel('Accuracies')


with open('train_accuracies_{1}_{2}_{3}_{4}'.format(max_depth, num_layers, stride, kernel), 'wb') as f:
    pickle.dump(train_accuracies)
with open('test_accuracies_{1}_{2}_{3}_{4}'.format(max_depth, num_layers, stride, kernel), 'wb') as f:
    pickle.dump(test_accuracies)


    
#os.chdir('/home/michael/Documents/Scxript/Syclop-MINE-master/Trajectory_classification/')
#torch.save(net.state_dict, 'net_{}'.format(int(100.*correct/len(test_loader.dataset))))