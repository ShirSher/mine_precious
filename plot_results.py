#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Feb  6 10:28:24 2020

@author: michael
"""
import torch
import torch.optim as optim

import matplotlib.pyplot as plt
import numpy as np

import pandas as pd
import os

import MINE
import networks

max_value = 2
max_length = 100
opt_to_see = 1
fig = plt.figure(figsize=[12,8])
for key in name_dict:
    if name_dict[key][2] == opt_to_see:
        plt.plot(results[key],label=[name_dict[key][0],name_dict[key][1]])
        if results[key].max()+0.5 > max_value:
            max_value = results[key].max()+0.5
        if results[key].shape[0]>max_length:
            max_length = results[key].shape[0]
        
plt.axis([0, 900, 0, max_value])
if opt_to_see == 1:
    opt_to_see = 'SGD'
elif opt_to_see == 2:
    opt_to_see = 'Adam'
else:
    opt_to_see = 'RMSprop'
    
plt.legend(loc = 'lower right')
plt.title('Mutual Information Neural Estimator MNIST - MNIST Optimizer = {0}'.format(opt_to_see))
plt.xlabel('Epochs')
plt.ylabel('nats')

max_value = 2
max_length = 100
opt_to_see = 2
fig = plt.figure(figsize=[12,8])
for key in name_dict:
    if name_dict[key][2] == opt_to_see:
        plt.plot(results[key],label=[name_dict[key][0],name_dict[key][1]])
        if results[key].max()+0.5 > max_value:
            max_value = results[key].max()+0.5
        if results[key].shape[0]>max_length:
            max_length = results[key].shape[0]
        
plt.axis([0, 900, 0, max_value])
if opt_to_see == 1:
    opt_to_see = 'SGD'
elif opt_to_see == 2:
    opt_to_see = 'Adam'
else:
    opt_to_see = 'RMSprop'
    
plt.legend(loc = 'lower right')
plt.title('Mutual Information Neural Estimator MNIST - MNIST Optimizer = {0}'.format(opt_to_see))
plt.xlabel('Epochs')
plt.ylabel('nats')

max_value = 2
max_length = 100
opt_to_see = 3
fig = plt.figure(figsize=[12,8])
for key in name_dict:
    if name_dict[key][2] == opt_to_see:
        plt.plot(results[key],label=[name_dict[key][0],name_dict[key][1]])
        if results[key].max()+0.5 > max_value:
            max_value = results[key].max()+0.5
        if results[key].shape[0]>max_length:
            max_length = results[key].shape[0]
        
plt.axis([0, 900, 0, max_value])
if opt_to_see == 1:
    opt_to_see = 'SGD'
elif opt_to_see == 2:
    opt_to_see = 'Adam'
else:
    opt_to_see = 'RMSprop'
    
plt.legend(loc = 'lower right')
plt.title('Mutual Information Neural Estimator MNIST - MNIST Optimizer = {0}'.format(opt_to_see))
plt.xlabel('Epochs')
plt.ylabel('nats')