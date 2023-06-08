#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Feb  5 11:12:43 2020

@author: michael
"""

import torch
import torch.optim as optim
import pandas as pd
import numpy as np
import pickle 

import MINE 
import utils

import matplotlib.pyplot as plt

import os
cwd = os.getcwd()

def create_train_test(dr = cwd):
    with open(cwd+'/mnist_padded_act_full/mnist_padded_act_full1.pkl', 'rb') as f:
        # The protocol version used is detected automatically, so we do not
        # have to specify it.
        data1 = pickle.load(f)
    
    with open(cwd+'/mnist_padded_act_full/mnist_padded_act_full2.pkl', 'rb') as f:
        # The protocol version used is detected automatically, so we do not
        # have to specify it.
        data2 = pickle.load(f)
    
    
    data = data1[0] + data2[0]
    labels = data1[1] + data2[1]
    dataset = [data,labels]
    dataset = utils.one_hot_dataset(dataset)
    
    data1 = []
    data2 = []
    
    k = int(len(dataset)*0.09)
    train, test = utils.sampleFromClass(dataset,k)
    return train, test

#dataset = []
# Lets get some stats about the trajectories:
# most popular actions:

# most popular for each label

# recurrent combination of actions

#  
   
#one_hot1 = utils.one_hot(data1)
#one_hot2 = utils.one_hot(data2)
