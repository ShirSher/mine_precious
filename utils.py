#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jan 22 11:13:16 2020

@author: michael
"""


import torch
import numpy as np
import os
import gc
from random import shuffle

# CUDA for PyTorch
_device = 'cuda:3' if torch.cuda.is_available() else 'cpu'
_batch_size = 16

# creating needed output folders, if needed.
output_path = 'outputs/'
if not os.path.exists(output_path):
    os.makedirs(output_path)

# Network description for each epoch
# including bias and weights for each layer
epochs_path = 'outputs/net_epochs/'
if not os.path.exists(epochs_path):
        os.makedirs(epochs_path)


def run (mode, mine, generator, dataset, epoch_results, epoch_losses):

    # print('~~~~~~~~~~~~~~~')
    # print(mode)
    # print('~~~~~~~~~~~~~~~')

    prints = False
    results= []
    losses = []
    for i, sample in enumerate(generator):

        trajectory, joint, marginal = sample
        if (prints) :
            print('Sample (pre-mut)')
            print('trajectory:', trajectory.shape, type(trajectory))
            print('joint:     ', joint.shape, type(joint))
            print('marg:      ', marginal.shape, type(marginal))

        traj_inp = trajectory.permute(0,2,1).float()
        joint_inp = joint.permute(0,3,1,2).float()
        marg_inp = marginal.permute(0,3,1,2).float()
        if (prints) :
            print('Sample post-mut')
            print('trajectory: ', traj_inp.shape, type(traj_inp))
            print('joint:      ', joint_inp.shape, type(joint_inp))
            print('marginal:   ', marg_inp.shape, type(marg_inp))


        # where is loss recorded, managed
        NIM, loss = mine.run(mode, (traj_inp, joint_inp, marg_inp))
        if (prints) :
            print('MI',NIM.detach())
            print('loss',loss.detach())

        if torch.isnan(NIM.detach()):
            ix = _batch_size * i
            # which samples
            print('NaN samples {1}'.format(dataset.ix_list[ix:ix+_batch_size]))
            continue
        else:
            results.append(NIM.detach())
            losses.append(loss.detach())

    epoch_results.append(np.mean(results))
    epoch_losses.append(np.mean(losses))


def print_all_tensors():
    # prints currently alive Tensors and Variables
    num_tensors = 0
    num_cpu_tensors = 0
    num_gpu_tensors = 0
    for obj in gc.get_objects():
        try:
            if torch.is_tensor(obj) or (hasattr(obj, 'data') and torch.is_tensor(obj.data)) :
                print(type(obj), obj.size(), obj.get_device())
                num_tensors += 1
                num_cpu_tensors += (obj.get_device() < 0)
                num_gpu_tensors += (obj.get_device() > -1)
        except:
            pass
    print("#tensors ", num_tensors, "#cpu_tensors ", num_cpu_tensors, "#gpu_tensors ", num_gpu_tensors)


def conv1d_block_calculator(input_w, input_h = None, kernel = 5, stride = 0, padding = 0, pooling = 0):
    if not input_h:
        input_h = input_w
    dim_w = ((input_w + 2*padding - (kernel-1)-1)/stride) + 1
    dim_h = ((input_h + 2*padding - (kernel-1)-1)/stride) + 1
    if pooling:
        dim_w /=pooling
        dim_h /=pooling
    return np.floor(dim_w), np.floor(dim_h)

