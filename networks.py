#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jan 21 14:44:27 2020

@author: michael
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.autograd as autograd
import numpy as np

import utils

   # trajectories over images
class statistical_estimator(nn.Module):
    # MOD traject_input_dim from [9,1000,0]. make it flex to traj size which may vary between samples, participants => [14,300]
    def __init__(self, traject_input_dim ,p_conv=None, p_fc = None,
                 traject_max_depth = 128, traject_num_layers = 3, traject_stride = [3,1],
                 traject_kernel = 3, traject_padding = 0,
                 traject_pooling = [1,2],number_descending_blocks=4,
                 number_repeating_blocks = 3, repeating_blockd_size = 512, non_linearity = 'elu'):

        super().__init__()
        '''
        The statistical estimator for the MINE framework, recieve as input
        an image from MNIST dataset and a trajectory from the Syclop.
        Output - statistacl extimation of mutual information.
        flexible network that calculates the network dimentions to accomidate changes
        in all veriables.
        The traject runs through Conv1d blocks with batchnorm for its architecture:
            traject_input_dim - Input dimantions of the data [dimentions, dim_w, dim_h]
            traject_max_depth       - max depth of conv layers
            traject_num_layers      - number of layers in the conv part
            traject_stride          - stride of conv layers, could be a list
                                      or int if list size < network size
                                      will use the last value in the list for
                                      the rest
            traject_kernel          - kernel of conv layers, could be a list
                                      or int if list size < network size
                                      will use the last value in the list for
                                      the rest
            traject_padding         - padding of conv layers, could be a list
                                      or int if list size < network size
                                      will use the last value in the list for
                                      the rest
            padding_when_too_small  - is size of conv reduced below 4 to start
                                      same padding on all layers. defult - True
                                      if False, will stop at the last layer.
            traject_pooling         - a list of layers where to insert a pooling
                                      layer, if int will put on all layers.
                                      defult == 1 i.e. no pooling

        The image runs through Conv2d blocks with batchnorm
        They are concated on the fc layers.

        Variables:

            output_size            - output_size of classifier
            p_conv                 - dropout rate of conv layer
            p_fc                   - dropout rate of FC layers
        '''

        length_net = traject_num_layers
        if len(traject_stride) > length_net:
            traject_stride = traject_stride[:length_net]
        if len(traject_pooling) > length_net:
            traject_pooling = traject_pooling[:length_net]

        self.p_conv = p_conv
        self.p_fc = p_fc
        #conv_depth_array = np.linspace(traject_input_dim[0],traject_max_depth, traject_num_layers+1).astype(int)
        first_depth = 32
        conv_depth_array = [traject_input_dim[0],first_depth]
        for i in range(traject_num_layers):
            if conv_depth_array[-1]*2 > traject_max_depth:
                conv_depth_array.append(conv_depth_array[-1])
            else:
                conv_depth_array.append(conv_depth_array[-1]*2)
        if type(traject_stride) == int:
            self.stride = np.ones(traject_num_layers)*traject_stride
            self.stride = self.stride.astype(int).tolist()
        else:
            if len(traject_stride) < length_net:
                ''' stride should be an integar or a num_layers list taking first element only   '''
                self.stride = np.ones(length_net)*traject_stride[-1]
                self.stride = self.stride.astype(int).tolist()
                for i in range(len(traject_stride)):
                    self.stride[i] = traject_stride[i]
            else:
                self.stride = traject_stride
        if type(traject_kernel) == int:
            self.kernel = np.ones(length_net)*traject_kernel
            self.kernel = self.kernel.astype(int).tolist()
        else:
            if len(traject_kernel) < length_net:
                ''' stride should be an integar or a num_layers list taking first element only   '''
                self.kernel = np.ones(length_net)*traject_kernel[-1]
                self.kernel = self.kernel.astype(int).tolist()
                for i in range(len(traject_kernel)):
                    self.kernel[i] = traject_kernel[i]
            else:
                self.kernel = traject_kernel

        if type(traject_padding) == int:
            self.padding = np.ones(length_net)*traject_padding
            self.padding = self.padding.astype(int).tolist()
        else:
            if len(traject_kernel) < length_net:
                ''' stride should be an integar or a num_layers list taking first element only   '''
                self.padding = np.ones(length_net)*traject_padding[-1]
                self.padding = self.padding.astype(int).tolist()
                for i in range(len(traject_padding)):
                    self.padding[i] = traject_padding[i]
            else:
                self.padding = traject_padding
        ''' define the pooling layers:  '''
        if type(traject_pooling) == int:
            ''' Make sure pooling is not 0 '''
            if traject_pooling == 0:
                traject_pooling == 1
            self.pooling_array = np.ones(length_net)*traject_pooling
            self.pooling_array = self.pooling_array.astype(int).tolist()
        else:
            ''' Make sure there is no 0 in the pooling arrays. '''
            while 0 in traject_pooling:
                traject_pooling[traject_pooling.index(0)] = 1
            self.pooling_array = np.ones(length_net)*traject_pooling[-1]
            self.pooling_array = self.pooling_array.astype(int).tolist()
            for pool in range(len(traject_pooling)):
                self.pooling_array[pool] = traject_pooling[pool]
        ''' Define conv layers:  '''
        dim = []
        self.conv_layers = nn.ModuleList()
        dim_w = traject_input_dim[1]
        for i in range(traject_num_layers):
            dim_w_old = dim_w
            dim_w, _ = utils.conv1d_block_calculator(input_w = dim_w, kernel = self.kernel[i], stride = self.stride[i], padding = self.padding[i], pooling = self.pooling_array[i])
            #dim.append(dim_w)
            new_padding = self.padding[i]
            ''' making sure the minimal size of dim_w is 4 or greater '''
            if dim_w < 4:
                while dim_w < 4:
                    new_padding += 1
                    dim_w, _ = utils.conv1d_block_calculator(input_w = dim_w_old, kernel = self.kernel[i], stride = self.stride[i], padding = new_padding, pooling = self.pooling_array[i])

                self.conv_layers.append(nn.Conv1d(conv_depth_array[i],conv_depth_array[i+1],self.kernel[i], stride = self.stride[i], padding = new_padding))
            else:
                self.conv_layers.append(nn.Conv1d(conv_depth_array[i],conv_depth_array[i+1],self.kernel[i], stride = self.stride[i], padding = self.padding[i]))

            dim.append(dim_w)
            new_padding = 0

        self.final_conv_dim = dim_w.astype(int)
        self.fc_first_size = self.final_conv_dim*conv_depth_array[-1]

        if not p_conv:
            p_conv = 0
        '''  Creating a network with or without batchnorm/dropout '''
        if p_conv > 0:
            self.conv_drop = nn.ModuleList()
            for i in range(length_net):
                self.conv_drop.append(nn.Dropout(p=p_conv))
        else:
            self.conv_batchNorm = nn.ModuleList()
            for i in range(traject_num_layers):
                self.conv_batchNorm.append(nn.BatchNorm1d(conv_depth_array[i+1]))

        ''' Defining the Conv2d netwrok processing the image: '''
        if type(repeating_blockd_size) == int:
            repeating_blockd_size = [repeating_blockd_size]
        ''' Define the networks steps: '''
        ''' mod for 32 x 32 '''
        self.conv1 = nn.Conv2d(3, 16, 5, stride = 5, padding = 2)
        self.conv2 = nn.Conv2d(16, 32, 5, stride = 2, padding = 2)
        self.conv3 = nn.Conv2d(32, 64, 5, stride = 2, padding = 2)
        self.non_linearity = non_linearity
        if self.non_linearity == 'elu':
            self.non_linearity = nn.ELU(inplace = True)
        elif self.non_linearity == 'leaky':
            self.non_linearity = nn.LeakyReLU()
        '''  Defining the joint fully connected layers: '''
        fc = []
        in_dim = int(1024 + self.fc_first_size)
        if number_descending_blocks>7:
            print('number_descending_blocks should be <= 7, assigning descending_block_size = 7')
            number_descending_blocks = 7
        for i in range(number_descending_blocks):
            new_dim = int(in_dim/2)
            if in_dim in repeating_blockd_size:
                for j in range(number_repeating_blocks):
                    fc.append(nn.Linear(in_dim,in_dim))
            else:
                fc.append(nn.Linear(in_dim,new_dim))
            in_dim = new_dim

        self.fc = nn.ModuleList(fc)
        #self.fc12 = nn.Linear(1024,256)
        # MOD
        self.fc_last = nn.Linear(fc[-1].out_features,1)

    def forward(self, traject, image):

        if self.p_conv:
            for idx, conv_layer in enumerate(self.conv_layers):
                drop = self.conv_drop[idx]
                pool = nn.MaxPool1d(self.pooling_array[idx])
                traject = pool(self.non_linearity(drop(conv_layer(traject))))
        else:
            for idx, conv_layer in enumerate(self.conv_layers):
                pool = nn.MaxPool1d(self.pooling_array[idx])
                traject = pool(self.non_linearity(conv_layer(traject)))

        traject = traject.view(traject.size(0), self.fc_first_size)

        image = self.non_linearity(self.conv1(image))
        image = self.non_linearity(self.conv2(image))
        image = self.non_linearity(self.conv3(image))
        image = image.contiguous().view(image.size(0), -1)

        # ~~~~~~~~~
        # 1D vector
        # ~~~~~~~~~
        x = torch.cat((image,traject),1)
        if self.fc[0].in_features != x.shape[1]:
            y = nn.Linear(x.shape[1], self.fc[0].in_features).to(torch.device(utils._device), non_blocking=True)
            self.fc.insert(0, y)

        for idx, des_fc in enumerate(self.fc):
            x = self.non_linearity(des_fc(x))
        x = self.fc_last(x)

        return x