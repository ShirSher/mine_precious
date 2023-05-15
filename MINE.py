
'''

Using Mine to drew information from the 
MINE - Mutual Information Neural Estimator 

steps to take:
    1)draw two images for MINE
    2)create joint_distribution and marginal distribution 
    3)repeat:
        
        

'''

import os
import numpy as np
import pandas as pd
#import pyprind
import pickle

import matplotlib.pyplot as plt


import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.autograd as autograd
import torchvision

import networks 
import utils
import FullDriftDataset


# mod added traject_input_dim and hardcoded to dimenstions ()
# mod rm "net_num", "traject", "dataset_status"
class MINE():
    def __init__(self, traject_input_dim=[14,300], train = True,batch = 1000, lr = 3e-3, gamma = 0.001, optimizer=2,
                 traject_max_depth = 512, traject_num_layers = 6, traject_stride = [3,1],
                 traject_kernel = 5, traject_padding = 0,
                 traject_pooling = [1,2], number_descending_blocks = 3, 
                 number_repeating_blocks=0, repeating_blockd_size=512):
        # self.net_num = net_num
        # self.traject = traject
        # self.dataset_status = dataset_status
        # print(self.dataset_status)
        self.traject_max_depth = traject_max_depth 
        self.traject_num_layers = traject_num_layers 
        self.traject_stride = traject_stride
        self.traject_kernel = traject_kernel
        self.traject_padding = traject_padding
        self.traject_pooling = traject_pooling    
        self.number_descending_blocks = number_descending_blocks 
        self.number_repeating_blocks = number_repeating_blocks 
        self.repeating_blockd_size = repeating_blockd_size
        # mod
        self.traject_input_dim = traject_input_dim
        #Defining the network:
        self.net = networks.statistical_estimator(traject_max_depth = self.traject_max_depth,
                                                    traject_input_dim= self.traject_input_dim,
                                                      traject_num_layers = self.traject_num_layers, 
                                                      traject_stride = self.traject_stride,
                                                      traject_kernel = self.traject_kernel, 
                                                      traject_padding = self.traject_padding,
                                                      traject_pooling = self.traject_pooling, 
                                                      number_descending_blocks=self.number_descending_blocks, 
                                                      number_repeating_blocks = self.number_repeating_blocks, 
                                                      repeating_blockd_size = self.repeating_blockd_size)
        self.lr = lr
        self.optimizer = optimizer
        if type(optimizer) != int:
            optimizer = int(optimizer)
        if self.optimizer == 1:
            self.mine_net_optim = optim.SGD(self.net.parameters(), lr = self.lr)
            print('')
            print('Optimizer: SGD')
        elif self.optimizer == 2:
            self.mine_net_optim = optim.Adam(self.net.parameters(), lr = self.lr)
            print('')
            print('Optimizer: Adam')
        else:
            self.mine_net_optim = optim.RMSprop(self.net.parameters(), lr = self.lr)
            print('')
            print('Optimizer: SGD')
        #MOD TODO train/test
        self.train_value = train
        #self.dataloader = torch.utils.data.DataLoader(self.dataset,  batch_size = batch, shuffle = True)    
        self.batch_size = batch
        
        #self.scheduler = optim.lr_scheduler.StepLR(self.mine_net_optim, step_size=10*(len(self.dataset)/self.batch_size), gamma=gamma)
        self.gamma = gamma
        #self.scheduler2 = optim.lr_scheduler.ReduceLROnPlateau(self.mine_net_optim, mode='max', factor=0.5, patience=10, verbose=False, threshold=0.0001, threshold_mode='abs', cooldown=0, min_lr=0, eps=1e-08)    
        self.results = []
        
        #print all variables of the system:
        print('Learning rate = {}'.format(self.lr))
        print('Batch size = {}'.format(self.batch_size))
        #print('Gamma of lr decay = {}'.format(self.gamma))
        # print('Using Net {}'.format(self.net_num))
        # if self.net_num == 3:
        #     print('Number of Descending Blocks is {}'.format(self.number_descending_blocks))
        #     print('Number of times to repeat a block = {}'.format(self.number_repeating_blocks))
        #     print('The fully connected layer to repeat - {}'.format(self.repeating_blockd_size))
        
    def restart_network(self):
        self.net = networks.statistical_estimator(traject_max_depth = self.traject_max_depth,
                traject_num_layers = self.traject_num_layers, traject_stride = self.traject_stride,
                traject_kernel = self.traject_kernel, traject_padding = self.traject_padding,
                traject_pooling = self.traject_pooling, number_descending_blocks=self.number_descending_blocks, 
                number_repeating_blocks = self.number_repeating_blocks, repeating_blockd_size = self.repeating_blockd_size)
        print('Restarted Network')
        if self.optimizer == 1:
            self.mine_net_optim = optim.SGD(self.net.parameters(), lr = self.lr)
            print('Optimizer: SGD')
        elif self.optimizer == 2: 
            self.mine_net_optim = optim.Adam(self.net.parameters(), lr = self.lr)
            print('Optimizer: Adam')
        else:
            self.mine_net_optim = optim.RMSprop(self.net.parameters(), lr = self.lr)
            print('Optimizer: RMSprop')
        #self.scheduler = optim.lr_scheduler.StepLR(self.mine_net_optim, step_size=10*(len(self.dataset)/self.batch_size), gamma=self.gamma)
        print('Learning rate = {}'.format(self.lr))
        print('Batch size = {}'.format(self.batch_size))
        #print('Gamma of lr decay = {}'.format(self.gamma))
       
            
    def mutual_information(self,joint1, joint2, marginal):
        # TODO Confirm non "traject"
        # if self.traject == 'traject':
        #     obj = torch.cat((joint1,joint2),1)
        #     T = self.net(obj)
        #     obj2 = torch.cat((joint1, marginal),1)
        #     eT = torch.exp(self.net(obj2))
        # else:
        T = self.net(joint1, joint2)
        eT = torch.exp(self.net(joint1, marginal))
        NIM = torch.mean(T) - torch.log(torch.mean(eT)) #The neural information measure by the Donskar-Varadhan representation
        return NIM, T, eT
# MOD
    def validate_mine(self, batch,ma_rate=0.01):
        # batch is a tuple of (joint1, joint2, marginal (from the dataset of joint 2))
        joint1 = torch.autograd.Variable(batch[0])
        joint2 = torch.autograd.Variable(batch[1])
        marginal = torch.autograd.Variable(batch[2]) #the uneven parts of the dataset are the labels 
        if torch.cuda.is_available():
            joint1 = joint1.to('cuda', non_blocking=True)
            joint2 = joint2.to('cuda', non_blocking=True)
            marginal = marginal.to('cuda', non_blocking=True)
            self.net = self.net.cuda()
        #joint = torch.autograd.Variable(torch.FloatTensor(joint))
        #marginal = torch.autograd.Variable(torch.FloatTensor(marginal))
        
        NIM , T, eT = self.mutual_information(joint1, joint2, marginal)
        
        #Using exponantial moving average to correct bias 
        ma_eT = (1-ma_rate)*eT + (ma_rate)*torch.mean(eT) 
        # unbiasing 
        loss = -(torch.mean(T) - (1/ma_eT.mean()).detach()*torch.mean(eT))
        # use biased estimator
        # loss = - mi_lb
        
        # self.mine_net_optim.zero_grad()
        # autograd.backward(loss)
        # self.mine_net_optim.step()
        #self.scheduler.step()
        #self.scheduler2.step(NIM)
        # if cuda, why put it on the cpu?
        if torch.cuda.is_available():
            NIM = NIM.cpu() 
            loss = loss.cpu()
        return NIM, loss


    def learn_mine(self,batch, ma_rate=0.01):
        # batch is a tuple of (joint1, joint2, marginal (from the dataset of joint 2))
        joint1 = torch.autograd.Variable(batch[0])
        # mod from [2] to [1]
        joint2 = torch.autograd.Variable(batch[1])
        # mod from [4] to [2]
        marginal = torch.autograd.Variable(batch[2]) #the uneven parts of the dataset are the labels 
        if torch.cuda.is_available():
            joint1 = joint1.to('cuda', non_blocking=True)
            joint2 = joint2.to('cuda', non_blocking=True)
            marginal = marginal.to('cuda', non_blocking=True)
            self.net = self.net.cuda()
        #joint = torch.autograd.Variable(torch.FloatTensor(joint))
        #marginal = torch.autograd.Variable(torch.FloatTensor(marginal))
        
        NIM , T, eT = self.mutual_information(joint1, joint2, marginal)
        
        #Using exponantial moving average to correct bias 
        ma_eT = (1-ma_rate)*eT + (ma_rate)*torch.mean(eT) 
        # unbiasing 
        loss = -(torch.mean(T) - (1/ma_eT.mean()).detach()*torch.mean(eT))
        # use biased estimator
        # loss = - mi_lb
        
        self.mine_net_optim.zero_grad()
        autograd.backward(loss)
        self.mine_net_optim.step()
        #self.scheduler.step()
        #self.scheduler2.step(NIM)
        # if cuda, why put it on the cpu?
        if torch.cuda.is_available():
            NIM = NIM.cpu()
            loss = loss.cpu()
        return NIM, loss
    # mod insert ix_selection
    def epoch(self,ix_selection,num_epoch = 1):
        # data is x or y
        result = list()
        nan = None 
        
        dataset = FullDriftDataset.FullDataset(ix_dict=ix_selection)
        dataloader = torch.utils.data.DataLoader(dataset,  batch_size = self.batch_size, shuffle = True)    
        #bar = pyprind.ProgBar(len(dataloader), monitor = True)
        temp_results = []
        for idx, batch in enumerate(dataloader):
            NIM, loss = self.learn_mine(batch)
            temp_results.append(NIM.detach())
            if torch.isnan(temp_results[-1]):
                print(temp_results[-6:-1])
                print('Got to NaN in epoch {0} batch {1}'.format(num_epoch,idx))
                #plt.plot(result)
                nan = True
                break
            #bar.update()
            result.append(np.mean(temp_results))
        #result = np.mean(result)
        return np.mean(result), nan
    
    def train(self,epochs = 1):
        results = []
        #bar = pyprind.ProgBar(epochs, monitor = True)
        for epoch in range(epochs):
            result, nan = self.epoch(epoch)
            if nan:
                break
            print('  ',result)
            results.append(result)
         #   bar.update()
            
        self.results.append(results)
        
    
    def ma(a, window_size=100):
        return [np.mean(a[i:i+window_size]) for i in range(0,len(a)-window_size)]