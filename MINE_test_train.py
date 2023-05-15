
'''

Using Mine to drew information from the 
MINE - Mutual Information Neural Estimator 

steps to take:
    1)drew two images for MINE
    2)create joint_disterbution and marginal distribution 
    3)repeat:
        

In this implimintation we compare the test and train set in order to see
coherence of results and to point where the algorithm overfits       

'''

import os
import numpy as np
import pandas as pd
#import pyprind
import pickle
import torch
import torch.optim as optim
import torch.autograd as autograd

import networks 
import utils



class MINE():
    def __init__(self, train = True,traject = True, batch = 1000, lr = 3e-3, gamma = 0.001, optimizer=2, net_num = 3,
                 traject_max_depth = 512, traject_num_layers = 6, traject_stride = [3,1],
                 traject_kernel = 5, traject_padding = 0,
                 traject_pooling = [1,2], number_descending_blocks = 3, 
                 number_repeating_blocks=0, repeating_blockd_size=512,
                 dataset_status = 'same'):
        self.net_num = net_num
        self.traject = traject
        self.dataset_status = dataset_status
        print(self.dataset_status)
        self.traject_max_depth = traject_max_depth 
        self.traject_num_layers = traject_num_layers 
        self.traject_stride = traject_stride
        self.traject_kernel = traject_kernel
        self.traject_padding = traject_padding
        self.traject_pooling = traject_pooling    
        self.number_descending_blocks = number_descending_blocks 
        self.number_repeating_blocks = number_repeating_blocks 
        self.repeating_blockd_size = repeating_blockd_size
        #Defining the netwrok:
        if self.traject == 'combined':
            print('MNIST Trajectory MINE network')
            self.net = networks.statistical_estimator(traject_max_depth = self.traject_max_depth,
                                                      traject_num_layers = self.traject_num_layers, 
                                                      traject_stride = self.traject_stride,
                                                      traject_kernel = self.traject_kernel, 
                                                      traject_padding = self.traject_padding,
                                                      traject_pooling = self.traject_pooling, 
                                                      number_descending_blocks=self.number_descending_blocks, 
                                                      number_repeating_blocks = self.number_repeating_blocks, 
                                                      repeating_blockd_size = self.repeating_blockd_size)
        elif self.traject == 'traject':
            print('Traject MINE network')
            self.net = networks.conv1d_classifier_(input_dim=[18,1000,0], output_size = 1,
                                                   p_conv = 0, p_fc = 0, 
                                                   max_depth = self.traject_max_depth, 
                                                   num_layers = self.traject_num_layers, 
                                                   conv_depth_type = 'decending',
                                                   repeating_block_depth = 5, 
                                                   repeating_block_size = 0,
                                                   stride = self.traject_stride, 
                                                   kernel = self.traject_kernel, 
                                                   padding = self.traject_padding,
                                                   pooling = self.traject_pooling, 
                                                   BN = False)
        else:
            print('MNIST MINE network')
            if self.net_num == 1:
                self.net = networks.statistical_estimator_DCGAN(input_size = 2, output_size = 1)
            elif self.net_num == 2:
                self.net = networks.statistical_estimator_DCGAN_2(input_size = 1, output_size = 1)
            else:
                self.net = networks.statistical_estimator_DCGAN_3(input_size = 1, output_size = 1,number_descending_blocks = self.number_descending_blocks, 
                     number_repeating_blocks=self.number_repeating_blocks, repeating_blockd_size = self.repeating_blockd_size)
        
        #self.input1 = input1
        #self.input2 = input2
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
        print('Using Net {}'.format(self.net_num))
        if self.net_num == 3:
            print('Number of Descending Blocks is {}'.format(self.number_descending_blocks))
            print('Number of times to repeat a block = {}'.format(self.number_repeating_blocks))
            print('The fully connected layer to repeat - {}'.format(self.repeating_blockd_size))
        
    def restart_network(self):

        if self.traject == 'combine':
            print('MNIST Trajectory MINE network')
            self.net = networks.statistical_estimator(traject_max_depth = self.traject_max_depth,
                 traject_num_layers = self.traject_num_layers, traject_stride = self.traject_stride,
                 traject_kernel = self.traject_kernel, traject_padding = self.traject_padding,
                 traject_pooling = self.traject_pooling, number_descending_blocks=self.number_descending_blocks, 
                 number_repeating_blocks = self.number_repeating_blocks, repeating_blockd_size = self.repeating_blockd_size)
        elif self.traject == 'traject':
            print('Traject MINE network')
            self.net = networks.conv1d_classifier_(input_dim=[18,1000,0], output_size = 1,
                                                   max_depth = self.traject_max_depth, 
                                                   p_conv = 0, p_fc = 0, BN = False)
        else:
            print('MNIST MINE network')
            if self.net_num == 1:
                self.net = networks.statistical_estimator_DCGAN(input_size = 2, output_size = 1)
            elif self.net_num == 2:
                self.net = networks.statistical_estimator_DCGAN_2(input_size = 1, output_size = 1)
            else:
                self.net = networks.statistical_estimator_DCGAN_3(input_size = 1, output_size = 1,number_descending_blocks = self.number_descending_blocks, 
                     number_repeating_blocks=self.number_repeating_blocks, repeating_blockd_size = self.repeating_blockd_size)
            
        print('')
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
        print('Using Net {}'.format(self.net_num))
        if self.net_num == 3:
            print('Number of Descending Blocks is {}'.format(self.number_descending_blocks))
            print('Number of times to repeat a block = {}'.format(self.number_repeating_blocks))
            print('The fully connected layer to repeat - {}'.format(self.repeating_blockd_size))
            
            
    def mutual_information(self, joint1, joint2, marginal,train = True):
        if self.traject == 'traject':
            obj = torch.cat((joint1,joint2),1)
            T = self.net(obj)
            obj2 = torch.cat((joint1, marginal),1)
            eT = torch.exp(self.net(obj2))
        else:
            T = self.net(joint1, joint2)
            eT = torch.exp(self.net(joint1, marginal))
        NIM = torch.mean(T) - torch.log(torch.mean(eT)) #The neural information measure by the Donskar-Varadhan representation
        return NIM, T, eT

    def learn_mine(self,batch, ma_rate=0.01, train = True):
        # batch is a tuple of (joint1, joint2, marginal (from the dataset of joint 2))
        joint1 = torch.autograd.Variable(batch[0])
        joint2 = torch.autograd.Variable(batch[2])
        marginal = torch.autograd.Variable(batch[4]) #the uneven parts of the dataset are the labels 
        if torch.cuda.is_available():
            joint1 = joint1.to('cuda', non_blocking=True)
            joint2 = joint2.to('cuda', non_blocking=True)
            marginal = marginal.to('cuda', non_blocking=True)
            self.net = self.net.cuda()
        #joint = torch.autograd.Variable(torch.FloatTensor(joint))
        #marginal = torch.autograd.Variable(torch.FloatTensor(marginal))
        
        NIM , T, eT = self.mutual_information(joint1, joint2, marginal, train = train)
        
        #Using exponantial moving average to correct bias 
        ma_eT = (1-ma_rate)*eT + (ma_rate)*torch.mean(eT) 
        # unbiasing 
        loss = -(torch.mean(T) - (1/ma_eT.mean()).detach()*torch.mean(eT))
        # use biased estimator
        # loss = - mi_lb
        if train:
            self.mine_net_optim.zero_grad()
            autograd.backward(loss)
            self.mine_net_optim.step()
        #self.scheduler.step()
        #self.scheduler2.step(NIM)
        if torch.cuda.is_available():
            NIM = NIM.cpu()
            loss = loss.cpu()
        return NIM, loss
    
    def epoch(self,num_epoch = 1):
        # data is x or y
        
        
        result = list()
        test_results = list()
        nan = None 
        
        if self.traject == 'combined':
            dataset = utils.MNIST_TRAJECT_MINE(dataset_status = self.dataset_status)
            dataset_test = utils.MNIST_TRAJECT_MINE(dataset_status = self.dataset_status, train = False)
        elif self.traject == 'traject':
            dataset = utils.TRAJECT_MINE2()
        else:
            dataset = utils.MNIST_for_MINE(train=self.train_value, trans = None)
            dataset_test = utils.MNIST_for_MINE(train=self.train_value, trans = None)
        
        dataloader = torch.utils.data.DataLoader(dataset,  batch_size = self.batch_size, shuffle = True)
        dataloader_test = torch.utils.data.DataLoader(dataset_test,  batch_size = self.batch_size, shuffle = True)
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
        
        if num_epoch%100 == 0:
            temp_results = []
            for idx, batch in enumerate(dataloader_test):
                NIM, loss = self.learn_mine(batch)
                temp_results.append(NIM.detach())
                if torch.isnan(temp_results[-1]):
                    print(temp_results[-6:-1])
                    print('Got to NaN in epoch {0} batch {1}'.format(num_epoch,idx))
                    #plt.plot(result)
                    nan = True
                    break
                #bar.update()
                test_results.append(np.mean(temp_results))
            test_results = [np.mean(test_results)]
        
        #result = np.mean(result)
        return np.mean(result), nan, test_results
    
    def train(self,epochs = 1):
        results = []
        test_results = []
        #bar = pyprind.ProgBar(epochs, monitor = True)
        for epoch in range(epochs):
            result, nan, test_results = self.epoch(epoch)
            if nan:
                break
            print('  ',result)
            results.append(result)
            if len(test_results) == 1:
                test_results.append(test_results[0])
         #   bar.update()
            
        self.results.append(results)
        self.test_results.append(test_results)
        
    
    def ma(a, window_size=100):
        return [np.mean(a[i:i+window_size]) for i in range(0,len(a)-window_size)]


        
    

