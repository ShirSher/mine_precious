import torch
import torchvision
import numpy as np
import pandas as pd
import os
import pickle
from datetime import datetime
from random import shuffle, choice
import utils
import numpy as np
from scipy.fft import fft2, fftshift

# in folder need - normsXY, events, 
_base_path = '../mine_data/'
'''
    125 hz
    first 100 stimuli * 3 seconds each
    timedelta starts with 0 st second 300 is the right edge
    add one more time bracket for interpolation before trimming
'''
END_DELTA = utils._nobs # 300
_nStimuli = 100

nChannels = 2 # [normX, normY]

_device = utils._device

'''
    Selection are tables where for each participant
    there's an array with numbers 1-100 shuffled
    the numbers represent labels in Cifar100 dataset.
'''
def get_selection ():
    # ================
    # DATA
    # ================
    testing = True
    data_path = _base_path
    participants_set = set([_dir for _dir in os.listdir(data_path) if not _dir.startswith(".")])
    exclude_set = set([]) # set(['DL', 'NH', 'NC', 'OL', 'OA', 'SM', 'TL', 'VT'])
    participants_list = list(participants_set - exclude_set)
    participants_list = [p for p in participants_list if 'try' not in p.lower()]
    if testing==True:
        participants_list = participants_list[:2] # for testing  - rm for train.
        print('Testing on participants,', participants_list)
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # ^^ aks what are those numbers?
    # 300 observations per participant - 3sec
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    print(_nStimuli * len(participants_list), 'total observations')

    selection = {participant:np.array(range(0, _nStimuli)) for participant in participants_list}
    for k in selection.keys():
        # shuffle each individually. inplace function. returns None
        shuffle(selection[k])
    # train set
    # 80 random observations of each participant
    cut = int(_nStimuli * .80)
    train_selection = {k:selection[k][:cut] for k in selection.keys()}
    # validation set
    # 20 random observations of each participant
    val_selection = {k:selection[k][cut:] for k in selection.keys()}

    # once upon a time there was a test selection array. Abandoned, forgotten and unused.
    # more can be found in git :)

    # save selections for future runs
    with open(utils.output_path+'train_selection.pickle', 'wb') as handle:
        pickle.dump(train_selection, handle, protocol=pickle.HIGHEST_PROTOCOL)

    with open(utils.output_path+'val_selection.pickle', 'wb') as handle:
        pickle.dump(val_selection, handle, protocol=pickle.HIGHEST_PROTOCOL)

    return train_selection, val_selection


'''
    converting from the order Unity loads files:
    0,1,10,11,...2,20,21,...99
    to
    0,1,2,3,...,99 
'''
def conv_alph_to_num(num):
    if num<2 or num>89:
        return num
    units = num%10
    tens = num//10
    if (units - tens) == 1:
        return units
    temp = tens*10 + tens + 1
    if (num > temp):
        temp += 11
    return num + (10 - temp%10)

#  _nStimuli+1?
# base path


class Participant:

    def __init__(self, part_name):
        self.part_name = part_name
        self.simset = 0
        self.img_list = []
        self.normX = np.array([-1] * (_nStimuli+1)) 
        self.normY = np.array([-1] * (_nStimuli+1)) 

    def set_simset(self):
        events_path = _base_path + self.part_name + '/Events.csv'
        events_df = pd.read_csv(events_path)
        self.simset = int(events_df.columns[1].split('_')[1])

    def set_norms(self):
        norms_path = _base_path + self.part_name + '/normXY.csv'
        norms_df = pd.read_csv(norms_path)

        for index, row in norms_df.iterrows():
                if index > (_nStimuli+1):
                    break
                # Access the values of each column in the current row
                image = conv_alph_to_num(row['image'])
                self.img_list.append(image)
                self.normX[image] = row['norm_x']
                self.normY[image] = row['norm_y']

class DriftDataset(torch.utils.data.Dataset):

    def __init__(self, ix_dict, path=_base_path, sample_size=END_DELTA ):

        # ix_dict is a dictionary where key = participant's name and value = array of selected labels
        self.part_list = list(ix_dict)
        shuffle(self.part_list)
        
        # Moving ix_list from dictionary to pairs
        self.ix_list = [(participant, ix) for participant, ix_list in ix_dict.items() for ix in ix_list]
        shuffle(self.ix_list)

        self.part_dict = None 
        self.imgset = []
        self.imgset_labels = []
        self.sample_size = sample_size
        self.create_datasets()

    def create_datasets(self):

        cifar_data = torchvision.datasets.CIFAR100('./cifar100data/',train=True,download=True) 
        for i in range(len(cifar_data)):
            self.imgset.append(np.array(cifar_data[i][0]))
            self.imgset_labels.append(cifar_data[i][1])

        self.imgset = np.array(self.imgset)
        self.imgset_labels = np.array(self.imgset_labels)

        for part_name in self.part_list:

            participant = Participant(part_name)
            # change img with fucntion 
            participant.set_simset()
            participant.set_norms()
            self.part_dict[part_name] = participant

    def decode_image_from_simset_and_label(self, part_name, event_i):
        participant = self.part_dict[part_name]
        simset = participant.simset
        label = participant.img_list[event_i]
        loc = np.where(self.imgset_labels==label)[0][simset-1]
        return self.imgset[loc]
    
    def get_trajectory(self, part_name, event_i):
        participant = self.part_dict[part_name]
        label = participant.img_list[event_i]
        traj_normX = participant.normX[label][:self.sample_size]
        traj_normY = participant.normY[label][:self.sample_size]
        traj = [traj_normX, traj_normY] # TODO CHECK
        return traj

    def __len__(self):
        # 'Denotes the total number of samples'
        return len(self.ix_list)

    # event_i != label
    def __getitem__(self, index):

        part, event_i = self.ix_list[index]
        joint_img = self.decode_image_from_simset_and_label(part, event_i)
        self.joint_tens = torch.tensor(joint_img).to(_device)
        
        # Picking a randome image from the current's mode (train/val) set 
        marg_part, marg_eventi = choice(self.ix_list)
        marg_img = self.decode_image_from_simset_and_label(marg_part, marg_eventi)
        self.marg_tens = torch.tensor(marg_img).to(_device)
        
        traj = self.get_trajectory(part, event_i)
        self.traj_tens = torch.tensor(traj).to(_device)

        return (self.traj_tens, self.joint_tens, self.marg_tens)

