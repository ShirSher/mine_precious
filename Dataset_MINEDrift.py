import torch
import numpy as np
import pandas as pd
import tensorflow as tf
import os
from scipy.interpolate import CubicSpline
from datetime import datetime, date
from random import choice, shuffle
import logging


BASE_PATH = '/Volumes/Experiment Data/Sasha and Or/'
logging.basicConfig(filename='./samplelog.log', level=logging.INFO)

# list of participants - ignore hidden files/folders (.DS_Store)
part_set = set([_dir for _dir in os.listdir(BASE_PATH) if not _dir.startswith(".")])
exclude_set = set(['DL','OL','SM'])
PART_LIST = list(part_set - exclude_set)

# 125 hz
FREQ = 0.008
START_DELTA = 0.0 
# first 100 stimuli * 3 seconds each
# timedelta starts with 0 st second 300 is the right edge
# add one more time bracket for interpolation before trimming
NSTIMULI = 100
END_DELTA = 300.0 
END_MINUTE = 5

GAZE_COLS = ['norm_pos_x','norm_pos_y']
HEAD_COLS = ['head rot x','head rot y','head rot z','head_dir_x','head_dir_y','head_dir_z','head_right_x','head_right_y','head_right_z','head_up_x','head_up_y','head_up_z']


# keeps one participant's gaze and data in memory at a time
class Dataset_MINEDrift(torch.utils.data.Dataset):
  # 'Characterizes a dataset for PyTorch'
    def __init__(self, participantIDs=PART_LIST,gaze_cols=GAZE_COLS,head_cols=HEAD_COLS,freq=FREQ,path=BASE_PATH):
        self.part_list = shuffle(participantIDs)
        # self.labels = labels
        # self.list_IDs = list_IDs
        
        self.BASE_PATH = path
        
        self.GAZE_COLS = gaze_cols 
        self.HEAD_COLS = head_cols 
        self.FREQ = freq
        
        # returns onset times for the first 101 events (for bookending the 100th event)
        self.events_series,self.offset_dt, self.simset, self.imgsetIDs_arr = None,None,None,None
        # self.imgsetIDs_copy = None
        # self.img0 = -1 
        self.random_shuffle = list(range(0,NSTIMULI))
        shuffle(self.random_shuffle)
        
        self.traj_df = None
        # self.gaze_df = None
        # self.head_df = None
        
        self.part_pointer = 0
        self.sample_pointer = -1 
        
        (self.imgset, self.imgset_labels), (x_test, y_test) = tf.keras.datasets.cifar100.load_data()

    def load_events(part):
        events_path = BASE_PATH+part+'/Events.csv'
        _events_df = pd.read_csv(events_path,on_bad_lines='warn')
        
        simset = int(_events_df.columns[1].split('_')[1])
        
        imgs = np.array([int(_events_df.loc[img][0]) for img in _events_df.index])
        
        _events = pd.Series([event[0] for event in _events_df.index],name='unity_time')
        # unity time
        events = _events.apply(lambda x:x/1E7) 
        
        _onsets = events.copy()
        
        # time from start as datetime
        _onsets -= _onsets[0]
        onsets = pd.to_datetime(_onsets,unit='s')
        # first line contains a timestamp but no "Looking" tag or eye data - ignore?
        #_events_onsets.insert(0,int(_events_df.columns[0]))
        
        # print(str(len(_imgs)),'stimuli,',str(len(_onsets)),'events onsets' ,'- participant',part)
        # load only the first 100
        return events[:NSTIMULI+1], onsets[:NSTIMULI+1], simset, imgs[:NSTIMULI+1]

    def load_binocular(self,part,events,cols=None):
        events_path = BASE_PATH+part+'/Gaze data.csv'
        
        _cols = cols if cols else GAZE_COLS.copy()
        _cols.extend(['EyeID','unity_time'])
        _cols = list(set(_cols))
        
        _gaze_df = pd.read_csv(events_path,on_bad_lines='warn',usecols=_cols)
        
        # if it is heavy on the loading, usecols
        BINgaze_df = _gaze_df[_gaze_df['EyeID']=='Binocular'][_cols]
        BINgaze_df['unity_time'] = BINgaze_df['unity_time'].apply(lambda x: x/1E7)
            
        # time since the Beginning in unity-time seconds
        # unity time associates gaze data in Gaze data.csv with events onsets in Events.csv
        BINgaze_df['timedelta'] = BINgaze_df['unity_time'] - events[0]
        
        # cast as datetime for resampling, interpolation
        # "pre 1970" is negative time, time before first event onset
        BINgaze_df['timedelta_dt'] = pd.to_datetime(BINgaze_df['timedelta'],unit='s')
        # first data point from onset 
        # start_delta = 0.0
        BINgaze_df = BINgaze_df[BINgaze_df['timedelta'] >= START_DELTA ]
        BINgaze_df = BINgaze_df.reset_index(drop=True)
    
        return BINgaze_df

    def load_head(self,part,events,cols=None):
        head_path = BASE_PATH+part+'/position.csv'
        
        _cols = cols if cols else HEAD_COLS.copy()
        _cols.extend(['Time'])
        _cols = list(set(_cols))
            
        head_df = pd.read_csv(head_path,on_bad_lines='warn',usecols=_cols)

        # take intersection of head direction and gaze data?
        head_df['unity_time'] = head_df['Time'].apply(lambda x: x/1E7) 
        head_df['timedelta'] = head_df['unity_time'] - events[0]
        head_df['timedelta_dt'] = pd.to_datetime(head_df['timedelta'],unit='s')
        
        head_df = head_df[head_df['timedelta'] >= START_DELTA ]
        head_df = head_df.reset_index(drop=True)
        return head_df

    def decode_image_from_simset_and_label(self,label):
        return self.imgset[np.where(self.imgset_labels==label)[0][self.simset-1]]

    def fillfirstna(self,resampled_df,orig_df):
    # gaze_sample.loc[ar_gaze_sample['timedelta']>0].index[0] should be equivalent to ar_gaze_sample.iloc[0] if reset index to start_delta ie timedelta >=0
        if resampled_df.iloc[0].isna().all() and orig_df.iloc[0]['timedelta'] >= START_DELTA:
                resampled_df.iloc[0] = orig_df.iloc[0][resampled_df.columns]
        # TODO elif second clause make more flex for closest in time
        return resampled_df


    def resample_and_interpolate(self,df,interp_cols,ffill=False,linear=True,hz=FREQ):
        
        # 100 sample session ends at 6 minutes
        end = datetime(year=1970,month=1,day=1,minute=END_MINUTE,second=0)
        interp_cols = interp_cols.copy()
        # mod ffill method to only return resampled time points 
        if ffill:
            resampled_df = df.set_index('timedelta_dt').resample('{hz}S'.format(hz=hz)).ffill()
            df = df.set_index('timedelta_dt')
            # if no data at the 0th time mark, will be na because ffill 
            # fill onset gaze loc with "original" (closest) value 
            resampled_df = self.fillfirstna(resampled_df,df)
            # resampledInterspersed_df = resampledCat_df.sort_values(by='timedelta_dt')
            # resampled_df = resampled_df[(resampled_df.index.minute <= END_MINUTE)]
            resampled_df = resampled_df[(resampled_df.index <= end)][interp_cols]
            
        elif linear:
            # returns empty dataframe indexed at frequency
            NaNresampled_df = df.set_index('timedelta_dt').resample('{hz}S'.format(hz=hz)).interpolate()
            df = df.set_index('timedelta_dt')
            # should be zero overlap between timestamps since original report is to nanosecond and resampling is by the millisecond
            # concats back-to-back
            resampledCat_df = pd.concat([NaNresampled_df,df])
            # order by time - nan rows interspersed with reported values
            resampledInterspersed_df = resampledCat_df.sort_values(by='timedelta_dt')
            # (Linear) interpolation between resampled points
            linInterp_df = resampledInterspersed_df[interp_cols].interpolate(method='linear')
            # linInterp_df = resampledInterspersed_df.interpolate(method='linear')
            
            # take only the resampled time rows
            resampled_df = linInterp_df.loc[NaNresampled_df.index]
            
            resampled_df = self.fillfirstna(resampled_df,df)
            
            # resampled_df = resampled_df[resampled_df.index.minute < END_MINUTE]
            # keep right fencepost
            resampled_df = resampled_df[(resampled_df.index <= end)]
            
        # cubic spline interpolation
        else:
            # returns empty dataframe indexed at frequency
            NaNresampled_df = df.set_index('timedelta_dt').resample('{hz}S'.format(hz=hz)).interpolate()
            df = df.set_index('timedelta_dt')
            # should be zero overlap between timestamps since original report is to nanosecond and resampling is by the millisecond
            # concats back-to-back
            resampledCat_df = pd.concat([NaNresampled_df,df])
            # order by time - nan rows interspersed with reported values
            resampledInterspersed_df = resampledCat_df.sort_values(by='timedelta_dt')
            # (Cubic spline) interpolation between resampled points
            CSresampled_df = resampledInterspersed_df[interp_cols].interpolate(method='cubicspline')
            
            # take only the resampled time rows
            resampled_df = CSresampled_df.loc[NaNresampled_df.index]
            
            resampled_df = self.fillfirstna(resampled_df,df)
            
            # resampled_df = resampled_df[resampled_df.index.minute < END_MINUTE]
            # keep right fencepost
            resampled_df = resampled_df[(resampled_df.index <= end)]
            
        return resampled_df 
        
    def __len__(self):
        # 'Denotes the total number of samples'
        # return len(self.list_IDs)
        # or len self.part_list?
        return NSTIMULI * len(self.part_list)

    def __getitem__(self, index):
        # 'Generates one sample of data'
        # Select sample
        # cut sample size at 300 points for uniformity
        sample_size = 300
        
        if self.sample_pointer >= 99:
            self.part_pointer += 1
            self.sample_pointer = -1
        
        if self.sample_pointer < 0:
            part = self.part_list[self.part_pointer]
            # logging.info(f"================= \n Participant ID {part} \n =================")
            
            # convert events series in unity_time to datetime
            self.events_series, self.offset_dt, self.simset, self.imgsetIDs_arr = self.load_events(part)
            # self.imgsetIDs_copy = list(self.imgsetIDs_arr.copy())
            
            #Load df, linearly interpolate at resampled time periods,
            gaze_df = self.resample_and_interpolate(self.load_binocular(part=part,events=self.events_series),linear=True,interp_cols=self.GAZE_COLS.copy(),hz=self.FREQ)
            head_df = self.resample_and_interpolate(self.load_head(part=part,events=self.events_series),linear=True,interp_cols=self.HEAD_COLS.copy(),hz=self.FREQ)
            # join on shared, resampled timedelta_dt index
            self.traj_df = gaze_df.join(head_df)
            
            # save the first image for the last mutual set 
            self.img0 = self.imgsetIDs_arr[self.random_shuffle[0]]
            
        self.sample_pointer += 1
        # iterate over img Ids 
        jointImg_id = self.imgsetIDs_arr[self.random_shuffle[self.sample_pointer]]
        joint_img = self.decode_image_from_simset_and_label(label=jointImg_id)
        joint_tens = torch.tensor(joint_img)
        # select an alternative image ID from the participant's simset
        # self.imgsetIDs_copy.remove(jointImgs_id)
        # np.delete is not in-place
        # self.imgsetIDs_copy = np.delete(self.imgsetIDs_arr,jointImgs_id)
        
        # if it is the last img (and )
#         if self.sample_pointer < 99 else self.img0
        mutualImg_id = choice(np.delete(self.imgsetIDs_arr,jointImg_id)) 
        mutual_img = self.decode_image_from_simset_and_label(label=mutualImg_id)
        mutual_tens = torch.tensor(mutual_img)
        
        start_event = self.offset_dt[self.random_shuffle[self.sample_pointer]] + pd.Timedelta(seconds=0.5)
        # dataset is prefenceposted 
        end_event = self.offset_dt[self.random_shuffle[self.sample_pointer] + 1]    
        
        traj = self.traj_df[(self.traj_df.index >= start_event) & (self.traj_df.index < end_event)][:sample_size]
        traj_tens = torch.tensor(traj.values)
        print(jointImg_id,mutualImg_id)
        logging.info(f"(Joint-Image ID, Mutual-Image ID) :{jointImg_id},{mutualImg_id}")
        _start_str = date.strftime(start_event,"%M:%S.%f")
        _end_str = date.strftime(end_event,"%M:%S.%f")
        logging.info(f"Offset from start duration [{_start_str},{_end_str})")
        # return tuple of (trajectory, joint img, random img)
        # batch is a tuple of (joint1, joint2, marginal (from the dataset of joint 2)) - MINE.py
        return (traj_tens,joint_tens,mutual_tens)