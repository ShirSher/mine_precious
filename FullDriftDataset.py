import torch
import numpy as np
import pandas as pd
import torchvision
from datetime import datetime
from random import shuffle, choice
import utils

_base_path = '../VRDrift/'
'''
    125 hz
    first 100 stimuli * 3 seconds each
    timedelta starts with 0 st second 300 is the right edge
    add one more time bracket for interpolation before trimming
'''
FREQ = 0.008
START_DELTA = 0.0
END_DELTA = 300
_nStimuli = 100
END_MINUTE = 5
_gaze_cols = ['norm_pos_x',
             'norm_pos_y']
_head_cols = ['head rot x',
             'head rot y',
             'head rot z',
             'head_dir_x',
             'head_dir_y',
             'head_dir_z',
             'head_right_x',
             'head_right_y',
             'head_right_z',
             'head_up_x',
             'head_up_y',
             'head_up_z']

_device = 'cuda:3' if torch.cuda.is_available() else 'cpu'

# given a time-segmented sample, removes the mean from each time series column
def zero_center(df, return_cols=None):
    def zero_center_series(series):
         return series - np.mean(series)

    if type(df) is pd.Series: return zero_center_series(df)
    return_cols = return_cols if return_cols is not None else df.columns
    norm_df = pd.DataFrame().reindex_like(df[return_cols])
    for col in norm_df.columns:
        norm_df[col] = zero_center_series(df[col])
    return norm_df


def fillfirstna(resampled_df, orig_df):
    if resampled_df.iloc[0].isna().all() and orig_df.iloc[0]['timedelta'] >= START_DELTA:
            resampled_df.iloc[0] = orig_df.iloc[0][resampled_df.columns]
    return resampled_df


# linear interpolation
def resample_and_interpolate(df, interp_cols, hz=FREQ):
    # 100 sample session ends at 5 minutes (3seconds * 100)/60
    # mod minute = END_MINUTE + 10 seconds for 100th stimulus which extends into the 6th minute (ie 101 stimulus at 5:02)
    end = datetime(year=1970,month=1,day=1,minute=END_MINUTE, second= 10)
    interp_cols = interp_cols.copy()
    # returns empty dataframe indexed at frequency
    NaNresampled_df = df.set_index('timedelta_dt').resample('{hz}S'.format(hz=hz)).interpolate()
    # undo before returning
    df = df.set_index('timedelta_dt')
    # should be zero overlap between timestamps since original report is to nanosecond and resampling is by the millisecond
    # concats back-to-back
    resampledCat_df = pd.concat([NaNresampled_df,df])
    # order by time - nan rows interspersed with reported values
    resampledInterspersed_df = resampledCat_df.sort_values(by='timedelta_dt')
    # (Linear) interpolation between resampled points
    linInterp_df = resampledInterspersed_df[interp_cols].interpolate(method='linear')
    # take only the resampled time rows
    resampled_df = linInterp_df.loc[NaNresampled_df.index]
    resampled_df = fillfirstna(resampled_df,df)
    # keep right fencepost
    resampled_df = resampled_df[(resampled_df.index <= end)]

    # return timedelta_dt to its rightful columnar place
    resampled_df = resampled_df.reset_index()

    # resampled_df['timedelta_dt'] = [t for t in resampled_df.index]

    return resampled_df


# why 101  ? aldo why not _nStimuli+1?
def load_events(part, nStimuli=101):
    events_path = _base_path  +part + '/Events.csv'
    # _events_df = pd.read_csv(events_path,on_bad_lines='warn')
    _events_df = pd.read_csv(events_path)
    simset = int(_events_df.columns[1].split('_')[1])
    imgs = np.array([int(_events_df.loc[img][0]) for img in _events_df.index])
    _events = pd.Series([event[0] for event in _events_df.index],name='unity_time')
    # unity time
    events = _events.apply(lambda x:x/1E7)
    _onsets = events.copy()
    # time from start as datetime
    _onsets -= _onsets[0]
    onsets = pd.to_datetime(_onsets,unit='s')
    # load only the first 100
    events_df = pd.DataFrame({'unity_time':events, 'Offset_dt':onsets,'ImgID':imgs,'Simset':[simset for i in range(len(events))], 'ParticipantID':[part for i in range(len(events))]})
    return events_df.iloc[:nStimuli]
    # return events[:101], onsets[:101], simset, imgs[:101]


def load_binocular(part, events, cols=None):
    events_path = _base_path + part + '/Gaze data.csv'
    _cols = cols if cols else _gaze_cols.copy()
    _cols.extend(['EyeID','unity_time'])
    _cols = list(set(_cols))
    # _gaze_df = pd.read_csv(events_path,on_bad_lines='warn',usecols=_cols)
    _gaze_df = pd.read_csv(events_path, usecols=_cols)
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

    BINgaze_df['ParticipantID'] = part

    BINgaze_df = BINgaze_df.reset_index(drop=True)
    return BINgaze_df


def load_head(part,events,cols=None):
    head_path = _base_path + part + '/position.csv'
    _cols = cols if cols else _head_cols.copy()
    _cols.extend(['Time'])
    _cols = list(set(_cols))
    # head_df = pd.read_csv(head_path,on_bad_lines='warn',usecols=_cols)
    head_df = pd.read_csv(head_path,usecols=_cols)
    # take intersection of head direction and gaze data?
    head_df['unity_time'] = head_df['Time'].apply(lambda x: x/1E7)
    head_df['timedelta'] = head_df['unity_time'] - events[0]
    head_df['timedelta_dt'] = pd.to_datetime(head_df['timedelta'],unit='s')
    head_df = head_df[head_df['timedelta'] >= START_DELTA ]

    head_df['ParticipantID'] = part
    head_df = head_df.reset_index(drop=True)
    return head_df

class FullDataset(torch.utils.data.Dataset):

    def __init__(self, ix_dict, gaze_cols=_gaze_cols.copy(), head_cols=_head_cols.copy(), freq=FREQ, path=_base_path, sample_size=END_DELTA ):

        self.part_list = list(ix_dict)
        shuffle(self.part_list)
        self.BASE_PATH = path
        self.GAZE_COLS = gaze_cols
        self.HEAD_COLS = head_cols
        self.freq = freq
        # clip to 300 datapoints for consistent trajectory sizes for batch sizing
        self.SAMPLE_SIZE = sample_size
        # returns onset times for the first 101 events (for bookending the 100th event)
        # self.events_df = None
        self.events_dict = None
        self.traj_df = None

        # self.sample_pointer = -1

        self.ix_list = [(participant, ix) for participant, ix_list in ix_dict.items() for ix in ix_list]
        shuffle(self.ix_list)

        self.imgset, self.imgset_labels = [], []

        self.build_datasets()


    def build_datasets(self):
        # (self.imgset, self.imgset_labels), (x_test, y_test) = tf.keras.datasets.cifar100.load_data()

        cifar_data = torchvision.datasets.CIFAR100('./cifar100data/',train=True,download=True)
        for i in range(len(cifar_data)):
            self.imgset.append(np.array(cifar_data[i][0]))
            self.imgset_labels.append(cifar_data[i][1])

        self.imgset = np.array(self.imgset)
        self.imgset_labels = np.expand_dims(np.array(self.imgset_labels),axis=1)
        # build events_df and trajectory_df ; imgset
        # fullgaze_df, fullhead_df, fullevents_df = [], [], []
        # fullgaze_dict, fullhead_dict, fullevents_dict = {}, {}, {}
        fullgaze_df, fullhead_df, fullevents_dict = [], [], {}

        for part in self.part_list:
            # events_series, offset_dt, simset, imgsetIDs_arr = load_events(part)
            events_df = load_events(part)
            gaze_df = resample_and_interpolate(load_binocular(part=part,events= events_df['unity_time']),interp_cols=self.GAZE_COLS.copy(),hz=self.freq)
            gaze_df['ParticipantID'] = part
            # TODO figure out how/if to incorporate if we want to maintain info re
            # gaze_df['ParticipantID'] = part
            head_df = resample_and_interpolate(load_head(part=part,events= events_df['unity_time']),interp_cols=self.HEAD_COLS.copy(),hz=self.freq)
            head_df['ParticipantID'] = part
            # head_df['ParticipantID'] = part

            # fullgaze_dict[part] = gaze_df
            # fullhead_dict[part] = head_df
            fullevents_dict[part] = events_df

            fullgaze_df.append(gaze_df)
            fullhead_df.append(head_df)
            # fullevents_df.append(events_df)

        # ignore_index = False
        fullgaze_df = pd.concat(fullgaze_df)
        # multi-index
        fullgaze_df = fullgaze_df.set_index(['ParticipantID','timedelta_dt'])

        fullhead_df = pd.concat(fullhead_df)
        fullhead_df = fullhead_df.set_index(['ParticipantID','timedelta_dt'])
        # reindex 0-n
        # self.events_df = pd.concat(fullevents_df,ignore_index=True)
        self.events_dict = fullevents_dict
        self.traj_df = fullgaze_df.join(fullhead_df,on=['ParticipantID','timedelta_dt'])


    def decode_image_from_simset_and_label(self,simset,label):
        return self.imgset[np.where(self.imgset_labels==label)[0][simset-1]]


    def __len__(self):
        # 'Denotes the total number of samples'
        return len(self.ix_list)


    def __getitem__(self, index):
        part, event_i = self.ix_list[index]
        _simset = self.events_dict[part].iloc[event_i]['Simset']
        joint_img = self.decode_image_from_simset_and_label(simset=_simset,label=self.events_dict[part].iloc[event_i]['ImgID'])
        self.joint_tens = torch.tensor(joint_img).to(_device)

        # np.delete is not in-place
        marg_event = choice(np.delete(np.array(range(0,_nStimuli)),event_i))
        marg_img = self.decode_image_from_simset_and_label(simset=_simset,label=self.events_dict[part].iloc[marg_event]['ImgID'])
        self.marg_tens = torch.tensor(marg_img).to(_device)


        start_event = self.events_dict[part].iloc[event_i]['Offset_dt'] + pd.Timedelta(seconds=0.5)
        # dataset is prefenceposted
        end_event = self.events_dict[part].iloc[(event_i + 1)]['Offset_dt']
        # multi-index: participant.index => timedelta:
        traj = self.traj_df.loc[part][(self.traj_df.loc[part].index >= start_event) & (self.traj_df.loc[part].index < end_event)][:self.SAMPLE_SIZE]
        self.traj_tens = torch.tensor(traj.values).to(_device)

        return (self.traj_tens, self.joint_tens, self.marg_tens)

