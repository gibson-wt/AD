import numpy as np
import roughpy as rp
import pandas as pd
from pathlib import Path
import pickle
from functools import partial
import random
import argparse

from Utils import get_files, find_start_time, to_array,restrict_sig_array,find_start_time_dp, stream_from_file

# Set paths for where Data is stored
local_data_folder = Path("C:\\Users\\gibso\\Downloads\\DALE_data")
file_collection = "AllData"


# Import power data of individual appliances
def import_data(channel):
    channel_data_path = str(Path(local_data_folder, "house_1/channel_{}.DAT".format(channel)))
    df = pd.read_csv(channel_data_path, delimiter=' ', header=None)
    df.columns=["unix_time","power"]
    return df

# Import threshold data for when an appliance is deemed on or off
def find_standby_data():
    data_path = str(Path(local_data_folder, "standby_values.csv"))
    return pd.read_csv(data_path)


def calc_var(data_list):
    power_data=[data.P.item() for data in data_list]
    mean_power = 1/5*sum(power_data)
    max_min = max(power_data)-min(power_data)
    return mean_power,max_min

def restrict(df_1,df_2):
    df = pd.concat([df_1,df_2])
    mask = []
    for i, row in df_1.iterrows():
        start_time = row.start_time
        end_time = row.end_time
        test = df[(df.end_time > start_time) & (df.start_time < end_time)]
        if test.shape[0]==1:
            mask.append(True)
        else:
            mask.append(False)
    return df_1.iloc[mask]

def change_bounds(row):
    row.start_time -=7
    return row

def check_anom(anoms, candidate): 
    overlap= False
    for i,row in anoms.iterrows():
        if (candidate[0].unix_time.item() < row.end_time) & (candidate[4].unix_time.item() > row.start_time):
            overlap=True
    return overlap

def update_normal_intervals(normal_intervals,count,intervals,candidate,i,file):
    power_seq=[candidate[i].P.item() for i in range(5)]
    if max(power_seq)-min(power_seq) < 0.05*(power_seq[0]):
        normal_intervals.append({'file': file,'start_time': candidate[0].unix_time.item(), 'end_time': candidate[4].unix_time.item(), 'power': power_seq})
        count+=1
        del intervals[i]
    return normal_intervals,count,intervals

def check_varying(vi,candidate):
    overlap= False
    for time in vi:
        if (candidate[0].unix_time.item() < time) & (candidate[4].unix_time.item() > time-20):
            return True
    return overlap

def find_power_seq(df_power,row):
    ub = row.end_time +1.5 #+4 to due to sliding window, +0.5 due to rounding of individual appliance data
    lb = row.start_time -10.5
    return list(df_power[(df_power.unix_time> lb)&(df_power.unix_time< ub)].P)



# Find Intervals on which on-off events occur
class IntervalFinder:
    def __init__(self,file_collection):
        # Import 50kHz data -  labelled by start time
        self.files = get_files(file_collection)
        self.start_times = [find_start_time(file) for file in self.files]

        data_path = str(Path(local_data_folder, "house_1/mains.DAT"))
        self.start_time=self.start_times[0]
        self.end_time=self.start_times[-1]+3600

        # Import whole house 1s power data restricted to times where we have 50kHz data as well
        df=pd.read_csv(data_path,delimiter=' ',nrows=self.end_time-1363547563,header=None) 
        df.columns=['unix_time','P','S','V']
        self.df=df[df.unix_time>self.start_time]

        # Import on-off thresholds for each appliance
        self.standby_data = find_standby_data()


    def find_on_off(self,df_appliance,thres):
        '''
        Find intervals in which on-off events occur.
        df_appliance - Dataframe of individual appliance power consumption
        thres -  
        '''
        data = df_appliance[(df_appliance.unix_time>=self.start_time) & (df_appliance.unix_time <= self.end_time)]
   
        on_times = data[data.power > thres].unix_time
        before = data[data.shift(-1).power > thres]
        after = data[data.shift(1).power > thres]
            

        if (on_times.size != 0):
            diffs = (on_times.index.diff() != 1)
            if before.shape[0] != on_times.shape[0]: # on at start of time, which we don't include as an event
                diffs = diffs[1:]
                on_times = on_times.iloc[1:]
                after=after.iloc[1:]
            if after.shape[0] != on_times.shape[0]: # on at the end, which we don't include as an event
                diffs=diffs[:-1]
                on_times = on_times.iloc[:-1]
                before=before.iloc[:-1]

            ##### exclude if at edge of missing data ######
            # if on_times.unix_time.diff() > 30??? - is there a correct choice for this

 
            starts = on_times[diffs].reset_index()
            off_power = before[diffs].power.values
            power_change_s = pd.Series(after[diffs].power.values - off_power)  # 2 apart -as 1 might still be loading 
            starts = pd.concat([before[diffs].unix_time.reset_index(drop=True),starts.unix_time,power_change_s],axis=1)
            starts.columns=['start_time','end_time','power_change']
            starts = starts[starts.power_change > 10] # ensures no singular peaks - find that these can be faults in measuring 

            diff_ = list(diffs[1:])
            diff_.append(True)
            ends = on_times[diff_].reset_index()
            off_power = after[diff_].power.values
            power_change_e = pd.Series(before[diff_].power.values - off_power)
            ends = pd.concat([before[diff_].unix_time.reset_index(drop=True),after[diff_].unix_time.reset_index(drop=True),power_change_e],axis=1)
            ends.columns=['start_time','end_time','power_change']
            ends = ends[ends.power_change > 10]
        
            return starts,ends
        else:
            return None
        
    def find_intervals(self,events,event_type='start'):  
        intervals=[]
        for _,row in events.iterrows():
            index= self.df[(self.df.unix_time <= row.end_time +0.5)].index.values[-1] # iloc[-1] ensures final occurence
            power=self.df[self.df.index == index].P.item()
            if row.power_change > 0.05* power: # event not in noise floor

                change_started=False
    
                count=0
                while count<20:
                    count+=1
                    index-=1
                    temp=self.df[self.df.index==index].P.item()
                    if event_type=='start':
                        change= power-temp
                    else:
                        change=temp-power

                    if (change > 0.3* row.power_change) & (change < 1.3* row.power_change):
                        if ~change_started:
                            change_started=True
            
        
                    if change_started:  
                        if change < 10:
                            start_time = self.df[self.df.index==index-1].unix_time.item() #0.05 due to rounding 
                            end_time = self.df[self.df.index==index+4].unix_time.item()
                            intervals.append({'start_time': start_time, 'end_time': end_time})
                            count=20 # exits
                    power = temp
                    
                    # else:
                    #     if (temp-power > 0.3* row.power_change) & (temp-power < 1.3* row.power_change):
                    #         increase_started=True

                    #     if increase_started:
                    #         if temp-power < 10:
                    #             start_time = self.df[self.df.index==index-1].unix_time.item()-0.5 #0.05 due to rounding 
                    #             end_time = self.df[self.df.index==index+2].unix_time.item()+0.5
                    #             intervals.append({'start_time': start_time, 'end_time': end_time})
                    #             count=20 # exits

                  

        return pd.DataFrame(intervals)
    
    def find_intervals_given_channel(self,channel):
        df_ = import_data(channel)
        thres = self.standby_data[self.standby_data.channel==channel].standby_power.values[0]
        events = self.find_on_off(df_,thres)
        if events is None:
            return None
        else:
            start_intervals=self.find_intervals(events[0],event_type='start')
            end_intervals=self.find_intervals(events[1],event_type='end')
            start_intervals['channel'] = channel
            end_intervals['channel'] = channel
            return start_intervals.apply(change_bounds,axis=1),end_intervals
        
    def find_events_given_channel(self,channel):
        df_ = import_data(channel)
        thres = self.standby_data[self.standby_data.channel==channel].standby_power.values[0]
        events = self.find_on_off(df_,thres)

        if events is None:
            return None
        else:
            start_intervals=events[0]
            start_intervals['event_type']='on'
            end_intervals=events[1]
            end_intervals['event_type']='off'
            start_intervals['channel'] = channel
            end_intervals['channel'] = channel
            return start_intervals,end_intervals
        
    def find_events(self,channels):
        start_events=[]
        end_events=[]
        for channel in self.standby_data.channel.unique():
            events = self.find_events_given_channel(channel)
            if events is not None:
                start_events.append(events[0])
                end_events.append(events[1])
        start_events=pd.concat(start_events)
        end_events=pd.concat(end_events)
        
        self.events = pd.concat([start_events,end_events]) 
        start_events = start_events[start_events.channel.isin(channels)]
        end_events = end_events[end_events.channel.isin(channels)]
        temp = restrict(start_events,end_events)
        self.end_events = restrict(end_events,start_events)
        self.start_events=temp

    def find_anom_ps(self):
    
        fps=partial(find_power_seq,self.df)
        self.start_events['power'] = self.start_events.apply(fps,axis=1)
        self.end_events['power'] = self.end_events.apply(fps,axis=1)

    def find_outliers(self,channels):
        start_intervals=[]
        end_intervals=[]
        for channel in channels:
            events = self.find_intervals_given_channel(channel)
            if events is not None:
                start_intervals.append(events[0])
                end_intervals.append(events[1])

        start_events = pd.concat(start_intervals)
        end_events = pd.concat(end_intervals)   
        self.start_events = restrict(start_events,end_events)
        self.end_events = restrict(end_events,start_events)

        return self.start_events,self.end_events

    def find_normality(self):
        df = self.df[(self.df.unix_time>=self.start_time) & (self.df.unix_time <= self.end_time)].reset_index()
        # Load where anomalies occur in 6s power
        # vi=self.find_vi()
        events=self.events

        # Decrease lower bound due to misalignment
        events_ = events.apply(change_bounds,axis=1)

        normal_intervals=[]

        for file in self.files:
            start_time = find_start_time_dp(file)

            data = df[(df.unix_time >= start_time) & (df.unix_time <= start_time+3580)].reset_index()
            intervals= {i: [data.iloc[j] for j in range(i,i+5,1)] for i in range(0,data.shape[0]-5,5)}

            searching=True

            count_s=0
            count_m=0
            count_l=0
            count=0
            while searching:
                i = random.choice(list(intervals.keys()))
                candidate=intervals[i]

                # if check_varying(vi_,candidate):
                #     pass
                if check_anom(events_,candidate):
                    pass
                else:
                    if (candidate[0].P < 600) & (count_s < 5):
                        normal_intervals,count_s,intervals = update_normal_intervals(normal_intervals,count_s,intervals,candidate,i,file)
                    elif (candidate[0].P > 1500) & (count_l < 5):
                        normal_intervals,count_l,intervals = update_normal_intervals(normal_intervals,count_l,intervals,candidate,i,file)
                    elif count_m < 5:
                        normal_intervals,count_m,intervals = update_normal_intervals(normal_intervals,count_m,intervals,candidate,i,file)

                    if  (count_s == 5) & (count_m == 5) & (count_l == 5):
                        searching=False

                if count > 100:
                    searching=False

                count+=1

        return pd.DataFrame(normal_intervals)
    
    def update_end_time(self,end_time):
        self.end_time=end_time

def offsetor(offset,row):
    return rp.RealInterval(row.start_time-offset,row.end_time-offset)


def find_signature(stream,frequency,depth,offset,row):
    '''
    '''
    #  split interval into cycles, take floor and ceil so intervals not cut off.
    cycles = [rp.RealInterval(i/frequency,(i+1)/frequency) for i in range(int(np.floor(frequency*(row.start_time-offset))),int(np.ceil(frequency*(row.end_time-offset))),1)]
    return [stream.signature(cycle,depth=depth) for cycle in cycles]

def find_overlapped_signature(stream,frequency,depth,offset,row):
    cycles = [rp.RealInterval(i/frequency,(i+2)/frequency) for i in range(int(np.floor(frequency*(row.start_time-offset))),int(np.ceil(frequency*(row.end_time-offset)))-1,1)]
    return [stream.signature(cycle,depth=depth) for cycle in cycles]

def signatures(intvs,files,channel_given=True,overlapping=False):
    '''
    '''
    tester=[]
    for file in files:
        offset = find_start_time_dp(file)
        stream = stream_from_file(file,args.resolution)

        # Restrict to events in an individual 1 hour long 50kHz file.
        events = intvs[(intvs.start_time > offset) & (intvs.end_time < offset + stream.support.sup())]
        if events.empty:
            pass
        else:
            ofstr = partial(offsetor,offset)
            event_interval = events.apply(ofstr,axis=1)
            if overlapping:
                find_sigs = partial(find_overlapped_signature,stream,args.frequency,args.depth,offset)
            else:
                find_sigs = partial(find_signature,stream,args.frequency,args.depth,offset)
            signatures = events.apply(find_sigs,axis=1)
            if channel_given:
                channel=events.channel
            else:
                channel= None
    
            tester.append(pd.DataFrame({'file': file,'channel': channel, 'event_interval': event_interval, 'signatures': signatures}))
    return pd.concat(tester)

def main():
    # Initialise Interval Finder
    intfind = IntervalFinder(file_collection)
    # intfind.find_events(args.channels)

    # intfind.find_anom_ps()

    # with open(Path(local_data_folder, 'On_Event_Intervals_{freq}_hp_test.pkl'.format(freq=args.frequency)),'wb') as f:
    #     pickle.dump(intfind.start_events,f)

    # with open(Path(local_data_folder, 'Off_Event_Intervals_{freq}_hp.pkl'.format(freq=args.frequency)),'wb') as f:
    #     pickle.dump(intfind.end_events,f)

    # if args.find_normal_corpus:
    # normal_intvs = intfind.find_normality()
    # with open(Path(local_data_folder, 'Normal_Event_Intervals_{freq}.pkl'.format(freq=args.frequency)),'wb') as f:
    #     pickle.dump(normal_intvs,f)
    

    # Find Intervals where appliances turn on or off according to 1s power data
    outliers = intfind.find_outliers(args.channels)
    print(outliers[0].shape)

    # Construct Dataframe of Signatures, channel, event interval when appliance is turning on.
    anoms_split=signatures(outliers[0],intfind.files)
    anoms_overlapped=signatures(outliers[0],intfind.files,overlapping=True)

    # Pickle on-off event signature data from to save for later use.
    with open(Path(local_data_folder, 'On_Off_Event_Sigs_{freq}_split.pkl'.format(freq=args.frequency)),'wb') as f:
        pickle.dump(anoms_split,f)

    with open(Path(local_data_folder, 'On_Off_Event_Sigs_{freq}_overlapped.pkl'.format(freq=args.frequency)),'wb') as f:
        pickle.dump(anoms_overlapped,f)

    # Find normal corpus sigantures in the same manner
    if args.find_normal_corpus == 'True':
        
        normal_intervals = intfind.find_normality()
        normal_sigs = signatures(normal_intervals,intfind.files,channel_given=False)

        # Pickle normal corpus sigantures to use for later
        with open(Path(local_data_folder, 'Normal_Event_Sigs_{freq}.pkl'.format(freq=args.frequency)),'wb') as f:
            pickle.dump(normal_sigs,f)



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--channels', type=list, help='channels of appliances we want to restrict looking at', default = [5,6,10,11,13])
    parser.add_argument('--depth', type=int, help='Depth of signatures to be calculated', default=5)
    parser.add_argument('--resolution', type=int, help='Roughpy resolution', default=14)
    parser.add_argument('--frequency', type=int, help='frequency of interval', default=50)
    parser.add_argument('--find_normal_corpus', type=str, help='find signatures of normal corpus', default='False')
    args = parser.parse_args()

    main()
