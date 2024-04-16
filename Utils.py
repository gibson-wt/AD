import numpy as np
import roughpy as rp
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
import joblib


data_folder = Path("C:\\scratch\\dale_data")

find_start_time = lambda file_name: int(str(file_name).split('/')[-1].split('-')[1].split('_')[0])

class Scaler:
    def __init__(self,thres=1e-10):
        self.thres=thres
    
    def fit(self,X):
        self.mu = X.mean(0)
        X=X-self.mu
        U, S, Vt = np.linalg.svd(X)
        k = np.sum(S > self.thres)  # detected numerical rank
        self.numerical_rank = k
        self.Vt = Vt[:k]
        self.S = S[:k]
        
    def transform(self,X):
        x = X - self.mu
        return x @ self.Vt.T  / self.S


def get_files(file_):
    # Directory path
    directory = Path(data_folder, file_)

    # Loop through files in the directory
    files=[]
    for file in directory.iterdir():
        if file.is_file():
            files.append(file)
    return sorted(files)


# turn list of signatures into an array with signatures stacked ontop of each other
def to_array(row):
    return np.vstack([np.array(sig) for sig in row])

def restrict_sig_array(sigs_array,depth,width=2):
    return sigs_array[:,:(width ** (depth+1) -1)]

def find_start_time_dp(file_name): # decimal precision
    time = str(file_name).split('/')[-1].split('-')[1].split('_')
    int_part = int(time[0])
    decimal_part = int(time[1].split('.')[0])* 10 ** (-1 * (len(time[1])-5)) # 5 corespounding to '.''f''l''a''c'
    return int_part+decimal_part

def stream_from_file(file,resolution,width=2,depth=6):
    stream = rp.ExternalDataStream.from_uri(
            str(file),
            channel_types=[rp.ValueChannel, rp.IncrementChannel],
            width=2, depth=6, dtype=rp.DPReal, resolution=resolution)
    return stream

def restrict_interval_for_power(before,after,row):
    return row.signatures[max(0,row.index+before):min(len(row.signatures), row.index+after)]

def create_power_sequence(sig_list):
    power_seq = []
    for sig in sig_list:
        power_seq.append(np.array(sig)[4])
    return np.vstack(power_seq)

def augment(path,augmentations):
    for augmentation in augmentations:
        path = augmentation.apply(path)
    return path


def compute_signature(depth,augmentations,path):
    stream = rp.LieIncrementStream.from_increments(augment(path,augmentations), depth=6)
    sigs = np.array(stream.signature(stream.support,depth=depth))
    return sigs

def determine_thres(dist_fn,C1,C2,epsilon):
    dist_fn.fit(C1)
    thres = np.percentile(dist_fn.conformance(C2),100-epsilon)
    return thres
