import numpy as np
import roughpy as rp
import pandas as pd
from pathlib import Path
import pickle
from functools import partial
import argparse
import random

from Utils import to_array,restrict_sig_array, determine_thres
from SigMahaKNN import SignatureMahalanobisKNN

# Set paths for where Data is stored
local_data_folder = Path("C:\\Users\\gibso\\Downloads\\DALE_data")

def construct_normal_corpus(df):
    corpus=[]
    for index in range(df.shape[0]):
        mask = random.sample(range(0,len(df.iloc[index].signatures)),2)
        corpus.append(np.vstack([np.array(df.iloc[index].signatures[i]) for i in mask])) 
    return np.vstack(corpus)


def find_anomalous_intervals(depth):
    with open(Path(local_data_folder,'normal_corpus_small.pkl'),'rb') as f:
        df_norm = pickle.load(f)

    with open(Path(local_data_folder,'hp_overlapped.pkl'),'rb') as f:
        df_anom = pickle.load(f)

    corpus = restrict_sig_array(construct_normal_corpus(df_norm),depth)


    

    sigmaha = SignatureMahalanobisKNN()
    indices=np.arange(corpus.shape[0])
    np.random.shuffle(indices)
    C1 = corpus[indices[:corpus.shape[0]//2]]
    C2 = corpus[indices[corpus.shape[0]//2:]]
    thres = determine_thres(sigmaha,C1,C2,0.25)
    
    sigmaha.fit(corpus)

    anoms=[]
    for i,row in df_anom.iterrows():
        sig_array = restrict_sig_array(to_array(row.signatures),depth)
        dists=sigmaha.conformance(sig_array)
        # minmax = MinMaxAugmentation(stream_from_file(row.file),resolution=12)
        index=np.argmax(dists>thres)
        anoms.append(index)

    # elif method == 'LSTM':
    #     lstm = LSTMAD
    df_anom['anom_index'] = anoms
    return df_anom