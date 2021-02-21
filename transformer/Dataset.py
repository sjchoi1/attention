from __future__ import print_function, division
import os
import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader
import pandas as pd
from tqdm import tqdm
import swifter

tid_to_idx = {}
pc_to_idx = {}
n_pc = 0

class CustomDataset(Dataset):
    """Custom dataset."""
    def __init__(self, csv_file, look_back, look_front, thread_cnt):
        """
        Args:
            csv_file (string): Path to the csv file
            look_back (int): How much page fault history
            look_front (int): How much page to predict
        """      
        def hex2vec(hex):
            multi_hot_vector = torch.zeros(len(hex), 16)
            for i, hc in enumerate(hex):
                multi_hot_vector[i, int(hc, 16)] = 1
            return multi_hot_vector.view(1, -1)

        # def hex2vec(hex):
        #     multi_hot_vector = torch.zeros(16, len(hex))
        #     for i, hc in enumerate(hex):
        #         multi_hot_vector[int(hc, 16), i] = 1
        #     return multi_hot_vector.view(1, -1)

        def preprocess_tid(tid):
            global tid_to_idx 
            one_hot_vector = torch.zeros(1, 16)
            one_hot_vector[0, tid_to_idx[tid]] = 1
            return one_hot_vector

        def preprocess_pc(pc):
            # global pc_to_idx
            # global n_pc
            # one_hot_vector = torch.zeros(1, n_pc)
            # one_hot_vector[0, pc_to_idx[pc]] = 1
            # return one_hot_vector
            return hex2vec(pc[-3:])

        def preprocess_addr(addr):
            return hex2vec(addr[2:-3])

        def preprocess_csv_file(f):
            global tid_to_idx
            global pc_to_idx
            raw = pd.read_csv(f, header=None)
            # raw = pd.read_csv(f, header=None, nrows=10000)
            # unique_pc = []
            # for index, row in raw.iterrows():
            #     if row[1] not in unique_pc:
            #         unique_pc.append(row[1])
            
            # n_pc = len(unique_pc)
            # pc_to_idx = dict(zip(unique_pc, range(len(unique_pc))))

            # print('[Info] Preprocssing tid')
            # raw[0] = raw[0].swifter.allow_dask_on_strings(enable=True).apply(np.vectorize(preprocess_tid))

            # print('[Info] Preprocssing pc')
            # raw[1] = raw[1].swifter.allow_dask_on_strings(enable=True).apply(np.vectorize(preprocess_pc))

            print('[Info] Preprocssing addr')
            raw[2] = raw[2].swifter.allow_dask_on_strings(enable=True).apply(np.vectorize(preprocess_addr))

            return raw
        
        # self.raw = pd.read_csv(csv_file, header=None, nrows=1000)
        self.data = preprocess_csv_file(csv_file)
        self.look_back = look_back
        self.look_front = look_front

    def __len__(self):
        return len(self.data) - self.look_back - self.look_front

    def __getitem__(self, idx):
        # src_tid = np.concatenate(list(self.data.iloc[idx : idx + self.look_back][0]))        
        # src_pc = np.concatenate(list(self.data.iloc[idx : idx + self.look_back][1]))
        src_addr = np.concatenate(list(self.data.iloc[idx : idx + self.look_back][2]))
        # src = torch.tensor(np.concatenate([src_tid, src_pc, src_addr], axis=1))
        src = torch.tensor(np.concatenate([src_pc, src_addr], axis=1))
        trg = torch.tensor(np.concatenate(list(self.data.iloc[idx + self.look_back : 
                            idx + self.look_back + self.look_front][2])))

        return {'src': src, 'trg': trg}
