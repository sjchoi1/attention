from __future__ import print_function, division
import os
import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader
import pandas as pd
from tqdm import tqdm
import swifter
import random

tid_to_idx = {}
pc_to_idx = {}
n_pc = 0
wrap_repeat = 0

class CustomDataset(Dataset):
    """Custom dataset."""
    def __init__(self, csv_file, look_back, look_front, wrap, length):
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
            one_hot_vector = torch.zeros(1, 16)
            one_hot_vector[0, int(tid)] = 1
            return one_hot_vector

        def preprocess_pc(pc):
            global wrap_repeat
            # global pc_to_idx
            # global n_pc
            # one_hot_vector = torch.zeros(1, 32)
            # one_hot_vector[0, pc_to_idx[pc]] = 1
            # return one_hot_vector
            return hex2vec(pc[-3:]).repeat(wrap_repeat, 1)

        def preprocess_addr(addr):
            # return hex2vec(addr[2:-3])
            return hex2vec(addr[2:-2])

        def preprocess_csv_file(f):
            global pc_to_idx
            global n_pc
            
            raw = pd.read_csv(f, header=None)
            # raw = pd.read_csv(f, header=None, nrows=100000)
            # unique_pc = []
            # for index, row in raw.iterrows():
            #     if row[1] not in unique_pc:
            #         unique_pc.append(row[1])
            
            # n_pc = len(unique_pc)
            # print(n_pc)
            # sys.exit()
            # pc_to_idx = dict(zip(unique_pc, range(len(unique_pc))))

            # print('[Info] Preprocssing tid')
            # raw[0] = raw[0].swifter.allow_dask_on_strings(enable=True).apply(np.vectorize(preprocess_tid))

            print('[Info] Preprocssing pc')
            raw[1] = raw[1].swifter.set_npartitions(16).allow_dask_on_strings(enable=True).apply(np.vectorize(preprocess_pc))

            print('[Info] Preprocssing addr')
            raw[2] = raw[2].swifter.set_npartitions(16).allow_dask_on_strings(enable=True).apply(np.vectorize(preprocess_addr))

            return raw
        global wrap_repeat
        wrap_repeat = 8 // wrap
        self.data = preprocess_csv_file(csv_file)
        # self.raw = pd.read_csv(csv_file, header=None, nrows=1000)
        self.look_back = look_back
        self.look_front = look_front
        self.length = length
        self.wrap_size = wrap * 16



    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        idx = random.randint(0, len(self.data) - self.look_back - self.look_front)

        # src_tid = np.concatenate(list(self.data.iloc[idx : idx + self.look_back][0]))        
        src_pc = torch.tensor(np.concatenate(list(self.data.iloc[idx : idx + self.look_back][1])))
        # src_addr = np.concatenate(list(self.data.iloc[idx : idx + self.look_back][2]))
        src = torch.tensor(np.concatenate(list(self.data.iloc[idx : idx + self.look_back][2]))).view(-1, self.wrap_size)
        # src = torch.tensor(np.concatenate([src_tid, src_pc, src_addr], axis=1))
        # src = torch.tensor(np.concatenate([src_pc, src_addr], axis=1))
        trg = torch.tensor(np.concatenate(list(self.data.iloc[idx + self.look_back : 
                            idx + self.look_back + self.look_front][2]))).view(-1, self.wrap_size)        
        trg = torch.cat((src[-1].unsqueeze(0), trg), dim=0)
        src = torch.cat((src_pc, src), dim=1)

        return {'src': src, 'trg': trg}
