#!/usr/bin/env python3
import torch
import numpy as np
import glob
import os
from torch.utils.data import Dataset

class CustomDataset(Dataset):

    def __init__(self, data_dir):
        self.data_dir = data_dir

        data_clean = np.sort(np.array(glob.glob(data_dir+"/*clean.txt")))
        data_noise = np.sort(np.array(glob.glob(data_dir+"/*noise.txt")))
        data_mixture = np.sort(np.array(glob.glob(data_dir+"/*mixture.txt")))
            

        print(data_clean)
        print(data_noise)
        print(data_mixture) 



        
    def __len__(self):
        return data_clean.shape[0]*3
    
    def __getitem__(self):
        pass
        

Dieter = CustomDataset('/informatik1/students/home/xmannwei/Beamformer/mp-2022/mp-2022/dnn/implementation/testfiles')

 