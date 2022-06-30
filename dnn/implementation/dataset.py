#!/usr/bin/env python3
import torch
import numpy as np
import glob
import os
import soundfile
from scipy.signal import get_window
from scipy.signal import stft
from torch.utils.data import Dataset

class CustomDataset(Dataset):

    def __init__(self, data_dir):
        self.data_dir = data_dir

        data_clean = np.sort(np.array(glob.glob(data_dir+"/*clean.wav")))
        data_noise = np.sort(np.array(glob.glob(data_dir+"/*noise.wav")))
        data_mixture = np.sort(np.array(glob.glob(data_dir+"/*mixture.wav")))

        print(data_clean)
        print(data_noise)
        print(data_mixture) 

        
    def __len__(self):
        return data_clean.shape[0]*3

    
    #TODO: Kontrolliere Fensterbreite... 512?
    def __getitem__(self, index):
        window1 = np.sqrt(get_window('hann', 512, fftbins=True))

        clean_read = soundfile.read(data_clean[index])
        noise_read = soundfile.read(data_noise[index])
        mixture_read = soundfile.read(data_mixture[index])

        clean_freqs, clean_times, clean_power = stft(clean[0], clean[1], window=window1)
        noise_freqs, noise_times, noise_power = stft(noise[0], noise[1], window=window1)
        mixture_freqs, mixture_times, mixture_power = stft(mixture[0], mixture[1], window=window1)

        clean_split_concatenate = self.split_power(clean_power)
        noise_split_concatenate = self.split_power(noise_power)
        mixture_split_concatenate = self.split_power(mixture_power)


    def split_power(self, power)

        imag = np.array((clean_power.shape[2]))
        real = np.array((clean_power.shape[2]))
        for i,_ in enumerate(clean_power):
            real[i]=clean_power[i].real
            imag[i]=clean_power[i].imag
        return np.concatenate(real, imag)


        

Dieter = CustomDataset('/informatik1/students/home/xmannwei/Beamformer/mp-2022/mp-2022/dnn/implementation/testfiles')

 