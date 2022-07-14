#!/usr/bin/env python3
from matplotlib.pyplot import axes
import numpy as np
import glob
import soundfile
from scipy.signal import get_window
from scipy.signal import stft
import torch
from torch.utils.data import Dataset

class CustomDataset(Dataset):

    def __init__(self, type, stft_lenght=512, stft_shift=256):
        super(CustomDataset).__init__()
        self.type = type
        self.stft_length = stft_lenght
        self.stft_shift = stft_shift

        if self.type == 'Training':
            self.data_dir = './soundfiles/Training'
        else:
            if self.type == 'Validation':
                self.data_dir = './soundfiles/Validation'
            else:
                self.data_dir = './soundfiles/Test'

        self.data_clean = np.sort(np.array(glob.glob(self.data_dir+"/*clean.wav")))
        self.data_noise = np.sort(np.array(glob.glob(self.data_dir+"/*noise.wav")))
        self.data_mixture = np.sort(np.array(glob.glob(self.data_dir+"/*mixture.wav")))


        #TODO: *3?
    def __len__(self):
        return self.data_clean.shape[0]


    
    #TODO: Kontrolliere Fensterbreite... 256?
    def __getitem__(self, index):
        window1 = torch.from_numpy(np.sqrt(get_window('hann', self.stft_length, fftbins=True)))


        clean_read,fs = soundfile.read(self.data_clean[index])
        noise_read,fs = soundfile.read(self.data_noise[index])
        mixture_read,fs = soundfile.read(self.data_mixture[index])

        #print(clean_read.shape)
        #print(noise_read.shape)
        #print(mixture_read.shape)

        clean_stft = torch.stft(torch.from_numpy(clean_read), self.stft_length, self.stft_shift, window = window1, return_complex=True)
        noise_stft = torch.stft(torch.from_numpy(noise_read), self.stft_length, self.stft_shift, window = window1, return_complex=True)
        mixture_stft = torch.stft(torch.from_numpy(mixture_read.T), self.stft_length, self.stft_shift, window = window1, return_complex=True)
      

        #print(clean_stft.shape)
        #print(noise_stft.shape)
        #print(mixture_stft.shape)


        clean_split_concatenate = torch.stack((torch.real(clean_stft), torch.imag(clean_stft)), dim=0)
        noise_split_concatenate = torch.stack((torch.real(noise_stft), torch.imag(noise_stft)), dim=0)
        mixture_split_concatenate = torch.cat((torch.real(mixture_stft), torch.imag(mixture_stft)), dim=0)


        #print(clean_split_concatenate.shape)
        #print(noise_split_concatenate.shape)
        print(mixture_split_concatenate.shape)

        return clean_split_concatenate, noise_split_concatenate, mixture_split_concatenate
test = np.zeros(1)
print(test.ndim)

Dataset = CustomDataset('Test')

Dataset.__getitem__(3)
Dataset.__getitem__(1)
Dataset.__getitem__(45)