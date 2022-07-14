#!/usr/bin/env python3
from matplotlib.pyplot import axes
import numpy as np
import glob
import soundfile
from scipy.signal import get_window
from scipy.signal import stft
from torch.utils.data import Dataset

class CustomDataset(Dataset):

    def __init__(self, type):
        self.type = type

        if self.type == 'Training':
            self.data_dir = './dnn/imlementation/soundfiles/Training'
        else:
            if self.type == 'Validation':
                self.data_dir = './dnn/implementation/soundfiles/Validation'
            else:
                self.data_dir = './dnn/implementation/soundfiles/Test'

        self.data_clean = np.sort(np.array(glob.glob(self.data_dir+"/*clean.wav")))
        self.data_noise = np.sort(np.array(glob.glob(self.data_dir+"/*noise.wav")))
        self.data_mixture = np.sort(np.array(glob.glob(self.data_dir+"/*mixture.wav")))


        #TODO: *3?
    def __len__(self):
        return self.data_clean.shape[0]


    
    #TODO: Kontrolliere Fensterbreite... 256?
    def __getitem__(self, index):
        window1 = np.sqrt(get_window('hann', 512, fftbins=True))


        clean_read = soundfile.read(self.data_clean[index])
        noise_read = soundfile.read(self.data_noise[index])
        mixture_read = soundfile.read(self.data_mixture[index])

        clean_freqs, clean_times, clean_power = stft(clean_read[0], clean_read[1], window=window1, nperseg=512)
        noise_freqs, noise_times, noise_power = stft(noise_read[0], noise_read[1], window=window1, nperseg=512)
        mixture_freqs, mixture_times, mixture_power = stft(mixture_read[0], mixture_read[1], window=window1, nperseg=512)

        print(clean_power.shape)
        print(noise_power.shape)
        print(mixture_power.shape)


        clean_split_concatenate = np.concatenate((np.real(clean_power), np.imag(clean_power)), axis=0)
        noise_split_concatenate = np.concatenate((np.real(noise_power), np.imag(noise_power)), axis=0)
        mixture_split_concatenate = np.concatenate((np.real(mixture_power), np.imag(mixture_power)), axis=0)


        print(clean_split_concatenate.shape)
        print(noise_split_concatenate.shape)
        print(mixture_split_concatenate.shape)

        return clean_split_concatenate, noise_split_concatenate, mixture_split_concatenate



Dataset = CustomDataset('Training')

Dataset.__getitem__(0)

