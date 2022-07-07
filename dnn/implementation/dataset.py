#!/usr/bin/env python3
import numpy as np
import glob
import soundfile
from scipy.signal import get_window
from scipy.signal import stft
from torch.utils.data import Dataset

class CustomDataset(Dataset):

    def __init__(self, type):
        self.type = type

        if self.type == 'training':
            self.data_dir = '/export/scratch/soundfiles/Training'
        else:
            if self.type == 'validation':
                self.data_dir = '/export/scratch/soundfiles/Validation'
            else:
                self.data_dir = '/export/scratch/soundfiles/Test'

        self.data_clean = np.sort(np.array(glob.glob(self.data_dir+"/*clean.wav")))
        self.data_noise = np.sort(np.array(glob.glob(self.data_dir+"/*noise.wav")))
        self.data_mixture = np.sort(np.array(glob.glob(self.data_dir+"/*mixture.wav")))


        #TODO: *3?
    def __len__(self):
        return self.data_clean.shape[0]

    
    #TODO: Kontrolliere Fensterbreite... 256?
    def __getitem__(self, index):
        window1 = np.sqrt(get_window('hann', 256, fftbins=True))


        clean_read = soundfile.read(self.data_clean[index])
        noise_read = soundfile.read(self.data_noise[index])
        mixture_read = soundfile.read(self.data_mixture[index])

        clean_freqs, clean_times, clean_power = stft(clean_read[0], clean_read[1], window=window1)
        noise_freqs, noise_times, noise_power = stft(noise_read[0], noise_read[1], window=window1)
        mixture_freqs, mixture_times, mixture_power = stft(mixture_read[0], mixture_read[1], window=window1)


        def split_power(power):

            imag = np.zeros((clean_power.shape))
            real = np.zeros((clean_power.shape))
            for i in range(0,clean_power.shape[0]):
                for j in range(0,clean_power.shape[1]):
                    real[i,j]=clean_power[i,j].real
                    imag[i,j]=clean_power[i,j].imag
            return np.concatenate((real, imag))

        #TODO: Überlegen wie wir das richtige Label für Mixxture berechnen...
        clean_split_concatenate = np.array((split_power(clean_power), [1]))
        noise_split_concatenate = np.array((split_power(noise_power), [-1]))
        mixture_split_concatenate = np.array((split_power(mixture_power), [0]))

        #print(clean_split_concatenate)
        #print(noise_split_concatenate)
        #print(mixture_split_concatenate)

        return clean_split_concatenate, noise_split_concatenate, mixture_split_concatenate



#Dataset = CustomDataset('test')

#Dataset.__getitem__(0)

