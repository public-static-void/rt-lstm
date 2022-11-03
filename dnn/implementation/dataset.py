#!/usr/bin/env python3
import glob
from random import sample

import numpy as np
import soundfile
import torch
from matplotlib.pyplot import axes
from pandas import cut
from scipy.signal import get_window, stft
from torch.utils.data import Dataset

"""This Dataset class is used to get single files from certain directories.

Returns:
        _type_: _tensor_
"""
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
        self.sample_rate = 16000

    
    
        """Defines the length of one file
        """
    def __len__(self):
        return self.data_clean.shape[0]

        """Function to cut a soundfile into a variable number of seconds.

        Returns: NDArray[Float64]

        """
    def __cut__(self, sound, samples_to_take, start_sample:int):
        
        if samples_to_take > sound.shape[0]:
            if sound.ndim > 1:
                sound_cut = np.zeros((samples_to_take,3))
            else:
                sound_cut = np.zeros(samples_to_take)
        else:
            
            sound_cut = sound[start_sample:start_sample+samples_to_take]

        return sound_cut


        """This function reads in the soundfiles, uses __cut__ and transforms the signal into the frequency are via stft.
        Then the complex values of the clean, noisy and mixed signal are getting split into real and imaginary parts. This parts are getting concatinated.
        """
    def __getitem__(self, index):
        cut_length = 3
        samples_to_take = cut_length * self.sample_rate

        window1 = torch.from_numpy(np.sqrt(get_window('hann', self.stft_length, fftbins=True)))


        clean_read,fs = soundfile.read(self.data_clean[index])
        noise_read,fs = soundfile.read(self.data_noise[index])
        mixture_read,fs = soundfile.read(self.data_mixture[index])

        start_sample = np.random.randint(0, mixture_read.shape[0]-samples_to_take)

        clean_read = self.__cut__(clean_read, samples_to_take, start_sample,)
        noise_read = self.__cut__(noise_read, samples_to_take, start_sample)
        mixture_read = self.__cut__(mixture_read, samples_to_take, start_sample)

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
        #print(mixture_split_concatenate.shape)

        return clean_split_concatenate, noise_split_concatenate, mixture_split_concatenate

#Dataset = CustomDataset('Test')

#Dataset.__getitem__(3)







