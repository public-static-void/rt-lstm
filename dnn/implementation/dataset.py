#!/usr/bin/env python3
from random import sample
from matplotlib.pyplot import axes
import numpy as np
import glob
from pandas import cut
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
        self.sample_rate = 16000


        #TODO: *3?
    def __len__(self):
        return self.data_clean.shape[0]


    def __cut__(self, sound, sec:int):
        samples_to_take = sec * self.sample_rate
        if samples_to_take > sound.shape[0]:
            if sound.ndim > 1:
                sound_cut = np.zeros((samples_to_take,3))
            else:
                sound_cut = np.zeros(samples_to_take)
        else:
            start_sample = np.random.randint(0, sound.shape[0]-samples_to_take)
            sound_cut = sound[start_sample:start_sample+samples_to_take]

        return sound_cut

    #TODO: Kontrolliere Fensterbreite... 256?
    def __getitem__(self, index):
        window1 = torch.from_numpy(np.sqrt(get_window('hann', self.stft_length, fftbins=True)))


        clean_read,fs = soundfile.read(self.data_clean[index])
        noise_read,fs = soundfile.read(self.data_noise[index])
        mixture_read,fs = soundfile.read(self.data_mixture[index])

        clean_read = self.__cut__(clean_read, 3)
        noise_read = self.__cut__(noise_read, 3)
        mixture_read = self.__cut__(mixture_read, 3)

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

        #print(mixture_split_concatenate.shape)

        #real = torch.chunk(mixture_split_concatenate, 2, 0)[0]
        #imag = torch.chunk(mixture_split_concatenate, 2, 0)[1]

        #print(real.shape)
        #print(imag.shape)

        #mixture_split_concatenate = torch.complex(real, imag)


        #print(clean_split_concatenate.shape)
        #print(noise_split_concatenate.shape)
        #print(mixture_split_concatenate.shape)

        return clean_split_concatenate, noise_split_concatenate, mixture_split_concatenate

    def __merge_to_complex__(self, tensor_to_merge):
        real = torch.chunk(tensor_to_merge, 2, 1)[0]
        imag = torch.chunk(tensor_to_merge, 2, 1)[1]
        return torch.complex(real, imag)





#Dataset = CustomDataset('Test')

#Dataset.__getitem__(3)







