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
import json

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

        print(self.type)
        if self.type == 'training':
            # self.data_dir = '/export/scratch/9hmoelle/generatedDatasets/Training'
            self.data_dir = 'soundfiles/generatedDatasets/Training'

        else:
            if self.type == 'validation':
                # self.data_dir = '/export/scratch/9hmoelle/generatedDatasets/Validation'
                self.data_dir = 'soundfiles/generatedDatasets/Validation'
            else:
                # self.data_dir = '/export/scratch/9hmoelle/generatedDatasets/Test'
                self.data_dir = 'soundfiles/generatedDatasets/Test'


        self.data_clean = np.sort(np.array(glob.glob(self.data_dir+"/*clean.wav")))
        self.data_noise = np.sort(np.array(glob.glob(self.data_dir+"/*noise.wav")))
        self.data_meta = np.sort(np.array(glob.glob(self.data_dir+"/*meta.json")))
        self.sample_rate = 16000

        """Defines the number of files in the dataset
        """
    def __len__(self):
        # number_of_files = len(self.data_clean) + len(self.data_noise)
        # print(number_of_files)
        # return number_of_files
        #print(len(self.data_clean))
        return len(self.data_clean)


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


        """This function reads in the soundfiles, uses __cut__ and transforms the signal into the frequency domain via stft.
        Then the complex values of the clean, noisy and mixed signal are getting split into real and imaginary parts. This parts are getting concatinated.
        """
    def __getitem__(self, index):

        #TODO: Metadaten, die von Henning in einer Datei geliefert werden, zusammen mit der SNR, mit an Vadim übergeben. Inkl. Zuordnung zum Dateinamen.


        #TODO: deacitvate when actually training
        reproducable = False

        cut_length = 4
        samples_to_take = cut_length * self.sample_rate

        window1 = torch.from_numpy(np.sqrt(get_window('hann', self.stft_length, fftbins=True)))

        if self.type == 'test':
            data_index = self.data_clean[index].split("Test/")[1][:-10]

        #Test if Tensor is empty
        clean_read,fs = soundfile.read(self.data_clean[index])
        noise_read,fs = soundfile.read(self.data_noise[index])
        while clean_read.sum() == 0 or noise_read.sum() == 0:
            if index < len(self):
                index+=1
            else:
                index=0
            clean_read,fs = soundfile.read(self.data_clean[index])
            noise_read,fs = soundfile.read(self.data_noise[index])

        #Read Meta Data
        meta_file = open(self.data_meta[index])
        meta_json = json.loads(meta_file.read())
        reverberation_rate = meta_json["reverberationRate"]
        min_distance_to_noise = meta_json["minimalNoiseDistance"]


        #To make comparing validation graphs possible (and for bug finding). If this if-clause is true, we always choose the same cut from each soundfile in each epoche.
        if reproducable or self.type=='validation':
            np.random.seed(index)

        SNR = np.random.uniform(-10, 5)
        power_clean = np.sum(np.square(clean_read))
        power_noise = np.sum(np.square(noise_read))

        factor_to_lower_noise = np.sqrt(power_clean / (10**((SNR/10))*power_noise))

        # In case the signals are of different length, pad the shorter one with
        # zeros to match length of the larger one.
        if len(clean_read) > len(noise_read):
            # Amount of zeros to pad.
            n = len(clean_read) - len(noise_read)
            # Make sure only to pad first axis.
            npad = [(0, 0)] * noise_read.ndim
            npad[0] = (0, n)
            noise_read = np.pad(noise_read, pad_width=npad, mode='constant')
        elif len(clean_read) < len(noise_read):
            # Amount of zeros to pad.
            n = len(noise_read) - len(clean_read)
            # Make sure only to pad first axis.
            npad = [(0, 0)] * clean_read.ndim
            npad[0] = (0, n)
            clean_read = np.pad(clean_read, pad_width=npad, mode='constant')


        mixture_read = clean_read + factor_to_lower_noise * noise_read

        #Disables cutting the soundfile into a 3sec clip, when the Dataset Type is 'test'.

        if self.type=='validation' or self.type=='training':

            start_sample = np.random.randint(0, mixture_read.shape[0]-samples_to_take)

            clean_read = self.__cut__(clean_read, samples_to_take, start_sample,)
            noise_read = self.__cut__(noise_read, samples_to_take, start_sample)
            mixture_read = self.__cut__(mixture_read, samples_to_take, start_sample)

        #print(clean_read.shape)
        #print(noise_read.shape)
        #print(mixture_read.shape)

        #soundfile.write("./soundfiles/Hearing/clean.wav", clean_read, 16000)
        #soundfile.write("./soundfiles/Hearing/noise.wav", noise_read, 16000)
        #soundfile.write("./soundfiles/Hearing/mixture.wav", mixture_read, 16000)

        clean_stft = torch.stft(torch.from_numpy(clean_read.T), self.stft_length, self.stft_shift, window = window1, return_complex=True)
        noise_stft = torch.stft(torch.from_numpy(noise_read.T), self.stft_length, self.stft_shift, window = window1, return_complex=True)
        mixture_stft = torch.stft(torch.from_numpy(mixture_read.T), self.stft_length, self.stft_shift, window = window1, return_complex=True)


        #print(clean_stft.shape)
        #print(noise_stft.shape)
        #print(mixture_stft.shape)


        clean_split_concatenate = torch.stack((torch.real(clean_stft[0]), torch.imag(clean_stft[0])), dim=0)
        noise_split_concatenate = torch.stack((torch.real(noise_stft[0]), torch.imag(noise_stft[0])), dim=0)
        mixture_split_concatenate = torch.cat((torch.real(mixture_stft), torch.imag(mixture_stft)), dim=0)

        clean_name = self.data_clean[index]
        noise_name = self.data_noise[index]
        mixture_name = clean_name.replace("clean", "mixture")

        #print(clean_split_concatenate.shape)
        #print(noise_split_concatenate.shape)


        #print(clean_split_concatenate.shape)
        #print(noise_split_concatenate.shape)
        #print(mixture_split_concatenate.shape)

        if self.type =='test':
            meta_data = {'data_index': data_index, 'SNR':SNR, 'reverberation_rate':reverberation_rate, 'min_distance_to_noise': min_distance_to_noise}
            return clean_split_concatenate, noise_split_concatenate, mixture_split_concatenate, meta_data
        else:
            return clean_split_concatenate, noise_split_concatenate, mixture_split_concatenate


#Dataset = CustomDataset('test')

#Dataset.__getitem__(7)
