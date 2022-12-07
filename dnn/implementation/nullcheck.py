#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Authors       : Vadim Titov
Matr.-Nr.     : 6021356
Created       : December 7th, 2022
Last modified : December 7th, 2022
Description   : Master's Project "Source Separation for Robot Control"
Topic         : Null check module of the LSTM RNN Project
"""

import numpy as np
import soundfile
import glob


type = 'training'
# type = 'validation'
# type = 'test'

def main():

    if type == 'training':
        # data_dir = '/export/scratch/9hmoelle/generatedDatasets/Training'
        data_dir = 'soundfiles/generatedDatasets/Training'

    else:
        if type == 'validation':
            # data_dir = '/export/scratch/9hmoelle/generatedDatasets/Validation'
            data_dir = 'soundfiles/generatedDatasets/Validation'
        else:
            # data_dir = '/export/scratch/9hmoelle/generatedDatasets/Test'
            data_dir = 'soundfiles/generatedDatasets/Test'


    data_clean = np.sort(np.array(glob.glob(data_dir+"/*clean.wav")))
    data_noise = np.sort(np.array(glob.glob(data_dir+"/*noise.wav")))

    clean_nulls = 0
    noise_nulls = 0

    for index, file in enumerate(data_clean):
        clean_read, fs = soundfile.read(data_clean[index])
        if clean_read.sum() == 0:
            clean_nulls += 1

    for index, file in enumerate(data_noise):
        noise_read, fs = soundfile.read(data_noise[index])
        if noise_read.sum() == 0:
            noise_nulls += 1

    print("clean nulls:", clean_nulls)
    print("noise nulls:", noise_nulls)

if __name__ == "__main__":
    main()
