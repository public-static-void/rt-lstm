#!/usr/bin/env python3
import numpy as np
import predict
import torch
from scipy.signal import get_window

class Inverse_stft_and_more():
    def __init__(self, prediction, mixture, istft_length=512, istft_shift=256):
        self.prediction = np.array()
        self.mixture = np.array()
        self.stft_length = istft_length
        self.stft_shift = istft_shift
    

    def __inverse_stft__(self, signal):
        window1 = torch.from_numpy(np.sqrt(get_window('hann', self.stft_length, fftbins=True)))

        istft_signal = torch.istft(torch.from_numpy(signal), self.stft_length, self.stft_shift, window = window1)
