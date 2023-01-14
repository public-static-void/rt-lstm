import numpy as np
from scipy import signal


def get_periodic_hann(M):
    n = np.arange(0, M)
    return 0.5 - 0.5*np.cos(2.0*np.pi*n/(M))


def get_butter_coeffs(ds_factor, lp_order):
    b, a = signal.butter(lp_order, 1/ds_factor, 'low')
    return b, a


def get_windowed_irfft(spec, window, fft_len):
    return window * np.fft.irfft(spec, n = fft_len)


def get_windowed_rfft(signal, window, fft_len):
    return np.fft.rfft(signal*window, n=fft_len)
