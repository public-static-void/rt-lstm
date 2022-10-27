#!/usr/bin/env python3
import numpy as np
import predict
import torch
from scipy.signal import get_window
import soundfile as sf

class Inverse_stft_and_more():
    def __init__(self, prediction, mixture, istft_length=512, istft_shift=256):
        self.prediction = np.array()
        self.mixture = np.array()
        self.stft_length = istft_length
        self.stft_shift = istft_shift


    def __inverse_stft__(self, signal):
        window1 = torch.from_numpy(np.sqrt(get_window('hann', self.stft_length, fftbins=True)))

        istft_signal = torch.istft(torch.from_numpy(signal), self.stft_length, self.stft_shift, window = window1)

    def __merge_to_complex__(self, tensor_to_merge):
        real = torch.chunk(tensor_to_merge, 2, 1)[0]
        imag = torch.chunk(tensor_to_merge, 2, 1)[1]
        return torch.complex(real, imag)

    def __compute_SNR__(signal: np.ndarray, axis=0, ddof=0) -> float:
        """Helper function. Computed the Signal To Noise Ratio (SNR) for the input
        signal.

        Parameters
        ----------
        signal : np.ndarray
            Input signal.
        axis : int, optional
            Signal array axis to perform computation on.
            By default 0.
        ddof : int, optional
            Delta Degrees Of Freedom.
            By default 0.
        """
        signal = np.asanyarray(signal)
        mean = signal.mean(axis)
        stdev = signal.std(axis=axis, ddof=ddof)
        return 20*np.log10(abs(np.where(stdev == 0, 0, mean / stdev)))

def write_wav(self, name, source, fs):
    # TODO: evtl fs zentral definieren?
    sf.write(name, source, fs)

def show_spectrogram(m_stft: np.ndarray, v_freq: np.ndarray,
                     v_time: np.ndarray, title: str):
    """Helper function. Creates a spectrogram of the input. The extent option
    tells matplotlib to use the entries of the vector v_time for the x-axis
    and v_freq for the y-axis. Here, the vector v_time contains the time
    instants for each block / each spectrum and the vector v_freq contains the
    frequency bin information.

    Parameters
    ----------
    m_stft : np.ndarray
        A matrix which stores the complex short-time spectra in each row.
    v_freq : np.ndarray
        A vector which contains the frequency axis (in units of Hertz)
        corresponding to the computed spectra.
    v_time : np.ndarray
        Time steps around which a frame is centered (as in previous exercise).
    title : str
        Plot title.
    """
    fig = plt.figure()
    ax = fig.add_subplot(111)
    im = ax.imshow(10 * np.log10(np.maximum(np.square(np.abs(m_stft)),
                                            10**(-15))), cmap='viridis',
                   origin='lower', extent=[v_time[0], v_time[-1], v_freq[0],
                                           v_freq[-1]], aspect='auto')
    fig.colorbar(im, orientation="vertical", pad=0.2)
    plt.title(title)
    ax.set_xlabel('Time (s)')
    ax.set_ylabel('Frequency (Hz)')
    # TODO: evtl. plt.show() ganz ans ende der performance evaluation
    plt.show()


