#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Authors       : Vadim Titov, Henning Möllers, Leon Mannweiler
Matr.-Nr.     : 6021356, ..., ...
Created       : May 12th, 2022
Last modified : May 12th, 2022
Description   : Master's Project "Source Separation for Robot Control"
Topic         : Beamformer: Filter-And-Sum Implementation
"""
import sys

import numpy as np
np.set_printoptions(threshold=sys.maxsize)
import pyroomacoustics as pra
from scipy.signal import get_window
from scipy.signal import stft
from scipy.signal import istft
import matplotlib.pyplot as plt
import soundfile as sf


def create_room(room_dim:np.ndarray, rt60:float) -> pra.room:
    """Helper function. Creates a room from input parameters and returns it.

    :param room_dim: np.ndarray
    :param rt60: float reverberation time in s
    :return: pra.room
    """
    e_absorbtion, max_order = pra.inverse_sabine(rt60=rt60, room_dim=room_dim)
    room = pra.ShoeBox(room_dim,materials=pra.Material(e_absorbtion), max_order=max_order)
    return room


def add_sources(room:pra.room, source_locs:np.ndarray, audio:np.ndarray, delay:float):
    """Helper function. Adds sources to an input room defined by input paramters.

    :param room: pra.room
    :param source_locs: np.ndarray
    :param audio: np.ndarray
    :param delay: float
    :return: None
    """

    # place the source in the room
    room.add_source(source_locs, signal=audio, delay=delay)


def add_mics(room:pra.room,beamformer:pra.beamforming.Beamformer):
    """Helper function. Adds mics to an input room from input parameters.

    :param room: pra.room
    :param beamformer: pra.beamforming.Beamformer
    :return: None
    """
    # define the locations of the microphones.
    # finally place the array in the room.
    room.add_microphone_array(beamformer)


def plot_signal(s: np.ndarray, t: np.ndarray, title: str):
    """Helper function. Plots a signal as magnitude `s` against time `t`.

    Parameters
    ----------
    s : np.ndarray
        Amplitude values of the input signal.
    t : np.ndarray
        Time values of the input signal.
    title : str
        Title for the plot.
    """

    # Define some general settings for the plot.
    rows = 1
    cols = 1
    fig = plt.figure(figsize=(19, 7))

    plt.suptitle(title)

    plt.subplots_adjust(left=0.05,
                        # bottom=0.1,
                        right=0.99,
                        # top=0.9,
                        # wspace=0.4,
                        hspace=0.4)

    # First subplot.
    ax = fig.add_subplot(rows, cols, 1)
    ax.plot(t, s, 'b', label="s(t)")
    ax.set_xlabel('t[s]')
    ax.set_ylabel('s(t)')
    ax.grid(True)
    ax.legend()


def plot_dft(S: np.ndarray, title: str):
    """Helper function. Plots amplitude of `S` in dB together with the
    amplitude of the corresponding filter `H`(z) in dB in one plot.

    Parameters
    ----------
    S : np.ndarray
        Amplitudes of input signal.
    H : np.ndarray
        Amplitudes of the corresponding filter.
    title : str
        Plot title.
    """

    fig, ax1 = plt.subplots()

    ax1.set_title(title)
    ax1.plot(S)

    ax1.set_ylabel('Amplitude [dB]', color='b')

    ax1.grid()
    ax1.axis('tight')
    ax1.legend()


def compute_SNR(signal: np.ndarray, axis=0, ddof=0) -> float:
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
    m = signal.mean(axis)
    stdev = signal.std(axis=axis, ddof=ddof)
    return np.where(stdev == 0, 0, m / stdev)


def main():

    # Define sample path.
    sample_path_speech = "./Samples/examples_input_samples_german_speech_8000.wav"
    sample_path_noise = "./Samples/whitenoise.wav"

    # Read samples from disk.
    audio_speech, fs_speech = sf.read(sample_path_speech)
    audio_noise, fs_noise = sf.read(sample_path_noise)

    # The desired reverberation time in seconds.
    rt60 = 0.5
    # Room dimensions x,y,z in meters.
    room_dim = [20, 30]

    # Define mic array.
    R = pra.linear_2D_array([2, 1.5], 4, 0, 0.1)

    # ROOM 1
    # Define source locations. (noise outside beampattern)
    source_locs1 = [2.5, 4.5]
    source_locs2 = [17.5, 22.5]

    # ROOM 2
    # Define source locations. (noise inside beampattern)
    source_locs3 = [2.5, 4.5]
    source_locs4 = [0.5, 3.0]

    # Create time domain values.
    t_speech = np.arange(0, len(audio_speech))/fs_speech
    t_noise = np.arange(0, len(audio_noise))/fs_noise

    # Frame length in ms.
    fl = 32

    # Transform frame length from ms to samples.
    N_speech = int(fs_speech*fl/1000)
    N_noise = int(fs_noise*fl/1000)

    # Define windowing functions.
    window1 = np.sqrt(get_window('hann', N_speech, fftbins=True))
    window2 = np.sqrt(get_window('hann', N_noise, fftbins=True))

    # Specify delay.
    delay = 1.3

    # Create room.
    room1 = create_room(room_dim, rt60)
    room2 = create_room(room_dim, rt60)

    # Create beamformer.
    beamformer = pra.Beamformer(R, room1.fs)
    # Add beamformer to room.
    add_mics(room1, beamformer)
    add_mics(room2, beamformer)

    # Add source(s).
    add_sources(room1, source_locs1, audio_speech, delay)
    add_sources(room1, source_locs2, audio_noise, delay)

    add_sources(room2, source_locs3, audio_speech, delay)
    add_sources(room2, source_locs4, audio_noise, delay)

    # Now compute the delay and sum weights for the beamformer
    #room1.mic_array.rake_delay_and_sum_weights(room1.sources[0][:1])
    room2.mic_array.rake_delay_and_sum_weights(room2.sources[0][:1])

    #beamformer.plot()

    # plot the room and resulting beamformer
    room1.plot(freq=[1000, 2000, 4000, 8000], img_order=0)
    room2.plot(freq=[1000, 2000, 4000, 8000], img_order=0)

    room1.compute_rir()
    room2.compute_rir()

    premix1 = room1.simulate(return_premix=True)
    premix2 = room2.simulate(return_premix=True)

    #sf.write("premix2noise.wav",premix2[1].T,fs_noise)
    #sf.write("premix2speech.wav", premix2[0].T, fs_noise)

    stft2noise = stft(premix1[1], fs_speech, window=window1)
    #print(stft2noise[0].shape) # f: ndarray Array of sample frequencies.
    #print(stft2noise[1].shape) # t: ndarray Array of segment times.
    #print(stft2noise[2].shape) # Zxx: ndarray STFT of x.

    stft2speech = stft(premix1[0], fs_speech, window=window1)

    # covariance matrix
    alpha = 0.8

    _, times, _ = stft2speech
    freqs, _, _ = stft2speech
    _, _, power = stft2speech

    cov_mat = np.zeros((times.shape[0], freqs.shape[0], power.shape[0], power.shape[0]), dtype=np.complex_)

    inv_cov_mat = np.zeros_like(cov_mat, dtype="complex128")

    eigen_mat = np.zeros((cov_mat.shape[0], cov_mat.shape[1], cov_mat.shape[2], 1), dtype="complex128")
    filter_mat = np.zeros((cov_mat.shape[0], cov_mat.shape[1], cov_mat.shape[2], 1), dtype="complex128")

    # compute covariance matrix (or matrices rather?)
    for t, _ in enumerate(times):
        for f, _ in enumerate(freqs):
            init_matrix = np.zeros((4, 4))
            np.fill_diagonal(init_matrix, 1)
            x = power[:, f, t]
            init_matrix = alpha * init_matrix + (1 - alpha * x * x.conj().T)
            cov_mat[t, f] = init_matrix
            # eigenvector used as steering vector
            # eigenvector element wise
            eigenvector, _ = np.linalg.eig(cov_mat[t, f])
            eigenvector = eigenvector[:, np.newaxis]
            eigen_mat[t, f] = eigenvector
            # invert element wise
            inv_cov = np.linalg.inv(cov_mat[t, f])
            inv_cov_mat[t, f] = inv_cov
            filter_coeff = (inv_cov @ eigenvector) / eigenvector.conj().T @ inv_cov @ eigenvector

            filter_mat[t, f] = filter_coeff

    _, _, unfiltered_speech = stft2speech

    filtered_speech = filter_mat.T * unfiltered_speech
    _, filtered_istft = istft(filtered_speech, fs_speech, window1)
    _, unfiltered_istft = istft(unfiltered_speech, fs_speech, window1)

    premix_noisy_speech = premix1[0] + premix1[1]
    stft2noisy_speech = stft(premix_noisy_speech, fs_speech, window=window1)
    _,_,noisy_speech = stft2noisy_speech
    _, noisy_istft = istft(noisy_speech, fs_speech, window1)

    #plt.figure()
    #plt.title("noisy speech signal (pre-processing for DEBUG purposes)")
    #plt.plot(noisy_speech)

    plt.figure()
    plt.title("clean speech signal")
    plt.plot(audio_speech)
    plt.figure()
    plt.title("noisy speech signal")
    plt.plot(noisy_istft[0, :])
    plt.figure()
    plt.title("filtered speech signal")
    plt.plot(filtered_istft[0, 0, :])
    plt.show()


    sf.write("clean_speech.wav", audio_speech, fs_noise)
    sf.write("noisy_speech.wav", noisy_istft[0, :], fs_speech)
    sf.write("filtered_speech.wav", filtered_istft[0, 0, :], fs_speech)


    #output = beamformer.process()
    #sf.write("output.wav", output, fs_noise)

    #room1.mic_array.to_wav(filename="room1.wav", mono=False)
    #room2.mic_array.to_wav(filename="room2.wav", mono=False)

    #TODO SNR
    clean_SNR = compute_SNR(audio_speech)
    noisy_SNR = compute_SNR(noisy_istft[0, :])
    filtered_SNR = compute_SNR(filtered_istft[0, 0, :])

    #print(clean_SNR)
    #print(noisy_SNR)
    #print(filtered_SNR)

if __name__ == "__main__":
    main()

