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

import numpy as np
import pyroomacoustics as pra
from scipy.signal import get_window
import matplotlib.pyplot as plt
import soundfile as sf

def create_room(room_dim:np.ndarray) -> pra.room:
    """Helper function. Creates a room from input parameters and returns it.

    :param room_dim: np.ndarray
    :return: pra.room
    """
    room = pra.ShoeBox(room_dim)
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



def main():

    # Sample rate.
    fs = 16000
    # The desired reverberation time in seconds.
    rt60 = 0.5
    # Room dimensions x,y,z in meters.
    room_dim = [4,6]

    # Define beamformer.
    R = pra.linear_2D_array([2, 1.5], 4, 0, 0.1)

    # Define source locations.
    source_locs = [2.5, 4.5]

    # Define sample path.
    sample_path1 = "./Samples/2213f081fd811e304423f283850b-orig.wav"
    sample_path2 = "./Samples/examples_input_samples_german_speech_8000.wav"
    sample_path3 = "./Samples/whitenoise.wav"

    # Read samples from disk.
    audio1,fs1 = sf.read(sample_path1)
    audio2,fs2 = sf.read(sample_path2)
    audio3, fs3 = sf.read(sample_path3)

    # Create time domain values.
    t1 = np.arange(0, len(audio1))/fs1
    t2 = np.arange(0, len(audio2))/fs2
    t3 = np.arange(0,len(audio3))/fs3

    # Frame length in ms.
    fl = 32

    # Transform frame length from ms to samples.
    N1 = int(fs1*fl/1000)
    N2 = int(fs2*fl/1000)
    N3 = int(fs3*fl/1000)

    # Specify frame shift length.
    hop1 = int(N1/2)
    hop2 = int(N2/2)
    hop3 = int(N3/2)

    # Define windowing functions.
    window1 = np.sqrt(get_window('hann', N1, fftbins=True))
    window2 = np.sqrt(get_window('hann', N2, fftbins=True))
    window3 = np.sqrt(get_window('hann', N3, fftbins=True))

    # Specify number of channels.
    n_chans1 = 2
    n_chans2 = 1
    n_chans3 = 1

    # Specify delay.
    delay = 1.3

    # Create classes to perform STFT transformations.
    mySTFT1 = pra.transform.stft.STFT(N=N1,hop=hop1,analysis_window=window1,synthesis_window=window1,channels=n_chans1)
    mySTFT2 = pra.transform.stft.STFT(N=N2, hop=hop2, analysis_window=window2, synthesis_window=window2,channels=n_chans2)
    mySTFT3 = pra.transform.stft.STFT(N=N3, hop=hop3, analysis_window=window3, synthesis_window=window3,channels=n_chans3)

    # Create room.
    room1 = create_room(room_dim)

    # Create beamformer.
    beamformer = pra.Beamformer(R, room1.fs)
    # Add beamformer to room.
    add_mics(room1,beamformer)

    # Add source(s).
    add_sources(room1,source_locs,audio2, delay)

    # Compute STFT, iSTFT and corresponding tine values.
    dft1 = mySTFT1.analysis(audio1)
    idft1 = mySTFT1.synthesis(dft1)
    t_idft_1 = np.arange(0, len(idft1))/fs1

    dft2 = mySTFT2.analysis(audio2)
    idft2 = mySTFT2.synthesis(dft2)
    t_idft_2 = np.arange(0, len(idft2))/fs2

    dft3 = mySTFT3.analysis(audio3)
    idft3 = mySTFT3.synthesis(dft3)
    t_idft_3 = np.arange(0, len(idft3)) / fs3

    # Write to disk.
    #sf.write("audio1.wav",idft1,fs1)
    #sf.write("audio2.wav",idft2, fs2)

    # Plot signals.
    #plot_signal(audio1,t1,"audio1")
    #plot_signal(idft1, t_idft_1, "audio1 synthesis")
    #plot_signal(audio2,t2,"audio2")
    #plot_signal(idft2, t_idft_2, "audio2 synthesis")
    #plot_signal(audio3, t3, "audio3")
    #plot_signal(idft3, t_idft_3, "audio3 synthesis")

    # Now compute the delay and sum weights for the beamformer
    room1.mic_array.rake_delay_and_sum_weights(room1.sources[0][:1])

    beamformer.plot()

    # plot the room and resulting beamformer
    room1.plot(freq=[1000, 2000, 4000, 8000], img_order=0)
    plt.show()

if __name__ == "__main__":
    main()

