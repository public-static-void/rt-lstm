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
from scipy.io import wavfile
from scipy.signal import get_window
import matplotlib.pyplot as plt
import soundfile as sf

def create_room(rt60:float,room_dim:np.ndarray,fs:int) -> pra.room:
    """Helper function. Creates a room from input parameters and returns it.

    :param rt60: float
    :param room_dim: np.ndarray
    :param fs: int
    :return: pra.room
    """
    # RT60: the time it takes for the RIR to decays by 60 dB.
    # We invert Sabine's formula to obtain the parameters for the ISM simulator.
    e_absorption, max_order = pra.inverse_sabine(rt60, room_dim)

    # Create the room.
    room = pra.ShoeBox(
        room_dim, fs=fs, materials=pra.Material(e_absorption), max_order=max_order
    )
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

def add_mics(room:pra.room,mic_locs:np.ndarray):
    """Helper function. Adds mics to an input room from input parameters.

    :param room: pra.room
    :param mic_locs: np.ndarray
    :return: None
    """
    # define the locations of the microphones.
    # finally place the array in the room.
    room.add_microphone_array(mic_locs)



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
    room_dim = [9, 7.5, 3.5]

    # Define mic locations.
    mic_locs = np.c_[
        [6.3, 4.87, 1.2],  # mic 1
        [6.3, 4.93, 1.2],  # mic 2
    ]
    # Define source locations.
    source_locs = [2.5, 3.73, 1.76]

    # Define sample path.
    sample_path1 = "./Samples/2213f081fd811e304423f283850b-orig.wav"
    sample_path2 = "./Samples/examples_input_samples_german_speech_8000.wav"



    #fs1, audio1 = wavfile.read(sample_path1)
    #fs2, audio2 = wavfile.read(sample_path2)

    audio1,fs1 = sf.read(sample_path1)
    audio2,fs2 = sf.read(sample_path2)


    #print(audio1.dtype)

    # Create time domain values.
    t1 = np.arange(0, len(audio1))/fs1
    t2 = np.arange(0, len(audio2))/fs2



    # Frame length in ms.
    fl = 32

    # Transform frame length from ms to samples.
    N1 = int(fs1*fl/1000)
    N2 = int(fs2*fl/1000)


    hop1 = int(N1/2)
    hop2 = int(N2/2)

    window1 = np.sqrt(get_window('hann', N1, fftbins=True))
    window2 = np.sqrt(get_window('hann', N2, fftbins=True))

    # Specify number of channels.
    n_chans1 = 2
    n_chans2 = 1

    mySTFT1 = pra.transform.stft.STFT(N=N1,hop=hop1,analysis_window=window1,synthesis_window=window1,channels=n_chans1)
    mySTFT2 = pra.transform.stft.STFT(N=N2, hop=hop2, analysis_window=window2, synthesis_window=window2,channels=n_chans2)

    # Specify delay.
    delay = 1.3

    # Create room.
    room1 = create_room(rt60,room_dim,fs)
    # Add mics.
    add_mics(room1,mic_locs)
    # Add source(s).
    add_sources(room1,source_locs,audio2, delay)


    dft1 = mySTFT1.analysis(audio1)
    idft1 = mySTFT1.synthesis(dft1)
    t_idft_1 = np.arange(0, len(idft1))/fs1

    plot_signal(audio1,t1,"audio1")
    plot_signal(idft1, t_idft_1, "audio1 synthesis")


    dft2 = mySTFT2.analysis(audio2)
    idft2 = mySTFT2.synthesis(dft2)
    t_idft_2 = np.arange(0, len(idft2))/fs2

    plot_signal(audio2,t2,"audio2")
    plot_signal(idft2, t_idft_2, "audio2 synthesis")

    #print(idft1.dtype)

    sf.write("audio1.wav",idft1,fs1)
    sf.write("audio2.wav",idft2, fs2)


    plt.show()


if __name__ == "__main__":
    main()

