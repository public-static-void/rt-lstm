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
from scipy.signal import stft
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



def main():

    # Define sample path.
    sample_path2 = "./Samples/examples_input_samples_german_speech_8000.wav"
    sample_path3 = "./Samples/whitenoise.wav"

    # Read samples from disk.
    audio2, fs2 = sf.read(sample_path2)
    audio3, fs3 = sf.read(sample_path3)

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
    t2 = np.arange(0, len(audio2))/fs2
    t3 = np.arange(0, len(audio3))/fs3

    # Frame length in ms.
    fl = 32

    # Transform frame length from ms to samples.
    N2 = int(fs2*fl/1000)
    N3 = int(fs3*fl/1000)

    # Specify frame shift length.
    hop2 = int(N2/2)
    hop3 = int(N3/2)

    # Define windowing functions.
    window2 = np.sqrt(get_window('hann', N2, fftbins=True))
    window3 = np.sqrt(get_window('hann', N3, fftbins=True))

    # Specify delay.
    delay = 1.3

    # Create room.
    room1 = create_room(room_dim, rt60)
    room2 = create_room(room_dim, rt60)

    # Create beamformer.
    beamformer = pra.Beamformer(R, room1.fs)
    # Add beamformer to room.
    add_mics(room1,beamformer)
    add_mics(room2,beamformer)

    # Add source(s).
    add_sources(room1, source_locs1,audio2, delay)
    add_sources(room1, source_locs2, audio3, delay)

    add_sources(room2, source_locs3, audio2, delay)
    add_sources(room2, source_locs4, audio3, delay)

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

    #sf.write("premix2noise.wav",premix2[1].T,fs3)
    #sf.write("premix2speech.wav", premix2[0].T, fs3)

    # pyroomaccoustics dft? stft:
    # (10, 706, 2)

    # scipy stft:
    # (129,)
    # (454,)
    # (4, 129, 454)

    stft2noise = stft(premix1[1],fs2,window=window2)
    #print(stft2noise[0].shape) # f: ndarray Array of sample frequencies.
    #print(stft2noise[1].shape) # t: ndarray Array of segment times.
    #print(stft2noise[2].shape) # Zxx: ndarray STFT of x.

    stft2speech = stft(premix1[0],fs2,window=window2)
    #print(stft2speech[0].shape)
    #print(stft2speech[1].shape)
    #print(stft2speech[2].shape)

    # TODO:
    # covariance matrix
    alpha = 0.8
    init_matrix = np.zeros((4,4))
    #print(init_matrix)
    #times = stft2noise[2][:,t,:]
    #freqs = stft2noise[2][:,:,f]
    #print(times.shape)
    #print(freqs)

    # TODO: loop over array
    # TODO: zu jeder Frequenz und zu jedem Zeitpunkt neue Matrix
    # TODO: zu jeder Frequenz eine initiale Matrix
    # TODO: -->iterativ<-- (in Abhängigkeit von der Zeit) zu jeder Frequenz und zu jedem Zeitpunkt eine Kovarianz?Matrix ausrechnen
    for t in stft2noise:
        for f in t:
                #print(x)
            pass
            #init_matrix = init_matrix + f(1 - alpha * (?))







    # TESTs
    # zum überprüfen der implementation
    #
    #covMatrix = np.cov(data, bias=True)
    # sample covariance (based on N-1)
    #covMatrix = np.cov(data, bias=False)

    #output = beamformer.process()
    #sf.write("output.wav", output, fs3)

    #room1.mic_array.to_wav(filename="room1.wav", mono=False)
    #room2.mic_array.to_wav(filename="room2.wav", mono=False)

    #plt.show()

if __name__ == "__main__":
    main()

