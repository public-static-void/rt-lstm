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

def add_sources(room:pra.room, source_locs:np.ndarray, audio_path:str):

    _, audio = wavfile.read(audio_path)
    # place the source in the room
    room.add_source(source_locs, signal=audio, delay=1.3)

def add_mics(room:pra.room,mic_locs:np.ndarray):
    """Helper function. Adds mics to an input room from input parameters.

    :param room: pra.room
    :param mic_locs: np.ndarray
    :return: None
    """
    # define the locations of the microphones.
    # finally place the array in the room.
    room.add_microphone_array(mic_locs)






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

    # Create room.
    room1 = create_room(rt60,room_dim,fs)
    # Add mics.
    add_mics(room1,mic_locs)
    # Add source(s).
    add_sources(room1,'speech.wav')



if __name__ == "__main__":
    main()

