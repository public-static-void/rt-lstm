#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Authors       : Vadim Titov
Matr.-Nr.     : 6021356
Created       : December 14th, 2022
Last modified : December 14th, 2022
Description   : Master's Project "Source Separation for Robot Control"
Topic         : Real-time audio processing module of the LSTM RNN Project
"""


import pyaudio
import torch

# Sampling frequency
FS = 44100
# Chunk size in ms.
CHUNK_SIZE_MS = 8
# Block size in ms.
BLOCK_SIZE_MS = 32
# Chunk size in samples.
CHUNK_SIZE = int(FS * CHUNK_SIZE_MS / 1000)
# Block size in samples.
BLOCK_SIZE = int(FS * BLOCK_SIZE_MS / 1000)
# Number of channels.
CHANNELS = 1


PA = pyaudio.PyAudio()


def start_stream() -> pyaudio.Stream:
    """Helper function.

    Opens a stream.

    Returns
    -------
    pyaudio.Stream
        Opened stream.
    """
    stream = PA.open(
        # FIXME: For some reason overflows quickly from time to time.
        format=pyaudio.paFloat32,
        channels=CHANNELS,
        rate=FS,
        input=True,
        frames_per_buffer=CHUNK_SIZE,
    )
    return stream


def stop_stream(stream: pyaudio.Stream) -> None:
    """Helper function.

    Closes the `stream` while keeping PyAudio instance alive.

    stream : pyaudio.Stream
        Stream to close.
    """
    stream.stop_stream()
    stream.close()


def close_pa() -> None:
    """Helper function.

    Terminates the PyAudio instance.
    """
    PA.terminate()


def init_block(block_size: int) -> torch.Tensor:
    """Helper function.

    Initializes an empty block with zeros to store chunks of stream data in.

    block_size : int
        Size of the block.

    Returns
    -------
    torch.Tensor
        Initialized block.
    """
    block = torch.zeros(block_size)
    return block


def read_chunk(stream: pyaudio.Stream) -> bytes:
    """Helper function.

    Reads a chunk from `stream`.

    stream : pyaudio.Stream


    Returns
    -------
    bytes

    """
    data = stream.read(CHUNK_SIZE)

    chunk = torch.frombuffer(data, dtype=torch.float32)
    return chunk


def add_chunk(block: torch.Tensor, stream: pyaudio.Stream) -> torch.Tensor:
    """Helper function.

    Reads a `chunk` from the `stream` and adds it to the `block`. Shifts
    previous content to the left and only overwrites the right with new chunk.

    block : torch.Tensor
        Block of multiple chunks but of constant size.
    chunk : torch.Tensor
        Chunk of new data from the `stream`.
    stream : pyaudio.Stream
        Stream to read a `chunk` of data from.

    Returns
    -------
    torch.Tensor
        Block with new `chunk` of data.
    """
    new_block = torch.clone(block)
    new_block[:-CHUNK_SIZE] = block[CHUNK_SIZE:]
    new_block[-CHUNK_SIZE:] = read_chunk(stream)
    return new_block


def compute_FFT(block: torch.Tensor) -> torch.Tensor:
    """Helper function.

    Computes the FFT of a given time domain signal `block`.

    block : torch.Tensor
        Block of time domain signal.

    Returns
    -------
    torch.Tensor
        FFT of the given `block`.
    """
    block_fft = torch.fft.fft(block)
    return block_fft


def main():
    block = init_block(BLOCK_SIZE)
    stream = start_stream()

    try:
        while True:
            block = add_chunk(block, stream)
            print(block)
            block_fft = compute_FFT(block)
            print(block_fft)

    except:
        print("error/interrupt")
        stop_stream(stream)
        close_pa()
        return


if __name__ == "__main__":
    main()
