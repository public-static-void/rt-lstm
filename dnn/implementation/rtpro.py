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

from typing import Generator

import torch
import torchaudio
from torchaudio.io import StreamReader

# Sampling frequency
FS = 48000
# Chunk size in ms.
CHUNK_SIZE_MS = 8
# Block size in ms.
BLOCK_SIZE_MS = 32
# Chunk size in samples.
CHUNK_SIZE = int(FS * CHUNK_SIZE_MS / 1000)
# Block size in samples.
BLOCK_SIZE = int(FS * BLOCK_SIZE_MS / 1000)
# Number of channels. TODO: is there a way to change it in torchaudio?
CHANNELS = 1
# Hardware source device identifier.
SRC = "hw:1"
FORMAT = "alsa"


def init_streamer() -> StreamReader:
    """Helper function.

    Initializes and returns an instance of `torchaudio.io.StreamReader`.

    Returns
    -------
        Initialized instance of `torchaudio.io.StreamReader`.
    """
    streamer = StreamReader(src=SRC, format=FORMAT)
    return streamer


def start_stream(streamer: StreamReader) -> Generator[bytes, None, None]:
    """Helper function.

    Opens a stream.

    Returns
    -------
    Generator
        Opened stream.
    """
    streamer.add_basic_audio_stream(
        frames_per_chunk=CHUNK_SIZE,
        buffer_chunk_size=CHUNK_SIZE,
        sample_rate=FS,
    )
    # print(streamer.get_src_stream_info(0))
    # print(streamer.get_out_stream_info(0))
    stream_iteartor = streamer.stream(timeout=-1, backoff=1.0)
    return stream_iteartor


def stop_stream(streamer: StreamReader, idx: int) -> None:
    """Helper function.

    Closes the `stream` while keeping StreamReader instance alive.

    streamer : StreamReader
        StreamReader where thr stream to close is attached to.
    idx : int
        Index of the stream to close.
    """
    streamer.remove_stream(idx)


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


def read_chunk(stream_iterator: Generator[bytes, None, None]) -> bytes:
    """Helper function.

    Reads a chunk from `stream`.

    stream : Generator
        `StreamReader` generator yielding `bytes`.

    Returns
    -------
    bytes
        Chunk of data.
    """
    data = next(stream_iterator)
    # Data comes as a list containing a tensor, so extract it.
    data_tensor = data[0]
    # Then, multiple channels are present, so just extract the first one.
    chunk = data_tensor[:, 0]
    return chunk


def add_chunk(
    block: torch.Tensor, stream: Generator[bytes, None, None]
) -> torch.Tensor:
    """Helper function.

    Reads a `chunk` from the `stream` and adds it to the `block`. Shifts
    previous content to the left and only overwrites the right with new chunk.

    block : torch.Tensor
        Block of multiple chunks but of constant size.
    chunk : torch.Tensor
        Chunk of new data from the `stream`.
    stream : Generator
        `StreamReader` generator to read a `chunk` of data from.

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
    streamer = init_streamer()
    stream = start_stream(streamer)
    i = 0
    try:
        while True:
            block = add_chunk(block, stream)
            print("input CHUNK:", i, block)
            block_fft = compute_FFT(block)
            print("CHUNK FFT:", i, block_fft)
            i += 1
    except:
        print("error/interrupt")
        stop_stream(streamer, streamer.default_audio_stream)
        return


if __name__ == "__main__":
    main()
