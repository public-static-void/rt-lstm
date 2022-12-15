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

from scipy.signal import get_window
import hyperparameters
import numpy as np
import torch
import torchaudio
from torchaudio.io import StreamReader

# Input device sampling frequency in Hz.
INPUT_FS = 48000
# Processing sampling frequency in Hz.
PROCESSING_FS = 16000
# Downsampling factor.
DSF = int(INPUT_FS / PROCESSING_FS)
# Chunk size in ms.
CHUNK_SIZE_MS = 8
# Block size in ms.
BLOCK_SIZE_MS = 32
# Input chunk size in samples.
IN_CHUNK_SIZE = int(INPUT_FS * CHUNK_SIZE_MS / 1000)
# Input cock size in samples.
IN_BLOCK_SIZE = int(INPUT_FS * BLOCK_SIZE_MS / 1000)
# Processing chunk size in samples.
PRO_CHUNK_SIZE = int(PROCESSING_FS * CHUNK_SIZE_MS / 1000)
# Processing block size in samples.
PRO_BLOCK_SIZE = int(PROCESSING_FS * BLOCK_SIZE_MS / 1000)
# Hardware source device identifier.
SRC = "hw:1"
FORMAT = "alsa"
# Window scaling factor.
WSF = 2  # 75% overlap.


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
        frames_per_chunk=IN_CHUNK_SIZE,
        buffer_chunk_size=IN_CHUNK_SIZE,
        sample_rate=INPUT_FS,
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


def resample(input: torch.Tensor, factor: float) -> torch.Tensor:
    """Helper function.

    Resamples input signal with provided resampling factor.

    input : torch.Tensor
        Input signal.
    factor : float
        Resampling factor.

    Returns
    -------
    torch.Tensor
        Resampled signal.
    """
    out = input[::factor]
    return out


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
    # Resample.
    rs_chunk = resample(chunk, DSF)
    return rs_chunk


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
    new_block[:-PRO_CHUNK_SIZE] = block[PRO_CHUNK_SIZE:]
    new_block[-PRO_CHUNK_SIZE:] = read_chunk(stream)
    return new_block


def overlap_add(
    block_old: torch.Tensor, block_new: torch.Tensor
) -> torch.Tensor:
    # Overlap add.
    out = block_old + block_new
    # Rescale overlapping part.
    out = out[:PRO_CHUNK_SIZE] / WSF


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


def compute_IFFT_from_block(
    block: torch.Tensor,
) -> torch.Tensor:
    """Helper function.

    Computes the IFFT of a given frequency domain signal `block`.

    block : torch.Tensor
        Block of frequency domain signal.

    Returns
    -------
    torch.Tensor
        IFFT of the given `block`.
    """
    block_ifft = torch.fft.ifft(block)
    return block_ifft


def apply_window_on_block(block: torch.Tensor) -> torch.Tensor:
    window = torch.from_numpy(
        np.sqrt(
            get_window(
                "hann", hyperparameters.stft_length, hyperparameters.fftbins
            )
        )
    )
    windowed_block = block * window

    return windowed_block


def get_overlapping_chunk_sum_from_blocks(
    block1: torch.Tensor,
    block2: torch.Tensor,
    block3: torch.Tensor,
    block4: torch.Tensor,
) -> torch.Tensor:
    # 1 block consists of 4 chunks
    chunk_sum = (
        block1[:PRO_CHUNK_SIZE]
        + block2[PRO_CHUNK_SIZE: PRO_CHUNK_SIZE * 2]
        + block3[PRO_CHUNK_SIZE * 2: PRO_CHUNK_SIZE * 3]
        + block4[PRO_CHUNK_SIZE * 3: PRO_CHUNK_SIZE * 4]
    )
    # Rescale overlapping part.
    return chunk_sum


def remove_first_block_and_reorder(blocks_cache: []):
    blocks_cache[0] = blocks_cache[1]
    blocks_cache[1] = blocks_cache[2]
    blocks_cache[2] = blocks_cache[3]
    # last index, 3 can now be overwritten by new block


def main():
    block = init_block(PRO_BLOCK_SIZE)
    streamer = init_streamer()
    stream = start_stream(streamer)
    i = 0
    try:
        # intialize blocks list for 4 blocks with zero-tensors of corresponding shapes (see net output)
        blocks_queue = [
            torch.zeros(1, 1, 1),
            torch.zeros(1, 1, 1),
            torch.zeros(1, 1, 1),
            torch.zeros(1, 1, 1),
        ]
        while True:
            block = add_chunk(block, stream)
            print("input CHUNK:", i, block)
            windowed_new_block_before_FFT = apply_window_on_block(block)
            block_fft = compute_FFT(windowed_new_block_before_FFT)
            print("BLOCK FFT:", i, block_fft)
            # TODO net processing
            net_output = block_fft

            ifft_block = compute_IFFT_from_block(net_output)
            windowed_new_block_after_net = apply_window_on_block(ifft_block)
            blocks_queue = remove_first_block_and_reorder(blocks_queue)
            blocks_queue[3] = windowed_new_block_after_net
            overlapping_chunk = get_overlapping_chunk_sum_from_blocks(
                blocks_queue
            )
            print(overlapping_chunk)
            # output streaming
            print(i)
            i += 1
    except:
        print("error/interrupt")
        stop_stream(streamer, streamer.default_audio_stream)
        return


if __name__ == "__main__":
    main()
