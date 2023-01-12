#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Authors       : Vadim Titov, Henning Möllers
Matr.-Nr.     : 6021356, ...
Created       : January 6th, 2023
Last modified : January 6th, 2023
Description   : Master's Project "Source Separation for Robot Control"
Topic         : Real-time audio processing module of the LSTM RNN Project
"""

import os
import sys
import threading
import traceback
from typing import List

import hyperparameters as hp
import jack
import numpy as np
import torch
from net import LitNeuralNet
from scipy.signal import get_window
from torchaudio.transforms import Resample

# NOTE: Set qjackctl to 48000 fs, 384 frames and 4 buffer periods.
# NOTE: Make sure batch size is set to 1 in hyperparameters.

# Processing sampling frequency in Hz.
PROCESSING_FS = hp.fs  # 16000
# Chunk size in ms.
CHUNK_SIZE_MS = 8
# Block size in ms.
BLOCK_SIZE_MS = 32
# Processing chunk size in samples.
PRO_CHUNK_SIZE = int(PROCESSING_FS * CHUNK_SIZE_MS / 1000)
# Processing block size in samples.
PRO_BLOCK_SIZE = int(PROCESSING_FS * BLOCK_SIZE_MS / 1000)

NUM_IN_CHANNELS = 2
NUM_OUT_CHANNELS = 1

# Init empty block.
block = torch.zeros(PRO_BLOCK_SIZE)
blocks_queue = [
    torch.zeros(hp.stft_length),
    torch.zeros(hp.stft_length),
    torch.zeros(hp.stft_length),
    torch.zeros(hp.stft_length),
]
WINDOW = torch.from_numpy(
    np.sqrt(get_window("hann", hp.stft_length, hp.fftbins))
)
trained_model = LitNeuralNet.load_from_checkpoint(
    checkpoint_path=hp.trained_model_path
)
trained_model.eval()
trained_model.freeze()
# Init hidden and cell state of time-dimension LSTM.
h_t = None
c_t = None


def add_chunk(block: torch.Tensor, chunk: torch.Tensor) -> torch.Tensor:
    """Helper function.

    Reads a `chunk` from the `stream` and adds it to the `block`. Shifts
    previous content to the left and only overwrites the right with new chunk.

    block : torch.Tensor
        Block of multiple chunks but of constant size.
    chunk : torch.Tensor
        Chunk of new data from the `stream`.

    Returns
    -------
    torch.Tensor
        Block with new `chunk` of data.
    """
    new_block = torch.clone(block)
    new_block[:-PRO_CHUNK_SIZE] = block[PRO_CHUNK_SIZE:]
    new_block[-PRO_CHUNK_SIZE:] = chunk
    return new_block


def apply_window_on_block(block: torch.Tensor) -> torch.Tensor:
    """Helper function.

    Applies a window to a signal tensor.

    block : torch.Tensor
        Time domain signal tensor.

    Returns
    -------
    torch.Tensor
        Windowed tensor.
    """
    windowed_block = block * WINDOW

    return windowed_block


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
    block_fft = torch.fft.rfft(block)
    return block_fft


def compute_IFFT_from_block(block: torch.Tensor) -> torch.Tensor:
    """Helper function.

    Computes the IFFT of a given frequency domain signal `block`.

    block : torch.Tensor
        Block of frequency domain signal.

    Returns
    -------
    torch.Tensor
        IFFT of the given `block`.
    """
    block_ifft = torch.fft.irfft(block)
    return block_ifft


def get_overlapping_chunk_sum_from_blocks(blocks_queue: List) -> torch.Tensor:
    """Helper function.

    Sums up overlapping  chunks of blocks.

    blocks_queue : List
        List of blocks.

    Returns
    -------
    torch.Tensor
        Summed up chunks.
    """
    # 1 block consists of 4 chunks
    block1 = blocks_queue[0]
    block2 = blocks_queue[1]
    block3 = blocks_queue[2]
    block4 = blocks_queue[3]
    chunk_sum = (
        block1[:PRO_CHUNK_SIZE]
        + block2[PRO_CHUNK_SIZE: PRO_CHUNK_SIZE * 2]
        + block3[PRO_CHUNK_SIZE * 2: PRO_CHUNK_SIZE * 3]
        + block4[PRO_CHUNK_SIZE * 3: PRO_CHUNK_SIZE * 4]
    )
    # Rescale overlapping part.
    return chunk_sum


def remove_first_block_and_reorder(blocks_cache: List) -> List:
    """Helper function.

    Shifts list of blocks by one block to the left.

    blocks_cache : List
        Input list of blocks.

    Returns
    -------
    List
        Output list of blocks.
    """
    blocks_cache[0] = blocks_cache[1]
    blocks_cache[1] = blocks_cache[2]
    blocks_cache[2] = blocks_cache[3]
    # last index, 3 can now be overwritten by new block
    return blocks_cache


def main():
    """Main function.

    Continuously reads data from input stream, processes it using a
    pretrained neural net in order to perform noise reduction and writes
    the results to an output stream.
    """

    argv = iter(sys.argv)
    # By default, use script name without extension as client name:
    defaultclientname = os.path.splitext(os.path.basename(next(argv)))[0]
    clientname = next(argv, defaultclientname)
    servername = next(argv, None)

    try:
        client = jack.Client(clientname, servername=servername)
    except jack.JackError:
        sys.exit("JACK server not running?")

    if client.status.server_started:
        print("JACK server started")
    if client.status.name_not_unique:
        print(f"unique name {client.name!r} assigned")

    downsample = Resample(
        client.samplerate, PROCESSING_FS, dtype=torch.float32)
    upsample = Resample(PROCESSING_FS, client.samplerate, dtype=torch.double)

    event = threading.Event()

    @client.set_process_callback
    def process(frames):
        global block
        global blocks_queue
        global h_t
        global c_t

        assert frames == client.blocksize

        # Read stream from microphone(s).
        in_channels = []
        for ic in range(0, NUM_IN_CHANNELS):
            input_buffer = client.inports[ic].get_array()
            in_channels.append(input_buffer)

        # Perform signal processing magic.
        # Read chunk from stream and resample it to match processing fs.
        chunk = torch.from_numpy(in_channels[1])
        # chunk = downsample(chunk)
        # # Add chunk to block.
        # block = add_chunk(block, chunk)
        # windowed_new_block_before_FFT = apply_window_on_block(block)
        # block_fft = compute_FFT(windowed_new_block_before_FFT)
        # # Simulate 3 channels.
        # fft_stack = torch.stack((block_fft, block_fft, block_fft), dim=0)
        # # Split imaginary and real parts of complex fft.
        # fft_split = torch.cat(
        #     (torch.real(fft_stack), torch.imag(fft_stack)), dim=0
        # )
        # # Add dummy batch and time dimensions.
        # net_input = fft_split[None, :, :, None]
        # # Net processing:
        # net_output, _, h_t, c_t = trained_model.predict_rt(
        #     batch=net_input, h_pre=h_t, c_pre=c_t
        # )
        # net_output = net_output[0, :, 0]
        # ifft_block = compute_IFFT_from_block(net_output)
        # windowed_new_block_after_net = apply_window_on_block(ifft_block)
        # blocks_queue = remove_first_block_and_reorder(blocks_queue)
        # blocks_queue[3] = windowed_new_block_after_net
        # overlapping_chunk = get_overlapping_chunk_sum_from_blocks(blocks_queue)
        # print(overlapping_chunk)
        # # Output streaming.
        # resampled_chunk = upsample(overlapping_chunk)
        # resampled_chunk.to(dtype=torch.int16)
        # output_buffer = resampled_chunk
        # print(output_buffer)

        output_buffer = chunk

        # Write stream to speaker(s).
        for oc in range(0, NUM_OUT_CHANNELS):
            client.outports[oc].get_array()[:] = output_buffer.numpy()

    @client.set_shutdown_callback
    def shutdown(status, reason):
        print("JACK shutdown!")
        print("status:", status)
        print("reason:", reason)
        event.set()

    # create two port pairs
    for number in 1, 2:
        client.inports.register(f"input_{number}")
        client.outports.register(f"output_{number}")

    with client:
        # When entering this with-statement, client.activate() is called.
        # This tells the JACK server that we are ready to roll.
        # Our process() callback will start running now.

        # Connect the ports.  You can't do this before the client is activated,
        # because we can't make connections to clients that aren't running.
        # Note the confusing (but necessary) orientation of the driver backend
        # ports: playback ports are "input" to the backend, and capture ports
        # are "output" from it.

        capture = client.get_ports(is_physical=True, is_output=True)
        if not capture:
            raise RuntimeError("No physical capture ports")

        for src, dest in zip(capture, client.inports):
            client.connect(src, dest)

        playback = client.get_ports(is_physical=True, is_input=True)
        if not playback:
            raise RuntimeError("No physical playback ports")

        for src, dest in zip(client.outports, playback):
            client.connect(src, dest)

        print("Press Ctrl+C to stop")

        try:
            while True:
                event.wait()

        except KeyboardInterrupt:
            print("\nInterrupted by user")
        except Exception:
            print(traceback.format_exc())

    # When the above with-statement is left (either because the end of the
    # code block is reached, or because an exception was raised inside),
    # client.deactivate() and client.close() are called automatically.


if __name__ == "__main__":
    main()
