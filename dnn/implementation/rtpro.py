#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Authors       : Vadim Titov, Henning Möllers
Matr.-Nr.     : 6021356, ...
Created       : January 19th, 2023
Last modified : January 19th, 2023
Description   : Master's Project "Source Separation for Robot Control"
Topic         : Real-time audio processing module of the LSTM RNN Project
"""

import os
import sys
import threading
import traceback

import hyperparameters as hp
import jack
import numpy as np
import torch
from dsp_utils import (
    get_butter_coeffs,
    get_periodic_hann,
    get_windowed_irfft,
    get_windowed_rfft,
)
from net import LitNeuralNet
from scipy import signal

# #############################
# DNN model configuration
# #############################

# Load pretrained model from checkpoint.
trained_model = LitNeuralNet.load_from_checkpoint(
    checkpoint_path=hp.trained_model_path, batch_size=1,
).to(hp.device)
trained_model.eval()
trained_model.freeze()
# Init hidden and cell state of time-dimension LSTM.
h_t = None
c_t = None

# #############################
# Configure DSP settings
# #############################

INPUT_FS = 48000
PROC_FS = 16000
FFT_LEN = 512  # 32 ms
FFT_SHIFT = 128  # 8 ms
WIN_SCALE = 2  # due to 75% overlap
LP_ORDER = 16
BLOCK_LEN = 384

ds_factor = int(INPUT_FS / PROC_FS)

NUM_IN_CHANNELS = 3
NUM_OUT_CHANNELS = 2

# #############################
# Set-up jack client
# #############################

argv = iter(sys.argv)
# By default, use script name without extension as client name:
defaultclientname = os.path.splitext(os.path.basename(next(argv)))[0]
clientname = next(argv, defaultclientname)
servername = next(argv, None)
event = threading.Event()

try:
    client = jack.Client(clientname, servername=servername)
except jack.JackError:
    sys.exit("JACK server not running?")

if client.status.server_started:
    print("JACK server started")
if client.status.name_not_unique:
    print(f"unique name {client.name!r} assigned")

# ###############################
# Global variables
# ###############################

block_in_buffer = np.zeros((NUM_IN_CHANNELS, BLOCK_LEN))
block_out_buffer = np.zeros(BLOCK_LEN)
fft_buffer = np.zeros((NUM_IN_CHANNELS, FFT_LEN))
overlap_add_buffer = np.zeros(FFT_LEN)
b, a = get_butter_coeffs(ds_factor, LP_ORDER)
filter_states_lp_down_sample = np.zeros((NUM_IN_CHANNELS, LP_ORDER))
filter_states_lp_up_sample = np.zeros(LP_ORDER)
window = get_periodic_hann(FFT_LEN)

# ###############################
#  Processing
# ###############################


def net_processing(
    fft_stack: torch.Tensor, h_t: torch.Tensor, c_t: torch.Tensor
) -> tuple:
    """Helper function.

    Performs net processing.

    Parameters
    ----------
    fft_stack : torch.Tensor

    h_t : torch.Tensor

    c_t : torch.Tensor


    Returns
    -------
    tuple
        net_output, h_t, c_t

    """
    # Split imaginary and real parts of complex fft.
    fft_split = torch.cat(
        (torch.real(fft_stack), torch.imag(fft_stack)), dim=0)
    # Add dummy batch and time dimensions.
    net_input = fft_split[None, :, :, None]
    # Net processing:
    net_output, _, h_t, c_t = trained_model.predict_rt(
        batch=net_input, h_pre=h_t, c_pre=c_t
    )
    net_output = net_output[0, :, 0]
    return net_output, h_t, c_t


def block_processing(input_buffer):
    global fft_buffer
    global overlap_add_buffer
    global h_t
    global c_t

    # Compute fft
    fft_buffer[:, :-FFT_SHIFT] = fft_buffer[:, FFT_SHIFT:]
    fft_buffer[:, -FFT_SHIFT:] = input_buffer
    fft_data = get_windowed_rfft(
        np.ascontiguousarray(fft_buffer), window, FFT_LEN
    )

    # Perform speech enhancement in the frequency domain
    signal, h_t, c_t = net_processing(torch.from_numpy(fft_data), h_t, c_t)
    signal = signal.to("cpu")
    # Overlap-add
    overlap_add_buffer += (
        get_windowed_irfft(signal, window, FFT_LEN) / WIN_SCALE
    )
    output_signal = overlap_add_buffer[:FFT_SHIFT]
    overlap_add_buffer[:-FFT_SHIFT] = overlap_add_buffer[FFT_SHIFT:]
    overlap_add_buffer[-FFT_SHIFT:] = 0.0

    return output_signal


@client.set_process_callback
def process(frames):
    global block_in_buffer
    global block_out_buffer
    global filter_states_lp_down_sample
    global filter_states_lp_up_sample

    # Collect data in block_buffer
    for i in range(NUM_IN_CHANNELS):
        block_in_buffer[i] = client.inports[i].get_array()

    # Down-sample from INPUT_FS to PROC_FS
    lp_block_buffer, filter_states_lp_down_sample = signal.lfilter(
        b, a, block_in_buffer, axis=-1, zi=filter_states_lp_down_sample
    )  # [D, T]
    ds_block_buffer = lp_block_buffer[:, ::ds_factor]  # [D, T']

    # Perform enhancement
    enhanced_signal = block_processing(ds_block_buffer)

    # Upsample from PROC_FS to INPUT_FS
    block_out_buffer[...] = 0
    block_out_buffer[::ds_factor] = enhanced_signal
    block_out_buffer, filter_states_lp_up_sample = signal.lfilter(
        ds_factor * b,
        a,
        block_out_buffer,
        axis=-1,
        zi=filter_states_lp_up_sample,
    )

    # Write stream to speaker(s).
    for oc in range(NUM_OUT_CHANNELS):
        client.outports[oc].get_array()[:] = block_out_buffer


@client.set_shutdown_callback
def shutdown(status, reason):
    print("JACK shutdown! status:", status, " reason:", reason)
    event.set()


def main():
    """Main function.

    Continuously reads data from input stream, processes it using a
    pretrained neural net in order to perform noise reduction and writes
    the results to an output stream.
    """

    for number in range(NUM_IN_CHANNELS):
        client.inports.register(f"input_{number}")
    for number in range(NUM_OUT_CHANNELS):
        client.outports.register(f"output_{number}")

    print("activating JACK")

    with client:
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


if __name__ == "__main__":
    main()
