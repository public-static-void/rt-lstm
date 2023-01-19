import numpy
import numpy as np
from scipy import signal
import jack
import torch
from net import LitNeuralNet
import hyperparameters as hp
#import threading

from dsp_utils import get_butter_coeffs, get_periodic_hann, get_windowed_rfft, get_windowed_irfft
# from nn_model import get_model, perform_enhancement

# #############################
# DNN model configuration
# #############################
# config_file = 'config.yaml'
# model = get_model(config_file)


# #############################
# Configure DSP settings
# #############################

INPUT_FS = 48000
PROC_FS = 16000
N_CHANNEL = 3
FFT_LEN = 512       # 32 ms
FFT_SHIFT = 128     # 8 ms
WIN_SCALE = 2       # due to 75% overlap
LP_ORDER = 16
BLOCK_LEN = 384

ds_factor = int(INPUT_FS/PROC_FS)
fft_bins = int(FFT_LEN/2 + 1)

trained_model = LitNeuralNet.load_from_checkpoint(
    checkpoint_path=hp.trained_model_path
)
trained_model.eval()
trained_model.freeze()
# Init hidden and cell state of time-dimension LSTM.
h_t = None
c_t = None

# #############################
# Set-up jack client
# #############################
client = jack.Client("MCSpeechEnhancement")

if client.status.server_started:
    print("JACK server started")
if client.status.name_not_unique:
    print("unique name {0!r} assigned".format(client.name))
#event = threading.Event()


# ###############################
# Global variables
# ###############################

block_in_buffer = np.zeros((N_CHANNEL, BLOCK_LEN))
block_out_buffer = np.zeros(BLOCK_LEN)
fft_buffer = np.zeros((N_CHANNEL, FFT_LEN))
overlap_add_buffer = np.zeros(FFT_LEN)
b, a = get_butter_coeffs(ds_factor, LP_ORDER)
filter_states_lp_down_sample = np.zeros((N_CHANNEL, LP_ORDER))
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
    fft_data = get_windowed_rfft(np.ascontiguousarray(fft_buffer), window, FFT_LEN)

    # Perform speech enhancement in the frequency domain
    # signal = perform_enhancement(model, fft_data, ref_channel=0)
    # signal = fft_data[0]
    signal, h_t, c_t = net_processing(torch.from_numpy(fft_data), h_t, c_t)

    # Overlap-add
    overlap_add_buffer += get_windowed_irfft(signal, window, FFT_LEN) / WIN_SCALE
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
    for i in range(N_CHANNEL):
        block_in_buffer[i] = client.inports[i].get_array()

    # Down-sample from INPUT_FS to PROC_FS
    lp_block_buffer, filter_states_lp_down_sample = signal.lfilter(b, a, block_in_buffer, axis=-1, zi=filter_states_lp_down_sample) #[D, T]
    ds_block_buffer = lp_block_buffer[:, ::ds_factor] #[D, T']

    # Perform enhancement
    enhanced_signal = block_processing(ds_block_buffer)

    # Upsample from PROC_FS to INPUT_FS
    block_out_buffer[...] = 0
    block_out_buffer[::ds_factor] = enhanced_signal
    block_out_buffer, filter_states_lp_up_sample = signal.lfilter(ds_factor*b, a, block_out_buffer, axis=-1, zi=filter_states_lp_up_sample)

    # Store result
    client.outports[0].get_array()[:] = block_out_buffer


@client.set_shutdown_callback
def shutdown(status, reason):
    print("JACK shutdown! status:", status, " reason:", reason )
    #event.set()


for i in range(N_CHANNEL):
    client.inports.register(F"mixed_speech_ch{i}")
client.outports.register("enhanced_speech")

print('activating JACK')
with client:
    print('#' * 80)
    print('press Return to quit')
    print('#' * 80)
    input()
    print('closing JACK')
