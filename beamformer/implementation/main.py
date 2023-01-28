#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Authors       : Vadim Titov, Henning Möllers, Leon Mannweiler
Matr.-Nr.     : 6021356, ..., ...
Created       : May 12th, 2022
Last modified : January 28th, 2023
Description   : Master's Project "Source Separation for Robot Control"
Topic         : Beamformer: MVDR Implementation
"""
import sys

import matplotlib.pyplot as plt
import numpy as np
import pyroomacoustics as pra
import soundfile as sf
from scipy.signal import get_window, istft, stft

np.set_printoptions(threshold=sys.maxsize)


def create_room(room_dim: np.ndarray, rt60: float) -> pra.room:
    """Helper function.

    Creates a room from input parameters and returns it.

    Parameters
    ----------
    room_dim : np.ndarray

    rt60 : float
        Reverberation time in s.

    Returns
    -------
    pra.room
    """
    e_absorbtion, max_order = pra.inverse_sabine(rt60=rt60, room_dim=room_dim)
    room = pra.ShoeBox(
        room_dim, materials=pra.Material(e_absorbtion), max_order=max_order
    )
    return room


def add_sources(
    room: pra.room, source_locs: np.ndarray, audio: np.ndarray, delay: float
) -> None:
    """Helper function.

    Adds sources to an input room defined by input parameters.

    Parameters
    ----------
    room: pra.room

    source_locs: np.ndarray

    audio: np.ndarray

    delay: float
    """
    # place the source in the room
    room.add_source(source_locs, signal=audio, delay=delay)


def add_mics(room: pra.room, beamformer: pra.beamforming.Beamformer) -> None:
    """Helper function.

    Adds mics to an input room from input parameters.

    Parameters
    ----------
    room: pra.room

    beamformer: pra.beamforming.Beamformer
    """
    # define the locations of the microphones.
    # finally place the array in the room.
    room.add_microphone_array(beamformer)


def plot_signal(s: np.ndarray, t: np.ndarray, title: str) -> None:
    """Helper function.

    Plots a signal as magnitude `s` against time `t`.

    Parameters
    ----------
    s: np.ndarray
        Amplitude values of the input signal.
    t: np.ndarray
        Time values of the input signal.
    title: str
        Title for the plot.
    """

    # Define some general settings for the plot.
    rows = 1
    cols = 1
    fig = plt.figure(figsize=(19, 7))

    plt.suptitle(title)

    plt.subplots_adjust(
        left=0.05,
        # bottom=0.1,
        right=0.99,
        # top=0.9,
        # wspace=0.4,
        hspace=0.4,
    )

    # First subplot.
    ax = fig.add_subplot(rows, cols, 1)
    ax.plot(t, s, "b", label="s(t)")
    ax.set_xlabel("t[s]")
    ax.set_ylabel("s(t)")
    ax.grid(True)
    ax.legend()


def plot_dft(S: np.ndarray, title: str) -> None:
    """Helper function.

    Plots amplitude of `S` in dB together with the amplitude of the
    corresponding filter `H`(z) in dB in one plot.

    Parameters
    ----------
    S: np.ndarray
        Amplitudes of input signal.
    H: np.ndarray
        Amplitudes of the corresponding filter.
    title: str
        Plot title.
    """

    fig, ax1 = plt.subplots()

    ax1.set_title(title)
    ax1.plot(S)

    ax1.set_ylabel("Amplitude [dB]", color="b")

    ax1.grid()
    ax1.axis("tight")
    ax1.legend()


def compute_SNR(signal: np.ndarray, axis=0, ddof=0) -> float:
    """Helper function.

    Computes the Signal To Noise Ratio (SNR) for the input signal.

    Parameters
    ----------
    signal: np.ndarray
        Input signal.
    axis: int, optional
        Signal array axis to perform computation on.
        By default 0.
    ddof: int, optional
        Delta Degrees Of Freedom.
        By default 0.

    Returns
    -------
    float
    """
    signal = np.asanyarray(signal)
    mean = signal.mean(axis)
    stdev = signal.std(axis=axis, ddof=ddof)
    return 20 * np.log10(abs(np.where(stdev == 0, 0, mean / stdev)))


def get_power(s: np.ndarray) -> float:
    """Helper function.

    Computes the power of input signal.

    Parameters
    ----------
    s: np.ndarray
        Input signal.

    Returns
    -------
    float
    """
    stdev = s.std(axis=0, ddof=0)
    var = stdev**2
    power = np.sum(var)
    return power


def show_spectrogram(
    m_stft: np.ndarray, v_freq: np.ndarray, v_time: np.ndarray, title: str
) -> None:
    """Helper function.

    Creates a spectrogram of the input. The extent option tells matplotlib to
    use the entries of the vector v_time for the x-axis and v_freq for the
    y-axis. Here, the vector v_time contains the time instants for each block /
    each spectrum and the vector v_freq contains the frequency bin information.

    Parameters
    ----------
    m_stft: np.ndarray
        A matrix which stores the complex short-time spectra in each row.
    v_freq: np.ndarray
        A vector which contains the frequency axis ( in units of Hertz)
        corresponding to the computed spectra.
    v_time: np.ndarray
        Time steps around which a frame is centered(as in previous exercise).
    title: str
        Plot title.
    """
    fig = plt.figure()
    ax = fig.add_subplot(111)
    im = ax.imshow(
        10 * np.log10(np.maximum(np.square(np.abs(m_stft)), 10 ** (-15))),
        cmap="viridis",
        origin="lower",
        extent=[v_time[0], v_time[-1], v_freq[0], v_freq[-1]],
        aspect="auto",
        vmin=-140,
        vmax=-20,
    )
    fig.colorbar(im, orientation="vertical", pad=0.2)
    plt.title(title)
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Frequency (Hz)")


def main():
    """Main function."""
    # Define sample path.
    sample_path_speech = (
        "./Samples/examples_input_samples_german_speech_8000.wav"
    )
    sample_path_noise = "./Samples/whitenoise.wav"

    # Read samples from disk.
    audio_speech, fs_speech = sf.read(sample_path_speech)
    audio_noise, fs_noise = sf.read(sample_path_noise)

    # The desired reverberation time in seconds.
    rt60 = 0.5
    # Room dimensions x,y,z in meters.
    room_dim = [20, 30]

    # Number of mics.
    N_mics = 3

    mics_locs = [10, 15]

    # Define mic array.
    R3 = pra.linear_2D_array(mics_locs, N_mics, 0, 0.1)

    # Define source locations.

    speech_source_locs = [5, 5]
    noise_source_locs = [17.5, 17.5]

    # Frame length in ms.
    fl = 32

    # Transform frame length from ms to samples.
    N_speech = int(fs_speech * fl / 1000)
    N_noise = int(fs_noise * fl / 1000)

    # Define windowing functions.
    window1 = np.sqrt(get_window("hann", N_speech, fftbins=True))

    # Specify delay.
    delay = 1.3

    # Create room.
    room = create_room(room_dim, rt60)

    # Create beamformer.
    beamformer = pra.Beamformer(R3, room.fs)

    # Add beamformer to room.
    add_mics(room, beamformer)

    # Add source(s).
    add_sources(room, speech_source_locs, audio_speech, delay)
    add_sources(room, noise_source_locs, audio_noise, delay)

    room.compute_rir()

    premix = room.simulate(return_premix=True)

    premix_speech = premix[0]
    premix_noise = premix[1]

    sp_freqs, sp_times, sp_power = stft(
        premix_speech, fs_speech, window=window1
    )
    n_freqs, n_times, n_power = stft(premix_noise, fs_speech, window=window1)

    # covariance matrix
    alpha = 0.8

    n_cov_mat = np.zeros(
        (
            n_times.shape[0],
            n_freqs.shape[0],
            n_power.shape[0],
            n_power.shape[0],
        ),
        dtype=np.complex_,
    )
    sp_cov_mat = np.zeros(
        (
            sp_times.shape[0],
            sp_freqs.shape[0],
            sp_power.shape[0],
            sp_power.shape[0],
        ),
        dtype=np.complex_,
    )

    n_inv_cov_mat = np.zeros_like(n_cov_mat, dtype="complex128")

    eigen_mat = np.zeros(
        (sp_cov_mat.shape[0], sp_cov_mat.shape[1], sp_cov_mat.shape[2], 1),
        dtype="complex128",
    )
    filter_mat = np.zeros(
        (n_cov_mat.shape[0], n_cov_mat.shape[1], n_cov_mat.shape[2], 1),
        dtype="complex128",
    )

    # Compute covariance matrix.
    for t, _ in enumerate(n_times):
        for f, _ in enumerate(n_freqs):
            # Adaptive filtering.
            if t == 0:
                sp_init_matrix = np.eye(3)
                n_init_matrix = np.eye(3)
            else:
                sp_init_matrix = sp_cov_mat[t - 1, f]
                n_init_matrix = n_cov_mat[t - 1, f]
            sp_x = sp_power[:, f, t]
            n_x = n_power[:, f, t]
            sp_init_matrix = alpha * sp_init_matrix + (
                (1 - alpha) * np.einsum("i,j->ij", sp_x, sp_x.conj())
            )
            n_init_matrix = alpha * n_init_matrix + (
                (1 - alpha) * np.einsum("i,j->ij", n_x, n_x.conj())
            )
            sp_cov_mat[t, f] = sp_init_matrix
            n_cov_mat[t, f] = n_init_matrix
            # Compute eigenvectors element-wise. Eigenvector used as steering
            # vector. Eigenvalues computed as byproduct but not needed.
            eigenvalues, eigenvectors = np.linalg.eigh(sp_cov_mat[t, f])
            eigenvector = eigenvectors[:, -1][:, np.newaxis]
            eigen_mat[t, f] = eigenvector
            # Invert element-wise.
            n_inv_cov = np.linalg.inv(n_cov_mat[t, f])
            n_inv_cov_mat[t, f] = n_inv_cov
            filter_coeff = (n_inv_cov @ eigenvector) / (
                eigenvector.conj().T @ n_inv_cov @ eigenvector
            )

            filter_mat[t, f] = filter_coeff

    noisy_speech = sp_power + n_power

    filtered_speech = np.einsum(
        "tfc,cft->ft", np.squeeze(filter_mat.conj()), noisy_speech
    )

    _, filtered_istft = istft(
        filtered_speech, fs_speech, window1, time_axis=1, freq_axis=0
    )
    _, unfiltered_istft = istft(
        noisy_speech, fs_speech, window1, time_axis=2, freq_axis=1
    )

    premix_noisy_speech = premix[0] + premix[1]

    sf.write("noisy_speech0-0.wav", premix_noisy_speech[0], fs_speech)
    sf.write("noisy_speech0-1.wav", premix_noisy_speech[1], fs_speech)
    sf.write("noisy_speech0-2.wav", premix_noisy_speech[2], fs_speech)
    sf.write("reconstr_filtered_speech0.wav", filtered_istft, fs_speech)

    # show_spectrogram(
    #     n_power[0], n_freqs, n_times, "Noise channel 0 spectrogram"
    # )

    show_spectrogram(
        sp_power[0], sp_freqs, sp_times, "Speech channel 0 spectrogram"
    )

    show_spectrogram(
        noisy_speech[0],
        sp_freqs,
        sp_times,
        "Noisy mixture channel 0 spectrogram",
    )

    show_spectrogram(
        filtered_speech,
        sp_freqs,
        sp_times,
        "Filtered speech spectrogram",
    )

    plt.show()


if __name__ == "__main__":
    main()
