#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Authors       : Vadim Titov
Matr.-Nr.     : 6021356
Created       : June 23rd, 2022
Last modified : December 12th, 2022
Description   : Master's Project "Source Separation for Robot Control"
Topic         : Net module of the LSTM RNN Project
"""

import hyperparameters as hp
import pytorch_lightning as pl
import soundfile as sf
import torch
import torch.nn as nn
import torchaudio
from matplotlib import pyplot as plt
from torchmetrics import ScaleInvariantSignalDistortionRatio as SI_SDR


class LitNeuralNet(pl.LightningModule):
    """Main class of this module.

    Implements an LSTM deep neural net by inheriting from Pytorch Lightning Net
    module."""

    def __init__(
        self,
        input_size: int,
        hidden_size_1: int,
        hidden_size_2: int,
        output_size: int,
    ):
        """Constructor.

        Initializes the LSTM deep neural net with hyperparameters.

        The net architecture is as follows:

        lstm1: [b, 2ch, f, t] -> [b*f, t, 2c]
        lstm2: [b*f, t, hidden1 (*2 if bidir)] -> [b*f, t, hidden2 (*2 if bidir)]
        ff:    [b*f, t, hidden2 (*2 if nidir)] -> [b*f, t, 2]
        tanh:  -> [b*f, t, 2]
        reshape to match output shape [b, 2, f, t]

        input_size : int
            Size of the net's input layer.
        hidden_size_1 : int
            Size of the first LSTM hidden layer's output.
        hidden_size_2 : int
            Size of the second LSTM hidden layer's output.
        output_size : int
            Size of the net's output layer.
        """
        super(LitNeuralNet, self).__init__()
        self.input_size = input_size
        self.hidden_size_1 = hidden_size_1
        self.hidden_size_2 = hidden_size_2
        self.output_size = output_size
        self.learning_rate = hp.learning_rate
        self.t_bidirectional = hp.t_bidirectional
        self.f_bidirectional = hp.f_bidirectional
        self.lstm1_in = self.input_size
        self.lstm1_out = self.hidden_size_1

        if self.t_bidirectional is True:
            self.lstm2_in = 2 * self.hidden_size_1
            self.lstm2_out = self.hidden_size_2
            self.dense_in = 2 * self.hidden_size_2
        else:
            self.lstm2_in = self.hidden_size_1
            self.lstm2_out = int(self.hidden_size_2 / 2)
            self.dense_in = self.hidden_size_2

        # LSTM Forward layer 1.
        self.lstm1 = nn.LSTM(
            self.lstm1_in,
            self.lstm1_out,
            bidirectional=self.t_bidirectional,
            batch_first=hp.batch_first,
        )
        # LSTM Forward layer 2.
        self.lstm2 = nn.LSTM(
            self.lstm2_in,
            self.lstm2_out,
            bidirectional=self.f_bidirectional,
            batch_first=hp.batch_first,
        )
        # Dense (= Fully connected) layer.
        self.dense = nn.Linear(self.dense_in, self.output_size)
        # Tanh layer.
        self.tanh = nn.Tanh()
        self.save_hyperparameters()

    def forward(self, x: torch.Tensor, h_pre: torch.Tensor, c_pre:
                torch.Tensor)-> tuple:
        """Implements the forward step functionality.

        x : torch.Tensor
            Input of the net.
        h_pre : torch.Tensor
            Hidden state of previous iteration.
        c_pre : torch.Tensor
            Cell state of previous iteration.

        Returns
        -------
        tuple
           Compressed mask, (current hidden state, current cell state).
        """
        n_batch, n_ch, n_f, n_t = x.shape
        x = x.permute(0, 3, 2, 1)
        x = x.reshape(n_batch * n_t, n_f, n_ch)
        x, (h_new, c_new) = self.lstm1(x, (h_pre, c_pre))
        x = x.reshape(n_batch, n_t, n_f, self.lstm2_in)
        x = x.permute(0, 2, 1, 3)
        x = x.reshape(n_batch * n_f, n_t, self.lstm2_in)
        x, _ = self.lstm2(x)
        x = x.reshape(n_batch, n_f, n_t, self.dense_in)
        x = self.dense(x)
        x = x.permute(0, 3, 1, 2)
        x = self.tanh(x)

        # Output = compessed mask.
        return x, h_new, c_new

    def comp_mse(self, pred: torch.Tensor, clean: torch.Tensor) -> float:
        """Helper function.

        Computes mean squared error (MSE) for complex valued tensors.

        pred : torch.Tensor
            Predicted signal tensor.
        clean : torch.Tensor
            Clean signal tensor.

        Returns
        -------
        float
            MSE.
        """
        loss = torch.mean(torch.square(torch.abs(pred - clean)))
        return loss

    def configure_optimizers(self):
        """Helper function.

        Configured the function used to update parameters.
        """
        # RMSProp optimizer.
        # return torch.optim.RMSprop(self.parameters(), lr=self.learning_rate)
        # Adam optimizer.
        return torch.optim.Adam(self.parameters(), lr=self.learning_rate)

    def common_step(self, batch: torch.Tensor, batch_idx: int) -> tuple:
        """Helper function.

        Implements functionality used in training, validation and prediction
        step functions of this class.

        batch : torch.Tensor
            Input of the net.
        batch_idx : int
            Index of the current batch.

        Returns
        -------
        tuple
            Loss, clean, mix, prediction.
        """
        torch.autograd.set_detect_anomaly(mode=hp.mode, check_nan=hp.check_nan)

        # Unpack and cast input data for further processing.
        clean, noise, mix = batch
        clean = clean.float()
        noise = noise.float()
        mix = mix.float()

        # Init LSTM hidden state and cell state.
        # TODO: only works for batch_size of 1.
        hidden_state_size = hp.batch_size * mix.shape[3]

        if self.t_bidirectional is True:
            h_0 = torch.randn(2, hidden_state_size,
                             self.hidden_size_1).to(hp.device)
            c_0 = torch.randn(2, hidden_state_size,
                             self.hidden_size_1).to(hp.device)
        else:
            h_0 = torch.randn(1, hidden_state_size,
                             self.hidden_size_1).to(hp.device)
            c_0 = torch.randn(1, hidden_state_size,
                             self.hidden_size_1).to(hp.device)

        # Init LSTM hidden state in first iteration.
        if batch_idx == 0:
            self.h_pre = h_0
            self.c_pre = c_0
        # Compute mask.
        comp_mask, h_new, c_new = self(mix, self.h_pre, self.c_pre)
        decomp_mask = -torch.log((hp.K - comp_mask) / (hp.K + comp_mask))
        mix_co = torch.complex(mix[:, 0], mix[:, 3])
        clean_co = torch.complex(clean[:, 0], clean[:, 1])
        mask_co = torch.complex(decomp_mask[:, 0], decomp_mask[:, 1])
        # Carry LSTM hidden state over to next iteration.
        self.h_pre = h_new.detach()
        self.c_pre = c_new.detach()

        # Apply mask to mixture (noisy) input signal.
        prediction = mask_co * mix_co

        # Compute loss.
        loss = self.comp_mse(prediction, clean_co)

        # TODO: alternative loss function.
        # clean_istft = torch.istft(
        #     clean_co, hp.stft_length, hp.stft_shift, window=hp.window)
        # pred_istft = torch.istft(
        #     prediction, hp.stft_length, hp.stft_shift, window=hp.window)
        # si_sdr = SI_SDR().to("cuda")
        # loss = -si_sdr(pred_istft, clean_istft)

        return loss, clean_co, mix_co, prediction

    def training_step(self, batch: torch.Tensor, batch_idx: int) -> float:
        """Implements the training step functionality.

        batch : torch.Tensor
            Input of the net.
        batch_idx : int
            Index of the current batch.

        Returns
        -------
        float
            Training loss.
        """
        # Forward pass.
        loss, _, _, _ = self.common_step(batch, batch_idx)

        # Logging.
        self.log(
            "train/loss",
            loss,
            on_step=hp.on_step,
            on_epoch=hp.on_epoch,
            logger=hp.logger,
        )

        return loss

    def validation_step(self, batch: torch.Tensor, batch_idx: int) -> float:
        """Implements the validation step functionality.

        batch : torch.Tensor
            Input of the net.
        batch_idx : int
            Index of the current batch.

        Returns
        -------
        float
            Training loss.
        """
        # Forward pass.
        loss, clean_co, mix_co, prediction = self.common_step(batch, batch_idx)

        # Logging.
        self.log(
            "val/loss",
            loss,
            on_step=hp.on_step,
            on_epoch=hp.on_epoch,
            logger=hp.logger,
        )
        si_sdr = SI_SDR().to("cuda")
        clean_istft = torch.istft(
            clean_co, hp.stft_length, hp.stft_shift, window=hp.window
        )
        pred_istft = torch.istft(
            prediction, hp.stft_length, hp.stft_shift, window=hp.window
        )
        si_sdr_val = si_sdr(pred_istft, clean_istft)
        self.log(
            "val/si_sdr",
            si_sdr_val,
            on_step=hp.on_step,
            on_epoch=hp.on_epoch,
            logger=hp.logger,
        )

        # Add spectrograms and audios to tensorboard.

        writer = self.logger.experiment

        for sample in range(0, mix_co.shape[0]):
            if sample == 1 and batch_idx == 0:
                mix_istft = torch.istft(
                    mix_co[sample],
                    hp.stft_length,
                    hp.stft_shift,
                    window=hp.window,
                )
                clean_istft = torch.istft(
                    clean_co[sample],
                    hp.stft_length,
                    hp.stft_shift,
                    window=hp.window,
                )
                pred_istft = torch.istft(
                    prediction[sample],
                    hp.stft_length,
                    hp.stft_shift,
                    window=hp.window,
                )

                transform = torchaudio.transforms.Spectrogram(
                    n_fft=hp.stft_length, win_length=hp.stft_shift
                )
                mix_spec = transform(mix_istft.to("cpu"))
                clean_spec = transform(clean_istft.to("cpu"))
                pred_spec = transform(pred_istft.to("cpu"))

                mix_spec = 10 * torch.log10(
                    torch.maximum(
                        torch.square(torch.abs(mix_co[sample])),
                        (10 ** (-15))
                        * torch.ones_like(mix_co[sample], dtype=torch.float32),
                    )
                )
                clean_spec = 10 * torch.log10(
                    torch.maximum(
                        torch.square(torch.abs(clean_co[sample])),
                        (10 ** (-15))
                        * torch.ones_like(
                            clean_co[sample], dtype=torch.float32
                        ),
                    )
                )
                pred_spec = 10 * torch.log10(
                    torch.maximum(
                        torch.square(torch.abs(prediction[sample])),
                        (10 ** (-15))
                        * torch.ones_like(
                            prediction[sample], dtype=torch.float32
                        ),
                    )
                )

                fig_mix = plt.figure()
                ax = fig_mix.add_subplot(111)
                im = ax.imshow(
                    mix_spec.to("cpu"), origin="lower", vmin=-80, vmax=20
                )
                ax.set_xlabel("Frequency [bin]")
                ax.set_ylabel("Time [bin]")
                fig_mix.colorbar(im, orientation="vertical", pad=0.1)
                plt.title("mix")

                fig_clean = plt.figure()
                ax = fig_clean.add_subplot(111)
                im = ax.imshow(
                    clean_spec.to("cpu"), origin="lower", vmin=-80, vmax=20
                )
                ax.set_xlabel("Frequency [bin]")
                ax.set_ylabel("Time [bin]")
                fig_clean.colorbar(im, orientation="vertical", pad=0.1)
                plt.title("clean")

                fig_pred = plt.figure()
                ax = fig_pred.add_subplot(111)
                im = ax.imshow(
                    pred_spec.to("cpu"), origin="lower", vmin=-80, vmax=20
                )
                ax.set_xlabel("Frequency [bin]")
                ax.set_ylabel("Time [bin]")
                fig_pred.colorbar(im, orientation="vertical", pad=0.1)
                plt.title("pred")

                writer.add_figure(
                    "fig-mix-" + str(batch_idx) + "-" + str(sample),
                    fig_mix,
                    self.current_epoch,
                )
                writer.add_figure(
                    "fig-clean-" + str(batch_idx) + "-" + str(sample),
                    fig_clean,
                    self.current_epoch,
                )
                writer.add_figure(
                    "fig-pred-" + str(batch_idx) + "-" + str(sample),
                    fig_pred,
                    self.current_epoch,
                )

                writer.add_audio(
                    "mix-" + str(batch_idx) + "-" + str(sample),
                    mix_istft,
                    self.current_epoch,
                    16000,
                )
                writer.add_audio(
                    "clean-" + str(batch_idx) + "-" + str(sample),
                    clean_istft,
                    self.current_epoch,
                    16000,
                )
                writer.add_audio(
                    "pred-" + str(batch_idx) + "-" + str(sample),
                    pred_istft,
                    self.current_epoch,
                    16000,
                )

        return loss

    def predict_step(self, batch: torch.Tensor, batch_idx: int) -> tuple:
        """Implements the prediction step functionality.

        batch : torch.Tensor
            Input of the net.
        batch_idx : int
            Index of the current batch.

        Returns
        -------
        tuple
            Prediction, clean, mix.
        """
        _, clean_co, mix_co, prediction = self.common_step(batch, batch_idx)

        # Generate sound files for mix, clean and prediction.

        for sample in range(0, mix_co.shape[0]):
            mix_istft = torch.istft(
                mix_co[sample], hp.stft_length, hp.stft_shift, window=hp.window
            )
            clean_istft = torch.istft(
                clean_co[sample],
                hp.stft_length,
                hp.stft_shift,
                window=hp.window,
            )
            pred_istft = torch.istft(
                prediction[sample],
                hp.stft_length,
                hp.stft_shift,
                window=hp.window,
            )

            sf.write(
                hp.OUT_DIR
                + "mix-"
                + str(batch_idx)
                + "-"
                + str(sample)
                + ".wav",
                mix_istft.cpu(),
                hp.fs,
            )
            sf.write(
                hp.OUT_DIR
                + "clean-"
                + str(batch_idx)
                + "-"
                + str(sample)
                + ".wav",
                clean_istft.cpu(),
                hp.fs,
            )
            sf.write(
                hp.OUT_DIR
                + "pred-"
                + str(batch_idx)
                + "-"
                + str(sample)
                + ".wav",
                pred_istft.cpu(),
                hp.fs,
            )

        return prediction, clean_co, mix_co
