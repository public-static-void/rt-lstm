#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Authors       : Vadim Titov
Matr.-Nr.     : 6021356
Created       : June 23th, 2022
Last modified : November 12th, 2022
Description   : Master's Project "Source Separation for Robot Control"
Topic         : Net module of the LSTM RNN Project
"""

import hyperparameters as hp
import pytorch_lightning as pl
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
        # LSTM Forward layer 1.
        self.lstm1 = nn.LSTM(
            input_size,
            hidden_size_1,
            bidirectional=hp.bidirectional,
            batch_first=hp.batch_first,
        )
        # LSTM Forward layer 2.
        self.lstm2 = nn.LSTM(
            2 * hidden_size_1,
            hidden_size_2,
            bidirectional=hp.bidirectional,
            batch_first=hp.batch_first,
        )
        # Dense (= Fully connected) layer.
        self.dense = nn.Linear(2 * hidden_size_2, output_size)
        # Tanh layer.
        self.tanh = nn.Tanh()
        self.save_hyperparameters()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Implements the forward step functionality.

        x : torch.Tensor
            Input of the net.

        Returns
        -------
        torch.Tensor
           Compressed mask.
        """
        n_batch, n_ch, n_f, n_t = x.shape
        x = x.permute(0, 3, 2, 1)
        x = x.reshape(n_batch * n_t, n_f, n_ch)
        x, _ = self.lstm1(x)
        x = x.reshape(n_batch, n_t, n_f, 2 * self.hidden_size_1)
        x = x.permute(0, 2, 1, 3)
        x = x.reshape(n_batch * n_f, n_t, 2 * self.hidden_size_1)
        x, _ = self.lstm2(x)
        x = x.reshape(n_batch, n_f, n_t, 2 * self.hidden_size_2)

        x = self.dense(x)
        x = x.permute(0, 3, 1, 2)
        x = self.tanh(x)

        # Output = compessed mask.
        return x

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
        loss = torch.mean((torch.abs(pred - clean)) ** 2)
        return loss

    def configure_optimizers(self):
        """Helper function.

        Configured the function used to update parameters.
        """
        # Adam optimizer.
        return torch.optim.Adam(self.parameters(), lr=hp.learning_rate)

    def common_step(self, batch: torch.Tensor) -> tuple:
        """Helper function.

        Implements functionality used in training, validation and prediction
        step functions of this class.

        batch : torch.Tensor
            Input of the net.

        Returns
        -------
        tuple
            Loss, clean, mix, prediction.
        """
        # Unpack and cast input data for further processing.
        clean, noise, mix = batch
        clean = clean.float()
        noise = noise.float()
        mix = mix.float()

        # Compute mask.
        comp_mask = self(mix)
        decomp_mask = -torch.log((hp.K - comp_mask) / (hp.K + comp_mask))
        mix_co = torch.complex(mix[:, 0], mix[:, 3])
        clean_co = torch.complex(clean[:, 0], clean[:, 1])
        mask_co = torch.complex(decomp_mask[:, 0], decomp_mask[:, 1])
        # Apply mask to mixture (noisy) input signal.
        prediction = mask_co * mix_co

        # Compute loss.
        loss = self.comp_mse(prediction, clean_co)
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
        loss, _, _, _ = self.common_step(batch)

        # Logging.
        tensorboard_logs = {f"train/loss": loss}
        self.log(f"train/loss", loss, on_step=False,
                 on_epoch=True, logger=True)

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
        loss, clean_co, mix_co, prediction = self.common_step(batch)

        # Logging.
        self.log(
            f"val/loss",
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
            f"val/si_sdr",
            si_sdr_val,
            on_step=hp.on_step,
            on_epoch=hp.on_epoch,
            logger=hp.logger,
        )

        # Add spectrograms and audios to tensorboard.
        # TODO: this way of trying to add spectrograms and audios to
        # tensorboard isn't working for some odd reason.

        writer = self.logger.experiment

        for sample in range(0, mix_co.shape[0]):
            if sample == 0 and batch_idx == 0:
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

                # DEBUG
                fig_mix = plt.figure()
                ax = fig_mix.add_subplot(111)
                ax.imshow(mix_spec.to("cpu"))
                plt.title("mix")
                # plt.show()

                fig_clean = plt.figure()
                ax = fig_clean.add_subplot(111)
                ax.imshow(clean_spec.to("cpu"))
                plt.title("clean")
                # plt.show()

                fig_pred = plt.figure()
                ax = fig_pred.add_subplot(111)
                ax.imshow(pred_spec.to("cpu"))
                plt.title("pred")
                # plt.show()

                writer.add_figure(
                    "clean-" + str(batch_idx) + "-" + str(sample),
                    fig_mix,
                    self.current_epoch,
                )
                writer.add_figure(
                    "pred-" + str(batch_idx) + "-" + str(sample),
                    fig_clean,
                    self.current_epoch,
                )
                writer.add_figure(
                    "mix-" + str(batch_idx) + "-" + str(sample),
                    fig_pred,
                    self.current_epoch,
                )

                writer.add_audio(
                    "mix-" + str(batch_idx) + "-" + str(sample),
                    mix_istft,
                    self.current_epoch,
                )
                writer.add_audio(
                    "clean-" + str(batch_idx) + "-" + str(sample),
                    clean_istft,
                    self.current_epoch,
                )
                writer.add_audio(
                    "pred-" + str(batch_idx) + "-" + str(sample),
                    pred_istft,
                    self.current_epoch,
                )

                # writer.add_image("clean-"+ str(batch_idx) + "-" + str(sample),
                #                  clean_spec.unsqueeze(0), self.current_epoch)
                # writer.add_image("pred-"+ str(batch_idx) + "-" + str(sample),
                #                  pred_spec.unsqueeze(0), self.current_epoch)
                # writer.add_image("mix-"+ str(batch_idx) + "-" + str(sample),
                #                  mix_spec.unsqueeze(0), self.current_epoch)

        # writer.close()

        return loss

    def predict_step(self, batch: torch.Tensor, batch_idx: int) -> torch.Tensor:
        """Implements the prediction step functionality.

        batch : torch.Tensor
            Input of the net.
        batch_idx : int
            Index of the current batch.

        Returns
        -------
        torch.Tensor
            Prediction.
        """
        _, clean_co, mix_co, prediction = self.common_step(batch)

        # Generate sound files for mix, clean and prediction.

        for sample in range(0, mix_co.shape[0]):
            mix_istft = torch.istft(
                mix_co[sample], hp.stft_length, hp.stft_shift, hp.window
            )
            clean_istft = torch.istft(
                clean_co[sample], hp.stft_length, hp.stft_shift, hp.window
            )
            pred_istft = torch.istft(
                prediction[sample], hp.stft_length, hp.stft_shift, hp.window
            )
            torchaudio.save(
                hp.OUT_DIR
                + "mix-"
                + str(batch_idx)
                + "-"
                + str(sample)
                + ".wav",
                mix_istft.float().cpu(),
                hp.fs,
            )
            torchaudio.save(
                hp.OUT_DIR
                + "clean-"
                + str(batch_idx)
                + "-"
                + str(sample)
                + ".wav",
                clean_istft.float().cpu(),
                hp.fs,
            )
            torchaudio.save(
                hp.OUT_DIR
                + "pred-"
                + str(batch_idx)
                + "-"
                + str(sample)
                + ".wav",
                pred_istft.float().cpu(),
                hp.fs,
            )

        return prediction

    def train_dataloader(self):
        """Initialize the data used in training."""
        train_dataset = hp.CustomDataset(type="training")

        train_loader = torch.utils.data.DataLoader(
            dataset=train_dataset,
            batch_size=hp.batch_size,
            num_workers=hp.num_workers,
            shuffle=True,
        )
        return train_loader

    def val_dataloader(self):
        """Initialize the data used in validation."""
        val_dataset = hp.CustomDataset(type="validation")

        val_loader = torch.utils.data.DataLoader(
            dataset=val_dataset,
            batch_size=hp.batch_size,
            num_workers=hp.num_workers,
            shuffle=False,
        )
        return val_loader

    def test_dataloader(self):
        """Initialize the data used in prediction."""
        test_dataset = hp.CustomDataset(type="test")

        test_loader = torch.utils.data.DataLoader(
            dataset=test_dataset,
            batch_size=hp.batch_size,
            num_workers=hp.num_workers,
            shuffle=False,
        )
        return test_loader
