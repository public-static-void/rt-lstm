#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Authors       : Vadim Titov
Matr.-Nr.     : 6021356
Created       : June 23th, 2022
Last modified : October 20th, 2022
Description   : Master's Project "Source Separation for Robot Control"
Topic         : Net module of the LSTM RNN Project
"""

import hyperparameters as hp
import pytorch_lightning as pl
import soundfile as sf
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchaudio
import torchvision.transforms as transforms
from torchmetrics import ScaleInvariantSignalDistortionRatio as SI_SDR
import numpy as np
from scipy.signal import get_window
from torch.utils.tensorboard import SummaryWriter

stft_length = 512
stft_shift = 256
window1 = torch.from_numpy(np.sqrt(get_window('hann', stft_length,
                                              fftbins=True))).to("cuda")

class LitNeuralNet(pl.LightningModule):
    def __init__(self, input_size, hidden_size_1, hidden_size_2, output_size):
        super(LitNeuralNet, self).__init__()
        self.input_size = input_size
        self.hidden_size_1 = hidden_size_1
        self.hidden_size_2 = hidden_size_2
        self.output_size = output_size
        # LSTM Forward layer 1.
        self.lstm1 = nn.LSTM(input_size, hidden_size_1, bidirectional=True, batch_first=True)
        # LSTM Forward layer 2.
        self.lstm2 = nn.LSTM(2*hidden_size_1, hidden_size_2, bidirectional=True, batch_first=True)
        # Dense (= Fully connected) layer.
        self.dense = nn.Linear(2*hidden_size_2, output_size)
        # Tanh layer.
        self.tanh = nn.Tanh()
        self.save_hyperparameters()

    def forward(self, x):
        # lstm1: [b, 2ch, f, t] -> [b*f, t, 2c]
        # lstm2: [b*f, t, hidden1 (*2 if bidir)] -> [b*f, t, hidden2 (*2 bidir)]
        # ff:    [b*f, t, hidden2 (*2 if nidir)] -> [b*f, t, 2]
        # tanh:  -> [b*f, t, 2]
        # reshape zum ausgangsshape [b, 2, f, t]
        # x = x.float()
        n_batch, n_ch, n_f, n_t = x.shape
        x = x.permute(0, 3, 2, 1)
        x = x.reshape(n_batch * n_t, n_f, n_ch)
        x, _ = self.lstm1(x)
        x = x.reshape(n_batch, n_t, n_f, 2*self.hidden_size_1)
        x = x.permute(0, 2, 1, 3)
        x = x.reshape(n_batch * n_f, n_t, 2*self.hidden_size_1)
        x, _ = self.lstm2(x)
        x = x.reshape(n_batch, n_f, n_t, 2*self.hidden_size_2)

        x = self.dense(x)
        x = x.permute(0, 3, 1, 2)
        x = self.tanh(x)

        # output = komprimierte maske
        return x

    def configure_optimizers(self):
        # Adam optimizer.
        return torch.optim.Adam(self.parameters(), lr=hp.learning_rate)

    def training_step(self, batch, batch_idx):
        clean, noise, mix = batch

        clean = clean.float()
        noise = noise.float()
        mix = mix.float()

        # input = input.reshape(-1, input_size)

        # Forward pass.
        outputs = self(mix)
        # MSE loss function.

        loss = F.mse_loss(outputs, clean)

        tensorboard_logs = {f"train/loss": loss}
        self.log(f"train/loss", loss, on_step=False,
                 on_epoch=True, logger=True)

        return loss

    def validation_step(self, batch, batch_idx):
        clean, noise, mix = batch
        # dimensionen: batch, channel, frequency, time
        # input = input.reshape(-1, input_size)

        clean = clean.float()
        noise = noise.float()
        mix = mix.float()

        # Forward pass.
        outputs = self(mix)
        # MSE loss function.
        loss = F.mse_loss(outputs, clean)
        self.log(f"val/loss", loss, on_step=False, on_epoch=True, logger=True)
        si_sdr = SI_SDR().to("cuda")
        si_sdr_val = si_sdr(outputs, clean)
        self.log(f"val/si_sdr", si_sdr_val, on_step=False, on_epoch=True, logger=True)

        # Add spectrograms and audios to tensorboard.

        comp_mask = self(mix)
        decomp_mask = -torch.log((hp.K - comp_mask) / (hp.K + comp_mask))

        mix_re = torch.chunk(mix, 2, 1)[0]
        mix_im = torch.chunk(mix, 2, 1)[1]

        clean_re = torch.chunk(clean, 2, 1)[0]
        clean_im = torch.chunk(clean, 2, 1)[1]

        mask_re = torch.chunk(decomp_mask, 2, 1)[0]
        mask_im = torch.chunk(decomp_mask, 2, 1)[1]

        mix_co = torch.complex(mix_re, mix_im)
        clean_co = torch.complex(clean_re, clean_im)
        mask_co = torch.complex(mask_re, mask_im)

        prediction = mask_co * mix_co[0]

        writer = SummaryWriter()

        for sample in range(0, mix.shape[0]):

            mix_istft = torch.istft(mix_co[sample], stft_length, stft_shift, window = window1)
            clean_istft = torch.istft(clean_co[sample], stft_length, stft_shift, window = window1)
            pred_istft = torch.istft(prediction[sample], stft_length, stft_shift, window = window1)

            writer.add_audio("mix-"+ batch_idx + "-" + sample, mix_istft, self.current_epoch)
            writer.add_audio("clean-"+ batch_idx + "-" + sample, clean_istft, self.current_epoch)
            writer.add_audio("pred-"+ batch_idx + "-" + sample, pred_istft, self.current_epoch)

            mix_spec = 10 * torch.log10(
                torch.maximum(torch.square(torch.abs(mix_co[sample])),
                              (10 ** (-15)) * torch.ones_like(combined_stfts,
                                                              dtype=torch.float32)))

            clean_spec = 10 * torch.log10(
                torch.maximum(torch.square(torch.abs(clean_co[sample])),
                              (10 ** (-15)) * torch.ones_like(combined_stfts,
                                                              dtype=torch.float32)))
            pred_spec = 10 * torch.log10(
                torch.maximum(torch.square(torch.abs(prediction[sample])),
                              (10 ** (-15)) * torch.ones_like(combined_stfts,
                                                              dtype=torch.float32)))

            writer.add_image("mix-"+ batch_idx + "-" + sample, mix_spec, self.current_epoch)
            writer.add_image("clean-"+ batch_idx + "-" + sample, clean_spec, self.current_epoch)
            writer.add_image("pred-"+ batch_idx + "-" + sample, pred_spec, self.current_epoch)

        writer.close()

        return loss

    def predict_step(self, batch, batch_idx):

        clean, noise, mix = batch

        clean = clean.float()
        noise = noise.float()
        mix = mix.float()

        comp_mask = self(mix)
        decomp_mask = -torch.log((hp.K - comp_mask) / (hp.K + comp_mask))

        # merge real and imaginary parts of mix (and mask) before multiplication with mask.

        mix_re = torch.chunk(mix, 2, 1)[0]
        mix_im = torch.chunk(mix, 2, 1)[1]

        clean_re = torch.chunk(clean, 2, 1)[0]
        clean_im = torch.chunk(clean, 2, 1)[1]

        mask_re = torch.chunk(decomp_mask, 2, 1)[0]
        mask_im = torch.chunk(decomp_mask, 2, 1)[1]

        mix_co = torch.complex(mix_re, mix_im)
        clean_co = torch.complex(clean_re, clean_im)
        mask_co = torch.complex(mask_re, mask_im)

        prediction = mask_co * mix_co[0]

        # generate sound files for mix, clean and prediction.

        # TODO: refactor

        # istft


        # print(mix.shape)

        for sample in range(0, mix.shape[0]):
            mix_istft = torch.istft(mix_co[sample], stft_length, stft_shift, window = window1)
            clean_istft = torch.istft(clean_co[sample], stft_length, stft_shift, window = window1)
            pred_istft = torch.istft(prediction[sample], stft_length, stft_shift, window = window1)

            # wavs

            # print(type(mix_istft.cpu().numpy()[0][0]))

            # sf.write("mix-" + str(batch_idx) + "-" + str(sample) + ".wav",
            #          mix_istft.cpu().numpy(), 16000)
            # sf.write("clean-" + str(batch_idx) + "-" + str(sample) + ".wav",
            #          clean_istft.cpu().numpy(), 16000)
            # sf.write("pred-" + str(batch_idx) + "-" + str(sample) + ".wav",
            #          pred_istft.cpu().numpy(), 16000)

            # print(pred_istft)
            print(mix_istft)

            torchaudio.save("./out/mix-" + str(batch_idx) + "-" + str(sample) + ".wav",
                     mix_istft.float().cpu(), 16000)
            torchaudio.save("./out/clean-" + str(batch_idx) + "-" + str(sample) + ".wav",
                     clean_istft.float().cpu(), 16000)
            torchaudio.save("./out/pred-" + str(batch_idx) + "-" + str(sample) + ".wav",
                     pred_istft.float().cpu(), 16000)

        return prediction

    def train_dataloader(self):
        train_dataset = hp.CustomDataset(type="training")

        train_loader = torch.utils.data.DataLoader(
            dataset=train_dataset,
            batch_size=hp.batch_size,
            num_workers=hp.num_workers,
            shuffle=True,
        )
        return train_loader

    def val_dataloader(self):
        val_dataset = hp.CustomDataset(type="validation")

        val_loader = torch.utils.data.DataLoader(
            dataset=val_dataset,
            batch_size=hp.batch_size,
            num_workers=hp.num_workers,
            shuffle=False,
        )
        return val_loader

    def test_dataloader(self):
        test_dataset = hp.CustomDataset(type="test")

        test_loader = torch.utils.data.DataLoader(
            dataset=test_dataset,
            batch_size=hp.batch_size,
            num_workers=hp.num_workers,
            shuffle=False,
        )
        return test_loader
