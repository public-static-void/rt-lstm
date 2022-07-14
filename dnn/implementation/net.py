#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Authors       : Vadim Titov
Matr.-Nr.     : 6021356
Created       : June 23th, 2022
Last modified : July 7th, 2022
Description   : Master's Project "Source Separation for Robot Control"
Topic         : Net module of the LSTM RNN Project
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl
import hyperparameters as hp
import torchvision.transforms as transforms

class LitNeuralNet(pl.LightningModule):
    def __init__(self, input_size, n_t, hidden_size_1, hidden_size_2, output_size):
        super(LitNeuralNet, self).__init__()
        # LSTM Forward layer 1.
        self.lstm1 = nn.LSTM(input_size, n_t*hidden_size_1, batch_first=True)
        # LSTM Forward layer 2.
        self.lstm2 = nn.LSTM(n_t*hidden_size_1, n_t*hidden_size_2, batch_first=True)
        # Dense (= Fully connected) layer.
        self.dense = nn.Linear(n_t*hidden_size_2, output_size)
        # Tanh layer.
        self.tanh = nn.Tanh()

    def forward(self, x):
        # lstm1: [b, 2ch, f, t] -> [b*f, t, 2c]
        # lstm2: [b*f, t, hidden1 (*2 if bidir)] -> [b*f, t, hidden2 (*2 bidir)]
        # ff:    [b*f, t, hidden2 (*2 if nidir)] -> [b*f, t, 2]
        # tanh:  -> [b*f, t, 2]
        # reshape zum ausgangsshape [b, 2, f, t]
        n_batch, n_ch, n_f, n_t = x.shape
        x = x.permute(0, 2, 3, 1)
        x = x.reshape(n_batch*n_f, n_t, n_ch)
        x, _ = self.lstm1(x)
        x = x.reshape(n_batch * n_f, n_t, self.hidden_size1)
        x, _ = self.lstm2(x)
        x = x.reshape(n_batch * n_f, n_t, self.hidden_size2)
        x = self.linear(x)
        x = x.reshape(n_batch * n_f, n_t, 2)
        x = self.tanh(x)
        x = x.reshape(n_batch, n_f, n_t, 2)
        x = x.permute(0, 3, 1, 2)

        # output = komprimierte maske
        return x

    def configure_optimizers(self):
        # Adam optimizer.
        return torch.optim.Adam(self.parameters(), lr=hp.learning_rate)

    def training_step(self, batch, batch_idx):

        clean, noise, mix = batch

        #input = input.reshape(-1, input_size)

        # Forward pass.
        outputs = self(mix)
        # MSE loss function.
        loss = F.mse_loss(outputs, clean)

        tensorboard_logs = {f'train/loss': loss}
        self.log(f'train/loss', loss, on_step=False,
                 on_epoch=True, logger=True)

        return {f'train/loss': loss, 'log': tensorboard_logs}

    def validation_step(self, batch, batch_idx):
        clean, noise, mix = batch
        # dimensionen: batch, channel, frequency, time
        #input = input.reshape(-1, input_size)

        # Forward pass.
        outputs = self(input)
        # MSE loss function.
        loss = F.mse_loss(outputs, clean)
        self.log(f'val/loss', loss, on_step=False, on_epoch=True, logger=True)

        return {f'val/loss': loss}

    def validation_epoch_end(self, outputs):
        avg_loss = torch.stack([x[f'val/loss'] for x in outputs]).mean()
        tensorboard_logs = {f'avg_val/loss': avg_loss}

        return {f'avg_val/loss': avg_loss, 'log': tensorboard_logs}

    def predict_step(self, batch, batch_idx, dataloader_idx=0):
        compressed_mask = self(batch)
        mix = batch[2]
        prediction = compressed_mask * mix
        return prediction


    def train_dataloader(self):
        train_dataset = hp.CustomDataset(type='training')

        train_loader = torch.utils.data.DataLoader(dataset=train_dataset,
                                                   batch_size=hp.batch_size,
                                                   num_workers=hp.num_workers,
                                                   shuffle=True)
        return train_loader

    def val_dataloader(self):
        val_dataset = hp.CustomDataset(type='validation')

        val_loader = torch.utils.data.DataLoader(dataset=val_dataset,
                                                 batch_size=hp.batch_size,
                                                 num_workers=hp.num_workers,
                                                 shuffle=False)
        return val_loader

    def test_dataloader(self):
        test_dataset = hp.CustomDataset(type='test')

        test_loader = torch.utils.data.DataLoader(dataset=test_dataset,
                                                 batch_size=hp.batch_size,
                                                 num_workers=hp.num_workers,
                                                 shuffle=False)
        return test_loader

