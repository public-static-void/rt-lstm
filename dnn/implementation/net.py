#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Authors       : Vadim Titov
Matr.-Nr.     : 6021356
Created       : June 23th, 2022
Last modified : June 23th, 2022
Description   : Master's Project "Source Separation for Robot Control"
Topic         : Net module of the LSTM RNN Project
"""

import pytorch_lightning as pl
import torch.nn.functional as F


# Hyper-parameters
input_size = 512  # TODO: specify input size.
hidden_size_1 = 256
hidden_size_2 = 128
output_size = 1
batch_size = 512
num_epochs = 5  # TODO: specify number of epochs.
learning_rate = 0.001
num_workers = 4
num_devices = 1
device = 'gpu'
is_test_run = False


class LitNeuralNet(pl.LightningModule):
    def __init__(self, input_size, hidden_size_1, hidden_size_2, output_size):
        super(LitNeuralNet, self).__init__()
        self.input_size = input_size
        # LSTM Forward layer 1.
        self.lstm1 = nn.LSTM(input_size, hidden_size_1)
        # LSTM Forward layer 2.
        self.lstm2 = nn.LSTM(hidden_size_1, hidden_size_2)
        # Dense (= Fully connected) layer.
        self.dense = nn.linear(hidden_size_2, output_size)
        # Rectified RELU.
        self.relu = nn.RReLU()

    def forward(self, x):
        out = self.lstm1(x)
        out = self.lstm2(out)
        out = self.linear(out)
        out = self.relu(out)

        # Sigmoid activation and no softmax at the end.
        out = F.sigmoid(out)
        return out

    def training_step(self, batch, batch_idx):
        input, labels = batch
        input = input.reshape(-1, input_size)

        # Forward pass.
        outputs = self(input)
        # MSE loss function.
        loss = F.mse_loss(outputs, labels)

        tensorboard_logs = {'train_loss': loss}

        return {'loss': loss, 'log': tensorboard_logs}

    def configure_optimizers(self):
        # Adam optimizer.
        return torch.optim.Adam(self.parameters(), lr=learning_rate)

    # TODO: Get input data from outside this module.
    def train_dataloader(self, train_dataset):
        train_loader = torch.utils.data.DataLoader(dataset=train_dataset,
                                                   batch_size=batch_size,
                                                   num_workers=num_workers,
                                                   shuffle=True)

        return train_loader

    def validation_step(self, batch, batch_idx):
        input, labels = batch
        input = input.reshape(-1, input_size)

        # Forward pass.
        outputs = self(input)
        # MSE loss function.
        loss = F.mse_loss(outputs, labels)
        return {'val_loss': loss}

    # TODO: Get training data from outside this module.
    def val_dataloader(self, val_dataset):
        val_loader = torch.utils.data.DataLoader(dataset=val_dataset,
                                                 batch_size=batch_size,
                                                 num_workers=num_workers,
                                                 shuffle=False)

        return val_loader

    def validation_epoch_end(self, outputs):
        avg_loss = torch.stack([x['val_loss'] for x in outputs]).mean()
        tensorboard_logs = {'avg_val_loss': avg_loss}
        return {'val_loss': avg_loss, 'log': tensorboard_logs}


def main():

    # train model

    # Running in fast_dev_run mode: will run a full train, val, test and
    # prediction loop using 1 batch(es).
    trainer = pl.Trainer(fast_dev_run=is_test_run, accelerator=device,
                         devices=num_devices, max_epochs=num_epochs)
    model = LitNeuralNet(input_size, hidden_size, num_classes)
    print(model)
    trainer.fit(model)


if __name__ == '__main__':
    main()
