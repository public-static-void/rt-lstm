#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt

import pytorch_lightning as pl
import torch.nn.functional as F

# Hyper-parameters
input_size = 784  # 28x28
hidden_size = 512
num_classes = 10
num_epochs = 5
batch_size = 64
learning_rate = 0.001
num_workers = 4
num_devices = 1
device = 'gpu'
is_test_run = False


class LitNeuralNet(pl.LightningModule):
    def __init__(self, input_size, hidden_size, num_classes):
        super(LitNeuralNet, self).__init__()
        self.input_size = input_size
        self.l1 = nn.Linear(input_size, hidden_size)
        self.relu1 = nn.ReLU()
        self.l2 = nn.Linear(hidden_size, hidden_size)
        self.relu2 = nn.ReLU()
        self.l3 = nn.Linear(hidden_size, num_classes)

    def forward(self, x):
        out = self.l1(x)
        out = self.relu1(out)
        out = self.l2(out)
        out = self.relu2(out)
        out = self.l3(out)

        # no activation and no softmax at the end
        return out

    def training_step(self, batch, batch_idx):
        images, labels = batch
        images = images.reshape(-1, input_size)

        # Forward pass
        outputs = self(images)
        loss = F.cross_entropy(outputs, labels)

        tensorboard_logs = {'train_loss': loss}

        return {'loss': loss, 'log': tensorboard_logs}

    def configure_optimizers(self):
        # return torch.optim.Adam(self.parameters(), lr=learning_rate)
        return torch.optim.SGD(self.parameters(), lr=learning_rate)

    def train_dataloader(self):
        train_dataset = torchvision.datasets.MNIST(root='./data',
                                                   train=True,
                                                   transform=transforms.ToTensor(),
                                                   download=True)

        train_loader = torch.utils.data.DataLoader(dataset=train_dataset,
                                                   batch_size=batch_size,
                                                   num_workers=num_workers,
                                                   shuffle=True)

        return train_loader

    def validation_step(self, batch, batch_idx):
        images, labels = batch
        images = images.reshape(-1, input_size)

        # Forward pass
        outputs = self(images)
        loss = F.cross_entropy(outputs, labels)
        return {'val_loss': loss}

    def val_dataloader(self):
        val_dataset = torchvision.datasets.MNIST(root='./data',
                                                 train=False,
                                                 transform=transforms.ToTensor(),
                                                 download=True)

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
