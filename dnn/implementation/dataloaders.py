#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Authors       : Vadim Titov
Matr.-Nr.     : 6021356
Created       : November 26th, 2022
Last modified : November 26th, 2022
Description   : Master's Project "Source Separation for Robot Control"
Topic         : Dataloaders module of the LSTM RNN Project
"""

import hyperparameters as hp
import torch
from dataset import CustomDataset


def train_dataloader():
    """Initialize the data used in training."""
    train_dataset = CustomDataset(type="training")

    train_loader = torch.utils.data.DataLoader(
        dataset=train_dataset,
        batch_size=hp.batch_size,
        num_workers=hp.num_workers,
        shuffle=True,
    )
    return train_loader


def val_dataloader():
    """Initialize the data used in validation."""
    val_dataset = CustomDataset(type="validation")

    val_loader = torch.utils.data.DataLoader(
        dataset=val_dataset,
        batch_size=hp.batch_size,
        num_workers=hp.num_workers,
        shuffle=False,
    )
    return val_loader


def test_dataloader():
    """Initialize the data used in prediction."""
    test_dataset = CustomDataset(type="test")

    test_loader = torch.utils.data.DataLoader(
        dataset=test_dataset,
        batch_size=hp.batch_size,
        num_workers=hp.num_workers,
        shuffle=False,
    )
    return test_loader
