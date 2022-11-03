#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Authors       : Vadim Titov
Matr.-Nr.     : 6021356
Created       : June 23th, 2022
Last modified : October 20th, 2022
Description   : Master's Project "Source Separation for Robot Control"
Topic         : Hyperparameters module of the LSTM RNN Project
"""

import pytorch_lightning as pl
import torch.nn as nn
import torch.nn.functional as F
from dataset import CustomDataset
from pytorch_lightning import loggers as pl_loggers
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint

# Hyper-parameters
input_size = 6
hidden_size_1 = 256
hidden_size_2 = 128
output_size = 2  # 2 channels: 1 Re 1 Im ?.
# TODO: how to ensure each part ends up where it belongs?
batch_size = 5
K = 1  # decompression constant for mask decompression
num_epochs = 100  # TODO: specify number of epochs.
learning_rate = 0.00001
num_workers = 4
num_devices = 1
device = "gpu"
# Running in fast_dev_run mode: will run a full train, val, test and
# prediction loop using 1 batch(es).
is_test_run = False

LOG_DIR = "logs/"
CHECKPOINT_DIR = "checkpoints/"
DATA_DIR = "soundfiles/"

# LOGGING
tb_logger = pl_loggers.TensorBoardLogger(LOG_DIR, log_graph=False)

# Callbacks/Checkpoints.
early_stopping = EarlyStopping(monitor="val/loss", patience=10, mode="min")
checkpointing = ModelCheckpoint(
    dirpath=CHECKPOINT_DIR,
    filename="{epoch}-{step}",
    save_top_k=2,
    mode="min",
    monitor="val/loss",
    every_n_epochs=1,
)
