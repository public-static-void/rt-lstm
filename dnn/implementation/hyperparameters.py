#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Authors       : Vadim Titov
Matr.-Nr.     : 6021356
Created       : June 23th, 2022
Last modified : November 12th, 2022
Description   : Master's Project "Source Separation for Robot Control"
Topic         : Hyperparameters module of the LSTM RNN Project
"""

import pytorch_lightning as pl
import torch
import torch.nn as nn
import torch.nn.functional as F
from dataset import CustomDataset
from pytorch_lightning import loggers as pl_loggers
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint
from scipy.signal import get_window
import numpy as np

###########################
# General global settings #
###########################

device = "gpu"
num_devices = 1
num_workers = 4
DATA_DIR = "soundfiles/"
OUT_DIR = "out/"

# STFT settings.
fs = 16000
stft_length = 512
stft_shift = 256
fftbins = True
window = torch.from_numpy(np.sqrt(get_window("hann", stft_length, fftbins))).to(
    "cuda"
)

#########################
# LSTM Hyper-parameters #
#########################

input_size = 6  # 3 microphone channels * 2 (Re + Im).
hidden_size_1 = 256
hidden_size_2 = 128
output_size = 2  # 1 channel * 2 (Re + Im).
bidirectional = True
batch_size = 3
batch_first = True
num_epochs = 100
learning_rate = 0.00001
K = 1  # Decompression constant for mask decompression.

###########
# Logging #
###########

LOG_DIR = "logs/"
logger = True
on_step = False
on_epoch = True
log_every_n_steps = 1
tb_logger = pl_loggers.TensorBoardLogger(LOG_DIR, log_graph=False)

#################
# Checkpointing #
#################

CHECKPOINT_DIR = "checkpoints/"
checkpoint_name = "epoch=73-step=148000.ckpt"
enable_checkpointing = True

#############
# Callbacks #
#############

early_stopping = EarlyStopping(monitor="val/loss", patience=100, mode="min")

checkpointing = ModelCheckpoint(
    dirpath=CHECKPOINT_DIR,
    filename="{epoch}-{step}",
    save_top_k=2,
    mode="min",
    monitor="val/loss",
    every_n_epochs=1,
)

##################
# Debug settings #
##################

# Running in fast_dev_run mode: will run a full train, val, test and
# prediction loop using 1 batch(es).
is_test_run = False
# Limit batches for debug training runs.
limit_train_batches = 0.1
