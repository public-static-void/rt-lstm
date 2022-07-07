#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Authors       : Vadim Titov
Matr.-Nr.     : 6021356
Created       : June 23th, 2022
Last modified : July 7th, 2022
Description   : Master's Project "Source Separation for Robot Control"
Topic         : Hyperparameters module of the LSTM RNN Project
"""

import pytorch_lightning as pl
import torch.nn as nn
import torch.nn.functional as F
from dataset import CustomDataset
from pytorch_lightning import loggers as pl_loggers
from pytorch_lightning.callbacks import EarlyStopping
from pytorch_lightning.callbacks import ModelCheckpoint

# Hyper-parameters
input_size = 512  # TODO: specify input size.
hidden_size_1 = 256
hidden_size_2 = 128
output_size = 2  # 2 channels: 1 Re 1 Im ?.
# TODO: how to ensure each part ends up where it belongs?
batch_size = 1
num_epochs = 1  # TODO: specify number of epochs.
learning_rate = 0.001
num_workers = 4
num_devices = 1
device = 'gpu'
is_test_run = True

LOG_DIR = 'logs/'
CHECKPOINT_DIR = 'checkpoints/'
DATA_DIR = 'soundfiles/'

# LOGGING
tb_logger = pl_loggers.TensorBoardLogger(LOG_DIR,
                                         name='Speech Enhancement LSTM RNN',
                                         log_graph=False)

# Callbacks/Checkpoints.
early_stopping = EarlyStopping(monitor='val/loss', patience=5, mode='min')
checkpointing = ModelCheckpoint(dirpath=CHECKPOINT_DIR, filename='{epoch}-{step}',
                                save_top_k=2, monitor='val/loss',
                                every_n_epochs=1)