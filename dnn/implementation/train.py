#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Authors       : Vadim Titov
Matr.-Nr.     : 6021356
Created       : June 23th, 2022
Last modified : July 7th, 2022
Description   : Master's Project "Source Separation for Robot Control"
Topic         : Training module of the LSTM RNN Project
"""

from net import LitNeuralNet
import hyperparameters

def main():

    # train model

    # Running in fast_dev_run mode: will run a full train, val, test and
    # prediction loop using 1 batch(es).
    trainer = pl.Trainer(fast_dev_run=is_test_run, accelerator=device,
                         devices=num_devices, max_epochs=num_epochs,
                         enable_checkpointing=True, callbacks=[early_stopping,
                                                               checkpointing],
                         log_every_n_steps=1, logger=tb_logger)
    model = LitNeuralNet(input_size, hidden_size, num_classes)
    # print(model)
    trainer.fit(model, LitNeuralNet.train_dataloader, LitNeuralNet.val_dataloader)

if __name__ == '__main__':
    main()