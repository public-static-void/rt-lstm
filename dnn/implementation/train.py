#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Authors       : Vadim Titov
Matr.-Nr.     : 6021356
Created       : June 23th, 2022
Last modified : October 20th, 2022
Description   : Master's Project "Source Separation for Robot Control"
Topic         : Training module of the LSTM RNN Project
"""

import hyperparameters as hp
import pytorch_lightning as pl
from net import LitNeuralNet


def main():
    # train model

    trainer = pl.Trainer(
        default_root_dir=hp.CHECKPOINT_DIR,
        fast_dev_run=hp.is_test_run,
        accelerator=hp.device,
        devices=hp.num_devices,
        max_epochs=hp.num_epochs,
        enable_checkpointing=True,
        callbacks=[hp.early_stopping, hp.checkpointing],
        log_every_n_steps=1,
        logger=hp.tb_logger,
    )
    model = LitNeuralNet(
        hp.input_size, hp.hidden_size_1, hp.hidden_size_2, hp.output_size
    )
    print(model)
    trainer.fit(
        model,
        LitNeuralNet.train_dataloader(model),
        LitNeuralNet.val_dataloader(model),
    )


if __name__ == "__main__":
    main()
