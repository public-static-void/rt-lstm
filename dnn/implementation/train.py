#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Authors       : Vadim Titov
Matr.-Nr.     : 6021356
Created       : June 23rd, 2022
Last modified : January 26th, 2023
Description   : Master's Project "Source Separation for Robot Control"
Topic         : Training module of the LSTM RNN Project
"""

import hyperparameters as hp
import pytorch_lightning as pl
from data import HDF5DataModule
from net import LitNeuralNet


def main():
    """Main function.

    Performs training functionality for the LSTM.
    """
    # Initialize trainer.
    trainer = pl.Trainer(
        # default_root_dir=hp.CHECKPOINT_DIR,
        fast_dev_run=hp.is_test_run,
        accelerator=hp.device,
        devices=hp.num_devices,
        max_epochs=hp.num_epochs,
        enable_checkpointing=hp.enable_checkpointing,
        callbacks=[hp.early_stopping, hp.checkpointing],
        log_every_n_steps=hp.log_every_n_steps,
        logger=hp.tb_logger,
        limit_train_batches=hp.limit_train_batches,
        limit_val_batches=hp.limit_val_batches,
        overfit_batches=hp.overfit_batches,
        auto_lr_find=hp.auto_lr_find,
    )
    # Initialize net.
    model = LitNeuralNet(
        input_size=hp.input_size,
        hidden_size_1=hp.hidden_size_1,
        hidden_size_2=hp.hidden_size_2,
        output_size=hp.output_size,
        batch_size=hp.batch_size,
    )
    print(model)
    if hp.auto_lr_find is True:
        # Let lightning try to find ideal learning rate and batch size.
        trainer.tune(
            model,
        )
        if hp.auto_lr_find is True:
            model.learning_rate
    # Train model.
    trainer.fit(
        model,
    )


if __name__ == "__main__":
    main()
