#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Authors       : Vadim Titov
Matr.-Nr.     : 6021356
Created       : June 23rd, 2022
Last modified : November 26th, 2022
Description   : Master's Project "Source Separation for Robot Control"
Topic         : Prediction module of the LSTM RNN Project
"""

import dataloaders
import hyperparameters as hp
import pytorch_lightning as pl
from net import LitNeuralNet
import performance

def main():
    """Main function.

    Performs prediction functionality for a pretrained LSTM.
    """
    # Load pretrained net from checkpoint.
    trained_model = LitNeuralNet.load_from_checkpoint(
        # TODO: Automatically load best checkpoint? At the moment it's
        # done manually by explicitly passing path and filename.
        checkpoint_path=hp.CHECKPOINT_DIR
        + hp.checkpoint_name
    )
    trained_model.eval()
    trained_model.freeze()

    # Initialize trainer.
    trainer = pl.Trainer(
        fast_dev_run=hp.is_test_run,
        accelerator=hp.device,
        devices=hp.num_devices,
        max_epochs=hp.num_epochs,
        enable_checkpointing=hp.enable_checkpointing,
        callbacks=[hp.early_stopping, hp.checkpointing],
        log_every_n_steps=hp.log_every_n_steps,
        logger=hp.tb_logger,
        limit_predict_batches=hp.limit_predict_batches,
    )

    # Perform prediction.
    predictions = trainer.predict(trained_model, dataloaders.test_dataloader(),
                                  None, None)
    # performance.__calculate_pesq__()


if __name__ == "__main__":
    main()
