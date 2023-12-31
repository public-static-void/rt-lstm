#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Authors       : Vadim Titov
Matr.-Nr.     : 6021356
Created       : June 23rd, 2022
Last modified : January 26th, 2023
Description   : Master's Project "Source Separation for Robot Control"
Topic         : Prediction module of the LSTM RNN Project
"""

import hyperparameters as hp
import performance
import pytorch_lightning as pl
from net import LitNeuralNet


def main():
    """Main function.

    Performs prediction functionality for a pretrained LSTM.
    """
    # Load pretrained net from checkpoint.
    trained_model = LitNeuralNet.load_from_checkpoint(
        checkpoint_path=hp.trained_model_path,
        batch_size=1,
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
    predictions = trainer.predict(trained_model, None, None)
    performance.__calculate_pesq__()


if __name__ == "__main__":
    main()
