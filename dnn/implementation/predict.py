#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Authors       : Vadim Titov
Matr.-Nr.     : 6021356
Created       : June 23th, 2022
Last modified : October 20th, 2022
Description   : Master's Project "Source Separation for Robot Control"
Topic         : Prediction module of the LSTM RNN Project
"""

import hyperparameters as hp
import pytorch_lightning as pl
from net import LitNeuralNet


def main():
    # load model
    trained_model = LitNeuralNet.load_from_checkpoint(
        # checkpoint_path=hp.CHECKPOINT_DIR + hp.CHECKPOINT_NAME
        checkpoint_path=hp.checkpointing.best_model_path
    )

    # predict
    trained_model.eval()
    trained_model.freeze()

    trainer = pl.Trainer(
        fast_dev_run=hp.is_test_run,
        accelerator=hp.device,
        devices=hp.num_devices,
        max_epochs=hp.num_epochs,
        enable_checkpointing=True,
        callbacks=[hp.early_stopping, hp.checkpointing],
        log_every_n_steps=1,
        logger=hp.tb_logger,
    )
    predictions = trainer.predict(trained_model, LitNeuralNet.test_dataloader)


if __name__ == "__main__":
    main()
