#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Authors       : Vadim Titov
Matr.-Nr.     : 6021356
Created       : June 23th, 2022
Last modified : July 7th, 2022
Description   : Master's Project "Source Separation for Robot Control"
Topic         : Prediction module of the LSTM RNN Project
"""

from net import LitNeuralNet
import hyperparameters

def main():

    # load model
    trained_model = LitNeuralNet.load_from_checkpoint(
        #checkpoint_path=CHECKPOINT_DIR)
        checkpoint_path=checkpointing.best_model_path)

    # predict
    trained_model.eval()
    trained_model.freeze()

    trainer = pl.Trainer(fast_dev_run=is_test_run, accelerator=device,
                         devices=num_devices, max_epochs=num_epochs,
                         enable_checkpointing=True, callbacks=[early_stopping,
                                                               checkpointing],
                         log_every_n_steps=1, logger=tb_logger)
    predictions = trainer.predict(trained_model, LitNeuralNet.test_dataloader)

if __name__ == '__main__':
    main()