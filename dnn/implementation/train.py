#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Authors       : Vadim Titov
Matr.-Nr.     : 6021356
Created       : June 23rd, 2022
Last modified : November 26th, 2022
Description   : Master's Project "Source Separation for Robot Control"
Topic         : Training module of the LSTM RNN Project
"""

import dataloaders
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
        default_root_dir=hp.CHECKPOINT_DIR,
        fast_dev_run=hp.is_test_run,
        accelerator=hp.device,
        devices=hp.num_devices,
        max_epochs=hp.num_epochs,
        enable_checkpointing=hp.enable_checkpointing,
        callbacks=[hp.early_stopping, hp.checkpointing],
        log_every_n_steps=hp.log_every_n_steps,
        logger=hp.tb_logger,
        limit_train_batches=hp.limit_train_batches,
        overfit_batches=hp.overfit_batches,
        auto_lr_find=hp.auto_lr_find,
    )
    # Initialize net.
    model = LitNeuralNet(
        hp.input_size, hp.hidden_size_1, hp.hidden_size_2, hp.output_size
    )
    print(model)
    if hp.auto_lr_find is True:
        # Let lightning try to find ideal learning rate.
        trainer.tune(
            model,
            dataloaders.train_dataloader(),
            dataloaders.val_dataloader(),
        )
        model.learning_rate
    # Train model.
    trainer.fit(
        model,
        # Select dataloaders.
        dataloaders.train_dataloader(),
        dataloaders.val_dataloader(),
        # HDF5DataModule(
        #     batch_size=hp.batch_size,
        #     prep_files={
        #         "data": "/data/test/prep_mix_mix_ch3_sp5_dir0.hdf5",
        #         "meta": "/data/test/prep_mix_meta_mix_ch3_sp5_dir0.json",
        #     },
        #     stft_length_samples=hp.stft_length,
        #     stft_shift_samples=hp.stft_shift,
        #     snr_range=None,
        #     meta_frame_length=3 * 16000,
        #     n_workers=hp.num_workers,
        #     n_speakers=1,
        #     fs=16000,
        # ),
    )


if __name__ == "__main__":
    main()
