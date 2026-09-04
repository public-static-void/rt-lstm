"""Smoke test for the pytorch-lightning 2.6.5 training loop.

Exercises the LitNeuralNet model with synthetic data through a full
train/val loop using fast_dev_run. Does not require the project's
soundfiles dataset.
"""
import sys
import os

# Force CPU so the model's hidden states and data stay on the same device.
os.environ["CUDA_VISIBLE_DEVICES"] = ""

# Make the project's implementation modules importable.
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import torch
import pytorch_lightning as pl
from torch.utils.data import Dataset, DataLoader

import hyperparameters as hp
from net import LitNeuralNet


class SyntheticDataset(Dataset):
    """Produces tensors shaped like the real STFT dataset."""

    def __init__(self, n_samples=4, n_freq=257, n_time=32):
        self.n_samples = n_samples
        self.n_freq = n_freq
        self.n_time = n_time

    def __len__(self):
        return self.n_samples

    def __getitem__(self, idx):
        # clean/noise/mix each: [6, n_freq, n_time] (3 ch x 2 re/im)
        clean = torch.randn(6, self.n_freq, self.n_time)
        noise = torch.randn(6, self.n_freq, self.n_time)
        mix = clean + noise
        return clean, noise, mix


def main():
    print("=== Smoke test: training loop with fast_dev_run ===")
    model = LitNeuralNet(
        input_size=6,
        hidden_size_1=16,
        hidden_size_2=8,
        output_size=2,
        batch_size=2,
    )
    print("Model instantiated:", type(model).__name__)

    dataset = SyntheticDataset()
    loader = DataLoader(dataset, batch_size=2, shuffle=False)

    # The model's validation_step hardcodes SI_SDR().to("cuda"); patch the
    # method to target the actual device so the smoke test can run on CPU.
    import net as net_module
    def _cpu_validation_step(self, batch, batch_idx, h_pre_t=None, c_pre_t=None):
        import torchmetrics
        si_sdr = torchmetrics.ScaleInvariantSignalDistortionRatio().to(hp.device)
        loss, clean_co, mix_co, prediction, _, _ = self.common_step(
            batch, batch_idx, h_pre_t, c_pre_t
        )
        self.log("val/loss", loss, on_step=hp.on_step, on_epoch=hp.on_epoch, logger=hp.logger)
        clean_istft = torch.istft(clean_co, hp.stft_length, hp.stft_shift, window=hp.window)
        pred_istft = torch.istft(prediction, hp.stft_length, hp.stft_shift, window=hp.window)
        si_sdr_val = si_sdr(pred_istft, clean_istft)
        self.log("val/si_sdr", si_sdr_val, on_step=hp.on_step, on_epoch=hp.on_epoch, logger=hp.logger)
        return loss
    net_module.LitNeuralNet.validation_step = _cpu_validation_step

    # Override dataloaders to use synthetic data.
    model.train_dataloader = lambda: loader
    model.val_dataloader = lambda: loader

    trainer = pl.Trainer(
        fast_dev_run=True,
        accelerator="cpu",
        devices=1,
        max_epochs=1,
        enable_checkpointing=False,
        logger=False,
    )
    trainer.fit(model)
    print("=== Training loop smoke test PASSED ===")


if __name__ == "__main__":
    main()
