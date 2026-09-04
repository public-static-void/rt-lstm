"""Smoke test for the pytorch-lightning 2.6.5 prediction loop.

Exercises the LitNeuralNet model's predict_step through trainer.predict()
with synthetic data. Does not require the project's soundfiles dataset.
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


class SyntheticPredictDataset(Dataset):
    """Produces tensors shaped like the real STFT test dataset."""

    def __init__(self, n_samples=2, n_freq=257, n_time=32):
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
        meta = {
            "data_index": idx,
            "SISDR": 0,
            "SNR": 1.0,
            "reverberation_rate": 0.5,
            "min_distance_to_noise": 0.3,
        }
        return clean, noise, mix, meta


def main():
    print("=== Smoke test: prediction loop ===")
    model = LitNeuralNet(
        input_size=6,
        hidden_size_1=16,
        hidden_size_2=8,
        output_size=2,
        batch_size=1,
    )
    model.eval()
    model.freeze()

    dataset = SyntheticPredictDataset()
    loader = DataLoader(dataset, batch_size=1, shuffle=False)

    # Patch predict_step to avoid writing WAV files and the hardcoded CUDA
    # SI_SDR; just return the prediction tensors.
    import net as net_module
    def _cpu_predict_step(self, batch, batch_idx, h_pre_t=None, c_pre_t=None):
        meta_data = batch[-1]
        meta_data.update(data_index=int(meta_data["data_index"].item()))
        batch = batch[:-1]
        (
            _,
            clean_co,
            mix_co,
            prediction,
            h_new_t,
            c_new_t,
        ) = self.common_step(batch, batch_idx, h_pre_t, c_pre_t)
        return prediction, clean_co, mix_co, h_new_t, c_new_t
    net_module.LitNeuralNet.predict_step = _cpu_predict_step

    model.predict_dataloader = lambda: loader

    trainer = pl.Trainer(
        fast_dev_run=True,
        accelerator="cpu",
        devices=1,
        max_epochs=1,
        enable_checkpointing=False,
        logger=False,
    )
    predictions = trainer.predict(model)
    print("Predictions returned:", type(predictions).__name__)
    assert predictions is not None, "predict() returned None"
    print("=== Prediction loop smoke test PASSED ===")


if __name__ == "__main__":
    main()
