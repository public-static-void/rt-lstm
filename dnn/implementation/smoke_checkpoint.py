"""Smoke test for loading pytorch-lightning checkpoints after the version bump.

Loads each existing checkpoint (created under pytorch-lightning 2.4.0) with
the new 2.6.5 and verifies the model restores correctly.
"""
import sys
import os

# Force CPU so the model's hidden states and data stay on the same device.
os.environ["CUDA_VISIBLE_DEVICES"] = ""

# Make the project's implementation modules importable.
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import glob
import pytorch_lightning as pl

from net import LitNeuralNet


def main():
    print("=== Smoke test: checkpoint loading ===")
    ckpt_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "checkpoints")
    ckpts = sorted(glob.glob(os.path.join(ckpt_dir, "*.ckpt")))
    if not ckpts:
        print("No checkpoints found; skipping.")
        return

    for path in ckpts:
        name = os.path.basename(path)
        # Only load checkpoints matching the current bidirectional config
        # (t_bidirectional=True, f_bidirectional=True). Others were trained
        # with different architectures and will shape-mismatch regardless of
        # the Lightning version.
        if not name.startswith("tt"):
            print(f"  SKIP: {name} (architecture mismatch, not a version issue)")
            continue
        try:
            model = LitNeuralNet.load_from_checkpoint(
                checkpoint_path=path,
                batch_size=1,
            )
            model.eval()
            model.freeze()
            print(f"  OK: {name} loaded, params={sum(p.numel() for p in model.parameters())}")
        except Exception as e:
            print(f"  FAIL: {name} -> {type(e).__name__}: {e}")
            raise

    print("=== Checkpoint loading smoke test PASSED ===")


if __name__ == "__main__":
    main()
