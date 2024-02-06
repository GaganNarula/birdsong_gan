"""A Vector Quantized Variational Autoencoder (VQ-VAE) for training on bird song spectrograms."""

import os
import json
from dataclasses import dataclass
import torch
import numpy as np
import matplotlib.pyplot as plt
from datasets import load_from_disk
from diffusers import VQModel
from torch.utils.data import DataLoader
import wandb
from birdsong_gan.utils.audio_utils import random_time_crop_spectrogram


l2loss = torch.nn.MSELoss()


def make_experiment_dir(base_path: str) -> str:
    """Create an experiment directory."""
    import os
    from datetime import datetime

    now = datetime.now()
    dt_string = now.strftime("%Y-%m-%d_%H-%M-%S")
    experiment_dir = os.path.join(base_path, dt_string)
    os.makedirs(experiment_dir, exist_ok=True)
    return experiment_dir


@dataclass
class TrainingConfig:
    """Configuration for training VQ-VAE."""

    lr: float = 2e-5
    batch_size: int = 200
    gradient_accumulation_steps: int = 1
    alpha: float = 10.0
    weight_decay: float = 0.0
    log_every: int = 200
    log_rich_media_every: int = 500
    num_epochs: int = 5
    num_workers: int = 4
    device: str = "cuda"
    spec_dtype: str = "float32"
    ntimeframes: int = 16
    log_scale: bool = True
    vq_embed_dim: int = 512
    num_vq_embeddings: int = 2000
    layers_per_block: int = 3
    dataset_path: str = (
        "/media/gagan/Gagan_external/songbird_data/age_resampled_hfdataset/"
    )
    base_dir: str = "/home/gagan/ek_experiments/vqvae"
    checkpoint_every: int = 1000
    max_grad_norm: float = 1.0


class SpectrogramSnippets(torch.utils.data.Dataset):
    """Dataset to extract random snippets from spectrograms.
    Time duration of the snippet is determined by `ntimeframes`.
    """

    def __init__(
        self,
        ds,
        ntimeframes: int = 16,
        spec_dtype: str = "float32",
        log_scale: bool = True,
    ):
        super().__init__()
        self.ds = ds
        self.ntimeframes = ntimeframes
        self.log_scale = log_scale
        if spec_dtype == "float16":
            self.spec_dtype = torch.float16
        else:
            self.spec_dtype = torch.float32

    def __len__(self) -> int:
        return len(self.ds)

    def __getitem__(self, index: int) -> torch.Tensor:
        x = np.array(self.ds[index]["spectrogram"])
        x = random_time_crop_spectrogram(x, self.ntimeframes)
        if self.log_scale:
            x = np.log1p(x)
        x = torch.from_numpy(x).to(self.spec_dtype)
        x = x.view(1, x.shape[0], x.shape[1])  # add channel dim
        return x


## Helper functions


def plot_reconstruction_and_original(
    x: torch.Tensor, xhat: torch.Tensor, epoch: int, batch: int
) -> None:
    """Use wandb.plot to log original and reconstructed spectrograms.
    Input spectrgrams have shape (batch_size, nfreq, ntimeframes).
    Converts them to (nfreq, ntimeframes x batch_size) for plotting.
    """
    x = x.view(x.shape[0], x.shape[1] * x.shape[2])
    xhat = xhat.view(xhat.shape[0], xhat.shape[1] * xhat.shape[2])
    wandb.log(
        {f"reconstruction_{epoch}_{batch}": wandb.Image(xhat, caption="reconstructed")}
    )
    wandb.log({f"original_{epoch}_{batch}": wandb.Image(x, caption="original")})


def compute_histograms(indices: np.ndarray, num_embeddings: int) -> np.ndarray:
    """Compute histograms of indices for embeddings."""
    histogram = np.zeros(num_embeddings)
    for i in range(num_embeddings):
        histogram[i] = np.sum(indices == i)
    return histogram


def plot_histogram_wandb(histogram) -> None:
    """Use wandb.log to log histogram of indices."""
    wandb.Histogram(np_histogram=histogram)


def train(dataloader: DataLoader, model: VQModel, optimizer, epoch, config):
    """Train VQ-VAE model for one epoch.

    Args:
        dataloader (torch.utils.data.DataLoader): dataloader for training data
        model (VQModel): VQ-VAE model
        optimizer (torch.optim.Optimizer): optimizer
        epoch (int): current epoch
        config (TrainingConfig): training configuration

    Returns:
        VQModel: trained VQ-VAE model
    """
    encoding_indices = []
    running_avg_l2 = 0.0
    running_avg_comm = 0.0

    for i, x in enumerate(dataloader):

        x = x.to(model.device)

        # commitment loss
        ze = model.encode(x)
        zq, commloss, (_, _, min_encoding_indices) = model.quantize(ze.latents)

        encoding_indices.append(min_encoding_indices.detach().cpu().numpy())

        # recon loss
        xhat = model.decode(zq).sample

        l2 = l2loss(xhat, x)

        total_loss = l2 + config.alpha * commloss

        total_loss /= config.gradient_accumulation_steps
        total_loss.backward()

        running_avg_l2 += float(l2.detach())  # detach to avoid memory leak
        running_avg_comm += float(commloss.detach())

        if (i + 1) % config.gradient_accumulation_steps == 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), config.max_grad_norm)
            optimizer.step()
            optimizer.zero_grad()

        if (i + 1) % config.log_every == 0:
            print(f"L2 loss at epoch={epoch}, batch={i} is = {running_avg_l2 / i: .4f}")
            print(
                f"Comm loss at epoch={epoch}, batch={i} is = {running_avg_comm / i: .4f}"
            )
            # log to wandb
            wandb.log(
                {
                    "running_avg_L2_loss": running_avg_l2 / i,
                    "running_avg_commitment_loss": running_avg_comm / i,
                    "L2_loss": running_avg_l2,
                    "commitment_loss": running_avg_comm,
                }
            )

        if (i + 1) % config.log_rich_media_every == 0:
            plot_reconstruction_and_original(x, xhat, epoch, i)
            encoding_indices = np.concatenate(encoding_indices)
            histogram = compute_histograms(encoding_indices, config.num_vq_embeddings)
            plot_histogram_wandb(histogram)
            encoding_indices = []

        if (i + 1) % config.checkpoint_every == 0:
            torch.save(
                model.state_dict(),
                f"vqvae_checkpoint_{epoch}_{i}.pt",
            )

    return model


def main():
    """Main function for training VQ-VAE."""

    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--lr", type=float, default=None)
    parser.add_argument("--batch_size", type=int, default=None)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=None)
    parser.add_argument("--alpha", type=float, default=None)
    parser.add_argument("--weight_decay", type=float, default=None)
    parser.add_argument("--log_every", type=int, default=None)
    parser.add_argument("--log_rich_media_every", type=int, default=None)
    parser.add_argument("--num_epochs", type=int, default=None)
    parser.add_argument("--num_workers", type=int, default=None)
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--spec_dtype", type=str, default=None)
    parser.add_argument("--ntimeframes", type=int, default=None)
    parser.add_argument("--log_scale", type=bool, default=None)
    parser.add_argument("--vq_embed_dim", type=int, default=None)
    parser.add_argument("--num_vq_embeddings", type=int, default=None)
    parser.add_argument("--layers_per_block", type=int, default=None)
    parser.add_argument("--dataset_path", type=str, default=None)
    parser.add_argument("--base_dir", type=str, default=None)

    args = parser.parse_args()

    config = TrainingConfig()

    # overwrite config with args
    for k, v in vars(args).items():
        if v is not None:
            setattr(config, k, v)

    # load dataset
    ds = load_from_disk(config.dataset_path)

    train_ds = SpectrogramSnippets(
        ds,
        ntimeframes=config.ntimeframes,
        spec_dtype=config.spec_dtype,
        log_scale=config.log_scale,
    )
    print(
        f"Loaded dataset from {config.dataset_path}, number of samples: {len(train_ds)}"
    )

    # create dataloader
    train_loader = DataLoader(
        train_ds,
        batch_size=config.batch_size,
        shuffle=True,
        num_workers=config.num_workers,
    )

    # create model
    model = VQModel(
        in_channels=1,
        out_channels=1,
        latent_channels=1,
        num_vq_embeddings=config.num_vq_embeddings,
        vq_embed_dim=config.vq_embed_dim,
        layers_per_block=config.layers_per_block,
        sample_size=config.batch_size,
    )
    num_params = model.num_parameters()
    print(f"Number of parameters in model: {num_params}")

    optimizer = torch.optim.AdamW(
        model.parameters(), lr=config.lr, weight_decay=config.weight_decay
    )

    model.to(config.device)
    model.train()

    experiment_dir = make_experiment_dir(config.base_dir)
    print(f"Experiment directory created at: {experiment_dir}")

    # INIT PROJECT
    wandb.init(project="vqvae", config=vars(config), dir=experiment_dir)

    # save config
    with open(os.path.join(experiment_dir, "experiment_config.json"), "w") as f:
        json.dump(vars(config), f)

    for epoch in range(config.num_epochs):

        model = train(train_loader, model, optimizer, epoch, config)


if __name__ == "__main__":

    main()
