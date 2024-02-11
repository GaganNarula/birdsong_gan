"""A Vector Quantized Variational Autoencoder (VQ-VAE) for training on bird song spectrograms."""

import os
import json
import argparse
from datetime import datetime
from dataclasses import dataclass
import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
from datasets import load_from_disk
from diffusers import VQModel
from torch.utils.data import DataLoader
from transformers import get_scheduler
from scipy.stats import entropy
import wandb
from birdsong_gan.utils.audio_utils import (
    random_time_crop_spectrogram,
    rescale_spectrogram,
)


l2loss = torch.nn.MSELoss()


def make_experiment_dir(base_path: str) -> str:
    """Create an experiment directory."""
    now = datetime.now()
    dt_string = now.strftime("%Y-%m-%d_%H-%M-%S")
    experiment_dir = os.path.join(base_path, dt_string)
    os.makedirs(experiment_dir, exist_ok=True)
    return experiment_dir


class VQVAEModel(nn.Module):
    """Wrap model to use new quantize function."""

    def __init__(self, vq, l2_normalize_latents: bool = False, scale: float = None):
        super().__init__()
        self.vq = vq
        self.l2_normalize_latents = l2_normalize_latents
        self.output_activation = torch.nn.ReLU()
        if scale is not None:
            self.scale = scale
        else:
            self.scale = self.vq.quantize.n_e
        # special initialization
        self.reinit_code_vectors()

    def reinit_code_vectors(self) -> None:
        self.vq.quantize.embedding.weight.data.uniform_(
            -1.0 / self.scale, 1.0 / self.scale
        )

    def reinit_code_vector(self, code: int) -> None:
        """Reinitialize a code vector at index `code`."""
        self.vq.quantize.embedding.weight.data[code].uniform_(
            -1.0 / self.scale, 1.0 / self.scale
        )

    def encode(
        self, x: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Encodes input into latent space and computes nearest neighbor codevectors."""
        ze = self.vq.encode(x)
        if self.l2_normalize_latents:
            ze.latents = ze.latents / ze.latents.norm(dim=-1, keepdim=True)
        zq, commloss, (perplexity, _, codes) = self.vq.quantize(ze.latents)
        return zq, commloss, codes, perplexity

    def decode(self, latents: torch.Tensor) -> torch.Tensor:
        """Decodes latents into output space."""
        xhat = self.vq.decode(latents).sample
        xhat = self.output_activation(xhat)
        return xhat

    @torch.no_grad()
    def infer_latent_code(self, x: torch.Tensor) -> torch.Tensor:
        """Infer latent code for input."""
        _, _, codes, _ = self.encode(x)
        return codes

    @torch.no_grad()
    def reconstruct(self, x: torch.Tensor) -> torch.Tensor:
        """Reconstruct input."""
        zq, _, _, _ = self.encode(x)
        return self.decode(zq)

    def forward(
        self, x: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Forward pass. Encodes input, computes reconstruction and loss."""
        # encode
        zq, commloss, codes, perplexity = self.encode(x)
        # decode
        xhat = self.decode(zq)
        l2 = l2loss(xhat, x)
        return xhat, l2, commloss, codes, perplexity


@dataclass
class TrainingConfig:
    """Configuration for training VQ-VAE."""

    lr: float = 2e-5
    scheduler: str = "cosine"
    last_checkpoint_path: str = None
    num_warmup_steps: int = 2000
    num_training_steps: int = 10200
    batch_size: int = 250
    gradient_accumulation_steps: int = 1
    alpha: float = 10.0  # weight for commitment loss
    vq_beta: float = (
        0.25  # balances between pushing codebook to latents vs pushing latents to codebook
    )
    weight_decay: float = 0.0
    l2_normalize_latents: bool = True
    log_every: int = 20
    log_rich_media_every: int = 20
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
    experiment_dir: str = ""  # will be set later
    checkpoint_every: int = 1000
    check_for_unused_codes_every: int = (
        20  # from https://proceedings.mlr.press/v202/huh23a/huh23a.pdf
    )
    replace_unused_codes_threshold: int = (
        1  # from https://proceedings.mlr.press/v202/huh23a/huh23a.pdf
    )
    max_grad_norm: float = 1.0

    def from_dict(self, d: dict) -> None:
        """Update config from dictionary."""
        for k, v in d.items():
            if k in self.__dict__:
                setattr(self, k, v)


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


def plot_reconstruction_and_original_wandb(
    x: torch.Tensor, xhat: torch.Tensor, epoch: int, batch: int
) -> None:
    """Use wandb.plot to log original and reconstructed spectrograms.
    Input spectrgrams have shape (batch_size, nfreq, ntimeframes).
    Converts them to (nfreq, ntimeframes x batch_size) for plotting.
    """
    x = x.squeeze()
    x = x.view(x.shape[1], x.shape[0] * x.shape[2]).cpu().detach().numpy()
    xhat = xhat.squeeze()
    xhat = (
        xhat.view(xhat.shape[1], xhat.shape[0] * xhat.shape[2]).cpu().detach().numpy()
    )
    x = rescale_spectrogram(x)
    xhat = rescale_spectrogram(xhat)
    wandb.log(
        {f"reconstruction_{epoch}_{batch}": wandb.Image(xhat, caption="reconstructed")}
    )
    wandb.log({f"original_{epoch}_{batch}": wandb.Image(x, caption="original")})


def plot_reconstruction_and_original(
    x: torch.Tensor,
    xhat: torch.Tensor,
    epoch: int,
    batch: int,
    experiment_dir: str,
    figsize: tuple[int, int] = (20, 6),
) -> None:
    """Plot original and reconstructed spectrograms.

    Input spectrgrams have shape (batch_size, nfreq, ntimeframes).
    Converts them to (nfreq, ntimeframes x batch_size) for plotting.

    :param x: original spectrogram
    :type x: torch.Tensor
    :param xhat: reconstructed spectrogram
    :type xhat: torch.Tensor
    :param epoch: current epoch
    :type epoch: int
    :param batch: current batch
    :type batch: int
    :param experiment_dir: directory to save plots
    :type experiment_dir: str
    :param figsize: figure size, defaults to (12, 8)
    :type figsize: tuple[int, int], optional

    """
    x = x.squeeze()
    x = np.concatenate([xx.cpu().detach().numpy() for xx in x], axis=-1)
    xhat = xhat.squeeze()
    xhat = np.concatenate([xx.cpu().detach().numpy() for xx in xhat], axis=-1)
    x = rescale_spectrogram(x)
    xhat = rescale_spectrogram(xhat)
    fig, axes = plt.subplots(nrows=2, ncols=1, figsize=figsize)
    axes[0].imshow(x, origin="lower", cmap="gray")
    axes[0].set_title("Original")
    axes[1].imshow(xhat, origin="lower", cmap="gray")
    axes[1].set_title("Reconstructed")
    # Adjust the space between the subplots
    plt.subplots_adjust(hspace=0.2)  # Adjusts the horizontal space
    plt.savefig(os.path.join(experiment_dir, f"reconstruction_{epoch}_{batch}.png"))
    plt.close(fig)


def compute_histograms(indices: np.ndarray, num_embeddings: int) -> np.ndarray:
    """Compute histograms of indices for embeddings."""
    histogram = np.zeros(num_embeddings)
    for i in range(num_embeddings):
        histogram[i] = np.sum(indices == i)
    return histogram


def plot_codebook_histogram(
    num_embeddings: int,
    experiment_dir: str,
    epoch: int,
    batch: int,
    indices: np.ndarray = None,
    histogram: np.ndarray = None,
) -> None:
    """Plot histogram of indices."""
    if histogram is None and indices is None:
        raise ValueError("Either indices or histogram must be provided.")
    if histogram is None:
        histogram = compute_histograms(indices, num_embeddings)
    plt.bar(np.arange(num_embeddings), np.log10(histogram))
    plt.savefig(
        os.path.join(experiment_dir, f"codebook_log_histogram_{epoch}_{batch}.png")
    )
    plt.close()


def plot_histogram_wandb(indices) -> None:
    """Use wandb.log to log histogram of indices."""
    wandb.log({"codebook_histogram": wandb.Histogram(indices, num_bins=512)})


def replace_unused_codes(
    model, codebook_histogram: np.ndarray, threshold: int = 10
) -> tuple[torch.nn.Module, int]:
    """Replace unused codes in codebook with random embeddings."""
    unused_codes = np.where(codebook_histogram < threshold)[0]
    for code in unused_codes:
        model.reinit_code_vector(code)

    return model.train(), len(unused_codes)


def logging(
    model: torch.nn.Module,
    x: torch.Tensor,
    xhat: torch.Tensor,
    encoding_indices: list[np.ndarray],
    running_avg_comm: float,
    running_avg_l2: float,
    l2: float,
    commloss: float,
    epoch: int,
    batch: int,
    config,
) -> None:
    """Logs losses to wandb and prints / saves figures to disk, checkpoints model."""

    if (batch + 1) % config.log_every == 0:
        print(
            f"L2 loss at epoch={epoch}, batch={batch} is = {running_avg_l2 / batch: .4f}"
        )
        print(
            f"Comm loss at epoch={epoch}, batch={batch} is = {running_avg_comm / batch: .4f}"
        )
        # log to wandb
        wandb.log(
            {
                "running_avg_L2_loss": running_avg_l2 / batch,
                "running_avg_commitment_loss": running_avg_comm / batch,
                "L2_loss": float(l2.detach()),
                "commitment_loss": float(commloss.detach()),
            }
        )

    if (batch + 1) % config.log_rich_media_every == 0:
        plot_reconstruction_and_original(x, xhat, epoch, batch, config.experiment_dir)

    if (batch + 1) % config.checkpoint_every == 0:
        torch.save(
            model.state_dict(),
            os.path.join(config.experiment_dir, f"model_checkpoint_{epoch}_{batch}.pt"),
        )

    if (batch + 1) % config.check_for_unused_codes_every == 0:

        code_histogram = compute_histograms(
            np.concatenate(encoding_indices).flatten(), config.num_vq_embeddings
        )
        # plot_codebook_histogram(
        #     num_embeddings=config.num_vq_embeddings,
        #     experiment_dir=config.experiment_dir,
        #     epoch=epoch,
        #     batch=batch,
        #     histogram=code_histogram,
        # )
        # compute entropy from code_histogram
        entropy_ = entropy(code_histogram)
        # normalize by maximum entropy possible
        entropy_ /= np.log2(config.num_vq_embeddings)
        print(f"Normalized entropy of codebook histogram: {entropy_}")
        wandb.log({"codebook_entropy": entropy_})

        if config.replace_unused_codes_threshold is not None:
            model, num_codes_replaced = replace_unused_codes(
                model, code_histogram, config.replace_unused_codes_threshold
            )
            print(f"Replaced {num_codes_replaced} unused codes.")
            encoding_indices = []

    return model, encoding_indices


def train(
    dataloader: DataLoader,
    model: VQVAEModel,
    optimizer: torch.optim.Optimizer,
    scheduler,
    epoch: int,
    config: TrainingConfig,
):
    """Train VQ-VAE model for one epoch.

    Args:
        dataloader (torch.utils.data.DataLoader): dataloader for training data
        model (VQVAEModel): VQ-VAE model
        optimizer (torch.optim.Optimizer): optimizer
        scheduler (torch.optim.lr_scheduler): learning rate scheduler
        epoch (int): current epoch
        config (TrainingConfig): training configuration

    Returns:
        VQModel: trained VQ-VAE model
    """
    encoding_indices = []
    running_avg_l2 = 0.0
    running_avg_comm = 0.0

    for i, x in enumerate(dataloader):

        x = x.to(config.device)

        xhat, l2, commloss, codes, _ = model(x)

        encoding_indices.append(codes.detach().cpu().numpy())

        total_loss = l2 + config.alpha * commloss

        total_loss /= config.gradient_accumulation_steps
        total_loss.backward()

        running_avg_l2 += float(l2.detach())  # detach to avoid memory leak
        running_avg_comm += float(commloss.detach())

        if (i + 1) % config.gradient_accumulation_steps == 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), config.max_grad_norm)
            optimizer.step()
            optimizer.zero_grad()
            scheduler.step()
            if (i + 1) % config.log_every == 0:
                print(f"Latest learning rate: {scheduler.get_last_lr()}")

        model, encoding_indices = logging(
            model,
            x,
            xhat,
            encoding_indices,
            running_avg_comm,
            running_avg_l2,
            l2,
            commloss,
            epoch,
            i,
            config,
        )

    return model


def main():
    """Main function for training VQ-VAE."""

    parser = argparse.ArgumentParser()
    parser.add_argument("--lr", type=float, default=None)
    parser.add_argument("--scheduler", type=str, default=None)
    parser.add_argument("--num_warmup_steps", type=int, default=None)
    parser.add_argument("--num_training_steps", type=int, default=None)
    parser.add_argument("--vq_beta", type=float, default=None)
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
    parser.add_argument("--checkpoint_every", type=int, default=None)
    parser.add_argument("--last_checkpoint_path", type=str, default=None)
    parser.add_argument("--check_for_unused_codes_every", type=int, default=None)
    parser.add_argument("--replace_unused_codes_threshold", type=int, default=None)
    parser.add_argument("--max_grad_norm", type=float, default=None)

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
    vq = VQModel(
        in_channels=1,
        out_channels=1,
        latent_channels=1,
        num_vq_embeddings=config.num_vq_embeddings,
        vq_embed_dim=config.vq_embed_dim,
        layers_per_block=config.layers_per_block,
        sample_size=config.batch_size,
    )
    num_params = vq.num_parameters()
    print(f"Number of parameters in model: {num_params}")

    vq.quantize.legacy = False  # use new quantize function
    vq.quantize.beta = config.vq_beta

    model = VQVAEModel(vq)
    if config.last_checkpoint_path is not None:
        model.load_state_dict(torch.load(config.last_checkpoint_path))
    model.to(config.device)
    model.train()

    optimizer = torch.optim.AdamW(
        model.parameters(), lr=config.lr, weight_decay=config.weight_decay
    )

    scheduler = get_scheduler(
        config.scheduler,
        optimizer,
        num_warmup_steps=config.num_warmup_steps,
        num_training_steps=config.num_training_steps,
    )

    experiment_dir = make_experiment_dir(config.base_dir)
    config.experiment_dir = experiment_dir
    print(f"Experiment directory created at: {experiment_dir}")

    # INIT PROJECT
    wandb.init(project="vqvae", config=vars(config), dir=experiment_dir)

    # save config
    with open(
        os.path.join(experiment_dir, "experiment_config.json"), "w", encoding="utf-8"
    ) as f:
        json.dump(vars(config), f)

    for epoch in range(config.num_epochs):

        model = train(train_loader, model, optimizer, scheduler, epoch, config)


if __name__ == "__main__":

    main()
