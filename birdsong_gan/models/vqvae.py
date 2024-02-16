import os
import json
import torch
import torch.nn as nn
from diffusers import VQModel
from diffusers.models.unets.unet_2d_blocks import Downsample2D, Upsample2D

l2loss = nn.MSELoss()


class VQVAEModel(nn.Module):
    """Wrap model to use new quantize function."""

    def __init__(
        self,
        vq: VQModel = None,
        l2_normalize_latents: bool = False,
        scale: float = None,
        downsample: bool = False,
    ):
        super().__init__()

        if vq is None:
            print("Creating new VQVAE model with default parameters.")
            vq = VQModel()

        self.vq = vq

        if downsample:
            self.downsample = nn.Sequential(
                Downsample2D(
                    channels=1, use_conv=True, out_channels=1, kernel_size=3, padding=1
                ),
                nn.SiLU(),
                Downsample2D(
                    channels=1, use_conv=True, out_channels=1, kernel_size=3, padding=0
                ),
            )
            self.upsample = nn.Sequential(
                Upsample2D(
                    channels=1, use_conv=True, out_channels=1, kernel_size=3, padding=1
                ),
                nn.SiLU(),
                Upsample2D(
                    channels=1, use_conv=True, out_channels=1, kernel_size=3, padding=1
                ),
            )

        else:
            self.downsample = None
            self.upsample = None

        self.l2_normalize_latents = l2_normalize_latents
        self.output_activation = torch.nn.ReLU()
        if scale is not None:
            self.scale = scale
        else:
            self.scale = self.vq.quantize.n_e
        # special initialization
        self.reinit_code_vectors()

    @classmethod
    def from_pretrained(
        cls, model_dir: str, checkpoint_path: str = None
    ) -> "VQVAEModel":
        """Load pretrained model."""
        # first load config
        with open(os.path.join(model_dir, "experiment_config.json"), "r") as f:
            config = json.load(f)
        # create model
        vq = VQModel(
            in_channels=1,
            out_channels=1,
            latent_channels=1,
            num_vq_embeddings=config["num_vq_embeddings"],
            vq_embed_dim=config["vq_embed_dim"],
            layers_per_block=config["layers_per_block"],
            sample_size=config["batch_size"],
        )
        num_params = vq.num_parameters()
        print(f"Number of parameters in model: {num_params}")

        vq.quantize.legacy = False  # use new quantize function
        vq.quantize.beta = config["vq_beta"]

        model = cls(
            vq,
            l2_normalize_latents=config["l2_normalize_latents"],
            scale=config["scale"],
            downsample=config["downsample"],
        )

        if checkpoint_path is None and config["last_checkpoint_path"] is not None:
            model.load_state_dict(torch.load(config.last_checkpoint_path), strict=False)
        elif checkpoint_path is not None:
            model.load_state_dict(torch.load(checkpoint_path), strict=False)
        else:
            print(
                "No checkpoint path provided. Model will be initialized with random weights."
            )

        model.to(config["device"])
        return model

    def reinit_code_vector(self, code: int) -> None:
        """Reinitialize a code vector at index `code`."""
        self.vq.quantize.embedding.weight.data[code].uniform_(
            -1.0 / self.scale, 1.0 / self.scale
        )

    def reinit_code_vectors(self) -> None:
        """Reinitialize all code vectors."""
        for i in range(self.vq.quantize.n_e):
            self.reinit_code_vector(i)

    def encode(
        self, x: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Encodes input into latent space and computes nearest neighbor codevectors."""
        ze = self.vq.encode(x)
        if self.downsample is not None:
            ze.latents = self.downsample(ze.latents)
        if self.l2_normalize_latents:
            ze.latents = ze.latents / ze.latents.norm(dim=-1, keepdim=True)
        zq, commloss, (perplexity, _, codes) = self.vq.quantize(ze.latents)
        return zq, commloss, codes, perplexity

    def decode(self, latents: torch.Tensor) -> torch.Tensor:
        """Decodes latents into output space."""
        if self.upsample is not None:
            latents = self.upsample(latents)
            latents = nn.pad(latents, (0, 0, 1, 0))
        xhat = self.vq.decode(latents).sample
        xhat = self.output_activation(xhat)
        return xhat

    @torch.no_grad()
    def infer_latent_code(self, x: torch.Tensor) -> torch.Tensor:
        """Infer latent code for input. The input is a spectrogram snippet
        of size (batch_size, 1, nfft//2, ntimeframes).
        """
        _, _, codes, _ = self.encode(x)
        return codes

    def chunk_spectrogram(
        self, x: torch.Tensor, chunk_frame_length: int = 16
    ) -> torch.Tensor:
        """Chunk spectrogram into smaller snippets. Input is shape (nfft//2, timeframes).
        Output is shape (n, 1, nfft//2, chunk_frame_length). where n is the number of chunks.
        """
        xt = []
        i = 0
        for t in range(x.shape[-1] // chunk_frame_length):
            xt.append(x[:, i : i + chunk_frame_length].unsqueeze(0))
            i += chunk_frame_length
        return torch.stack(xt)

    def unchunk_spectrogram(self, x: torch.Tensor) -> torch.Tensor:
        """Unchunk spectrogram into a single tensor. Input is shape (n, 1, nfft//2, chunk_frame_length).
        Output is shape (1, nfft//2, timeframes).
        """
        x = x.squeeze()
        return torch.cat([y for y in x], dim=-1)

    @torch.no_grad()
    def infer_latent_codes_for_spectrogram(
        self, spectrogram: torch.Tensor
    ) -> torch.Tensor:
        """Infer latent codes for input spectrogram.

        Input is shape (nfft//2, timeframes).
        Output is shape is (n, 1, nfft//2, chunk_frame_length).
        """
        self.vq.quantize.sane_index_shape = True

        x = self.chunk_spectrogram(spectrogram)
        codes = self.infer_latent_code(x)

        self.vq.quantize.sane_index_shape = False

        return codes

    @torch.no_grad()
    def decode_spectrogram_from_codes(self, codes: torch.Tensor) -> torch.Tensor:
        """Decode spectrogram from latent codes."""
        latents = self.vq.quantize.embedding(codes)
        xhat = self.decode(latents)  # shape (n, 1, nfft//2, chunk_frame_length)
        return self.unchunk_spectrogram(xhat)  # shape (1, nfft//2, timeframes)

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
