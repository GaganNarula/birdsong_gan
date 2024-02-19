"""VQ-VAE model with optional 4x downsampling and 4x upsampling."""

import os
import json
import torch
import torch.nn as nn
from diffusers import VQModel
from diffusers.models.unets.unet_2d_blocks import Downsample2D, Upsample2D

l2loss = nn.MSELoss()


class VQVAEModel(nn.Module):
    """Wrap model to use new quantize function.

    Args:
        l2_normalize_latents: Whether to L2 normalize latents.
        scale: Scale for code vectors.
        downsample: Whether to downsample latents.
        layers_per_block: Number of layers per block.
        num_vq_embeddings: Number of VQ embeddings.
        vq_embed_dim: VQ embedding dimension.
        vq_beta: VQ beta.

    """

    def __init__(
        self,
        l2_normalize_latents: bool = False,
        scale: float = None,
        downsample: bool = False,
        layers_per_block: int = 3,
        num_downsample_layers: int = 2,
        num_vq_embeddings: int = 128,
        vq_embed_dim: int = 1,
        vq_beta: float = 0.1,
        nfft_half: int = 129,
        ntimeframes: int = 16,
    ):
        super().__init__()

        vq = VQModel(
            in_channels=1,
            out_channels=1,
            latent_channels=1,
            num_vq_embeddings=num_vq_embeddings,
            vq_embed_dim=vq_embed_dim,
            layers_per_block=layers_per_block,
        )

        vq.quantize.legacy = False  # use new quantize function
        vq.quantize.beta = vq_beta

        self.vq = vq
        self.latent_height = nfft_half
        self.latent_width = ntimeframes

        if downsample:
            # downsample for nlayers
            downsample = []
            for layer in range(num_downsample_layers):
                downsample.append(
                    Downsample2D(
                        channels=1,
                        use_conv=True,
                        out_channels=1,
                        kernel_size=3,
                        padding=0,
                    )
                )
                # do not add silu if last layer
                if layer != num_downsample_layers - 1:
                    downsample.append(nn.SiLU())

                self.latent_height = self.latent_height // 2
                self.latent_width = self.latent_width // 2

            self.downsample = nn.Sequential(*downsample)

            # upsample for same num of layers
            upsample = []
            for layer in range(num_downsample_layers):
                upsample.append(
                    Upsample2D(
                        channels=1,
                        use_conv=True,
                        out_channels=1,
                        kernel_size=3,
                        padding=1,
                    )
                )
                # do not add silu if last layer
                if layer != num_downsample_layers - 1:
                    upsample.append(nn.SiLU())

            self.upsample = nn.Sequential(*upsample)

            print("Downsampled latent shape:", self.latent_height, self.latent_width)

        else:
            self.downsample = None
            self.upsample = None

        self.l2_normalize_latents = l2_normalize_latents
        self.nfft_half = nfft_half
        self.ntimeframes = ntimeframes
        self.output_activation = torch.nn.ReLU()

        if scale is not None:
            self.scale = scale
        else:
            self.scale = self.vq.quantize.n_e

        self.config = {
            "l2_normalize_latents": l2_normalize_latents,
            "scale": scale,
            "downsample": downsample,
            "layers_per_block": layers_per_block,
            "num_vq_embeddings": num_vq_embeddings,
            "vq_embed_dim": vq_embed_dim,
            "vq_beta": vq_beta,
            "device": "cuda" if torch.cuda.is_available() else "cpu",
            "last_checkpoint_path": None,
            "nfft_half": nfft_half,
            "ntimeframes": ntimeframes,
        }

        # special initialization
        self.reinit_code_vectors()

    @classmethod
    def from_pretrained(
        cls, model_dir: str, checkpoint_path: str = None
    ) -> "VQVAEModel":
        """Load pretrained model."""
        # first load config
        with open(
            os.path.join(model_dir, "experiment_config.json"), "r", encoding="utf-8"
        ) as f:
            config = json.load(f)

        # then load model
        model = cls(
            l2_normalize_latents=config["l2_normalize_latents"],
            scale=config["scale"],
            downsample=config["downsample"],
            num_downsample_layers=config["num_downsample_layers"],
            layers_per_block=config["layers_per_block"],
            num_vq_embeddings=config["num_vq_embeddings"],
            vq_embed_dim=config["vq_embed_dim"],
            vq_beta=config["vq_beta"],
            nfft_half=config["nfft_half"],
            ntimeframes=config["ntimeframes"],
        )

        if checkpoint_path is None and config["last_checkpoint_path"] is not None:
            model.load_state_dict(
                torch.load(config.last_checkpoint_path), strict=True, assign=False
            )
        elif checkpoint_path is not None:
            model.load_state_dict(
                torch.load(checkpoint_path), strict=True, assign=False
            )
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
            latents = nn.functional.pad(latents, (0, 0, 1, 0))

        xhat = self.vq.decode(latents).sample

        return self.output_activation(xhat)

    @torch.no_grad()
    def infer_latent_code(self, x: torch.Tensor) -> torch.Tensor:
        """Infer latent code for input. The input is a spectrogram snippet
        of size (batch_size, 1, nfft//2, ntimeframes).
        """
        _, _, codes, _ = self.encode(x)
        return codes

    def chunk_spectrogram(self, x: torch.Tensor, ntimeframes: int = 16) -> torch.Tensor:
        """Chunk spectrogram into smaller snippets. Input is shape (nfft//2, timeframes).
        Output is shape (n, 1, nfft//2, chunk_frame_length). where n is the number of chunks.
        """
        xt = []
        i = 0
        for _ in range(x.shape[-1] // ntimeframes):
            xt.append(x[:, i : i + ntimeframes].unsqueeze(0))
            i += ntimeframes
        return torch.stack(xt)

    def unchunk_spectrogram(self, x: torch.Tensor) -> torch.Tensor:
        """Unchunk spectrogram into a single tensor.
        Input is shape (n, 1, nfft//2, ntimeframes).
        Output is shape (1, nfft//2, ntimeframes * n).
        """
        x = x.squeeze()
        return torch.cat([y for y in x], dim=-1)

    @torch.no_grad()
    def infer_latent_codes_for_spectrogram(
        self, spectrogram: torch.Tensor
    ) -> torch.Tensor:
        """Infer latent codes for input spectrogram.

        Input is shape (nfft//2, timeframes).
        Output is shape is (n, 1, nfft//2, self.ntimeframes).
        """
        self.vq.quantize.sane_index_shape = True

        x = self.chunk_spectrogram(spectrogram, self.ntimeframes)
        codes = self.infer_latent_code(x)

        self.vq.quantize.sane_index_shape = False

        return codes

    @torch.no_grad()
    def decode_spectrogram_from_codes(self, codes: torch.Tensor) -> torch.Tensor:
        """Decode spectrogram from latent codes."""
        latents = self.vq.quantize.embedding(codes)
        xhat = self.decode(latents)  # shape (bsz, 1, nfft//2, ntimeframes)
        return self.unchunk_spectrogram(xhat)  # shape (1, nfft//2,  ntimeframes * bsz)

    @torch.no_grad()
    def reconstruct(self, x: torch.Tensor) -> torch.Tensor:
        """Reconstruct input."""
        zq, _, _, _ = self.encode(x)
        return self.decode(zq)

    def sample_uniform_random_codes(
        self, size: tuple[int, int, int, int]
    ) -> torch.Tensor:
        """Sample codes."""
        return torch.randint(0, self.vq.quantize.n_e, size=size).to(
            self.config["device"]
        )

    def sample(self, size: tuple[int, int, int, int]) -> torch.Tensor:
        """Sample from model. First, sample codes from a uniform categorical distribution,
        then fetch the latents, finally, decode the latents into the output space.
        """
        codes = self.sample_uniform_random_codes(size)
        latents = self.vq.quantize.embedding(codes)
        return self.decode(latents)  # shape (bsz, 1, nfft//2, ntimeframes)

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
