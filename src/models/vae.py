from typing import Tuple

import torch
from torch import nn


class ConvVAE(nn.Module):
    """Simple convolutional VAE intended for pipeline validation.

    Architecture experimentation guide:
    - Safe to change:
      - `hidden_dims`: controls encoder/decoder channel widths and depth.
      - activation/batchnorm choices in encoder/decoder blocks.
      - `latent_dim`: bottleneck size.
      - conv kernel/stride/padding values.
    - Must stay consistent:
      - Decoder must upsample back to input spatial size.
      - Final output channels must equal `in_channels`.
      - Final activation must match training target normalization:
        - `Sigmoid` -> targets in [0, 1]
        - `Tanh` -> targets in [-1, 1]
    """

    def __init__(
        self,
        image_size: Tuple[int, int] = (64, 64),
        in_channels: int = 3,
        latent_dim: int = 64,
        hidden_dims: Tuple[int, ...] = (32, 64, 128, 256),
    ) -> None:
        super().__init__()
        self.image_size = image_size
        self.in_channels = in_channels
        self.latent_dim = latent_dim
        self.hidden_dims = hidden_dims

        # Encoder design space:
        # - You can replace Conv2d blocks with residual/attention/downsampling blocks.
        # - Keep the final feature tensor compatible with the MLP heads below.
        encoder_layers = []
        current_channels = in_channels
        for h_dim in hidden_dims:
            encoder_layers.extend(
                [
                    nn.Conv2d(current_channels, h_dim, kernel_size=4, stride=2, padding=1),
                    nn.BatchNorm2d(h_dim),
                    nn.LeakyReLU(0.2, inplace=True),
                ]
            )
            current_channels = h_dim
        self.encoder = nn.Sequential(*encoder_layers)

        # Shape inference keeps the model robust to many architecture changes.
        # If you modify encoder depth/strides, this adapts `fc_mu/fc_logvar` sizes.
        with torch.no_grad():
            dummy = torch.zeros(1, in_channels, image_size[0], image_size[1])
            enc_out = self.encoder(dummy)
            self._enc_shape = enc_out.shape[1:]
            enc_flat_dim = int(enc_out.numel())

        # Latent heads:
        # - Safe to replace with deeper MLPs or separate feature projections.
        # - `latent_dim` can be varied directly from config.
        self.fc_mu = nn.Linear(enc_flat_dim, latent_dim)
        self.fc_logvar = nn.Linear(enc_flat_dim, latent_dim)
        self.decoder_input = nn.Linear(latent_dim, enc_flat_dim)

        # Decoder design space mirrors encoder:
        # - You can change upsampling strategy (transpose conv, resize+conv, etc.).
        # - Ensure total upsampling factor restores original HxW.
        decoder_layers = []
        rev_dims = list(hidden_dims[::-1])
        for i in range(len(rev_dims) - 1):
            decoder_layers.extend(
                [
                    nn.ConvTranspose2d(
                        rev_dims[i],
                        rev_dims[i + 1],
                        kernel_size=4,
                        stride=2,
                        padding=1,
                    ),
                    nn.BatchNorm2d(rev_dims[i + 1]),
                    nn.LeakyReLU(0.2, inplace=True),
                ]
            )
        self.decoder = nn.Sequential(*decoder_layers)

        # IMPORTANT: output activation must match image transform normalization.
        # Current pipeline uses ImageToTensor in [0,1], so Sigmoid is correct.
        # If you switch to Tanh here, switch training/eval image normalization to [-1,1].
        self.final_layer = nn.Sequential(
            nn.ConvTranspose2d(
                hidden_dims[0], hidden_dims[0], kernel_size=4, stride=2, padding=1
            ),
            nn.BatchNorm2d(hidden_dims[0]),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(hidden_dims[0], in_channels, kernel_size=3, padding=1),
            nn.Sigmoid(),
        )

    def encode(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        features = self.encoder(x).flatten(start_dim=1)
        return self.fc_mu(features), self.fc_logvar(features)

    def reparameterize(self, mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        x = self.decoder_input(z)
        x = x.view(-1, *self._enc_shape)
        x = self.decoder(x)
        return self.final_layer(x)

    def forward(self, x: torch.Tensor):
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        recon = self.decode(z)
        return recon, mu, logvar
