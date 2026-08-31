"""Diffusion policy over action chunks, for the ACT ablation.

Same observation encoder as ACT -- the ResNet features, state token and
transformer encoder are reused verbatim -- so the comparison isolates exactly
one variable: how the action chunk is generated. ACT decodes it in a single
transformer pass through a CVAE latent; this model learns to denoise it,
following "Diffusion Policy" (Chi et al., 2023) with a transformer denoiser
and DDPM training / DDIM sampling.

The point of the ablation is honesty about model class, not leaderboard
chasing: both models see the same data, the same cameras and the same
normalisation, and are scored by the same simulator.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F

from handrobot.policy.act import (
    ACTConfig,
    ResNetBackbone,
    sinusoidal_position_embedding_1d,
    sinusoidal_position_embedding_2d,
)


@dataclass
class DiffusionConfig:
    """Shape of the denoiser; observation side mirrors :class:`ACTConfig`."""

    state_dim: int = 6
    action_dim: int = 6
    chunk_size: int = 32
    cameras: Sequence[str] = ("front_cam", "wrist_cam")
    hidden_dim: int = 512
    dim_feedforward: int = 2048
    n_heads: int = 8
    n_encoder_layers: int = 4
    n_denoiser_layers: int = 6
    dropout: float = 0.1
    pretrained_backbone: bool = True
    n_tasks: int = 1
    train_timesteps: int = 100
    sample_steps: int = 10

    def to_dict(self) -> dict:
        payload = dict(self.__dict__)
        payload["cameras"] = list(self.cameras)
        return payload

    @classmethod
    def from_dict(cls, payload: dict) -> "DiffusionConfig":
        payload = dict(payload)
        payload["cameras"] = tuple(payload["cameras"])
        return cls(**payload)


def cosine_alphas_cumprod(timesteps: int) -> torch.Tensor:
    """The cosine noise schedule of Nichol & Dhariwal (2021)."""
    steps = torch.arange(timesteps + 1, dtype=torch.float64) / timesteps
    f = torch.cos((steps + 0.008) / 1.008 * math.pi / 2) ** 2
    alphas_cumprod = (f / f[0])[1:]
    return alphas_cumprod.clamp(1e-5, 1.0).float()


class DiffusionChunkPolicy(nn.Module):
    """Predicts the noise on a corrupted action chunk, conditioned on the scene."""

    def __init__(self, config: DiffusionConfig) -> None:
        super().__init__()
        self.config = config
        hidden = config.hidden_dim

        self.backbones = nn.ModuleList(
            [ResNetBackbone(config.pretrained_backbone) for _ in config.cameras]
        )
        self.image_proj = nn.Conv2d(ResNetBackbone.OUTPUT_CHANNELS, hidden, kernel_size=1)
        self.state_proj = nn.Linear(config.state_dim, hidden)
        self.state_token_embed = nn.Parameter(torch.zeros(1, hidden))
        nn.init.normal_(self.state_token_embed, std=0.02)
        self.task_embed = (
            nn.Embedding(config.n_tasks, hidden) if config.n_tasks > 1 else None
        )

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden, nhead=config.n_heads, dim_feedforward=config.dim_feedforward,
            dropout=config.dropout, activation="relu", batch_first=True, norm_first=False,
        )
        self.encoder = nn.TransformerEncoder(
            encoder_layer, config.n_encoder_layers, enable_nested_tensor=False
        )

        self.action_proj = nn.Linear(config.action_dim, hidden)
        self.time_embed = nn.Sequential(
            nn.Linear(hidden, hidden), nn.SiLU(), nn.Linear(hidden, hidden)
        )
        denoiser_layer = nn.TransformerDecoderLayer(
            d_model=hidden, nhead=config.n_heads, dim_feedforward=config.dim_feedforward,
            dropout=config.dropout, activation="relu", batch_first=True, norm_first=False,
        )
        self.denoiser = nn.TransformerDecoder(denoiser_layer, config.n_denoiser_layers)
        self.noise_head = nn.Linear(hidden, config.action_dim)
        # Zero-init: the model starts by predicting zero noise, so the first
        # gradient steps learn the mean rather than fighting random output.
        nn.init.zeros_(self.noise_head.weight)
        nn.init.zeros_(self.noise_head.bias)

        self.register_buffer(
            "chunk_pos", sinusoidal_position_embedding_1d(hidden, config.chunk_size),
            persistent=False,
        )
        self.register_buffer(
            "time_pos",
            sinusoidal_position_embedding_1d(hidden, config.train_timesteps),
            persistent=False,
        )
        self.register_buffer(
            "alphas_cumprod", cosine_alphas_cumprod(config.train_timesteps),
            persistent=False,
        )
        self._image_pos_cache: dict[tuple[int, int], torch.Tensor] = {}

    # -- conditioning --------------------------------------------------------

    def encode_observation(
        self, images: Sequence[torch.Tensor], state: torch.Tensor,
        task: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if len(images) != len(self.config.cameras):
            raise ValueError(f"expected {len(self.config.cameras)} cameras, got {len(images)}")
        state_token = self.state_proj(state)
        if self.task_embed is not None:
            if task is None:
                raise ValueError("this policy is multi-task; a task id is required")
            state_token = state_token + self.task_embed(task.to(state.device))
        tokens = [(state_token + self.state_token_embed).unsqueeze(1)]
        for backbone, image in zip(self.backbones, images):
            features = self.image_proj(backbone(image))
            _, _, height, width = features.shape
            flat = features.flatten(2).transpose(1, 2)
            key = (height, width)
            pos = self._image_pos_cache.get(key)
            if pos is None or pos.device != flat.device:
                pos = sinusoidal_position_embedding_2d(
                    self.config.hidden_dim, height, width
                ).to(flat.device)
                self._image_pos_cache[key] = pos
            tokens.append(flat + pos)
        return self.encoder(torch.cat(tokens, dim=1))

    def predict_noise(
        self, noisy_actions: torch.Tensor, timestep: torch.Tensor, memory: torch.Tensor
    ) -> torch.Tensor:
        tokens = self.action_proj(noisy_actions) + self.chunk_pos.unsqueeze(0)
        tokens = tokens + self.time_embed(self.time_pos[timestep]).unsqueeze(1)
        return self.noise_head(self.denoiser(tokens, memory))

    # -- training ------------------------------------------------------------

    def compute_loss(
        self,
        images: Sequence[torch.Tensor],
        state: torch.Tensor,
        actions: torch.Tensor,
        is_pad: torch.Tensor,
        kl_weight: float = 0.0,
        task: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        """DDPM noise-prediction loss; signature mirrors ACT so the training
        loop drives either model unchanged (``kl_weight`` is ignored)."""
        batch = actions.shape[0]
        timestep = torch.randint(
            0, self.config.train_timesteps, (batch,), device=actions.device
        )
        noise = torch.randn_like(actions)
        alpha_bar = self.alphas_cumprod[timestep].view(batch, 1, 1)
        noisy = alpha_bar.sqrt() * actions + (1 - alpha_bar).sqrt() * noise

        memory = self.encode_observation(images, state, task)
        predicted = self.predict_noise(noisy, timestep, memory)
        valid = (~is_pad).unsqueeze(-1).float().expand_as(noise)
        loss = (F.mse_loss(predicted, noise, reduction="none") * valid).sum() / valid.sum().clamp(min=1.0)
        return {"loss": loss, "l1": loss, "kl": torch.zeros((), device=loss.device)}

    # -- sampling ------------------------------------------------------------

    @torch.no_grad()
    def forward(
        self,
        images: Sequence[torch.Tensor],
        state: torch.Tensor,
        actions: torch.Tensor | None = None,
        is_pad: torch.Tensor | None = None,
        task: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        """DDIM sampling from pure noise to an action chunk."""
        batch = state.shape[0]
        memory = self.encode_observation(images, state, task)
        chunk = torch.randn(
            batch, self.config.chunk_size, self.config.action_dim, device=state.device
        )
        steps = torch.linspace(
            self.config.train_timesteps - 1, 0, self.config.sample_steps,
            device=state.device,
        ).long()
        for i, t in enumerate(steps):
            timestep = t.expand(batch)
            noise = self.predict_noise(chunk, timestep, memory)
            alpha_bar = self.alphas_cumprod[t]
            x0 = (chunk - (1 - alpha_bar).sqrt() * noise) / alpha_bar.sqrt()
            # At the noisiest steps 1/sqrt(alpha_bar) is enormous and any
            # imperfection in the predicted noise explodes x0. Actions are
            # z-normalised, so clamping the estimate to a generous +-5 sigma
            # is a statement of fact about the data, not a fudge.
            x0 = x0.clamp(-5.0, 5.0)
            if i + 1 < len(steps):
                alpha_next = self.alphas_cumprod[steps[i + 1]]
                chunk = alpha_next.sqrt() * x0 + (1 - alpha_next).sqrt() * noise
            else:
                chunk = x0
        return {"actions": chunk, "mu": None, "logvar": None}
