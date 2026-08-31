"""Action Chunking Transformer.

A faithful implementation of the architecture from "Learning Fine-Grained
Bimanual Manipulation with Low-Cost Hardware" (Zhao et al., 2023), sized for a
single 6-DoF arm and two 128x128 cameras.

Two networks are trained jointly:

* A **conditional VAE encoder**, used only during training. It reads the joint
  state together with the ground-truth action chunk and produces a style latent.
  Its job is to absorb the variation between demonstrations of the same task --
  the parts of a human's motion that the images do not explain -- so the decoder
  is not forced to average over them.
* A **transformer decoder** that reads image features and the joint state and
  emits the next ``chunk_size`` actions in one shot. At inference the latent is
  set to the prior mean, which yields the modal trajectory.

Predicting a chunk rather than a single step is what makes the policy robust to
compounding error: it commits to a short plan instead of re-deciding every frame.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


@dataclass
class ACTConfig:
    """Shape of the network. Must match between training and inference."""

    state_dim: int = 6
    action_dim: int = 6
    chunk_size: int = 32
    cameras: Sequence[str] = ("front_cam", "wrist_cam")
    hidden_dim: int = 512
    dim_feedforward: int = 2048
    n_heads: int = 8
    n_encoder_layers: int = 4
    n_decoder_layers: int = 6
    n_vae_layers: int = 4
    latent_dim: int = 32
    dropout: float = 0.1
    pretrained_backbone: bool = True
    #: 1 means unconditioned (and keeps old checkpoints loading unchanged);
    #: more than 1 adds a learned task embedding to the conditioning.
    n_tasks: int = 1

    def to_dict(self) -> dict:
        payload = dict(self.__dict__)
        payload["cameras"] = list(self.cameras)
        return payload

    @classmethod
    def from_dict(cls, payload: dict) -> "ACTConfig":
        payload = dict(payload)
        payload["cameras"] = tuple(payload["cameras"])
        payload.setdefault("n_tasks", 1)
        return cls(**payload)


def sinusoidal_position_embedding_2d(hidden_dim: int, height: int, width: int) -> torch.Tensor:
    """(H*W, hidden_dim) fixed 2D sine-cosine embedding for image feature grids."""
    if hidden_dim % 4 != 0:
        raise ValueError("hidden_dim must be divisible by 4 for a 2D embedding")
    quarter = hidden_dim // 4
    frequencies = torch.exp(
        torch.arange(quarter, dtype=torch.float32) * (-math.log(10000.0) / max(quarter - 1, 1))
    )
    ys = torch.arange(height, dtype=torch.float32).unsqueeze(1) * frequencies
    xs = torch.arange(width, dtype=torch.float32).unsqueeze(1) * frequencies
    y_embed = torch.cat([ys.sin(), ys.cos()], dim=1)          # (H, hidden/2)
    x_embed = torch.cat([xs.sin(), xs.cos()], dim=1)          # (W, hidden/2)
    grid = torch.cat(
        [
            y_embed.unsqueeze(1).expand(height, width, hidden_dim // 2),
            x_embed.unsqueeze(0).expand(height, width, hidden_dim // 2),
        ],
        dim=2,
    )
    return grid.reshape(height * width, hidden_dim)


def sinusoidal_position_embedding_1d(hidden_dim: int, length: int) -> torch.Tensor:
    """(length, hidden_dim) fixed sine-cosine embedding for token sequences."""
    position = torch.arange(length, dtype=torch.float32).unsqueeze(1)
    frequencies = torch.exp(
        torch.arange(0, hidden_dim, 2, dtype=torch.float32) * (-math.log(10000.0) / hidden_dim)
    )
    embedding = torch.zeros(length, hidden_dim)
    embedding[:, 0::2] = torch.sin(position * frequencies)
    embedding[:, 1::2] = torch.cos(position * frequencies)
    return embedding


class ResNetBackbone(nn.Module):
    """ResNet-18 trunk that returns a spatial feature map instead of a class score."""

    OUTPUT_CHANNELS = 512
    STRIDE = 32

    def __init__(self, pretrained: bool = True) -> None:
        super().__init__()
        from torchvision.models import ResNet18_Weights, resnet18

        weights = ResNet18_Weights.IMAGENET1K_V1 if pretrained else None
        model = resnet18(weights=weights)
        self.stem = nn.Sequential(
            model.conv1, model.bn1, model.relu, model.maxpool,
            model.layer1, model.layer2, model.layer3, model.layer4,
        )

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        """(B, 3, H, W) in [0, 1] -> (B, 512, H/32, W/32)."""
        mean = torch.tensor(IMAGENET_MEAN, device=images.device).view(1, 3, 1, 1)
        std = torch.tensor(IMAGENET_STD, device=images.device).view(1, 3, 1, 1)
        return self.stem((images - mean) / std)


class ACTPolicy(nn.Module):
    """The full CVAE policy."""

    def __init__(self, config: ACTConfig) -> None:
        super().__init__()
        self.config = config
        hidden = config.hidden_dim

        self.backbones = nn.ModuleList(
            [ResNetBackbone(config.pretrained_backbone) for _ in config.cameras]
        )
        self.image_proj = nn.Conv2d(ResNetBackbone.OUTPUT_CHANNELS, hidden, kernel_size=1)

        self.state_proj = nn.Linear(config.state_dim, hidden)
        self.latent_proj = nn.Linear(config.latent_dim, hidden)
        # Task conditioning, only materialised for multi-task training so that
        # every existing single-task checkpoint still loads bit-for-bit. The
        # embedding is added to the state token on both the observation and the
        # CVAE side: one vector that tells the whole network which objective
        # this trajectory serves.
        self.task_embed = (
            nn.Embedding(config.n_tasks, hidden) if config.n_tasks > 1 else None
        )
        # One learned embedding per non-image token, so the transformer can tell
        # the latent token from the proprioception token.
        self.extra_token_embed = nn.Parameter(torch.zeros(2, hidden))
        nn.init.normal_(self.extra_token_embed, std=0.02)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden, nhead=config.n_heads, dim_feedforward=config.dim_feedforward,
            dropout=config.dropout, activation="relu", batch_first=True, norm_first=False,
        )
        # enable_nested_tensor=False: the fast path builds a NestedTensor from the
        # padding mask, and that operator is not implemented on MPS. It is only a
        # memory optimisation, so disabling it costs nothing but a little speed.
        self.encoder = nn.TransformerEncoder(
            encoder_layer, config.n_encoder_layers, enable_nested_tensor=False
        )

        decoder_layer = nn.TransformerDecoderLayer(
            d_model=hidden, nhead=config.n_heads, dim_feedforward=config.dim_feedforward,
            dropout=config.dropout, activation="relu", batch_first=True, norm_first=False,
        )
        self.decoder = nn.TransformerDecoder(decoder_layer, config.n_decoder_layers)
        self.query_embed = nn.Embedding(config.chunk_size, hidden)
        self.action_head = nn.Linear(hidden, config.action_dim)

        # --- CVAE encoder (training only) ---
        self.vae_cls = nn.Parameter(torch.zeros(1, 1, hidden))
        nn.init.normal_(self.vae_cls, std=0.02)
        self.vae_state_proj = nn.Linear(config.state_dim, hidden)
        self.vae_action_proj = nn.Linear(config.action_dim, hidden)
        vae_layer = nn.TransformerEncoderLayer(
            d_model=hidden, nhead=config.n_heads, dim_feedforward=config.dim_feedforward,
            dropout=config.dropout, activation="relu", batch_first=True, norm_first=False,
        )
        self.vae_encoder = nn.TransformerEncoder(
            vae_layer, config.n_vae_layers, enable_nested_tensor=False
        )
        self.latent_head = nn.Linear(hidden, 2 * config.latent_dim)

        self.register_buffer(
            "vae_pos", sinusoidal_position_embedding_1d(hidden, config.chunk_size + 2),
            persistent=False,
        )
        self._image_pos_cache: dict[tuple[int, int], torch.Tensor] = {}

    # -- pieces -------------------------------------------------------------

    def _image_position_embedding(self, height: int, width: int, device: torch.device) -> torch.Tensor:
        key = (height, width)
        cached = self._image_pos_cache.get(key)
        if cached is None or cached.device != device:
            cached = sinusoidal_position_embedding_2d(self.config.hidden_dim, height, width).to(device)
            self._image_pos_cache[key] = cached
        return cached

    def _task_vector(self, task: torch.Tensor | None, batch: int, device) -> torch.Tensor | None:
        if self.task_embed is None:
            return None
        if task is None:
            raise ValueError("this policy is multi-task; a task id is required")
        return self.task_embed(task.to(device))

    def encode_latent(
        self, state: torch.Tensor, actions: torch.Tensor, is_pad: torch.Tensor,
        task: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Posterior over the style latent, given the demonstrated action chunk."""
        batch = state.shape[0]
        state_token = self.vae_state_proj(state)
        conditioning = self._task_vector(task, batch, state.device)
        if conditioning is not None:
            state_token = state_token + conditioning
        tokens = torch.cat(
            [
                self.vae_cls.expand(batch, -1, -1),
                state_token.unsqueeze(1),
                self.vae_action_proj(actions),
            ],
            dim=1,
        )
        tokens = tokens + self.vae_pos[: tokens.shape[1]].unsqueeze(0)
        # The CLS and state tokens are always real; padded actions are masked out.
        key_padding_mask = torch.cat(
            [torch.zeros(batch, 2, dtype=torch.bool, device=state.device), is_pad], dim=1
        )
        encoded = self.vae_encoder(tokens, src_key_padding_mask=key_padding_mask)
        mu, logvar = self.latent_head(encoded[:, 0]).chunk(2, dim=-1)
        return mu, logvar.clamp(-8.0, 8.0)

    def encode_observation(
        self, images: Sequence[torch.Tensor], state: torch.Tensor, latent: torch.Tensor,
        task: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Build the transformer memory from cameras, proprioception and the latent."""
        if len(images) != len(self.config.cameras):
            raise ValueError(f"expected {len(self.config.cameras)} camera tensors, got {len(images)}")

        state_token = self.state_proj(state)
        conditioning = self._task_vector(task, state.shape[0], state.device)
        if conditioning is not None:
            state_token = state_token + conditioning
        tokens = [
            (self.latent_proj(latent) + self.extra_token_embed[0]).unsqueeze(1),
            (state_token + self.extra_token_embed[1]).unsqueeze(1),
        ]
        for backbone, image in zip(self.backbones, images):
            features = self.image_proj(backbone(image))
            batch, channels, height, width = features.shape
            flat = features.flatten(2).transpose(1, 2)
            tokens.append(flat + self._image_position_embedding(height, width, flat.device))
        return self.encoder(torch.cat(tokens, dim=1))

    def decode_actions(self, memory: torch.Tensor) -> torch.Tensor:
        batch = memory.shape[0]
        queries = self.query_embed.weight.unsqueeze(0).expand(batch, -1, -1)
        decoded = self.decoder(queries, memory)
        return self.action_head(decoded)

    # -- forward ------------------------------------------------------------

    def forward(
        self,
        images: Sequence[torch.Tensor],
        state: torch.Tensor,
        actions: torch.Tensor | None = None,
        is_pad: torch.Tensor | None = None,
        task: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        """Predict an action chunk.

        With ``actions`` supplied the CVAE posterior is used and the KL terms are
        returned for the loss; without it the latent is set to the prior mean,
        which is what inference should do.
        """
        if actions is not None:
            if is_pad is None:
                is_pad = torch.zeros(actions.shape[:2], dtype=torch.bool, device=actions.device)
            mu, logvar = self.encode_latent(state, actions, is_pad, task)
            latent = mu + torch.randn_like(mu) * torch.exp(0.5 * logvar)
        else:
            mu = logvar = None
            latent = torch.zeros(state.shape[0], self.config.latent_dim, device=state.device)

        memory = self.encode_observation(images, state, latent, task)
        predicted = self.decode_actions(memory)
        return {"actions": predicted, "mu": mu, "logvar": logvar}

    def compute_loss(
        self,
        images: Sequence[torch.Tensor],
        state: torch.Tensor,
        actions: torch.Tensor,
        is_pad: torch.Tensor,
        kl_weight: float,
        task: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        """L1 action loss over unpadded steps, plus the KL regulariser."""
        output = self.forward(images, state, actions, is_pad, task)
        valid = (~is_pad).unsqueeze(-1).float()
        l1 = (F.l1_loss(output["actions"], actions, reduction="none") * valid).sum() / valid.sum().clamp(min=1.0)
        mu, logvar = output["mu"], output["logvar"]
        kl = (-0.5 * (1 + logvar - mu.pow(2) - logvar.exp())).sum(-1).mean()
        return {"loss": l1 + kl_weight * kl, "l1": l1, "kl": kl}
