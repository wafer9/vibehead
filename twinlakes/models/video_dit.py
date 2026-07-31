import math
from dataclasses import dataclass
from typing import Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass(frozen=True)
class VideoDiTConfig:
    latent_channels: int = 16
    llm_hidden_size: int = 1024
    hidden_size: int = 512
    num_heads: int = 8
    num_layers: int = 10
    ffn_dim: int = 2048
    patch_size: Tuple[int, int, int] = (1, 4, 4)
    context_tokens: int = 8
    latent_height: int = 64
    latent_width: int = 64
    frequency_embedding_size: int = 256
    norm_eps: float = 1e-6

    @classmethod
    def from_dict(cls, values, llm_hidden_size):
        values = dict(values or {})
        values["llm_hidden_size"] = llm_hidden_size
        if "patch_size" in values:
            values["patch_size"] = tuple(values["patch_size"])
        return cls(**values)


def sinusoidal_embedding_1d(dim, position):
    if dim % 2:
        raise ValueError("frequency embedding dimension must be even")
    half = dim // 2
    frequencies = torch.exp(
        -math.log(10000.0)
        * torch.arange(half, device=position.device, dtype=torch.float32)
        / half
    )
    angles = position.float().unsqueeze(-1) * frequencies
    return torch.cat([angles.cos(), angles.sin()], dim=-1)


class RMSNorm(nn.Module):
    def __init__(self, dim, eps=1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x):
        dtype = x.dtype
        x = x.float()
        x = x * torch.rsqrt(x.square().mean(dim=-1, keepdim=True) + self.eps)
        return x.to(dtype) * self.weight


def _split_rope_dims(head_dim):
    if head_dim % 2:
        raise ValueError("attention head dimension must be even for RoPE")
    pairs = head_dim // 2
    spatial = pairs // 3
    return pairs - 2 * spatial, spatial, spatial


def _rope_angles_3d(grid_size, head_dim, device):
    frames, height, width = grid_size
    t, h, w = torch.meshgrid(
        torch.arange(frames, device=device),
        torch.arange(height, device=device),
        torch.arange(width, device=device),
        indexing="ij",
    )
    coordinates = (t.reshape(-1), h.reshape(-1), w.reshape(-1))
    angle_parts = []
    for coordinate, dim in zip(coordinates, _split_rope_dims(head_dim)):
        if dim == 0:
            continue
        inv_freq = torch.exp(
            -math.log(10000.0)
            * torch.arange(dim, device=device, dtype=torch.float32)
            / dim
        )
        angle_parts.append(coordinate.float().unsqueeze(-1) * inv_freq)
    return torch.cat(angle_parts, dim=-1)


def apply_rope_3d(x, grid_size):
    # x: [B, L, num_heads, head_dim]
    angles = _rope_angles_3d(grid_size, x.shape[-1], x.device)
    if angles.shape[0] != x.shape[1]:
        raise ValueError(
            f"RoPE grid has {angles.shape[0]} tokens, but attention input has {x.shape[1]}"
        )
    cos = angles.cos()[None, :, None, :]
    sin = angles.sin()[None, :, None, :]
    pairs = x.float().reshape(*x.shape[:-1], -1, 2)
    real, imag = pairs.unbind(dim=-1)
    rotated = torch.stack([real * cos - imag * sin, real * sin + imag * cos], dim=-1)
    return rotated.flatten(-2).to(x.dtype)


def scaled_dot_product_attention(q, k, v, attention_mask=None):
    # q/k/v: [B, sequence, heads, head_dim]
    q = q.transpose(1, 2)
    k = k.transpose(1, 2)
    v = v.transpose(1, 2)
    output = F.scaled_dot_product_attention(q, k, v, attn_mask=attention_mask)
    return output.transpose(1, 2).flatten(2)


class SelfAttention(nn.Module):
    def __init__(self, dim, num_heads, eps):
        super().__init__()
        if dim % num_heads:
            raise ValueError("hidden_size must be divisible by num_heads")
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.q = nn.Linear(dim, dim)
        self.k = nn.Linear(dim, dim)
        self.v = nn.Linear(dim, dim)
        self.o = nn.Linear(dim, dim)
        self.norm_q = RMSNorm(dim, eps)
        self.norm_k = RMSNorm(dim, eps)

    def forward(self, x, grid_size, token_mask=None):
        batch, sequence, _ = x.shape
        q = self.norm_q(self.q(x)).view(batch, sequence, self.num_heads, self.head_dim)
        k = self.norm_k(self.k(x)).view(batch, sequence, self.num_heads, self.head_dim)
        v = self.v(x).view(batch, sequence, self.num_heads, self.head_dim)
        q = apply_rope_3d(q, grid_size)
        k = apply_rope_3d(k, grid_size)
        attention_mask = None
        if token_mask is not None:
            attention_mask = token_mask[:, None, None, :]
        return self.o(scaled_dot_product_attention(q, k, v, attention_mask))


class CrossAttention(nn.Module):
    def __init__(self, dim, num_heads, eps):
        super().__init__()
        if dim % num_heads:
            raise ValueError("hidden_size must be divisible by num_heads")
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.q = nn.Linear(dim, dim)
        self.k = nn.Linear(dim, dim)
        self.v = nn.Linear(dim, dim)
        self.k_reference = nn.Linear(dim, dim)
        self.v_reference = nn.Linear(dim, dim)
        self.o = nn.Linear(dim, dim)
        self.norm_q = RMSNorm(dim, eps)
        self.norm_k = RMSNorm(dim, eps)
        self.norm_k_reference = RMSNorm(dim, eps)

    def _shape(self, x):
        return x.view(x.shape[0], x.shape[1], self.num_heads, self.head_dim)

    def forward(self, x, condition, reference):
        q = self._shape(self.norm_q(self.q(x)))
        k = self._shape(self.norm_k(self.k(condition)))
        v = self._shape(self.v(condition))
        output = scaled_dot_product_attention(q, k, v)

        k_reference = self._shape(
            self.norm_k_reference(self.k_reference(reference))
        )
        v_reference = self._shape(self.v_reference(reference))
        output = output + scaled_dot_product_attention(q, k_reference, v_reference)
        return self.o(output)


class VideoDiTBlock(nn.Module):
    def __init__(self, dim, num_heads, ffn_dim, eps):
        super().__init__()
        self.self_attn = SelfAttention(dim, num_heads, eps)
        self.cross_attn = CrossAttention(dim, num_heads, eps)
        self.norm1 = nn.LayerNorm(dim, eps=eps, elementwise_affine=False)
        self.norm2 = nn.LayerNorm(dim, eps=eps, elementwise_affine=False)
        self.norm3 = nn.LayerNorm(dim, eps=eps)
        self.ffn = nn.Sequential(
            nn.Linear(dim, ffn_dim),
            nn.GELU(approximate="tanh"),
            nn.Linear(ffn_dim, dim),
        )
        self.modulation = nn.Parameter(torch.randn(1, 1, 6, dim) / dim**0.5)

    def forward(
        self,
        x,
        condition,
        reference,
        timestep_modulation,
        grid_size,
        token_mask=None,
    ):
        batch, frames, spatial_tokens, dim = x.shape
        modulation = self.modulation.to(timestep_modulation) + timestep_modulation
        shift_attn, scale_attn, gate_attn, shift_ffn, scale_ffn, gate_ffn = (
            modulation.unbind(dim=2)
        )

        normalized = self.norm1(x)
        normalized = normalized * (1 + scale_attn.unsqueeze(2)) + shift_attn.unsqueeze(2)
        attended = self.self_attn(
            normalized.flatten(1, 2), grid_size, token_mask=token_mask
        )
        attended = attended.view(batch, frames, spatial_tokens, dim)
        x = x + gate_attn.unsqueeze(2) * attended

        query = self.norm3(x).reshape(batch * frames, spatial_tokens, dim)
        cross = self.cross_attn(query, condition, reference)
        x = x + cross.view(batch, frames, spatial_tokens, dim)

        normalized = self.norm2(x)
        normalized = normalized * (1 + scale_ffn.unsqueeze(2)) + shift_ffn.unsqueeze(2)
        x = x + gate_ffn.unsqueeze(2) * self.ffn(normalized)
        return x


class VideoDiTHead(nn.Module):
    def __init__(self, dim, out_dim, patch_size, eps):
        super().__init__()
        self.patch_size = patch_size
        self.norm = nn.LayerNorm(dim, eps=eps, elementwise_affine=False)
        self.proj = nn.Linear(dim, out_dim * math.prod(patch_size))
        self.modulation = nn.Parameter(torch.randn(1, 1, 2, dim) / dim**0.5)

    def forward(self, x, timestep_embedding):
        modulation = self.modulation.to(timestep_embedding)
        modulation = modulation + timestep_embedding.unsqueeze(2)
        shift, scale = modulation.unbind(2)
        x = self.norm(x) * (1 + scale.unsqueeze(2)) + shift.unsqueeze(2)
        return self.proj(x)


class VideoDiT(nn.Module):
    """A compact FlashHead-style video diffusion transformer.

    Args follow the training interface directly:
      noisy_latent: [B, C, T, H, W]
      llm_condition: [B, T, D_lm]
      reference: [B, C, 1, H, W] or [B, C, H, W]
      timestep: scalar, [B], or [B, T]
    """

    def __init__(self, config):
        super().__init__()
        self.config = config
        if config.hidden_size <= 0 or config.num_heads <= 0:
            raise ValueError("hidden_size and num_heads must be positive")
        if config.hidden_size % config.num_heads:
            raise ValueError("hidden_size must be divisible by num_heads")
        if (config.hidden_size // config.num_heads) % 2:
            raise ValueError("attention head dimension must be even for RoPE")
        if config.num_layers <= 0 or config.ffn_dim <= 0:
            raise ValueError("num_layers and ffn_dim must be positive")
        if config.latent_channels <= 0 or config.llm_hidden_size <= 0:
            raise ValueError("latent_channels and llm_hidden_size must be positive")
        if config.frequency_embedding_size <= 0 or config.frequency_embedding_size % 2:
            raise ValueError("frequency_embedding_size must be a positive even number")
        if config.context_tokens <= 0:
            raise ValueError("context_tokens must be positive")
        patch_t, patch_h, patch_w = config.patch_size
        if min(config.patch_size) <= 0:
            raise ValueError("patch_size values must be positive")
        if patch_t != 1:
            raise ValueError("VideoDiT currently requires temporal patch size 1")
        if config.latent_height % patch_h or config.latent_width % patch_w:
            raise ValueError("latent dimensions must be divisible by spatial patch size")

        self.patch_embedding = nn.Conv3d(
            config.latent_channels,
            config.hidden_size,
            kernel_size=config.patch_size,
            stride=config.patch_size,
        )
        self.reference_embedding = nn.Conv2d(
            config.latent_channels,
            config.hidden_size,
            kernel_size=(patch_h, patch_w),
            stride=(patch_h, patch_w),
        )
        reference_tokens = (
            config.latent_height // patch_h
        ) * (config.latent_width // patch_w)
        self.reference_position = nn.Parameter(
            torch.randn(1, reference_tokens, config.hidden_size) * 0.02
        )
        self.condition_norm = nn.LayerNorm(config.llm_hidden_size)
        self.condition_projection = nn.Linear(
            config.llm_hidden_size,
            config.context_tokens * config.hidden_size,
        )
        self.time_embedding = nn.Sequential(
            nn.Linear(config.frequency_embedding_size, config.hidden_size),
            nn.SiLU(),
            nn.Linear(config.hidden_size, config.hidden_size),
        )
        self.time_projection = nn.Sequential(
            nn.SiLU(), nn.Linear(config.hidden_size, config.hidden_size * 6)
        )
        self.blocks = nn.ModuleList(
            [
                VideoDiTBlock(
                    config.hidden_size,
                    config.num_heads,
                    config.ffn_dim,
                    config.norm_eps,
                )
                for _ in range(config.num_layers)
            ]
        )
        self.head = VideoDiTHead(
            config.hidden_size,
            config.latent_channels,
            config.patch_size,
            config.norm_eps,
        )
        nn.init.zeros_(self.head.proj.weight)
        nn.init.zeros_(self.head.proj.bias)

    def _prepare_timesteps(self, timestep, batch, frames, device):
        timestep = torch.as_tensor(timestep, device=device)
        if timestep.ndim == 0:
            timestep = timestep.expand(batch, frames)
        elif timestep.ndim == 1:
            if timestep.numel() == batch:
                timestep = timestep[:, None].expand(batch, frames)
            elif batch == 1 and timestep.numel() == frames:
                timestep = timestep[None, :]
            else:
                raise ValueError("1D timestep must have B elements (or T elements when B=1)")
        elif timestep.shape != (batch, frames):
            raise ValueError(
                f"timestep must have shape {(batch, frames)}, "
                f"got {tuple(timestep.shape)}"
            )
        return timestep

    def _unpatchify(self, x, grid_size):
        batch, frames, _, _ = x.shape
        patch_t, patch_h, patch_w = self.config.patch_size
        _, grid_h, grid_w = grid_size
        x = x.view(
            batch,
            frames,
            grid_h,
            grid_w,
            patch_t,
            patch_h,
            patch_w,
            self.config.latent_channels,
        )
        x = x.permute(0, 7, 1, 4, 2, 5, 3, 6).contiguous()
        return x.view(
            batch,
            self.config.latent_channels,
            frames * patch_t,
            grid_h * patch_h,
            grid_w * patch_w,
        )

    def forward(
        self,
        noisy_latent,
        llm_condition,
        reference,
        timestep,
        frame_mask=None,
    ):
        if noisy_latent.ndim != 5:
            raise ValueError("noisy_latent must have shape [B, C, T, H, W]")
        batch, channels, frames, height, width = noisy_latent.shape
        if channels != self.config.latent_channels:
            raise ValueError(
                f"expected {self.config.latent_channels} latent channels, got {channels}"
            )
        if llm_condition.shape[:2] != (batch, frames):
            raise ValueError(
                f"llm_condition must start with {(batch, frames)}, got {tuple(llm_condition.shape)}"
            )
        if llm_condition.shape[-1] != self.config.llm_hidden_size:
            raise ValueError(
                f"expected LLM hidden size {self.config.llm_hidden_size}, "
                f"got {llm_condition.shape[-1]}"
            )
        if (height, width) != (self.config.latent_height, self.config.latent_width):
            raise ValueError(
                "latent spatial shape does not match VideoDiTConfig: "
                f"expected {(self.config.latent_height, self.config.latent_width)}, "
                f"got {(height, width)}"
            )

        x = self.patch_embedding(noisy_latent)
        grid_size = tuple(x.shape[2:])
        x = x.permute(0, 2, 3, 4, 1).reshape(batch, frames, -1, self.config.hidden_size)
        token_mask = None
        if frame_mask is not None:
            if frame_mask.shape != (batch, frames):
                raise ValueError(
                    f"frame_mask must have shape {(batch, frames)}, "
                    f"got {tuple(frame_mask.shape)}"
                )
            token_mask = frame_mask.to(device=x.device, dtype=torch.bool).clone()
            empty_samples = ~token_mask.any(dim=1)
            if empty_samples.any():
                token_mask[empty_samples, 0] = True
            token_mask = token_mask[:, :, None].expand(-1, -1, x.shape[2]).flatten(1)

        condition = self.condition_projection(self.condition_norm(llm_condition))
        condition = condition.view(
            batch * frames,
            self.config.context_tokens,
            self.config.hidden_size,
        )

        if reference.ndim == 5:
            if reference.shape[2] != 1:
                raise ValueError("5D reference must contain exactly one latent frame")
            reference = reference[:, :, 0]
        if reference.shape != (batch, channels, height, width):
            raise ValueError(
                f"reference must have shape {(batch, channels, height, width)}, "
                f"got {tuple(reference.shape)}"
            )
        reference = self.reference_embedding(reference).flatten(2).transpose(1, 2)
        if reference.shape[1] != self.reference_position.shape[1]:
            raise ValueError("reference token count does not match configured latent size")
        reference = reference + self.reference_position.to(reference)
        reference = reference[:, None].expand(-1, frames, -1, -1)
        reference = reference.reshape(batch * frames, reference.shape[2], reference.shape[3])

        timestep = self._prepare_timesteps(timestep, batch, frames, noisy_latent.device)
        time_frequency = sinusoidal_embedding_1d(
            self.config.frequency_embedding_size, timestep
        ).to(noisy_latent.dtype)
        time_embedding = self.time_embedding(time_frequency)
        timestep_modulation = self.time_projection(time_embedding).view(
            batch, frames, 6, self.config.hidden_size
        )

        for block in self.blocks:
            x = block(
                x,
                condition,
                reference,
                timestep_modulation,
                grid_size,
                token_mask=token_mask,
            )
        x = self.head(x, time_embedding)
        return self._unpatchify(x, grid_size)


__all__ = ["VideoDiT", "VideoDiTConfig"]
