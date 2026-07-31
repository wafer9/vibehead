# Copyright (c) Kyutai, all rights reserved.
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import logging
from typing import Dict, List, Optional

import torch
from torch import nn

from transformers import PretrainedConfig, PreTrainedModel
from transformers import AutoConfig
from transformers import AutoModelForCausalLM, AutoTokenizer

from vibevoice.modular.modeling_vibevoice import SpeechConnector
from vibevoice.modular.modeling_vibevoice_inference import VibeVoiceForConditionalGenerationInference
from vibevoice.schedule.dpm_solver import DPMSolverMultistepScheduler
from vibevoice.modular.configuration_vibevoice import VibeVoiceConfig
from twinlakes.models.video_dit import VideoDiT, VideoDiTConfig
from twinlakes.vae.wan import WanVAE


logger = logging.getLogger(__name__)


def make_pad_mask(lengths: torch.Tensor, max_len: int = 0) -> torch.Tensor:
    """Make mask tensor containing indices of padded part.

    See description of make_non_pad_mask.

    Args:
        lengths (torch.Tensor): Batch of lengths (B,).
    Returns:
        torch.Tensor: Mask tensor containing indices of padded part.

    Examples:
        >>> lengths = [5, 3, 2]
        >>> make_pad_mask(lengths)
        masks = [[0, 0, 0, 0 ,0],
                 [0, 0, 0, 1, 1],
                 [0, 0, 1, 1, 1]]
    """
    batch_size = lengths.size(0)
    max_len = max_len if max_len > 0 else lengths.max().item()
    seq_range = torch.arange(0,
                             max_len,
                             dtype=torch.int64,
                             device=lengths.device)
    seq_range_expand = seq_range.unsqueeze(0).expand(batch_size, max_len)
    seq_length_expand = lengths.unsqueeze(-1)
    mask = seq_range_expand >= seq_length_expand
    return mask


class WanLatentCompressorQFormer(nn.Module):
    """(B, 16, T, 64, 64) -> (B, T, 1024), 用1个query抽取整帧"""
    def __init__(self, in_ch=16, hidden=512, out_dim=1024, num_heads=8):
        super().__init__()
        self.in_proj = nn.Linear(in_ch, hidden)
        self.pos_embed = nn.Parameter(torch.randn(1, 64 * 64, hidden) * 0.02)
        self.query = nn.Parameter(torch.randn(1, 1, hidden) * 0.02)
        self.attn = nn.MultiheadAttention(hidden, num_heads, batch_first=True)
        self.norm = nn.LayerNorm(hidden)
        self.out_proj = nn.Linear(hidden, out_dim)

    def forward(self, x):                          # (B, C, T, H, W)
        B, C, T, H, W = x.shape
        x = x.permute(0, 2, 3, 4, 1).reshape(B * T, H * W, C)  # (B*T, 4096, 16)
        x = self.in_proj(x) + self.pos_embed       # (B*T, 4096, hidden)
        q = self.query.expand(B * T, -1, -1)       # (B*T, 1, hidden)
        out, _ = self.attn(q, x, x)                # (B*T, 1, hidden)
        out = self.out_proj(self.norm(out))        # (B*T, 1, out_dim)
        return out.reshape(B, T, -1)               # (B, T, out_dim)


class LMMConfig(PretrainedConfig):
    model_type = "lam"  # 用于保存模型时自动保存对应的代码
    _auto_class = "AutoConfig"  # 用于保存模型时自动保存对应的代码
    is_composition = True  # TODO:
    keys_to_ignore_at_inference = [
        "past_key_values", "hidden_states", "attentions"
    ]  # 推理时不需要保存的输出，避免显存占用

    def __init__(
        self,
        audio_config: Optional[Dict] = {},
        text_config: Optional[Dict] = {},
        num_special_tokens_add: Optional[int] = 0,
        **kwargs,
    ):
        """
        num_special_tokens_add: 相较于原始 text model，添加的 special tokens 数量
        """
        super().__init__(**kwargs)

        audio_model_type = audio_config.pop("model_type")
        self.audio_config = AutoConfig.for_model(audio_model_type, **audio_config)
        self.num_special_tokens_add = num_special_tokens_add



class LMModel(nn.Module):
    """Transformer-based language model on multiple streams of codes.

    Args:
        n_q (int): Number of parallel streams to model as input.
        dep_q (int): Number of parallel streams to model in the depformer.
        card (int): Cardinality, vocabulary size.
        text_card (int): Cardinality of the text vocabulary.
        dim (int): Dimension of the transformer encoder.
        num_heads (int): Number of heads for the transformer encoder.
        hidden_scale (int): Scale for hidden feed forward dimension of the transformer encoder.
        norm (str): Normalization method.
        norm_emb (bool): Whether to normalize embeddings.
        bias_proj (bool): Use bias for output projections.
        depformer_*: params used for the Depformer Transformer, all the other will be shared.
        depformer_multi_linear (bool): if True, uses one linear layer per codebook to project the
            output of the main transformer to the Depformer latent space.
        depformer_dim_feedforward (int| list[int]| None): If None, defaults to hidden_scale * depformer_dim.
        existing_text_padding_id (bool): if True, will use a different token for the initial text token, and
            the text padding token.
        same_initial (bool): if True, uses the same initial tokens for both text and audio mode.
        **kwargs: Additional parameters for the transformer encoder.
    """

    def __init__(
        self,
        config: Optional[VibeVoiceConfig] = None,
        language_model: Optional[PreTrainedModel] = None,
        acoustic_tokenizer: Optional[PreTrainedModel] = None,
        acoustic_connector: Optional[SpeechConnector] = None,
        video_dit: Optional[VideoDiT] = None,
        noise_scheduler: Optional[DPMSolverMultistepScheduler] = None,
        tokenizer: Optional[AutoTokenizer] = None,
        speech_scaling_factor: Optional[torch.tensor] = None,
        speech_bias_factor: Optional[torch.tensor] = None,
        vae: Optional[WanVAE] = None,
        dtype: Optional[torch.dtype] = None,
        **kwargs,
    ):
        super().__init__()
        self.config = config
        self.lm = language_model
        self.acoustic_tokenizer = acoustic_tokenizer
        self.acoustic_connector = acoustic_connector
        self.video_dit = video_dit
        self.noise_scheduler = noise_scheduler
        self.tokenizer = tokenizer

        self.register_buffer('speech_scaling_factor', speech_scaling_factor)
        self.register_buffer('speech_bias_factor', speech_bias_factor)

        self.lm_head = torch.nn.Linear(self.lm.config.hidden_size, 2)

        self.vae = vae

        self.comp = WanLatentCompressorQFormer(out_dim=self.lm.config.hidden_size)

        self.to(dtype=dtype)


    @classmethod
    def from_audio_text_pretrained(
        self,
        configs: Optional[Dict] = {},
    ):
        
        lm_path = configs['temporal_model']
        lm_config = AutoConfig.from_pretrained(lm_path, trust_remote_code=True)
        language_model = AutoModelForCausalLM.from_pretrained(lm_path, trust_remote_code=True)

        from vibevoice.modular.modular_vibevoice_text_tokenizer import VibeVoiceTextTokenizerFast
        tokenizer = VibeVoiceTextTokenizerFast.from_pretrained(lm_path)

        vibevoice = VibeVoiceForConditionalGenerationInference.from_pretrained(configs['vibevoice_path'])
        acoustic_tokenizer = vibevoice.model.acoustic_tokenizer
        acoustic_connector = SpeechConnector(acoustic_tokenizer.config.vae_dim, lm_config.hidden_size)

        if configs['dtype'] == 'bf16':
            dtype=torch.bfloat16
        else:
            dtype=torch.float32
        vae = WanVAE(vae_path=configs['vae_path'], dtype=dtype)


        video_dit_config = VideoDiTConfig.from_dict(
            configs.get('video_dit_conf'),
            llm_hidden_size=lm_config.hidden_size,
        )
        video_dit = VideoDiT(video_dit_config)
        noise_scheduler = DPMSolverMultistepScheduler(
            num_train_timesteps=vibevoice.config.diffusion_head_config.ddpm_num_steps,
            beta_schedule=vibevoice.config.diffusion_head_config.ddpm_beta_schedule,
            prediction_type=vibevoice.config.diffusion_head_config.prediction_type
        )

        model = self(
                    config=vibevoice.config,
                    language_model=language_model,
                    acoustic_tokenizer=acoustic_tokenizer,
                    acoustic_connector=acoustic_connector,
                    video_dit=video_dit,
                    noise_scheduler=noise_scheduler,
                    tokenizer=tokenizer,
                    speech_scaling_factor=vibevoice.speech_scaling_factor,
                    speech_bias_factor=vibevoice.speech_bias_factor,
                    vae=vae,
                    dtype=dtype,
                    )
        return model
    
    def get_embeds(self, input_ids):
        if self.num_special_tokens_add > 0:
            mask = input_ids >= self.text_model_vocab_size
            inputs_embeds = self.transformer.get_input_embeddings()(
                torch.masked_fill(input_ids, mask, 0))
            
            mask_asr = (input_ids >= self.text_model_vocab_size) \
                    & (input_ids < (self.text_model_vocab_size + self.num_asr_tokens_add))
            emd_add = self.asr_wte_add(torch.masked_fill(input_ids - self.text_model_vocab_size, ~mask_asr, 0))
            inputs_embeds[mask_asr] = emd_add[mask_asr]

            mask_tts = input_ids >= (self.text_model_vocab_size + self.num_asr_tokens_add)
            emd_add = self.tts_wte_add(
                torch.masked_fill(input_ids - self.text_model_vocab_size - self.num_asr_tokens_add, ~mask_tts, 0))
            inputs_embeds[mask_tts] = emd_add[mask_tts]

        else:
            inputs_embeds = self.transformer.get_input_embeddings()(input_ids)
        return inputs_embeds


    def forward(self,
                keys: List[str],
                input_ids: torch.Tensor,
                labels: torch.Tensor,
                wavs: torch.Tensor,
                wavs_lengths: torch.Tensor,
                audio_pos: List[torch.Tensor],
                videos: torch.Tensor,
                videos_lengths: torch.Tensor,
                ):
        batch_size, _ = input_ids.shape
        mask = (0 <= input_ids) & (input_ids < self.lm.vocab_size)
        x = self.lm.get_input_embeddings()(torch.masked_fill(input_ids, ~mask, self.tokenizer.eos_id))

        # Audio and 30 FPS video latents are both 7.5 Hz: 24000/3200 == 30/4.
        speech_valid_len = torch.ceil(wavs_lengths / 3200).to(torch.int64)
        video_valid_len = 1 + (videos_lengths.to(torch.int64) - 1) // 4

        speech_features = self.acoustic_tokenizer.encode(wavs.transpose(1,2)).mean
        speech_features = (speech_features + self.speech_bias_factor) * self.speech_scaling_factor
        speech_features = self.acoustic_connector(speech_features)

        with torch.no_grad():
            videos_latent = self.vae.encode(videos.transpose(1, 2))
        if videos_latent.ndim == 4:
            videos_latent = videos_latent.unsqueeze(0)
        if videos_latent.ndim != 5 or videos_latent.shape[0] != batch_size:
            raise ValueError(
                "Wan VAE must return [B, 16, T, H, W], got "
                f"{tuple(videos_latent.shape)}"
            )
        if videos_latent.shape[2] < 2:
            raise ValueError("video must contain a reference latent and at least one target latent")

        # The first causal Wan latent is the identity reference. Future latents
        # are teacher-forced into the LLM one step after their prediction state.
        reference_latent = videos_latent[:, :, :1]
        target_latent = videos_latent[:, :, 1:]
        target_frames = target_latent.shape[2]
        target_valid_len = (video_valid_len - 1).clamp(min=0, max=target_frames)
        video_features = self.comp(target_latent)

        video_positions = []
        for index, positions in enumerate(audio_pos):
            positions = positions.tolist()
            if len(positions) == 4:
                prompt_start, prompt_end, video_start, video_end = positions
                audio_count = min(
                    int(speech_valid_len[index]),
                    prompt_end - prompt_start - 1,
                    speech_features.shape[1],
                )
                if audio_count > 0:
                    x[index, prompt_start + 1:prompt_start + 1 + audio_count] = (
                        speech_features[index, :audio_count].to(x.dtype)
                    )
            elif len(positions) == 2:
                video_start, video_end = positions
            else:
                raise ValueError(
                    f"audio_pos for {keys[index]} must contain 2 or 4 positions, "
                    f"got {len(positions)}"
                )

            video_count = int(target_valid_len[index])
            available_positions = video_end - video_start - 1
            if video_count > available_positions:
                raise ValueError(
                    f"{keys[index]} has {video_count} video latents but only "
                    f"{available_positions} video placeholder positions"
                )
            if video_count > 0:
                x[index, video_start + 1:video_start + 1 + video_count] = (
                    video_features[index, :video_count].to(x.dtype)
                )
            video_positions.append((video_start, video_count))

        outputs = self.lm(inputs_embeds=x,
                            past_key_values=None,
                            attention_mask=None,
                            labels=None,
                            use_cache=None,
                            output_attentions=None,
                            output_hidden_states=True,
                            return_dict=True)

        hidden_states = outputs.hidden_states[-1]
        h_video = hidden_states.new_zeros(
            batch_size, target_frames, hidden_states.shape[-1]
        )
        for index, (video_start, video_count) in enumerate(video_positions):
            if video_count > 0:
                # h at position n predicts the clean video token inserted at n+1.
                h_video[index, :video_count] = hidden_states[
                    index, video_start:video_start + video_count
                ]

        if target_valid_len.sum().item() == 0:
            diffusion_loss = sum(p.sum() for p in self.video_dit.parameters()) * 0.0
            diffusion_loss += sum(p.sum() for p in self.comp.parameters()) * 0.0
            return {"loss": diffusion_loss, "diffusion_loss": diffusion_loss}

        noise = torch.randn_like(target_latent)
        timesteps = torch.randint(
            0,
            self.config.diffusion_head_config.ddpm_num_steps,
            (batch_size,),
            device=target_latent.device,
        )
        noisy_latent = self.noise_scheduler.add_noise(target_latent, noise, timesteps)
        valid_mask = (
            torch.arange(target_frames, device=target_latent.device)[None, :]
            < target_valid_len[:, None]
        )
        model_output = self.video_dit(
            noisy_latent=noisy_latent,
            llm_condition=h_video,
            reference=reference_latent,
            timestep=timesteps,
            frame_mask=valid_mask,
        )

        prediction_type = self.config.diffusion_head_config.prediction_type
        if prediction_type == "epsilon":
            target_for_loss = noise
        elif prediction_type == "v_prediction":
            target_for_loss = self.noise_scheduler.get_velocity(
                target_latent, noise, timesteps
            )
        else:
            raise NotImplementedError(f"unsupported prediction type: {prediction_type}")

        valid_mask = valid_mask[:, None, :, None, None]
        squared_error = (model_output.float() - target_for_loss.float()).square()
        squared_error = squared_error * valid_mask
        elements_per_frame = (
            target_latent.shape[1] * target_latent.shape[3] * target_latent.shape[4]
        )
        denominator = valid_mask.sum() * elements_per_frame
        diffusion_loss = squared_error.sum() / denominator.clamp_min(1)
        return {"loss": diffusion_loss, "diffusion_loss": diffusion_loss}



    @torch.no_grad()
    def sample_video_latents(
        self,
        llm_condition,
        reference,
        cfg_scale=1.0,
        num_inference_steps=20,
        generator=None,
    ):
        """Denoise a complete video-latent chunk from aligned LLM states."""
        if llm_condition.ndim != 3:
            raise ValueError("llm_condition must have shape [B, T, D]")
        batch_size, frames, _ = llm_condition.shape
        config = self.video_dit.config
        latent = torch.randn(
            batch_size,
            config.latent_channels,
            frames,
            config.latent_height,
            config.latent_width,
            device=llm_condition.device,
            dtype=llm_condition.dtype,
            generator=generator,
        )
        self.noise_scheduler.set_timesteps(
            num_inference_steps, device=llm_condition.device
        )
        for timestep in self.noise_scheduler.timesteps:
            if cfg_scale == 1.0:
                model_output = self.video_dit(
                    latent, llm_condition, reference, timestep
                )
            else:
                model_input = torch.cat([latent, latent], dim=0)
                condition = torch.cat(
                    [llm_condition, torch.zeros_like(llm_condition)], dim=0
                )
                reference_input = torch.cat([reference, reference], dim=0)
                model_output = self.video_dit(
                    model_input, condition, reference_input, timestep
                )
                conditional, unconditional = model_output.chunk(2, dim=0)
                model_output = unconditional + cfg_scale * (
                    conditional - unconditional
                )
            latent = self.noise_scheduler.step(
                model_output, timestep, latent
            ).prev_sample
        return latent
