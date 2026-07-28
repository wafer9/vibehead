# Copyright (c) Kyutai, all rights reserved.
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

from dataclasses import dataclass
from functools import partial
import logging

import torch
from torch import nn
import torch.nn.functional as F
from transformers.models.llama.modeling_llama import LlamaRMSNorm

from transformers import PretrainedConfig, PreTrainedModel
from typing import Optional
from typing import Optional, Dict, List
from torch.nn import CrossEntropyLoss
import torchaudio

from transformers import AutoConfig, AutoModel
from transformers import AutoModelForCausalLM, AutoTokenizer

import copy
import torch.distributed as dist

from vibevoice.modular.modeling_vibevoice import SpeechConnector
from vibevoice.modular.modeling_vibevoice_inference import VibeVoiceForConditionalGenerationInference
from vibevoice.modular.modular_vibevoice_diffusion_head import VibeVoiceDiffusionHead
from vibevoice.schedule.dpm_solver import DPMSolverMultistepScheduler
from vibevoice.modular.configuration_vibevoice import VibeVoiceConfig
from vibevoice.modular.modular_vibevoice_tokenizer import VibeVoiceTokenizerStreamingCache

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
        diffusion_head: Optional[VibeVoiceDiffusionHead] = None,
        noise_scheduler: Optional[DPMSolverMultistepScheduler] = None,
        tokenizer: Optional[AutoTokenizer] = None,
        speech_scaling_factor: Optional[torch.tensor] = None,
        speech_bias_factor: Optional[torch.tensor] = None,
        diffusion_head_proj: Optional[torch.nn.Linear] = None,
        **kwargs,
    ):
        super().__init__()
        self.config = config
        self.lm = language_model
        self.acoustic_tokenizer = acoustic_tokenizer
        self.acoustic_connector = acoustic_connector
        self.diffusion_head = diffusion_head
        self.diffusion_head_proj = diffusion_head_proj
        self.noise_scheduler = noise_scheduler
        self.tokenizer = tokenizer

        self.register_buffer('speech_scaling_factor', speech_scaling_factor)
        self.register_buffer('speech_bias_factor', speech_bias_factor)

        self.lm_head = torch.nn.Linear(self.lm.config.hidden_size, 2)

        self.to(torch.bfloat16)


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


        if configs['freeze_diffusion_head']:
            diffusion_head = vibevoice.prediction_head
        else:
            diffusion_head_config = copy.deepcopy(vibevoice.config.diffusion_head_config)
            diffusion_head_config.hidden_size = lm_config.hidden_size
            diffusion_head = AutoModel.from_config(diffusion_head_config)

        diffusion_head_proj = None
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
                    diffusion_head=diffusion_head,
                    noise_scheduler=noise_scheduler,
                    tokenizer=tokenizer,
                    speech_scaling_factor=vibevoice.speech_scaling_factor,
                    speech_bias_factor=vibevoice.speech_bias_factor,
                    diffusion_head_proj=diffusion_head_proj,
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
                prompt_wavs: torch.Tensor,
                wavs: torch.Tensor,
                prompt_wavs_lengths: torch.Tensor,
                wavs_lengths: torch.Tensor,
                audio_pos: List[torch.Tensor],
                ):

        B, T  = input_ids.shape
        mask = (0 <= input_ids) & (input_ids < self.lm.vocab_size)
        x = self.lm.get_input_embeddings()(torch.masked_fill(input_ids, ~mask, self.tokenizer.eos_id))


        prompt_speech_masks = ~make_pad_mask(torch.ceil(prompt_wavs_lengths/3200).to(dtype=torch.int64))
        prompt_speech_features = self.acoustic_tokenizer.encode(prompt_wavs.transpose(1,2)).mean
        prompt_speech_features = (prompt_speech_features + self.speech_bias_factor) * self.speech_scaling_factor
        prompt_speech_connect_features = self.acoustic_connector(prompt_speech_features)


        speech_masks = ~make_pad_mask(torch.ceil(wavs_lengths/3200).to(dtype=torch.int64))
        speech_features = self.acoustic_tokenizer.encode(wavs.transpose(1,2)).mean
        speech_features = (speech_features + self.speech_bias_factor) * self.speech_scaling_factor
        speech_connect_features = self.acoustic_connector(speech_features)



        prompt_wav_pad_mask = torch.zeros_like(input_ids)
        wav_pad_mask = torch.zeros_like(input_ids)
        wav_loss_mask = torch.zeros_like(input_ids)

        for idx in range(len(audio_pos)):
            if audio_pos[idx].shape[0] == 4:
                p_s, p_e, s, e = audio_pos[idx].tolist()
                prompt_wav_pad_mask[idx, p_s+1:p_e] = 1
                wav_pad_mask[idx, s+1:e] = 1
                wav_loss_mask[idx, s:e-1] = 1 # 包括
            elif audio_pos[idx].shape[0] == 2:
                s, e = audio_pos[idx].tolist()
                wav_pad_mask[idx, s+1:e] = 1
                wav_loss_mask[idx, s:e-1] = 1
                prompt_speech_masks[idx] = 0

        
        prompt_wav_pad_mask = prompt_wav_pad_mask.bool()
        wav_pad_mask = wav_pad_mask.bool()
        wav_loss_mask = wav_loss_mask.bool()


        x[prompt_wav_pad_mask] = prompt_speech_connect_features[prompt_speech_masks] # prompt 恒为声学
        x[wav_pad_mask] = speech_connect_features[speech_masks]                           # 只声学(基线)


        # cond 计算text loss
        cond_outputs = self.lm(inputs_embeds=x,
                            past_key_values=None,
                            attention_mask=None,
                            labels=None,
                            use_cache=None,
                            output_attentions=None,
                            output_hidden_states=True,
                            return_dict=True)

        con_hidden_states = cond_outputs.hidden_states[-1]
        logits = self.lm_head(con_hidden_states)

        loss_text = None
        if labels is not None:
            labels[labels==self.tokenizer.speech_diffusion_id] = 0
            labels[labels==self.tokenizer.eos_id] = 1
            labels = labels.to(logits.device)
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            loss_fct = CrossEntropyLoss()
            loss_text = loss_fct(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))


        # --- Diffusion Loss Calculation ---
        ddpm_batch_mul = 1
        diffusion_loss = None
        speech_features = speech_features[speech_masks]
        # This block is executed only if we are in a context that involves speech.
        if wav_loss_mask.sum().item() > 0:
            condition_features = con_hidden_states[wav_loss_mask]



            speech_len, latent_size = speech_features.shape
            
            noise = torch.randn(
                (speech_len * ddpm_batch_mul, latent_size),
                device=condition_features.device, dtype=condition_features.dtype
            )
            
            timesteps = torch.multinomial(
                torch.ones(self.config.diffusion_head_config.ddpm_num_steps),
                speech_len * ddpm_batch_mul,
                replacement=True,
            ).to(con_hidden_states.device)

            speech_features_repeated = speech_features.repeat_interleave(ddpm_batch_mul, dim=0)
            condition_features_repeated = condition_features.repeat_interleave(ddpm_batch_mul, dim=0)

            noisy_speech_features = self.noise_scheduler.add_noise(
                speech_features_repeated, noise, timesteps
            )
            
            model_output = self.diffusion_head(noisy_speech_features,  timesteps.type_as(x),  condition_features_repeated)

            target_for_loss = self.noise_scheduler.get_velocity(speech_features_repeated, noise, timesteps)

            diffusion_loss = torch.nn.functional.mse_loss(model_output.float(), target_for_loss.float(), reduction='sum')/speech_len

        else:
            # Dummy loss for DDP to work when there are no speech samples in a batch,
            # but we are in a speech context.
            diffusion_loss = sum(p.sum() for p in self.diffusion_head.parameters()) * 0.0
            diffusion_loss += sum(p.sum() for p in self.acoustic_connector.parameters()) * 0.0

        # --- End Diffusion Loss Calculation ---

        loss = loss_text + diffusion_loss

        return {"loss": loss, "loss_text": loss_text, "diffusion_loss": diffusion_loss}



    @torch.no_grad()
    def generate_cfg_ori(self,
                keys, prompt_wavs, prompt_wavs_lengths, input_ids, audio_pos, cfg_scale,
                spk_emb=None
                ):
        B, T  = input_ids.shape
        device = input_ids.device
        mask = (0 <= input_ids) & (input_ids < self.lm.vocab_size)
        x = self.lm.get_input_embeddings()(torch.masked_fill(input_ids, ~mask, self.tokenizer.eos_id))
        negative_x = x[:, -4:, :]

        prompt_speech_masks = ~make_pad_mask(torch.ceil(prompt_wavs_lengths/3200).to(dtype=torch.int64))
        prompt_speech_features = self.acoustic_tokenizer.encode(prompt_wavs.transpose(1,2)).mean
        prompt_speech_features = (prompt_speech_features + self.speech_bias_factor) * self.speech_scaling_factor
        prompt_speech_connect_features = self.acoustic_connector(prompt_speech_features)

        prompt_wav_pad_mask = torch.zeros_like(input_ids)
        for idx in range(len(audio_pos)):
            p_s, p_e, s = audio_pos[idx].tolist()
            prompt_wav_pad_mask[idx, p_s+1:p_e] = 1

        prompt_wav_pad_mask = prompt_wav_pad_mask.bool()
        x[prompt_wav_pad_mask] = prompt_speech_connect_features[prompt_speech_masks] # 拼接 prompt wav

        audios = []
        past_key_values = None
        negative_past_key_values = None
        diffusion_indices = torch.arange(B, device=device)
        acoustic_cache = VibeVoiceTokenizerStreamingCache()
        semantic_cache = VibeVoiceTokenizerStreamingCache() if self.use_semantic else None

        latents = []
        n = 0
        while True:
            outputs = self.lm(inputs_embeds=x,
                        past_key_values=past_key_values,
                        attention_mask=None,
                        labels=None,
                        use_cache=None,
                        output_attentions=None,
                        output_hidden_states=True,
                        return_dict=True)
            negative_outputs = self.lm(inputs_embeds=negative_x,
                        past_key_values=negative_past_key_values,
                        attention_mask=None,
                        labels=None,
                        use_cache=None,
                        output_attentions=None,
                        output_hidden_states=True,
                        return_dict=True)

            past_key_values = outputs.past_key_values
            hidden_states = outputs.hidden_states[-1][:, -1:, :]

            logits = self.lm_head(hidden_states)
            next_tokens = torch.argmax(logits, dim=-1)
            if next_tokens[0] == 1:
                break


            negative_past_key_values = negative_outputs.past_key_values
            negative_hidden_states = negative_outputs.hidden_states[-1][:, -1:, :]

            speech_latent = self.sample_speech_tokens_cfg(hidden_states[0], negative_hidden_states[0], cfg_scale=cfg_scale, spk_emb=spk_emb).unsqueeze(1)
            scaled_latent = speech_latent / self.speech_scaling_factor.to(speech_latent.device) - self.speech_bias_factor.to(speech_latent.device)
            latents.append(scaled_latent)
            
            acoustic_embed = self.acoustic_connector(speech_latent)
            x = acoustic_embed
            negative_x = acoustic_embed

            if n > 7.5*20:
                break
            n += 1

        all_latent = torch.cat(latents, dim=1)
        y = self.acoustic_tokenizer.decode(all_latent.to(self.acoustic_tokenizer.device))[0]  # 整条解码

        return y.float().detach().cpu()


    @torch.no_grad()
    def sample_speech_tokens_cfg(self, condition, neg_condition, cfg_scale=1.0,
                                 spk_emb=None):

        ddpm_inference_steps = 10
        self.noise_scheduler.set_timesteps(ddpm_inference_steps)

        # 说话人条件注入(与训练 forward 一致)：cond 拼真声纹、neg 拼全零，各自过 diffusion_head_proj。
        if self.use_spk_emb and self.diffusion_head_proj is not None:
            if spk_emb is None:
                spk_emb = torch.zeros(condition.shape[0], self.spk_emb_dim,
                                      device=condition.device, dtype=condition.dtype)
            spk_emb = spk_emb.to(condition.dtype)
            if spk_emb.shape[0] == 1 and condition.shape[0] > 1:
                spk_emb = spk_emb.expand(condition.shape[0], -1)
            condition = self.diffusion_head_proj(torch.cat([condition, spk_emb], dim=-1))
            neg_condition = self.diffusion_head_proj(
                torch.cat([neg_condition, torch.zeros_like(spk_emb)], dim=-1))

        condition = torch.cat([condition, neg_condition], dim=0).to(self.diffusion_head.device)
        speech = torch.randn(condition.shape[0], self.config.acoustic_vae_dim).to(condition)
        for t in self.noise_scheduler.timesteps:
            half = speech[: len(speech) // 2]
            combined = torch.cat([half, half], dim=0)
            eps = self.diffusion_head(combined, t.repeat(combined.shape[0]).to(combined), condition=condition)
            cond_eps, uncond_eps = torch.split(eps, len(eps) // 2, dim=0)
            half_eps = uncond_eps + cfg_scale * (cond_eps - uncond_eps)
            eps = torch.cat([half_eps, half_eps], dim=0)
            speech = self.noise_scheduler.step(eps, t, speech).prev_sample

        return speech[: len(speech) // 2]