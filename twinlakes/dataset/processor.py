# Copyright (c) 2021 Wenet Community. (authors: Binbin Zhang)
#               2023 Wenet Community. (authors: Dinghao Zhou)
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import io
import json
from subprocess import PIPE, Popen
from urllib.parse import urlparse
import logging
import librosa
import random

import torch
from torch.nn.utils.rnn import pad_sequence
import torchaudio
import torchaudio.compliance.kaldi as kaldi
import torch.nn.functional as F
import whisper
import re
from typing import List, Dict, Union
from twinlakes.utils.mask import make_pad_mask
import math
import base64
import numpy as np

try:
    torchaudio.utils.sox_utils.set_buffer_size(16500)
except AttributeError:
    pass

AUDIO_FORMAT_SETS = set(['flac', 'mp3', 'm4a', 'ogg', 'opus', 'wav', 'wma'])
from vibevoice.processor.vibevoice_processor import AudioNormalizer
audio_normalizer = AudioNormalizer()
SAMPLE_RATE=24000

logging.getLogger('langid').setLevel(logging.INFO)


def get_T_after_pool(L_in, model_type='whisper', dilation=1):
    conv = "[(0,16,8), (0,10,5), (0,10,5), (0,8,4), (0,4,2)]"    
    for (padding, kernel_size, stride) in eval(conv):
        L_out = L_in + 2 * padding - dilation * (kernel_size - 1) - 1
        L_out = 1 + L_out // stride
        L_in = L_out
    return L_out

def get_T_after_codec(L_in, stride=1920):
    L_out = torch.ceil(((L_in - 1) // 2 + 1)/4).to(dtype=torch.int64)
    return L_out

import os
try:
    cpu_info = os.popen("lscpu | grep 'Vendor ID'").read()
    # 0x48 --> HiSilicon
    if (cpu_info.rstrip().split(" ")[-1] == "0x48"):
        # NOTE (MengqingCao): set number of threads in the subprocesses to 1
        # Why? There may be some operators ultilizing multi-threads in processor,
        # causing possibly deadlock in Kunpeng.
        # Similar issue in PyTorch: https://github.com/pytorch/pytorch/issues/45198
        torch.set_num_threads(1)
except Exception as ex:
    logging.warning('Failed to set number of thread in Kunpeng, \
        this may cause segmentfault while dataloading, \
        ignore this warning if you are not using Kunpeng')


class UrlOpenError(Exception):

    def __init__(self, msg: str, *args: object) -> None:
        super().__init__(*args)
        self.err_msg = msg

    def __str__(self) -> str:
        return self.err_msg


def parse_json(elem):
    line = elem['line']
    obj = json.loads(line)
    obj['file_name'] = elem['file_name']
    return dict(obj)


def parse_url(elem):
    assert 'file_name' in elem
    assert 'line' in elem
    assert isinstance(elem, dict)
    url = elem['line']
    try:
        pr = urlparse(url)
        # local file
        if pr.scheme == '' or pr.scheme == 'file':
            stream = open(url, 'rb')
            # network file, such as HTTP(HDFS/OSS/S3)/HTTPS/SCP
        else:
            cmd = f'wget -q -O - {url}'
            process = Popen(cmd, shell=True, stdout=PIPE)
            elem.update(process=process)
            stream = process.stdout
        elem.update(stream=stream)
        return elem
    except Exception as ex:
        err_msg = 'Failed to open {}'.format(url)
        raise UrlOpenError(err_msg) from ex


def decode_wav(sample, max_per_line=None):
    """ Parse a json line holding a list of clips from one speaker and expand
        it into multiple training samples.

        Each chosen clip is used once as the target (wav + text); its voice
        prompt is another randomly picked clip from the same list. Only the
        clips actually used (targets + their prompts) are decoded, once each,
        so per-line IO / json parse is amortized without wasting decode CPU.

        Args:
            sample: dict with 'key' and 'text' (a json line carrying wavs/texts)
            max_per_line: cap on how many target samples to emit per line; None
                means use every clip as a target.

        Yields:
            {key, wav, prompt_wav, sample_rate, text}
    """
    assert 'key' in sample
    assert 'text' in sample

    obj = json.loads(sample['text'])
    wavs, texts = obj['wavs'], obj['texts']
    assert len(wavs) == len(texts) and len(wavs) >= 2
    n = len(wavs)

    # pick which clips serve as targets
    if max_per_line is not None and max_per_line < n:
        tgt_indices = random.sample(range(n), max_per_line)
    else:
        tgt_indices = list(range(n))

    # assign a distinct prompt to each target, then decode only what's needed
    prompt_of = {t: random.choice([j for j in range(n) if j != t])
                 for t in tgt_indices}
    cache = {}
    for idx in set(tgt_indices) | set(prompt_of.values()):
        with io.BytesIO(base64.b64decode(wavs[idx])) as file_obj:
            waveform, sample_rate = torchaudio.load(file_obj)
            if sample_rate != SAMPLE_RATE:
                waveform = torchaudio.transforms.Resample(sample_rate, SAMPLE_RATE)(waveform)
        cache[idx] = waveform

    for tgt_idx in tgt_indices:
        yield {
            'key': '{}_{}'.format(sample['key'], tgt_idx),
            'wav': cache[tgt_idx],
            'prompt_wav': cache[prompt_of[tgt_idx]],
            'sample_rate': SAMPLE_RATE,
            'text': texts[tgt_idx],
        }

def decode_wav_single(sample, max_per_line=None):
    """ Parse a json line holding a list of clips from one speaker and expand
        it into multiple training samples.

        Each chosen clip is used once as the target (wav + text); its voice
        prompt is another randomly picked clip from the same list. Only the
        clips actually used (targets + their prompts) are decoded, once each,
        so per-line IO / json parse is amortized without wasting decode CPU.

        Args:
            sample: dict with 'key' and 'text' (a json line carrying wavs/texts)
            max_per_line: cap on how many target samples to emit per line; None
                means use every clip as a target.

        Yields:
            {key, wav, prompt_wav, sample_rate, text}
    """
    assert 'key' in sample
    assert 'text' in sample

    obj = json.loads(sample['text'])
    key = obj['id']
    text = obj['text']
    wav = obj['wav']
    prompt_wav = obj['prompt_wav']

    with io.BytesIO(base64.b64decode(wav)) as file_obj:
        waveform, sample_rate = torchaudio.load(file_obj)
        if sample_rate != SAMPLE_RATE:
            waveform = torchaudio.transforms.Resample(sample_rate, SAMPLE_RATE)(waveform)
    wav = waveform

    with io.BytesIO(base64.b64decode(prompt_wav)) as file_obj:
        waveform, sample_rate = torchaudio.load(file_obj)
        if sample_rate != SAMPLE_RATE:
            waveform = torchaudio.transforms.Resample(sample_rate, SAMPLE_RATE)(waveform)
    prompt_wav = waveform

    sample = dict(key=key, wav=wav, text=text, prompt_wav=prompt_wav, sample_rate=SAMPLE_RATE)

    return sample


def decode_latent_single(sample, max_per_line=None):
    """新 latent shard：每个 .text 是 json，含 base64(np.save, fp16) 的 VAE latent，
    直接解码成张量,免去在线 encode。

    产出的 'wav'/'prompt_wav' 是 **latent** 张量 [T, vae_dim](不是音频),并置 is_latent=True，
    下游 tokenize/filter/padding/forward 会据此走 latent 分支(长度按帧数而非采样数)。
    """
    assert 'key' in sample
    assert 'text' in sample
    obj = json.loads(sample['text'])

    def _lat(b64):
        arr = np.load(io.BytesIO(base64.b64decode(b64)))     # [T, vae_dim], fp16
        return torch.from_numpy(np.ascontiguousarray(arr)).float()

    tts_latent = _lat(obj['tts_latent'])                     # [T, 64]
    prompt_latent = _lat(obj['prompt_latent'])               # [Tp, 64]

    # target 的 semantic latent [T,128]（shard 里预存则直读，训练用作 target wav 的语义条件）。
    # 帧数 T 与 tts_latent 对齐；缺失(旧 shard)则为 None。
    semantic = _lat(obj['semantic_latent']) if 'semantic_latent' in obj else None

    # prompt 段声纹(WavLM-ECAPA 256-d)：既用于 filter 的相似度清洗，也作 diffusion head 的说话人条件。
    spk_sim = obj['sim']
    spk_emb = None
    # if 'prompt_emb' in obj:
    #     spk_emb = _lat(obj['prompt_emb']).flatten()          # [256] prompt 段声纹
    # if 'tts_emb' in obj and spk_emb is not None:
    #     e1 = _lat(obj['tts_emb']).flatten()                  # [256] 目标段声纹
    #     spk_sim = torch.nn.functional.cosine_similarity(e1, spk_emb, dim=0).item()

    return dict(key=obj['id'], text=obj['text'],
                wav=tts_latent, prompt_wav=prompt_latent, semantic=semantic,
                is_latent=True, spk_sim=spk_sim, spk_emb=spk_emb, sample_rate=SAMPLE_RATE,
                prompt_text=obj['prompt_text'])


def decode_wav_raw(sample, resample=24000):
    """ Parse key/wav/txt from json line

        Args:
            sample: str, str is a json line has key/wav

        Returns:
            {key, wav, sample_rate, ...}
    """
    assert 'key' in sample
    assert 'text' in sample

    waveform, sample_rate = torchaudio.load(sample['prompt_wav'])
    if sample_rate != resample:
        waveform = torchaudio.transforms.Resample(sample_rate, resample)(waveform)
        sample_rate = resample
    sample['prompt_wav'] = waveform

    if 'wav' in sample:
        waveform, sample_rate = torchaudio.load(sample['wav'])
        if sample_rate != resample:
            waveform = torchaudio.transforms.Resample(sample_rate, resample)(waveform)
            sample_rate = resample
        sample['wav'] = waveform
    else:
        sample['wav'] = torch.empty(1, 0)
        sample_rate = resample

    sample['sample_rate'] = sample_rate
    sample['text'] = sample['text']
    return sample


def sort_by_feats(sample):
    assert 'input_ids' in sample
    return sample['input_ids'].size(0)


def tokenize(sample, tokenizer, is_inference=False, cfg_rate=0.0):
    speech_tok_compress_ratio = 3200
    is_latent = sample.get('is_latent', False)
    if is_latent:
        # latent 已是 [T, vae_dim]，vae token 数 = 帧数 = shape[0]
        prompt_vae_tok_len = sample['prompt_wav'].shape[0]
    else:
        prompt_vae_tok_len = math.ceil(sample['prompt_wav'].shape[1] / speech_tok_compress_ratio)
    prompt = " Voice input:\n<|vision_start|>%s<|vision_end|>\n Text input:\n%s\n Speech output:\n<|vision_start|>" \
                % ("<|vision_pad|>"*prompt_vae_tok_len, sample['text'])
    
    # alpha = random.random()
    # if alpha > cfg_rate:
    #     prompt = " Voice input:\n<|vision_start|>%s<|vision_end|>\n Text input:\n%s\n Speech output:\n<|vision_start|>" \
    #             % ("<|vision_pad|>"*prompt_vae_tok_len, sample['text'])
    # else:
    #     prompt = " Speech output:\n<|vision_start|>"
    
    if is_latent:
        vae_tok_len = sample['wav'].shape[0] if sample['wav'].numel() > 0 else 0
    else:
        vae_tok_len = math.ceil(sample['wav'].shape[1] / speech_tok_compress_ratio)
    label = "<|vision_pad|>"*vae_tok_len + "<|endoftext|>"

    prompt, label = [prompt], [label]


    if is_inference:
        encoding = tokenizer(text=prompt,
                        text_target=label,
                        padding=True,
                        return_tensors="pt",
                        return_attention_mask=True)
    else:
        encoding = tokenizer(
                            text=prompt,
                            text_pair=label,
                            add_special_tokens=True,
                            padding=True,  # truncation
                            return_tensors="pt",
                            return_token_type_ids=True,
                            return_attention_mask=True)
        token_type_ids = encoding["token_type_ids"]
        sample['labels'] = encoding["input_ids"].clone()
        sample["labels"][token_type_ids == 0] = -100
    prompt, label = prompt[0], label[0]

    sample['input_ids'] = encoding["input_ids"]

    start_pos = torch.where(encoding["input_ids"] == tokenizer.speech_start_id)
    end_pos = torch.where(encoding["input_ids"] == tokenizer.speech_end_id)
    eos_pos = torch.where(encoding["input_ids"] == tokenizer.eos_id)
    if start_pos[0].shape[0] == 2:
        audio_pos = torch.stack((start_pos[1][0], end_pos[1][0], start_pos[1][1], eos_pos[1][0]), dim=0)
    else:
        audio_pos = torch.stack((start_pos[1][0], eos_pos[1][0]), dim=0)

    if is_latent:
        # latent 直接透传 [T, vae_dim]，不做 audio_normalizer
        prompt_wav = sample['prompt_wav']
        wav = sample['wav']
    else:
        prompt_wav = torch.from_numpy(audio_normalizer(sample['prompt_wav'][0].numpy())).unsqueeze(0)
        wav = torch.from_numpy(audio_normalizer(sample['wav'][0].numpy())).unsqueeze(0) if sample['wav'].shape[1] > 0 else sample['wav']
    batch = {
        "key": sample['key'],
        "prompt": prompt,
        "label": label,
        "input_ids": sample['input_ids'][0],
        "label_ids": sample["labels"][0] if 'labels' in sample else torch.zeros([1, 0]),
        "prompt_wav": prompt_wav,
        "wav": wav,
        "audio_pos": audio_pos,
        "is_latent": is_latent,
        "spk_sim": sample.get('spk_sim', None),
        "spk_emb": sample.get('spk_emb', None),      # [256] prompt 段声纹，作 diffusion head 说话人条件
        "semantic": sample.get('semantic', None),    # [T,128] target 预存 semantic latent(仅 latent shard)
        "text": sample['text'],
        "prompt_text": sample['prompt_text']
    }
    return batch


def filter(sample, spk_sim_thresh=0.7):
    flag = True
    if sample['wav'] is not None:
        if sample.get('is_latent', False):
            # latent [T, vae_dim]，按帧数过滤(7.5Hz)：30s≈225 帧，0.5s≈4 帧
            T = sample['wav'].shape[0]
            if T > 30 * 8 or (0 < T < 4):
                flag = False
        else:
            if sample['wav'].shape[1] > 30*24000 or ( 0 < sample['wav'].shape[1] < 0.5*24000):
                flag = False
    # 声纹相似度过滤：prompt 与当前 wav 说话人相似度 < 阈值则丢弃(有 spk_sim 时生效)
    spk_sim = sample.get('spk_sim', None)
    if spk_sim is not None and (spk_sim < spk_sim_thresh or spk_sim > 0.98):
        flag = False
    if sample['text'] == sample['prompt_text']:
        flag = False
    return flag


def padding(data, ):

    sample = data
    assert isinstance(sample, list)
    input_ids_length = torch.tensor([x['input_ids'].size(0) for x in sample], dtype=torch.int32)
    order = torch.argsort(input_ids_length, descending=True)

    keys = [sample[i]['key'] for i in order]
    is_latent = sample[0].get('is_latent', False)
    if is_latent:
        # latent [T, vae_dim]：沿帧维(dim0)pad，长度=帧数；不 transpose
        prompt_wavs = [sample[i]['prompt_wav'] for i in order]
        wavs = [sample[i]['wav'] for i in order]
        prompt_wavs_lengths = torch.tensor([sample[i]['prompt_wav'].size(0) for i in order],
                                     dtype=torch.int32)
        wavs_lengths = torch.tensor([sample[i]['wav'].size(0) for i in order],
                                     dtype=torch.int32)
    else:
        prompt_wavs = [sample[i]['prompt_wav'].transpose(0,1) for i in order]
        wavs = [sample[i]['wav'].transpose(0,1) for i in order]
        prompt_wavs_lengths = torch.tensor([sample[i]['prompt_wav'].size(1) for i in order],
                                     dtype=torch.int32)
        wavs_lengths = torch.tensor([sample[i]['wav'].size(1) for i in order],
                                     dtype=torch.int32)

    input_ids = [sample[i]['input_ids'] for i in order]
    label_ids = [sample[i]['label_ids'] for i in order]

    padded_prompt_wavs = pad_sequence(prompt_wavs, batch_first=True, padding_value=0)
    padded_wavs = pad_sequence(wavs, batch_first=True, padding_value=0)
    
    input_ids = pad_sequence(input_ids, batch_first=True, padding_value=-100)
    label_ids = pad_sequence(label_ids, batch_first=True, padding_value=-100)
    
    audio_pos = [sample[i]['audio_pos'] for i in order]

    # prompt 段声纹 [B, spk_dim]，作 diffusion head 的说话人条件；缺失(无 emb 的样本)补零。
    SPK_DIM = 256
    spk_list = [sample[i].get('spk_emb', None) for i in order]
    if any(e is not None for e in spk_list):
        _dim = next(e.numel() for e in spk_list if e is not None)
        spk_embs = torch.stack(
            [e.float() if e is not None else torch.zeros(_dim) for e in spk_list], dim=0)
    else:
        spk_embs = torch.zeros(len(order), SPK_DIM)

    # target 预存 semantic latent [B, Tmax, 128]，沿帧维(dim0)pad，与 wavs 帧对齐(同 wavs_lengths)。
    # 仅 latent shard 且样本带 semantic 时构建；否则为 None(wav 输入走在线抽取)。
    sem_list = [sample[i].get('semantic', None) for i in order]
    if all(e is not None for e in sem_list):
        semantic_latents = pad_sequence(sem_list, batch_first=True, padding_value=0)
    else:
        semantic_latents = None

    batch = {
        "keys": keys,
        "prompt_wavs": padded_prompt_wavs,
        "wavs": padded_wavs,
        "input_ids": input_ids,
        "label_ids":label_ids,
        "prompt_wavs_lengths": prompt_wavs_lengths,
        "wavs_lengths": wavs_lengths,
        "audio_pos": audio_pos,
        "spk_embs": spk_embs,
        "semantic_latents": semantic_latents,
    }
    return batch


class DynamicBatchWindow:

    def __init__(self, max_frames_in_batch=12000):
        self.longest_frames = 0
        self.max_frames_in_batch = max_frames_in_batch

    def __call__(self, sample, buffer_size):
        assert isinstance(sample, dict)
        assert 'input_ids' in sample
        assert isinstance(sample['input_ids'], torch.Tensor)
        # new_sample_frames = sample['feat'].size(0)
        new_sample_frames = sample['input_ids'].size(0)
        self.longest_frames = max(self.longest_frames, new_sample_frames)
        frames_after_padding = self.longest_frames * (buffer_size + 1)
        if frames_after_padding > self.max_frames_in_batch:
            self.longest_frames = new_sample_frames
            return True
        return False
