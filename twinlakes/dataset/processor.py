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
from twinlakes.utils.mask import make_pad_mask
import math
import base64
import numpy as np

try:
    torchaudio.utils.sox_utils.set_buffer_size(16500)
except AttributeError:
    pass
from decord import VideoReader, cpu
import os

AUDIO_FORMAT_SETS = set(['flac', 'mp3', 'm4a', 'ogg', 'opus', 'wav', 'wma'])
from vibevoice.processor.vibevoice_processor import AudioNormalizer
audio_normalizer = AudioNormalizer()
SAMPLE_RATE=24000
_AUDIO_BACKEND = os.environ.get("AUDIO_BACKEND", "soundfile")

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


def decode_wav_raw(sample, resample=24000):
    """ Parse key/wav/txt from json line

        Args:
            sample: str, str is a json line has key/wav

        Returns:
            {key, wav, sample_rate, ...}
    """
    assert 'key' in sample

    waveform, sample_rate = torchaudio.load(sample['audio'], backend=_AUDIO_BACKEND)
    if sample_rate != SAMPLE_RATE:
        waveform = torchaudio.transforms.Resample(sample_rate, SAMPLE_RATE)(waveform)
        sample_rate = SAMPLE_RATE
    assert sample_rate == resample
    sample['wav'] = waveform[:1]
    sample['sample_rate'] = sample_rate

    vr = VideoReader(sample['video'], ctx=cpu(0), num_threads=8)
    
    frames = vr.get_batch(list(range(len(vr)))).asnumpy() # 读取全部视频帧：[T, H, W, C]，RGB uint8
    fps = float(vr.get_avg_fps())

    frames = torch.from_numpy(frames)
    frames = frames.permute(0, 3, 1, 2).float() # [T, H, W, C] -> [T, C, H, W]
    frames = frames / 127.5 - 1.0 # [0, 255] -> [-1, 1]
    video = frames.permute(1, 0, 2, 3).contiguous() # [T, C, H, W] -> [C, T, H, W]

    sample['video'] = video
    sample['fps'] = fps
    return sample

def filter(sample, fps_diff_thresh=1):
    flag = True
    sample_rate = sample['sample_rate']
    fps = sample['fps']
    if abs(sample['video'].shape[1] - int(sample['wav'].shape[1]/sample_rate*fps)) > fps_diff_thresh:
        flag = False
    return flag


def sort_by_feats(sample):
    return sample['video'].shape[1]


def tokenize(sample, tokenizer):
    speech_tok_compress_ratio = 3200
    prompt = " Reference image:<|image_pad|>\n Video output:\n<|vision_start|>"

    # 1. 计算原始 token 数
    vae_audio_tok_len = math.ceil(sample['wav'].shape[1] / speech_tok_compress_ratio)
    vae_video_tok_len = 1 + (sample['video'].shape[1] - 1) // 4

    # 2. 各自能凑出多少个完整 chunk, 取较小值(木桶原理)
    num_chunks = min(vae_audio_tok_len // 6, vae_video_tok_len // 5)

    # 3. 根据公共 chunk 数, 反算对齐后的 token 数
    audio_tok_clipped = num_chunks * 6
    video_tok_clipped = num_chunks * 5

    # 4. 裁剪音频和视频
    sample['wav'] = sample['wav'][:, :audio_tok_clipped * speech_tok_compress_ratio]
    sample['video'] = sample['video'][:, :(video_tok_clipped - 1) * 4 + 1, :, :]

    # 5. 校验对齐
    assert math.ceil(sample['wav'].shape[1] / speech_tok_compress_ratio) == audio_tok_clipped
    assert 1 + (sample['video'].shape[1] - 1) // 4 == video_tok_clipped
    assert audio_tok_clipped // 6 == video_tok_clipped // 5 == num_chunks

    
    label = ""
    for i in range(num_chunks):
        label += "<|vision_pad|>" * 6 + "<|vision_end|>" * 5
    label += "<|endoftext|>"

    prompt, label = [prompt], [label]

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
    eos_pos = torch.where(encoding["input_ids"] == tokenizer.eos_id)
    video_pos = torch.stack((start_pos[1][0], eos_pos[1][0]), dim=0)

    wav = torch.from_numpy(audio_normalizer(sample['wav'][0].numpy())).unsqueeze(0)
    batch = {
        "key": sample['key'],
        "prompt": prompt,
        "label": label,
        "input_ids": sample['input_ids'][0],
        "label_ids": sample["labels"][0] if 'labels' in sample else torch.zeros([1, 0]),
        "wav": wav,
        "video_pos": video_pos,
        "video": sample['video'],
        "num_chunks": num_chunks,
    }
    return batch


def padding(data, ):

    sample = data
    assert isinstance(sample, list)
    input_ids_length = torch.tensor([x['input_ids'].size(0) for x in sample], dtype=torch.int32)
    order = torch.argsort(input_ids_length, descending=True)

    keys = [sample[i]['key'] for i in order]
    wavs = [sample[i]['wav'].transpose(0,1) for i in order]

    videos = [sample[i]['video'].transpose(0,1) for i in order]

    input_ids = [sample[i]['input_ids'] for i in order]
    label_ids = [sample[i]['label_ids'] for i in order]

    padded_wavs = pad_sequence(wavs, batch_first=True, padding_value=0)
    wavs_lengths = torch.tensor([sample[i]['wav'].size(1) for i in order], dtype=torch.int32)

    padded_videos = pad_sequence(videos, batch_first=True, padding_value=0)
    videos_lengths = torch.tensor([sample[i]['video'].size(1) for i in order], dtype=torch.int32)
    
    
    input_ids = pad_sequence(input_ids, batch_first=True, padding_value=-100)
    label_ids = pad_sequence(label_ids, batch_first=True, padding_value=-100)
    
    video_pos = [sample[i]['video_pos'] for i in order]


    batch = {
        "keys": keys,
        "input_ids": input_ids,
        "label_ids":label_ids,
        "wavs": padded_wavs,
        "wavs_lengths": wavs_lengths,
        "video_pos": video_pos,
        "videos": padded_videos,
        "videos_lengths": videos_lengths,
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
