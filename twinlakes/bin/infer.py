# Copyright (c) Kyutai, all rights reserved.
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import argparse
import logging
from dataclasses import dataclass
import random
import copy
import yaml
import numpy as np
import torch

import torch.nn.functional as F
from decord import VideoReader, cpu
import imageio
from tqdm import tqdm

from twinlakes.vae.wan import WanVAE

vae_path = "/nfs-speech-cfs/wangzhou/tts/SoulX-FlashHead/models/SoulX-FlashHead-1_3B/VAE_Wan/Wan2.1_VAE.pth"
video_path = "/nfs-speech-cfs/wangzhou/tts/SoulX-FlashTalk/sample_results_batch/00000003_58a866e17fc0561876faa159e2a6fbf75d2227a8995d13c93bb52f04f6ce8377_0.mp4"

vae = WanVAE(
    vae_path=vae_path,
    dtype=torch.bfloat16,
    device="cuda:0",
    parallel=False,
)



def read_full_video(
    video_path,
    device="cuda",
    dtype=torch.bfloat16,
):
    vr = VideoReader(
        video_path,
        ctx=cpu(0),
        num_threads=8,
    )

    # 读取全部视频帧：[T, H, W, C]，RGB uint8
    frames = vr.get_batch(list(range(len(vr)))).asnumpy()
    fps = float(vr.get_avg_fps())

    # [T, H, W, C] -> [T, C, H, W]
    frames = torch.from_numpy(frames)
    frames = frames.permute(0, 3, 1, 2).float()
    # print(frames.shape)

    # [0, 255] -> [-1, 1]
    frames = frames / 127.5 - 1.0

    # [T, C, H, W] -> [1, C, T, H, W]
    video = frames.permute(1, 0, 2, 3).unsqueeze(0).contiguous()

    return video.to(device=device, dtype=dtype), fps


def save_video_tensor(video, output_path, fps=25):
    """
    video:
        [1, 3, T, H, W] 或 [3, T, H, W]
        数值范围通常为 [-1, 1]
    """
    if video.ndim == 5:
        assert video.shape[0] == 1
        video = video[0]

    assert video.ndim == 4
    assert video.shape[0] == 3

    # [-1, 1] -> [0, 255]
    video = (
        (video.float().clamp(-1, 1) + 1.0)
        * 127.5
    ).round().to(torch.uint8)

    # [C, T, H, W] -> [T, H, W, C]
    frames = video.permute(1, 2, 3, 0).cpu().numpy()

    with imageio.get_writer(
        output_path,
        format="FFMPEG",
        mode="I",
        fps=fps,
        codec="libx264",
        pixelformat="yuv420p",
        ffmpeg_params=["-crf", "18"],
        macro_block_size=None,
    ) as writer:
        for frame in frames:
            writer.append_data(frame)

# t = 0
# with open('data/vivi/video.list', 'r') as f:
#     for line in tqdm(f.readlines()[:10]):
#         video_path = line.strip()
#         # 一次性读取整个视频
#         video, fps = read_full_video(
#             video_path,
#         )

#         print(video.shape, fps, flush=True)   # THWC: (T, H, W, 3) 或 TCHW: (T, 3, H, W)
#         t += video.shape[2]/fps
# print(t)


# x = vae.encode(video)
# video_r = vae.decode(x)
# print(x.shape)
# print(video_r.shape)
# save_video_tensor(video=video_r, output_path="/nfs-speech-cfs/wangzhou/s2s/vibehead/1.mp4")


from vibevoice.modular.modeling_vibevoice_inference import VibeVoiceForConditionalGenerationInference
import torchaudio
vibevoice = VibeVoiceForConditionalGenerationInference.from_pretrained("/nfs-speech-cfs/wangzhou/.cache/models/VibeVoice-1.5B")
acoustic_tokenizer = vibevoice.model.acoustic_tokenizer
acoustic_tokenizer = acoustic_tokenizer.to(device="cuda:0")
acoustic_tokenizer.eval()


import os
_AUDIO_BACKEND = os.environ.get("AUDIO_BACKEND", "soundfile")
audio, sr = torchaudio.load("/nfs-speech-cfs/wangzhou/data/tts/VividHead/audios/82565.wav", backend=_AUDIO_BACKEND)

audio = audio[:1, :].unsqueeze(0)
print(audio.shape, sr)

latent = acoustic_tokenizer.encode(audio.to(device="cuda:0")).mean
print(latent.shape)

y = acoustic_tokenizer.decode(latent.to(acoustic_tokenizer.device))[0]  # 整条解码
y = y.float().detach().cpu()
print(y.shape)
torchaudio.save('/nfs-speech-cfs/wangzhou/s2s/vibehead/1.wav', y, sample_rate=sr, backend=_AUDIO_BACKEND)
