#!/usr/bin/env python3

# Copyright (c) 2021 Mobvoi Inc. (authors: Binbin Zhang)
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

"""
从「自包含 pair 列表」生成 shard, 存 VibeVoice 声学 VAE latent + 声纹向量。

数据源: /nfs-speech-cfs/wangzhou/s2s/TwinLakes_cfg/data/emilia_latent/data.list
  每行已是一个完整训练样本 (prompt 已配好), 字段包含:
    id, wav, text, duration, speaker, language, dnsmos, emb (当前 wav 声纹),
    prompt_id, prompt_wav, prompt_text, prompt_duration, prompt_emb (prompt wav 声纹)
  → 本脚本不再做 speaker 分组/配对, 逐行处理即可。

每条 shard 记录存:
    - tts_latent    : 当前 wav 的 VAE latent
    - prompt_latent : prompt wav 的 VAE latent
    - tts_emb       : 当前 wav 的声纹向量 (取自输入 emb)
    - prompt_emb    : prompt wav 的声纹向量 (取自输入 prompt_emb)
均为 np.float16 数组, 用 np.save 序列化后 base64 塞进 key.text 的 json。

latent 由 VibeVoice 声学 tokenizer 编码:
    acoustic_tokenizer.encode(audio[B,1,T]).mean -> [B, T/3200, vae_dim]
音频统一重采样到 24000 Hz (与训练 dataset 中 tts/prompt wav 的采样率一致)。

用法 (单卡):
    python3 make_shard_stage_latent.py \
        --num_utts_per_shard 1000 \
        --vibevoice_path /nfs-speech-cfs/wangzhou/.cache/models/VibeVoice-1.5B \
        --gpus 0 \
        /nfs-speech-cfs/wangzhou/s2s/TwinLakes_cfg/data/emilia_latent/data.list \
        <shards_dir> <shards_list>

用法 (多卡并行, 一条命令): --gpus 传逗号分隔的 GPU id, 每个 GPU 起一个进程,
按 chunk 编号取模切分, 各写 <shards_list>.<rank>, 主进程最后合并成 <shards_list>:
    python3 make_shard_stage_latent.py --gpus 0,1,2,3,4,5,6,7 \
        --batch_size 32 --max_batch_seconds 240 \
        --vibevoice_path /nfs-speech-cfs/wangzhou/.cache/models/VibeVoice-1.5B \
        /nfs-speech-cfs/wangzhou/s2s/TwinLakes_cfg/data/emilia_latent/data.list \
        <shards_dir> <shards_list>

关于提速: 该声学 VAE 是因果卷积, 编码耗时正比于音频长度, batch 对它几乎没有
加速 (实测同长 32 条仅 ~1.17x, 变长真实数据接近 1x)。真正的提速来自:
    1) unique 路径去重: 当前 wav 与 prompt wav 大量重合, 去重后每条只编码一次
       (write_file 内已做);
    2) 多卡 (--gpus)。
batch/padding 只会让每条的「最后一帧」与单条编码略有不同, 对解码音频的 log-mel
影响 <0.03, 远小于 VAE 自身的重建误差 (~0.76), 可忽略。
"""

import argparse
import io
import logging
import os
import sys
import tarfile
import time
import json
import base64
from collections import defaultdict

import numpy as np
import torch
import torchaudio
torchaudio.utils.sox_utils.set_buffer_size(16500)

from tqdm import tqdm


# emilia zh 原始音频根目录 (与 make_shard_stage1/2 一致)
AUDIO_ROOT = os.environ.get(
    "EMILIA_ZH_ROOT", "/nfs-speech-cfs/ASR/opensource/Emilia/ZH/")

# VibeVoice 仓库根目录 (提供 vibevoice 包)
VIBE_ROOT = os.environ.get(
    "VIBE_ROOT", "/nfs-speech-cfs/wangzhou/s2s/TwinLakes_vibevoice_lite")
if VIBE_ROOT not in sys.path:
    sys.path.insert(0, VIBE_ROOT)

# VAE latent 编码使用的采样率 (训练 dataset 中 tts/prompt wav 均为 24000)
VAE_SR = 24000


def build_acoustic_tokenizer(vibevoice_path, device):
    """加载 VibeVoice, 取出训练好的声学 tokenizer, 放到 device 上 (eval)。"""
    from vibevoice.modular.modeling_vibevoice_inference import (
        VibeVoiceForConditionalGenerationInference)

    vibevoice = VibeVoiceForConditionalGenerationInference.from_pretrained(
        vibevoice_path, torch_dtype=torch.float32)
    acoustic_tokenizer = vibevoice.model.acoustic_tokenizer
    acoustic_tokenizer = acoustic_tokenizer.to(device).eval()
    return acoustic_tokenizer


# 解码后端: 必须用线程安全的 soundfile (libsndfile) —— sox 后端有全局状态,
# 多线程并发 load 会几乎全部 "failed to open file"; ffmpeg 作为兜底。
_AUDIO_BACKEND = os.environ.get("AUDIO_BACKEND", "soundfile")


def load_wav_24k(wav_path):
    """读取音频 -> 单通道 -> 重采样到 24000 -> (1, 1, T) float32 tensor。"""
    try:
        waveform, sr = torchaudio.load(wav_path, backend=_AUDIO_BACKEND)   # (C, T)
    except Exception:
        waveform, sr = torchaudio.load(wav_path, backend="ffmpeg")         # 兜底
    if waveform.shape[0] > 1:
        waveform = waveform.mean(dim=0, keepdim=True)
    if sr != VAE_SR:
        waveform = torchaudio.transforms.Resample(sr, VAE_SR)(waveform)
    
    return waveform.unsqueeze(0).float()       # (1, 1, T)


# VAE 压缩率: 24000Hz 音频每 3200 采样点 -> 1 帧 latent (与 rq_transformer 一致)
VAE_COMPRESS = 3200


@torch.no_grad()
def _encode_buffer(acoustic_tokenizer, buf, device, batch_size, max_batch_samples, result):
    """把一个已解码好的 buffer (list of (path, wav_1d, len)) 按长度排序 + 动态分桶编码,
    结果写入 result[path]。参考 rq_transformer: 右侧 0-pad 到同长, 整批过 encoder,
    再按 ceil(len/3200) 取每条有效帧 (丢 padding)。"""
    buf.sort(key=lambda x: x[2])   # 按长度排序, 相近长度分到同一 batch, 减少 padding 浪费
    i = 0
    while i < len(buf):
        batch = [buf[i]]
        maxlen = buf[i][2]
        i += 1
        while i < len(buf) and len(batch) < batch_size:
            cand_max = max(maxlen, buf[i][2])
            if (len(batch) + 1) * cand_max > max_batch_samples:
                break
            batch.append(buf[i])
            maxlen = cand_max
            i += 1

        audio = torch.zeros(len(batch), 1, maxlen)
        for b, (p, w, l) in enumerate(batch):
            audio[b, 0, :l] = w
        mean = acoustic_tokenizer.encode(audio.to(device)).mean   # (B, T'max, vae_dim)
        mean = mean.detach().to(torch.float16).cpu().numpy()
        for b, (p, w, l) in enumerate(batch):
            frames = int(np.ceil(l / VAE_COMPRESS))
            result[p] = mean[b, :frames]


def encode_paths_batched(acoustic_tokenizer, paths, device, batch_size,
                         max_batch_samples, num_workers=8):
    """批量编码一组 wav 路径的 VAE latent。

    读/算重叠 (prefetch): num_workers 个后台线程并行解码 mp3 (torchaudio 解码在 C 层
    释放 GIL, 多线程有效), 通过有界队列喂给主线程; 主线程攒够一个窗口就排序分桶、
    上 GPU 编码。GPU 编码期间后台线程继续解码下一批, 消除了「先串行读完整 chunk
    再编码」时 GPU 长期 0% 的空档。有界队列提供背压, 内存受控。

    注: 该声学 VAE 是因果卷积, 计算量正比于音频长度, batch 提速有限; 主要提速靠
    unique 去重 + 读算重叠 + 多卡。padding 只影响每条「最后一帧」, 影响可忽略。

    返回 dict: path -> np.float16 (T', vae_dim); 读取失败的 path 值为 None。
    """
    import threading
    import queue as _queue

    result = {p: None for p in paths}
    n = len(paths)
    in_q = _queue.Queue()
    for item in enumerate(paths):
        in_q.put(item)
    out_q = _queue.Queue(maxsize=max(batch_size * 8, 256))   # 背压, 控内存

    def worker():
        while True:
            try:
                _, p = in_q.get_nowait()
            except _queue.Empty:
                return
            try:
                w = load_wav_24k(p).squeeze(0).squeeze(0)       # (T,)
                out_q.put((p, w, int(w.shape[-1])))
            except Exception as e:
                logging.warning('load fail {}: {}'.format(p, e))
                out_q.put((p, None, 0))

    threads = [threading.Thread(target=worker, daemon=True) for _ in range(max(1, num_workers))]
    for t in threads:
        t.start()

    # 攒够一个窗口再排序编码: 窗口内按长度排序即可获得接近全局排序的 padding 效率,
    # 同时窗口足够小以尽早开始编码、与后台解码重叠。
    sort_window = max(batch_size * 32, 512)
    buf = []
    got = 0
    while got < n:
        p, w, l = out_q.get()
        got += 1
        if w is None:
            continue
        buf.append((p, w, l))
        if len(buf) >= sort_window:
            _encode_buffer(acoustic_tokenizer, buf, device, batch_size, max_batch_samples, result)
            buf = []
    if buf:
        _encode_buffer(acoustic_tokenizer, buf, device, batch_size, max_batch_samples, result)

    for t in threads:
        t.join()
    return result


def latent_to_base64(arr):
    """np 数组 -> np.save 字节 -> base64 字符串 (可直接塞进 json)。"""
    buf = io.BytesIO()
    np.save(buf, arr)
    buf.seek(0)
    return base64.b64encode(buf.read()).decode("utf-8")


def write_file(data_list, tar_file, acoustic_tokenizer, device,
               batch_size, max_batch_samples, num_workers=8, index=0, total=1):
    """data_list: list of dict (自包含 pair, 每条含 wav/prompt_wav/emb/prompt_emb)。
    批量编码 latent 并写入 tar。

    先收集本 shard 内所有 unique wav 路径 (当前 wav 与 prompt wav 会大量重合, 去重后
    几乎省掉一半编码量), 批量编码成 path->latent, 再按条组装写盘。声纹 emb 直接取自
    输入 list (emb=当前 wav, prompt_emb=prompt wav), 一并存进 shard。
    """
    logging.info('Processing {} {}/{}'.format(tar_file, index, total))
    ts = time.time()

    seen = set()
    paths = []
    for item in data_list:
        p = AUDIO_ROOT + item['wav']
        if p not in seen:
            seen.add(p)
            paths.append(p)

    lat = encode_paths_batched(acoustic_tokenizer, paths, device,
                               batch_size, max_batch_samples, num_workers)

    n_ok = 0
    with tarfile.open(tar_file, "w") as tar:
        for item in data_list:
            key = item['id']
            tts_latent = lat.get(AUDIO_ROOT + item['wav'])
            # prompt_latent = lat.get(AUDIO_ROOT + item['prompt_wav'])
            if tts_latent is None:   # 某条 wav 读取失败
            # if tts_latent is None or prompt_latent is None:   # 某条 wav 读取失败
                logging.warning('skip {}: latent missing'.format(key))
                continue

            obj = {
                'id': item['id'],
                'text': item.get('text'),
                'duration': item.get('duration'),
                'speaker': item.get('speaker'),
                'language': item.get('language'),
                'dnsmos': item.get('dnsmos'),
                # 'prompt_id': item.get('prompt_id'),
                # 'prompt_text': item.get('prompt_text'),
                # 'prompt_duration': item.get('prompt_duration'),
                'tts_latent': latent_to_base64(tts_latent),
                'tts_latent_shape': list(tts_latent.shape),
                # 'prompt_latent': latent_to_base64(prompt_latent),
                # 'prompt_latent_shape': list(prompt_latent.shape),
            }
            # 声纹向量 (输入 list 已带), 与 latent 一样存 base64 fp16 npy
            if item.get('emb') is not None:
                tts_emb = np.asarray(item['emb'], dtype=np.float16)
                obj['tts_emb'] = latent_to_base64(tts_emb)
                obj['tts_emb_shape'] = list(tts_emb.shape)
            if item.get('prompt_emb') is not None:
                prompt_emb = np.asarray(item['prompt_emb'], dtype=np.float16)
                obj['prompt_emb'] = latent_to_base64(prompt_emb)
                obj['prompt_emb_shape'] = list(prompt_emb.shape)
            text = json.dumps(obj, ensure_ascii=False).encode('utf8')

            txt_file = key + '.text'
            text_info = tarfile.TarInfo(txt_file)
            text_info.size = len(text)
            tar.addfile(text_info, io.BytesIO(text))
            n_ok += 1

    logging.info('write {} ({} utts) {:.1f}s'.format(tar_file, n_ok, time.time() - ts))


def read_list(list_file):
    """读取自包含 pair 列表 (data.list): 每行一条, 已含 prompt_wav / emb / prompt_emb。"""
    items = []
    with open(list_file, 'r', encoding='utf8') as fin:
        for line in tqdm(fin, desc='read list'):
            line = line.strip()
            if not line:
                continue
            items.append(json.loads(line))
    return items


def run_worker(rank, world_size, device, args):
    """单个进程/单卡的工作: 只处理 chunk 编号 % world_size == rank 的分片。

    每个 worker 独立读 list + 编码 latent (read_list 结果确定, 各卡一致切分),
    并把自己写的 shard 名单落到 <shards_list>.<rank>, 由主进程合并。
    """
    logging.basicConfig(level=logging.INFO,
                        format='%(asctime)s [rank{}] %(levelname)s %(message)s'.format(rank))

    items = read_list(args.list_file)
    num = args.num_utts_per_shard
    chunks = [items[i:i + num] for i in range(0, len(items), num)]
    num_chunks = len(chunks)
    if rank == 0:
        logging.info('total items: {}, total chunks: {}'.format(len(items), num_chunks))

    acoustic_tokenizer = build_acoustic_tokenizer(args.vibevoice_path, device)
    max_batch_samples = args.max_batch_seconds * VAE_SR

    shards_list = []
    for i, chunk in enumerate(chunks):
        if i % world_size != rank:   # 只处理属于本进程的 chunk
            continue
        tar_file = os.path.join(args.shards_dir, '{}_{:09d}.tar'.format(args.prefix, i))
        shards_list.append(tar_file)
        write_file(chunk, tar_file, acoustic_tokenizer, device,
                   args.batch_size, max_batch_samples, args.num_workers, i, num_chunks)

    part_list = '{}.{}'.format(args.shards_list, rank)
    with open(part_list, 'w', encoding='utf8') as fout:
        for name in shards_list:
            fout.write(name + '\n')
    logging.info('rank {} done, {} shards -> {}'.format(rank, len(shards_list), part_list))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='')
    parser.add_argument('--num_utts_per_shard', type=int, default=5000,
                        help='num utts per shard')
    parser.add_argument('--prefix', default='shards',
                        help='prefix of shards tar file')
    parser.add_argument('--vibevoice_path',
                        default='/nfs-speech-cfs/wangzhou/.cache/models/VibeVoice-1.5B',
                        help='VibeVoice 模型路径 (提供声学 tokenizer)')
    parser.add_argument('--gpus', default='0',
                        help='用逗号分隔的 GPU id, 每个 GPU 起一个进程并行, 例: 0,1,2,3')
    parser.add_argument('--batch_size', type=int, default=32,
                        help='一个 batch 最多编码多少条 wav')
    parser.add_argument('--max_batch_seconds', type=int, default=240,
                        help='一个 batch padding 后的总时长上限(秒), 用于控制显存')
    parser.add_argument('--num_workers', type=int, default=8,
                        help='后台并行解码 mp3 的线程数 (读算重叠, 消除 GPU 读数据空档)')
    parser.add_argument('list_file', help='自包含 pair 列表 (data.list, 含 prompt_wav/emb/prompt_emb)')
    parser.add_argument('shards_dir', help='output shards dir')
    parser.add_argument('shards_list', help='output shards list file')
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO,
                        format='%(asctime)s %(levelname)s %(message)s')

    gpus = [g.strip() for g in args.gpus.split(',') if g.strip() != '']
    world_size = len(gpus)
    os.makedirs(args.shards_dir, exist_ok=True)

    if world_size <= 1:
        # 单卡: 直接在主进程跑
        run_worker(0, 1, 'cuda:{}'.format(gpus[0]) if gpus else 'cuda:0', args)
    else:
        # 多卡: 每个 GPU spawn 一个进程
        import torch.multiprocessing as mp
        ctx = mp.get_context('spawn')
        procs = []
        for rank, gpu in enumerate(gpus):
            p = ctx.Process(target=run_worker,
                            args=(rank, world_size, 'cuda:{}'.format(gpu), args))
            p.start()
            procs.append(p)
        for p in procs:
            p.join()
        for p in procs:
            if p.exitcode != 0:
                raise RuntimeError('worker exited with code {}'.format(p.exitcode))

    # 合并各 rank 的 shards_list
    with open(args.shards_list, 'w', encoding='utf8') as fout:
        for rank in range(world_size):
            part_list = '{}.{}'.format(args.shards_list, rank)
            if not os.path.exists(part_list):
                continue
            with open(part_list, 'r', encoding='utf8') as fin:
                for line in fin:
                    fout.write(line)
    logging.info('merged shards_list -> {}'.format(args.shards_list))
