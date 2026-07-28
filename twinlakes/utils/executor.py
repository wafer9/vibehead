# Copyright (c) 2020 Mobvoi Inc (Binbin Zhang)
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import copy
import datetime
import logging
import os
from contextlib import nullcontext

# if your python version < 3.7 use the below one
# from contextlib import suppress as nullcontext
import torch
from twinlakes.utils.common import StepTimer

from twinlakes.utils.train_utils import log_per_step, wenet_join
from torch.nn.utils import clip_grad_norm_
from twinlakes.utils.checkpoint import save_checkpoint, save_state_dict_and_infos
from deepspeed.utils.zero_to_fp32 import get_fp32_state_dict_from_zero_checkpoint

class Executor:

    def __init__(self,
                 global_step: int = -1,
                 device: torch.device = torch.device("cpu"),
                 runname: str=None,
                 ):
        self.step = global_step + 1
        self.train_step_timer = None
        self.cv_step_timer = None
        self.device = device
        self.runname = runname

    def train(self, model, optimizer, scheduler, train_data_loader,
              writer, configs, rank, group_join):
        ''' Train one epoch
        '''
        if self.train_step_timer is None:
            self.train_step_timer = StepTimer(self.step)
        accum_grad = configs.get('accum_grad', 50.0)
        clip = configs.get('grad_clip', 50.0)
        log_interval = configs.get('log_interval', 10)
        model.train()
        info_dict = copy.deepcopy(configs)
        logging.info('using accumulate grad, new batch size is {} times'
                     ' larger than before'.format(info_dict['accum_grad']))
        # A context manager to be used in conjunction with an instance of
        # torch.nn.parallel.DistributedDataParallel to be able to train
        # with uneven inputs across participating processes.
        if isinstance(model, torch.nn.parallel.DistributedDataParallel):
            model_context = model.join
        else:
            model_context = nullcontext
        info_dict["tag"] = "TRAIN"

        dtype = configs.get("dtype", "fp32")
        if dtype == "fp16":
            dtype = torch.float16
        elif dtype == "bf16":
            dtype = torch.bfloat16
        else:  # fp32
            dtype = torch.float32

        with model_context():
            for batch_idx, batch_dict in enumerate(train_data_loader):
                info_dict['runname'] = self.runname
                info_dict["step"] = self.step
                info_dict["batch_idx"] = batch_idx
                info_dict["log_interval"] = configs.get('log_interval', 10)

                if wenet_join(group_join, info_dict):
                    break
                
                prompt_wavs = batch_dict['prompt_wavs'].to(device=self.device, dtype=dtype)
                wavs = batch_dict['wavs'].to(self.device, dtype=dtype)
                prompt_wavs_lengths = batch_dict['prompt_wavs_lengths'].to(device=self.device)
                wavs_lengths = batch_dict['wavs_lengths'].to(device=self.device)

                # prompt 段声纹 [B, spk_dim]，作 diffusion head 说话人条件(use_spk_emb 时生效)
                spk_embs = batch_dict.get('spk_embs', None)
                if spk_embs is not None:
                    spk_embs = spk_embs.to(device=self.device, dtype=dtype)
                # target 预存 semantic latent [B,T,128](仅 latent shard；wav 输入时为 None，forward 在线抽)
                semantic_latents = batch_dict.get('semantic_latents', None)
                if semantic_latents is not None:
                    semantic_latents = semantic_latents.to(device=self.device, dtype=dtype)

                # text_ids = batch_dict['text_ids'].to(self.device)

                input_ids = batch_dict['input_ids'].to(self.device)
                labels = batch_dict['label_ids'].to(self.device)


                info_dict["bs"] = labels.size(0)
                info_dict['tokens'] = labels.size(0) * labels.size(1)

                # Disable gradient synchronizations across DDP processes.
                # Within this context, gradients will be accumulated on module
                # variables, which will later be synchronized.
                train_engine = configs.get("train_engine", "torch_ddp")
                if train_engine in ["torch_ddp", "torch_fsdp"] and (batch_idx + 1) % info_dict["accum_grad"] != 0:
                    context = model.no_sync
                # Used for single gpu training and DDP gradient synchronization
                # processes.
                else:
                    context = nullcontext
                
                with context():
                    if train_engine == "deepspeed":
                        
                        with torch.cuda.amp.autocast(enabled=dtype is not None,
                                     dtype=dtype,
                                     cache_enabled=False):
                            loss_dict = model(
                                      keys=batch_dict['keys'],
                                      input_ids=input_ids,
                                      labels=labels,
                                      prompt_wavs=prompt_wavs,
                                      wavs=wavs,
                                      prompt_wavs_lengths=prompt_wavs_lengths,
                                      wavs_lengths=wavs_lengths,
                                      audio_pos=batch_dict['audio_pos'],
                                      spk_embs=spk_embs,
                                      semantic_latents=semantic_latents,
                            )
                    else:
                        loss_dict = model(
                                    keys=batch_dict['keys'],
                                      input_ids=input_ids,
                                      labels=labels,
                                      prompt_wavs=prompt_wavs,
                                      wavs=wavs,
                                      prompt_wavs_lengths=prompt_wavs_lengths,
                                      wavs_lengths=wavs_lengths,
                                      audio_pos=batch_dict['audio_pos'],
                                      spk_embs=spk_embs,
                                      semantic_latents=semantic_latents,
                        )
                    info_dict['loss_dict'] = loss_dict
                    loss = info_dict['loss_dict']['loss']
                if train_engine == 'deepspeed':
                    loss = model.backward(loss)
                else:
                    loss.backward()
                if train_engine == 'deepspeed':
                    info_dict["is_gradient_accumulation_boundary"] = \
                        model.is_gradient_accumulation_boundary()
                    model.step()
                    grad_norm = model.get_global_grad_norm()
                elif (batch_idx + 1) % accum_grad == 0:
                    grad_norm = clip_grad_norm_(model.parameters(), clip)
                    if torch.isfinite(grad_norm):
                        optimizer.step()
                    optimizer.zero_grad()
                    scheduler.step()
                    grad_norm = grad_norm.item()
                else:
                    grad_norm = None

                info_dict["lrs"] = [x['lr'] for x in optimizer.param_groups]
                info_dict["grad_norm"] = 0 if grad_norm is None else grad_norm

                # write training: tensorboard && log
                log_per_step(writer, info_dict, timer=self.train_step_timer)

                if self.step % 5000 == 0 and self.step // 5000 > 0:
                    if configs['train_engine'] == "deepspeed":
                        model.save_checkpoint(save_dir=configs['model_dir'], tag="train")
                    if rank == 0:
                        save_model_path = os.path.join(configs['model_dir'], '{}_{}.pt'.format(configs['epoch'], self.step))
                        infos_ = {
                                    'epoch': configs['epoch'],
                                    'cv_loss': loss.item(),
                                    'step': self.step,
                                    'save_time': datetime.datetime.now().strftime('%d/%m/%Y %H:%M:%S'),
                                }
                        if train_engine == "deepspeed":
                            state_dict = get_fp32_state_dict_from_zero_checkpoint(configs['model_dir'], "train",
                                                                                exclude_frozen_parameters=False
                                                                                )
                            save_state_dict_and_infos(state_dict, save_model_path, infos_)
                        else:
                            save_checkpoint(model, save_model_path, infos_)
                self.step += 1


    def cv(self, model, cv_data_loader, writer, configs):
        ''' Cross validation on
        '''
        if self.cv_step_timer is None:
            self.cv_step_timer = StepTimer(0.0)
        else:
            self.cv_step_timer.last_iteration = 0.0
        model.eval()
        info_dict = copy.deepcopy(configs)
        num_seen_utts, loss_dict, total_loss = 1, {}, 0  # avoid division by 0
        dtype = configs.get("dtype", "fp32")
        if dtype == "fp16":
            dtype = torch.float16
        elif dtype == "bf16":
            dtype = torch.bfloat16
        else:  # fp32
            dtype = torch.float32
        with torch.no_grad():
            for batch_idx, batch_dict in enumerate(cv_data_loader):
                info_dict["tag"] = "CV"
                info_dict["step"] = self.step
                info_dict["batch_idx"] = batch_idx
                info_dict["cv_step"] = batch_idx

                prompt_wavs = batch_dict['prompt_wavs'].to(device=self.device, dtype=dtype)
                wavs = batch_dict['wavs'].to(self.device, dtype=dtype)
                prompt_wavs_lengths = batch_dict['prompt_wavs_lengths'].to(device=self.device)
                wavs_lengths = batch_dict['wavs_lengths'].to(device=self.device)

                # prompt 段声纹 [B, spk_dim]，作 diffusion head 说话人条件(use_spk_emb 时生效)
                spk_embs = batch_dict.get('spk_embs', None)
                if spk_embs is not None:
                    spk_embs = spk_embs.to(device=self.device, dtype=dtype)
                # target 预存 semantic latent [B,T,128](仅 latent shard；wav 输入时为 None，forward 在线抽)
                semantic_latents = batch_dict.get('semantic_latents', None)
                if semantic_latents is not None:
                    semantic_latents = semantic_latents.to(device=self.device, dtype=dtype)

                # text_ids = batch_dict['text_ids'].to(self.device)

                input_ids = batch_dict['input_ids'].to(self.device)
                labels = batch_dict['label_ids'].to(self.device)

                info_dict["bs"] = labels.size(0)
                info_dict['tokens'] = labels.size(0) * labels.size(1)

                num_utts = batch_dict["label_ids"].size(0)
                num_seen_utts += num_utts
                if num_utts == 0:
                    continue
                loss_dict = model(
                                keys=batch_dict['keys'],
                                input_ids=input_ids,
                                labels=labels,
                                prompt_wavs=prompt_wavs,
                                wavs=wavs,
                                prompt_wavs_lengths=prompt_wavs_lengths,
                                wavs_lengths=wavs_lengths,
                                audio_pos=batch_dict['audio_pos'],
                                spk_embs=spk_embs,
                                semantic_latents=semantic_latents,
                )
                total_loss += loss_dict['loss'].item() * num_utts
                # info_dict['loss_dict'] = loss_dict

        # write cv: log
        info_dict['loss_dict'] = dict(loss = total_loss/num_seen_utts)
        log_per_step(writer, info_dict, timer=self.cv_step_timer)
        
        return total_loss/num_seen_utts
