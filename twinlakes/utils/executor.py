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
                

                wavs = batch_dict['wavs'].to(self.device, dtype=dtype)
                wavs_lengths = batch_dict['wavs_lengths'].to(device=self.device)

                videos = batch_dict['videos'].to(self.device, dtype=dtype)
                videos_lengths = batch_dict['videos_lengths'].to(device=self.device)

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
                    loss_dict = model(
                                keys=batch_dict['keys'],
                                input_ids=input_ids,
                                labels=labels,
                                wavs=wavs,
                                wavs_lengths=wavs_lengths,
                                audio_pos=batch_dict['video_pos'],
                                videos=videos,
                                videos_lengths=videos_lengths,
                    )
                    info_dict['loss_dict'] = loss_dict
                    loss = info_dict['loss_dict']['loss']
                loss.backward()
                if (batch_idx + 1) % accum_grad == 0:
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
                    if rank == 0:
                        save_model_path = os.path.join(configs['model_dir'], '{}_{}.pt'.format(configs['epoch'], self.step))
                        infos_ = {
                                    'epoch': configs['epoch'],
                                    'cv_loss': loss.item(),
                                    'step': self.step,
                                    'save_time': datetime.datetime.now().strftime('%d/%m/%Y %H:%M:%S'),
                                }
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


                wavs = batch_dict['wavs'].to(self.device, dtype=dtype)
                wavs_lengths = batch_dict['wavs_lengths'].to(device=self.device)

                videos = batch_dict['videos'].to(self.device, dtype=dtype)
                videos_lengths = batch_dict['videos_lengths'].to(device=self.device)

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
                                wavs=wavs,
                                wavs_lengths=wavs_lengths,
                                audio_pos=batch_dict['video_pos'],
                                videos=videos,
                                videos_lengths=videos_lengths,
                )
                total_loss += loss_dict['loss'].item() * num_utts
                # info_dict['loss_dict'] = loss_dict

        # write cv: log
        info_dict['loss_dict'] = dict(loss = total_loss/num_seen_utts)
        log_per_step(writer, info_dict, timer=self.cv_step_timer)
        
        return total_loss/num_seen_utts
