from __future__ import print_function

import argparse
import datetime
import logging
import os
import torch
import yaml
import copy
import json
import deepspeed
from deepspeed.runtime.zero.stage_1_and_2 import (
    estimate_zero2_model_states_mem_needs_all_live)
from deepspeed.runtime.zero.stage3 import (
    estimate_zero3_model_states_mem_needs_all_live)
from deepspeed.utils.zero_to_fp32 import (
    convert_zero_checkpoint_to_fp32_state_dict,
    get_fp32_state_dict_from_zero_checkpoint)

import torch.distributed as dist

from torch.distributed.elastic.multiprocessing.errors import record

from twinlakes.utils.executor import Executor
# from twinlakes.models.rq_transformer import LMModel

from safetensors.torch import load_model

from twinlakes.dataset.dataset import Dataset
from torch.utils.data import DataLoader
from tensorboardX import SummaryWriter
import torch.optim as optim
from twinlakes.utils.scheduler import WarmupLR
from twinlakes.utils.checkpoint import load_checkpoint, save_checkpoint, save_state_dict_and_infos
from glob import glob
from twinlakes.utils.train_utils import send_dingtalk, check_modify_and_save_config


def get_args():
    parser = argparse.ArgumentParser(description='training your network')
    parser.add_argument('--train_engine',
                        default='torch_ddp',
                        choices=['torch_ddp', 'deepspeed'],
                        help='Engine for paralleled training')
    parser.add_argument('--config', required=True, help='config file')
    parser.add_argument('--model_dir', required=True, help='save model dir')
    parser.add_argument('--data_type',
                        default='raw',
                        choices=['raw', 'shard'],
                        help='train and cv data type')
    parser.add_argument('--train_data', required=True, help='train data file')
    parser.add_argument('--cv_data', required=True, help='cv data file')
    parser.add_argument('--num_workers',
                        default=0,
                        type=int,
                        help='num of subprocess workers for reading')
    parser.add_argument('--pin_memory',
                        action='store_true',
                        default=False,
                        help='Use pinned memory buffers used for reading')
    parser.add_argument('--prefetch',
                        default=2,
                        type=int,
                        help='prefetch number')
    parser.add_argument('--ddp.dist_backend',
                        dest='dist_backend',
                        default='nccl',
                        choices=['nccl', 'gloo', "hccl"],
                        help='distributed backend')
    parser.add_argument('--use_amp',
                        action='store_true',
                        default=False,
                        help='Use automatic mixed precision training')
    parser.add_argument('--tensorboard_dir',
                        default='tensorboard',
                        help='tensorboard log dir')
    parser.add_argument('--timeout',
                        default=30,
                        type=int,
                        help='timeout (in seconds) of wenet_join. ' +
                        '30s for aishell & 300s for wenetspeech')
    parser.add_argument('--deepspeed.save_states',
                        dest='save_states',
                        default='model_only',
                        choices=['model_only', 'model+optimizer'],
                        help='save model/optimizer states')
    parser = deepspeed.add_config_arguments(parser)
    # DeepSpeed automaticly add '--deepspeed' and '--deepspeed_config' to parser
    parser.add_argument('--local-rank',
                        type=int,
                        default=-1,
                        help='local rank passed from distributed launcher')

    args = parser.parse_args()
    if args.train_engine == "deepspeed":
        args.deepspeed = True
        assert args.deepspeed_config is not None
    return args


@record
def main():
    args = get_args()
    logging.basicConfig(level=logging.DEBUG,
                        format='%(asctime)s %(levelname)s %(message)s')

    # Set random seed
    seed=778
    generator = torch.Generator()
    generator.manual_seed(seed)

    # Read config
    with open(args.config, 'r') as fin:
        configs = yaml.load(fin, Loader=yaml.FullLoader)
    from twinlakes.models.rq_transformer import LMModel
    model = LMModel.from_audio_text_pretrained(configs)
    def _load_weights(m, path, skip_uncond_conv=False):
        """strict=False 载入权重，返回 sidecar .yaml 的 infos。
        skip_uncond_conv=True 时丢弃 checkpoint 里所有 uncond_conv.* → uncond_conv 保持随机初始化。
        """
        import re as _re
        sd = torch.load(path, map_location='cpu')
        if skip_uncond_conv:
            sd = {kk: vv for kk, vv in sd.items() if not kk.startswith('uncond_conv.')}
        miss, unexp = m.load_state_dict(sd, strict=False)
        print('[load] %s | skip_uncond=%s missing=%d unexpected=%d'
              % (path, skip_uncond_conv, len(miss), len(unexp)), flush=True)
        info_path = _re.sub(r'\.pt$', '.yaml', path)
        _infos = {}
        if os.path.exists(info_path):
            with open(info_path) as _f:
                _infos = yaml.load(_f, Loader=yaml.FullLoader) or {}
        return _infos

    checkpoints = glob(os.path.join(args.model_dir + '/*.pt'))
    if checkpoints:
        # 按文件名里的 step 排序取最新(epoch_step.pt)。别用 os.path.getatime——
        # 访问时间会被推理/rsync/备份等任何"读"操作刷新，导致加载到旧 checkpoint。
        def _ckpt_step(p):
            name = os.path.basename(p).split('.')[0]      # e.g. "1_150000"
            try:
                return int(name.split('_')[-1])           # 取末段 step
            except ValueError:
                return os.path.getmtime(p)                 # 兜底：按写入时间
        checkpoints = sorted(checkpoints, key=_ckpt_step)
        infos = _load_weights(model, checkpoints[-1])     # resume 本 run：全部加载(含 uncond_conv)
        print('[resume] loaded checkpoint:', checkpoints[-1], flush=True)
    else:
        infos = {}
        # 全新 run 且指定了 init_checkpoint：热启动(权重 only)。
        # uncond_conv 随机初始化(默认 skip)，其余(LLM/connector/head)从 checkpoint 加载。
        init_ckpt = configs.get('init_checkpoint', None)
        if init_ckpt:
            _load_weights(model, init_ckpt,
                          skip_uncond_conv=configs.get('init_skip_uncond_conv', True))
            print('[init] warm-start from', init_ckpt, flush=True)

    def _freeze_params(module):
        for param in module.parameters():
            param.requires_grad = False

    _freeze_params(model.acoustic_tokenizer)
    # _freeze_params(model.lm)
    if configs['freeze_diffusion_head']:
        _freeze_params(model.diffusion_head)



    world_size = int(os.environ.get('WORLD_SIZE', 1))
    local_rank = int(os.environ.get('LOCAL_RANK', 0))
    rank = int(os.environ.get('RANK', 0))

    if args.train_engine == "torch_ddp":
        torch.cuda.set_device(local_rank)
        dist.init_process_group(args.dist_backend)
    elif args.train_engine == "deepspeed":
        deepspeed.init_distributed(dist_backend=args.dist_backend)
    else:
        logging.error("not supported engine: {}".format(args.train_engine))

    configs = check_modify_and_save_config(args, configs)


    train_conf = configs['dataset_conf']
    train_conf['is_inference'] = False
    cv_conf = copy.deepcopy(train_conf)
    cv_conf['speed_perturb'] = False
    cv_conf['spec_aug'] = False
    cv_conf['spec_sub'] = False
    cv_conf['shuffle'] = False

    if rank == 0:
        print('[resume] loaded train dataset:', args.train_data, flush=True)
        print('[resume] loaded dev dataset:', args.cv_data, flush=True)
    train_dataset = Dataset(args.data_type, 
                            args.train_data, 
                            model.tokenizer,
                            train_conf, True)
    cv_dataset = Dataset(args.data_type,
                         args.cv_data,
                         model.tokenizer,
                         cv_conf,
                         partition=False)

    train_data_loader = DataLoader(train_dataset,
                                   batch_size=None,
                                   pin_memory=args.pin_memory,
                                   num_workers=args.num_workers,
                                   persistent_workers=True,
                                   generator=generator,
                                   prefetch_factor=args.prefetch)
    cv_data_loader = DataLoader(cv_dataset,
                                batch_size=None,
                                pin_memory=args.pin_memory,
                                num_workers=args.num_workers,
                                persistent_workers=True,
                                generator=generator,
                                prefetch_factor=args.prefetch)


    writer = None
    if rank == 0:
        print(model)
        num_params = sum(p.numel() for p in model.parameters())
        print('the number of model params: {:,d}'.format(num_params))

        num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print('the number of model requires_grad params : {:,d}'.format(num_params))

        # Writer
        os.makedirs(args.model_dir, exist_ok=True)
        exp_id = os.path.basename(args.model_dir)
        writer = SummaryWriter(os.path.join(args.tensorboard_dir, exp_id))

        saved_config_path = os.path.join(args.model_dir, 'train.yaml')
        with open(saved_config_path, 'w') as fout:
            data = yaml.dump(configs)
            fout.write(data)


    if args.train_engine == "torch_ddp":
        model.cuda()
        model = torch.nn.parallel.DistributedDataParallel(
            model, find_unused_parameters=True)
        device = torch.device("cuda")
        device = int(os.environ.get('LOCAL_RANK', 0))
    elif args.train_engine == "deepspeed":
        local_world_size = int(os.environ.get('LOCAL_WORLD_SIZE', 1))
        if int(os.environ.get('RANK', 0)) == 0:
            logging.info("Estimating model states memory needs (zero2)...")
            estimate_zero2_model_states_mem_needs_all_live(
                model,
                num_gpus_per_node=local_world_size,
                num_nodes=world_size // local_world_size)
            logging.info("Estimating model states memory needs (zero3)...")
            estimate_zero3_model_states_mem_needs_all_live(
                model,
                num_gpus_per_node=local_world_size,
                num_nodes=world_size // local_world_size)
        device = int(os.environ.get('LOCAL_RANK', 0))
    else:
        logging.error("not supported engine: {}".format(args.train_engine))


    optimizer = optim.Adam(model.parameters(), **configs['optim_conf'])
    scheduler = WarmupLR(optimizer, **configs['scheduler_conf'])
    if args.train_engine == "deepspeed":
        with open(args.deepspeed_config, 'r') as fin:
            ds_configs = json.load(fin)
        if "optimizer" in ds_configs:
            # NOTE(xcsong): Disable custom optimizer if it is set in ds_config,
            # extremely useful when enable cpu_offload, DeepspeedCpuAdam
            # could be 4~5x faster than torch native adam
            optimizer = None
            if "scheduler" in ds_configs:
                scheduler = None
            else:

                def scheduler(opt):
                    return WarmupLR(opt, **configs['scheduler_conf'])

        model, optimizer, _, scheduler = deepspeed.initialize(
            args=args,
            model=model,
            optimizer=optimizer,
            lr_scheduler=scheduler,
            model_parameters=model.parameters())

    executor = Executor(device=device, runname=configs['run_name'])

    step = infos.get('step', -1)
    executor.step = step
    scheduler.set_step(step)

    start_epoch = max(0, infos.get('epoch', -1) + 1)
    num_epochs = configs.get('max_epoch', 100)

    for epoch in range(start_epoch, num_epochs):
        configs['epoch'] = epoch
        configs['train_engine'] = args.train_engine
        lr = optimizer.param_groups[0]['lr']
        if rank == 0:
            logging.info('Epoch {} TRAIN info lr {}'.format(epoch, lr))
        # dist.barrier() # Ensure all ranks start Train at the same time.
        dist.barrier(
        )  # NOTE(xcsong): Ensure all ranks start Train at the same time.
        # NOTE(xcsong): Why we need a new group? see `train_utils.py::wenet_join`
        group_join = dist.new_group(
            backend="gloo", timeout=datetime.timedelta(seconds=args.timeout))
        executor.train(model, optimizer, scheduler, train_data_loader, writer, configs, rank, group_join)
        dist.destroy_process_group(group_join)
        dist.barrier() # Ensure all ranks start CV at the same time.
        loss = executor.cv(model, cv_data_loader, writer, configs)
        # loss = 0
        tag = 'train'
        if args.train_engine == "deepspeed":
            model.save_checkpoint(save_dir=args.model_dir, tag=tag)

        if rank == 0:
            logging.info('Epoch {} CV info lr {} cv_loss {}'.format(epoch, lr, loss))
            save_model_path = os.path.join(args.model_dir, '{}.pt'.format(epoch))
            infos_ = {
                        'epoch': epoch,
                        'lr': lr,
                        'cv_loss': loss,
                        'step': executor.step,
                        'save_time': datetime.datetime.now().strftime('%d/%m/%Y %H:%M:%S'),
                    }
            if args.train_engine == "deepspeed":
                state_dict = get_fp32_state_dict_from_zero_checkpoint(args.model_dir, tag,
                                                                    exclude_frozen_parameters=False
                                                                    )
                for key in state_dict.keys():
                    if 'mimi' in key:
                        print(key)
                save_state_dict_and_infos(state_dict, save_model_path, infos_)
                os.system("rm -rf {}/{}".format(args.model_dir, tag))
            else:
                save_checkpoint(model, save_model_path, infos_)
            send_dingtalk(configs['run_name'], epoch, executor.step, loss, lr)


if __name__ == '__main__':
    main()
