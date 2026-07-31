from __future__ import print_function

import argparse
import datetime
import logging
import os
import torch
import yaml
import copy
import json

import torch.distributed as dist

from torch.distributed.elastic.multiprocessing.errors import record

from twinlakes.utils.executor import Executor


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
    parser.add_argument('--local-rank',
                        type=int,
                        default=-1,
                        help='local rank passed from distributed launcher')

    args = parser.parse_args()
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
        infos = load_checkpoint(model, checkpoints[-1])     # resume 本 run：全部加载(含 uncond_conv)
        print('[resume] loaded checkpoint:', checkpoints[-1], flush=True)
    else:
        infos = {}

    def _freeze_params(module):
        for param in module.parameters():
            param.requires_grad = False

    _freeze_params(model.acoustic_tokenizer)
    # _freeze_params(model.lm)
    if configs.get('freeze_video_dit', configs.get('freeze_diffusion_head', False)):
        _freeze_params(model.video_dit)
    _freeze_params(model.vae.model)


    world_size = int(os.environ.get('WORLD_SIZE', 1))
    local_rank = int(os.environ.get('LOCAL_RANK', 0))
    rank = int(os.environ.get('RANK', 0))

    torch.cuda.set_device(local_rank)
    dist.init_process_group(args.dist_backend)

    device = torch.device(f"cuda:{local_rank}")
    


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


    model = model.to(device)
    model = torch.nn.parallel.DistributedDataParallel(
        model, 
        device_ids=[local_rank], output_device=local_rank,
        find_unused_parameters=True)


    optimizer = optim.Adam(model.parameters(), **configs['optim_conf'])
    scheduler = WarmupLR(optimizer, **configs['scheduler_conf'])

    executor = Executor(device=device, runname=configs['run_name'])

    step = infos.get('step', -1)
    executor.step = step
    scheduler.set_step(step)

    start_epoch = max(0, infos.get('epoch', -1) + 1)
    num_epochs = configs.get('max_epoch', 100)

    for epoch in range(start_epoch, num_epochs):
        configs['epoch'] = epoch
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
            save_checkpoint(model, save_model_path, infos_)
            send_dingtalk(configs['run_name'], epoch, executor.step, loss, lr)


if __name__ == '__main__':
    main()
