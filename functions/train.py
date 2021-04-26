"""
Distributed training script for stereo depth estimation
"""
import argparse
import datetime
import json
import os
import sys
import time
import random
import numpy as np
import warnings
import torch

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(BASE_DIR)
sys.path.append(ROOT_DIR)

import torch.distributed as dist

from torch.utils.tensorboard import SummaryWriter
from torch.nn.parallel import DistributedDataParallel
import torch.nn.functional as F

from datasets.sceneflow_dataset import SceneFlowDataset
from datasets.kitti_dataset import KITTIDataset
from models import build_stereo_depth
from utils.config import config, update_config
from utils.logger import setup_logger
from utils.lr_scheduler import get_scheduler
from utils.utils import AverageMeter, save_checkpoint, load_checkpoint
from utils.helper_custom_transform import Compose, ArrayToTensor, RandomCrop
from utils.helper_optimizer import get_optimizer
from utils.helper_evaluation import err_pix
import torch.nn.functional as F
from utils.helper_visualization import visual_img, visual_uncertainty, err_unc_plot, visual_disparity, pixerr_unc_plot
from utils.helper_visualization import visual_mask, visual_uncertainty_, disp_error_image
from utils.helper_data_processing import NestedTensor
from models.loss import *
from torch.cuda.amp import autocast as autocast, GradScaler
from models.utils import *
from models.PASMnet import PASMnet

warnings.filterwarnings('ignore')
###########################################################
#                      训练参数载入                         #
###########################################################

parser = argparse.ArgumentParser('Stereo Depth Estimation')
parser.add_argument('--cfg', type=str, required=True, help='cfg file')
parser.add_argument('--data_root', type=str, help='root director of dataset')
parser.add_argument('--data_name', type=str, help='dataset name')
parser.add_argument('--load_path', type=str, help='path to checkpoint')
parser.add_argument('--is_continue', type=bool, default=False, help='whether to continue to train using checkpoint')
parser.add_argument('--local_rank', type=int, help='local rank for DistributedDataParallel')
parser.add_argument('--debug', type=bool, default=False, help='debug mode')
parser.add_argument('--output_dir', type=str, help='path to output')
parser.add_argument('--trainlist', default=None, help='training list')
parser.add_argument('--testlist', default=None, help='testing list')


args, unparsed = parser.parse_known_args()
update_config(args.cfg)  # 将args接收到的命令行参数更新到config中
config.data_root = args.data_root
config.data_name = args.data_name
config.load_path = args.load_path
config.is_continue = args.is_continue
config.local_rank = args.local_rank
config.output_dir = args.output_dir

###########################################################
#                         初始化                           #
###########################################################

# 设置随机种子
torch.manual_seed(config.rng_seed)
torch.cuda.manual_seed_all(config.rng_seed)
random.seed(config.rng_seed)
np.random.seed(config.rng_seed)

# 设置GPU
torch.cuda.set_device(config.local_rank)
torch.distributed.init_process_group(backend='nccl')
torch.backends.cudnn.enabled = True
torch.backends.cudnn.benchmark = True
torch.backends.cudnn.deterministic = True

# 设置训练日志
log_model = args.cfg.split('.')[-2].split('/')[-1]  # 网络名称
log_data = args.cfg.split('.')[-2].split('/')[-2]  # 数据集名称

if args.debug:
    log_time = 'debug'
else:
    log_time = datetime.datetime.now().strftime("%m-%d-%H:%M")

config.log_dir = os.path.join(config.log_dir, log_model, log_data, log_time)
os.makedirs(config.log_dir, exist_ok=True)

# 设置logger
logger = setup_logger(output=config.log_dir, distributed_rank=dist.get_rank(), name=config.data_name)

# 设置配置文件保存
if dist.get_rank() == 0:
    path = os.path.join(config.log_dir, "config.json")
    with open(path, 'w') as f:
        json.dump(vars(args), f, indent=2)
        json.dump(vars(config), f, indent=2)
        os.system("cp {} {}".format(args.cfg, config.log_dir))
    logger.info("Full config saved to {}".format(path))

###########################################################
#                      数据部分                         #
###########################################################

if 'kitti' in config.data_name:
    train_dataset = KITTIDataset(datapath=config.data_root, list_filename=args.trainlist, training=True)
    valid_dataset = KITTIDataset(datapath=config.data_root, list_filename=args.testlist, training=False)
elif 'sceneflow' in config.data_name:
    train_dataset = SceneFlowDataset(config, datapath=config.data_root, list_filename=args.trainlist, training=True)
    valid_dataset = SceneFlowDataset(config, datapath=config.data_root, list_filename=args.testlist, training=False)
else:
    raise NotImplementedError("dataset {} not supported".format(config.config.data_name))

train_sampler = torch.utils.data.distributed.DistributedSampler(train_dataset, shuffle=True)
valid_sampler = torch.utils.data.distributed.DistributedSampler(valid_dataset, shuffle=False)

train_loader = torch.utils.data.DataLoader(train_dataset,
                                           batch_size=config.batch_size,
                                           num_workers=config.num_workers,
                                           pin_memory=True,
                                           sampler=train_sampler)

valid_loader = torch.utils.data.DataLoader(valid_dataset,
                                           batch_size=1,
                                           num_workers=config.num_workers,
                                           pin_memory=True,
                                           sampler=valid_sampler)

logger.info('length of training dataset: {}'.format(len(train_dataset.left_filenames)))
logger.info('length of validation dataset: {}'.format(len(valid_dataset.left_filenames)))

###########################################################
def print_param(model):
    """
    print number of parameters in the model
    """

    n_parameters = sum(p.numel() for n, p in model.named_parameters() if 'backbone' in n and p.requires_grad)
    print('number of params in backbone:', f'{n_parameters:,}')
    n_parameters = sum(p.numel() for n, p in model.named_parameters() if
                       'transformer' in n and 'regression' not in n and p.requires_grad)
    print('number of params in transformer:', f'{n_parameters:,}')
    n_parameters = sum(p.numel() for n, p in model.named_parameters() if 'tokenizer' in n and p.requires_grad)

    print('number of params in tokenizer:', f'{n_parameters:,}')
    n_parameters = sum(p.numel() for n, p in model.named_parameters() if 'regression' in n and p.requires_grad)
    print('number of params in regression:', f'{n_parameters:,}')
#                      创建网络                            #
###########################################################
model = PASMnet()
model.cuda()
print_param(model)
optimizer = get_optimizer(model, config)
scheduler = get_scheduler(optimizer, config)

model = DistributedDataParallel(model, device_ids=[config.local_rank], find_unused_parameters=True)
# model = DistributedDataParallel(model, device_ids=[config.local_rank])
# model = DDP(model, delay_allreduce=True)
###########################################################
#                      优化器                              #
###########################################################



###########################################################
#                      Tensorboard                        #
###########################################################

if dist.get_rank() == 0:
    summary_writer = SummaryWriter(log_dir=config.log_dir)
else:
    summary_writer = None

def main():
    if config.load_path:
        assert os.path.isfile(config.load_path)
        load_checkpoint(logger, config, model, optimizer, scheduler, config.is_continue)
        logger.info("==> checking loaded ckpt")
    if dist.get_rank() == 0:
        save_checkpoint(logger, config, -1, model, optimizer, scheduler, -1)
        logger.info('saved!')

    for epoch in range(config.start_epoch, config.epochs + 1):
        train_loader.sampler.set_epoch(epoch)
        valid_loader.sampler.set_epoch(epoch)
        tic = time.time()
        scheduler.step()

        logger.info("epoch: {}, lr: {:.6f}".format(epoch, optimizer.param_groups[0]['lr']))

        loss, epe, d3 = train(epoch, train_loader, model, optimizer, config)
        # 保存模型
        if dist.get_rank() == 0 and epoch % config.save_freq == 0:
            save_checkpoint(logger, config, epoch, model, optimizer, scheduler, -1)
        logger.info('**************************************************************')
        logger.info("epoch: {}, total time: {:.2f}, lr: {:.6f}, avg loss: {:.6f}, avg epe: {:.6f}, avg d3: {:.6f}".format(epoch, time.time() - tic,
                                                                                         optimizer.param_groups[0][
                                                                                             'lr'],
                                                                                         loss, epe, d3))
        logger.info('**************************************************************')
        # # 保存模型
        # if dist.get_rank() == 0 and epoch % config.save_freq == 0:
        #     save_checkpoint(logger, config, epoch, model, optimizer, scheduler, -1, amp=amp)

        # 将数据写入Tensorboard
        if summary_writer is not None:
            summary_writer.add_scalar('avg_loss', loss, epoch)
            summary_writer.add_scalar('epe', epe, epoch)
            summary_writer.add_scalar('d3', d3, epoch)
            summary_writer.add_scalar('learning_rate', optimizer.param_groups[0]['lr'], epoch)

    tic = time.time()
    logger.info("last, total time: {:.2f}, avg err: {:.3}%".format(time.time() - tic, err * 100))


def train(epoch, train_loader, model, optimizer, config):
    model.train()
    # model.eval()
    # criterion.train()

    loss_epoch = AverageMeter()
    epe_epoch = AverageMeter()
    d3_epoch = AverageMeter()
    loss_P_epoch = AverageMeter()
    loss_S_epoch = AverageMeter()
    loss_PAM_P_epoch = AverageMeter()
    loss_PAM_C_epoch = AverageMeter()
    loss_PAM_S_epoch = AverageMeter()
    loss_PAM_epoch = AverageMeter()

    # for idx, (img_left, img_right, disp_left, disp_right) in enumerate(valid_loader):
    for idx, (img_left, img_right, disp_left, disp_right) in enumerate(train_loader):
        # tic_start = time.time()
        # print('###############################', idx)
        img_left = img_left.cuda(non_blocking=True)
        img_right = img_right.cuda(non_blocking=True)
        disp_left = disp_left.cuda(non_blocking=True).unsqueeze(1)
        disp_right = disp_right.cuda(non_blocking=True).unsqueeze(1)
        # print(img_left.size())

        batch_size = img_left.size(0)
        if 'kitti' in config.data_name:
            mask_left = ((disp_left > 0) & (disp_left < 192)).float()
            mask_right = ((disp_right > 0) & (disp_right < 192)).float()
        elif 'sceneflow' in config.data_name:
            mask_left = ((disp_left > 0) & (disp_left < 192)).float()
            mask_right = ((disp_right > 0) & (disp_right < 192)).float()

        disp, att, att_cycle, valid_mask = model(img_left, img_right, max_disp=config.max_disp)

        if 'kitti' in config.data_name:
            # loss-P
            loss_P = loss_disp_unsupervised(img_left, img_right, disp,
                                            F.interpolate(valid_mask[-1][0], scale_factor=4, mode='nearest'))
            # loss-S
            loss_S = loss_disp_smoothness(disp, img_left)
            # loss-PAM
            loss_PAM_P = loss_pam_photometric(img_left, img_right, att, valid_mask)
            loss_PAM_C = loss_pam_cycle(att_cycle, valid_mask)
            loss_PAM_S = loss_pam_smoothness(att)
            loss_PAM = loss_PAM_P + 5 * loss_PAM_S + 5 * loss_PAM_C
            loss = loss_P + 0.5 * loss_S + loss_PAM
            # loss_PAM = loss_PAM_P + loss_PAM_S + loss_PAM_C
            # loss = loss_P + 0.1 * loss_S + loss_PAM
        elif 'sceneflow' in config.data_name:
            # loss-P
            loss_P = loss_disp_unsupervised(img_left, img_right, disp,
                                            F.interpolate(valid_mask[-1][0], scale_factor=4, mode='nearest'), mask_left)
            # loss-S
            loss_S = loss_disp_smoothness(disp, img_left)
            # loss-PAM
            loss_PAM_P = loss_pam_photometric(img_left, img_right, att, valid_mask, [mask_left, mask_right])
            loss_PAM_C = loss_pam_cycle(att_cycle, valid_mask)
            loss_PAM_S = loss_pam_smoothness(att)
            # loss_PAM = loss_PAM_P + loss_PAM_C + 1.6 * loss_PAM_S
            # losses
            # loss = 1.5 * loss_P + 0.1 * loss_S + 0.5 * loss_PAM

            loss_PAM = loss_PAM_P + loss_PAM_S + loss_PAM_C
            loss = loss_P + 0.1 * loss_S + loss_PAM


        # with autocast():
        #     scaler.scale(loss).backward()
        optimizer.zero_grad()
        loss.backward()
        # torch.nn.utils.clip_grad_norm_(model.parameters(), config.grad_clip_norm)
        optimizer.step()

        EPE = EPE_metric(disp, disp_left, mask_left)
        EPE = float(np.array(EPE).mean())
        D3 = D1_metric(disp[:, :, :, :], disp_left[:, :, :, :], mask_left[:, :, :, :], 3)
        D3 = float(np.array(D3).mean())

        loss_P_epoch.update(loss_P.item(), batch_size)
        loss_S_epoch.update(loss_S.item(), batch_size)
        loss_PAM_P_epoch.update(loss_PAM_P.item(), batch_size)
        loss_PAM_C_epoch.update(loss_PAM_C.item(), batch_size)
        loss_PAM_S_epoch.update(loss_PAM_S.item(), batch_size)
        loss_PAM_epoch.update(loss_PAM.item(), batch_size)

        if loss.item() < 10000:
            loss_epoch.update(loss.item(), batch_size)
        if EPE < 10000:
            epe_epoch.update(EPE, batch_size)
        if D3 <= 1.01:
            d3_epoch.update(D3, batch_size)


        # print(idx)
        # print(config.print_freq)
        if (idx % config.print_freq == 0) and (idx > 0):
            logger.info('########################################################')
            logger.info("Train:[{}/{}][{}/{}]   loss_P:{},  loss_S:{},  loss_PAM:{}\t".format(epoch, config.epochs + 1, idx,
                                                                                   len(train_loader)-1,
                                                                                   loss_P_epoch.avg, loss_S_epoch.avg, loss_PAM_epoch.avg))
            logger.info(
                "Train:[{}/{}][{}/{}]   loss_PAM_P:{},  loss_PAM_C:{},  loss_PAM_S:{}\t".format(epoch, config.epochs + 1, idx,
                                                                                      len(train_loader)-1,
                                                                                      loss_PAM_P_epoch.avg, loss_PAM_C_epoch.avg,
                                                                                      loss_PAM_S_epoch.avg))
            # logger.info("Train:[{}/{}][{}/{}]   loss:{},  epe:{},  d3:{}\t".format(epoch, config.epochs + 1, idx, len(train_loader),
            #                                                                        loss.item(), EPE, D3))
            logger.info("Train:[{}/{}][{}/{}]   loss_avg:{},  epe_avg:{},  d3_avg:{}\t".format(epoch, config.epochs + 1, idx, len(train_loader)-1,
                                                                  loss_epoch.avg, epe_epoch.avg, d3_epoch.avg))
            logger.info('########################################################')

    return loss_epoch.avg, epe_epoch.avg, d3_epoch.avg


if __name__ == '__main__':
    main()