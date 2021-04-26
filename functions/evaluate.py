"""
Distributed validation script for stereo depth estimation
"""
import argparse
import os
import sys
import time
import datetime
import warnings
import cv2
import json
import torch
import random
import numpy as np
import torch.nn.functional as F

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(BASE_DIR)
sys.path.append(ROOT_DIR)

import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel
from datasets import KITTIDataset, SceneFlowDataset
from models import build_stereo_depth
from utils.logger import setup_logger

from utils.config import config, update_config
from utils.utils import AverageMeter, load_checkpoint
from utils.helper_custom_transform import Compose, ArrayToTensor
from utils.helper_evaluation import err_pix
from utils.helper_visualization import visual_img, visual_uncertainty, err_unc_plot, visual_disparity, pixerr_unc_plot
from utils.helper_visualization import visual_mask, visual_uncertainty_, disp_error_image
from utils.helper_data_processing import img_to_pix
from utils.helper_data_processing import NestedTensor
from apex.parallel import DistributedDataParallel as DDP
from models.loss import *
from torch.cuda.amp import autocast as autocast, GradScaler
from models.utils import *
from models.PASMnet import PASMnet

warnings.filterwarnings('ignore')
###########################################################
#                      训练参数载入                         #
###########################################################

parser = argparse.ArgumentParser(description='Stereo Depth Estimation')
parser.add_argument('--cfg', type=str, required=True, help='cfg file')
parser.add_argument('--data_root', type=str, help='root director of dataset')
parser.add_argument('--data_name', type=str, help='dataset name')
parser.add_argument('--load_path', type=str, help='path to checkpoint')
parser.add_argument('--local_rank', type=int, help='local rank for DistributedDataParallel')
parser.add_argument('--debug', type=bool, default=False, help='debug mode')
parser.add_argument('--output_dir', type=str, help='path to output')
parser.add_argument('--uncertainty', type=bool, default=False, help='whether to estimate uncertainty or not')
parser.add_argument('--self_supervised', type=bool, default=False, help='self supervised way')
parser.add_argument('--selfada_mask', type=bool, default=False, help='path to dataset')
parser.add_argument('--apex', action='store_true', help='enable mixed precision training')
parser.add_argument('--downsample', default=-1, type=int, help='Ratio to downsample width/height')
parser.add_argument('--trainlist', default=None, help='training list')
parser.add_argument('--testlist', default=None, help='testing list')

args, unparsed = parser.parse_known_args()
update_config(args.cfg)  # 将args接收到的命令行参数更新到config中
config.data_root = args.data_root
config.data_name = args.data_name
config.load_path = args.load_path
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

# 设置训练结果保存
log_model = args.cfg.split('.')[-2].split('/')[-1]  # 网络名称
log_data = args.cfg.split('.')[-2].split('/')[-2]  # 数据集名称

if args.debug:
    log_time = 'debug'
else:
    log_time = datetime.datetime.now().strftime("%m-%d-%H:%M")

config.output_dir = os.path.join(args.output_dir, log_model, log_data, log_time)
os.makedirs(config.output_dir, exist_ok=True)

# 设置logger
logger = setup_logger(output=config.output_dir, distributed_rank=dist.get_rank(), name=config.data_name)

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
from utils.helper_custom_transform import Compose, ArrayToTensor, RandomCrop

if 'kitti' in config.data_name:
    train_dataset = KITTIDataset(datapath=config.data_root, list_filename=args.trainlist, training=True)
    valid_dataset = KITTIDataset(datapath=config.data_root, list_filename=args.testlist, training=False)
elif 'sceneflow' in config.data_name:
    train_dataset = SceneFlowDataset(config, datapath=config.data_root, list_filename=args.trainlist, training=True)
    valid_dataset = SceneFlowDataset(config, datapath=config.data_root, list_filename=args.testlist, training=False)
else:
    raise NotImplementedError("dataset {} not supported".format(config.config.data_name))

valid_sampler = torch.utils.data.distributed.DistributedSampler(valid_dataset, shuffle=False)

valid_loader = torch.utils.data.DataLoader(valid_dataset,
                                           batch_size=1,
                                           num_workers=config.num_workers,
                                           pin_memory=True,
                                           sampler=valid_sampler)


train_sampler = torch.utils.data.distributed.DistributedSampler(train_dataset, shuffle=False)
train_loader = torch.utils.data.DataLoader(train_dataset,
                                           batch_size=1,
                                           num_workers=config.num_workers,
                                           pin_memory=True,
                                           sampler=train_sampler)

logger.info('length of validation dataset: {}'.format(len(valid_dataset.left_filenames)))

###########################################################
#                      创建网络                            #
###########################################################
model = PASMnet()
model.cuda()
model = DistributedDataParallel(model, device_ids=[config.local_rank])
# model = DistributedDataParallel(model, device_ids=[config.local_rank], find_unused_parameters=True)


def main():
    assert os.path.isfile(config.load_path)
    load_checkpoint(logger, config, model, optimizer=None, scheduler=None, is_continue=False)
    ###to do###
    model.eval()
    # model.train()
    ### to do###
    err_total = AverageMeter()
    err_total_right = AverageMeter()
    tic = time.time()

    tl = iter(valid_loader)
    batches = []
    for _ in range(2):
        batches.append(next(tl))

    epe_epoch = AverageMeter()
    d3_epoch = AverageMeter()

    # for idx, (left_img, right_img, left_disp, occ_mask) in enumerate(batches):
    # for idx, (left_img, right_img, left_disp, occ_mask) in enumerate(train_loader):
    # for idx, (img_left, img_right, disp_left, disp_right) in enumerate(train_loader):
    for idx, (img_left, img_right, disp_left, disp_right) in enumerate(valid_loader):
        img_left = img_left.cuda(non_blocking=True)
        img_right = img_right.cuda(non_blocking=True)
        disp_left = disp_left.cuda(non_blocking=True).unsqueeze(1)
        disp_right = disp_right.cuda(non_blocking=True).unsqueeze(1)
        # padding_left = padding_left.cuda(non_blocking=True)
        # padding_right = padding_right.cuda(non_blocking=True)
        # print(img_left.size())

        batch_size = img_left.size(0)


        if 'kitti' in config.data_name:
            mask_left = ((disp_left > 0) & (disp_left < 192)).float()
            mask_right = ((disp_right > 0) & (disp_right < 192)).float()
        elif 'sceneflow' in config.data_name:
            mask_left = ((disp_left > 0) & (disp_left < 192)).float()
            mask_right = ((disp_right > 0) & (disp_right < 192)).float()

        # disp, _, _, _ = model(img_left, img_right, max_disp=config.max_disp)
        disp, att, att_cycle, valid_mask = model(img_left, img_right, max_disp=config.max_disp)
        mask_stage3 = F.interpolate(valid_mask[-1][0], scale_factor=4, mode='nearest')
        # print(disp.shape)

        EPE = EPE_metric(disp, disp_left, mask_left)
        EPE = float(np.array(EPE).mean())
        D3 = D1_metric(disp[:, :, :, :], disp_left[:, :, :, :], mask_left[:, :, :, :], 3)
        D3 = float(np.array(D3).mean())

        epe_epoch.update(EPE, batch_size)
        d3_epoch.update(D3, batch_size)

        logger.info("Valid: [{}/{}] epe:{}, d3:{}\t".format(idx, len(valid_loader), EPE, D3))

        # avg_err_r, count_r = err_pix(pred_disp_right, right_disp, threshold=3, mask=~occ_mask_right)
        # err_total_right.update(avg_err_r.item(), count_r)
        # logger.info("Valid: [{}/{}] err_rate_right:{}\t".format(idx, len(valid_loader), avg_err_r.item()))

        # pred_disp =  pred_disp * mask
        visual_disparity(disp, config, str(idx) + '_pred')
        visual_disparity(disp_left, config, str(idx) + '_gt')
        # visual_disparity(pred_disp_right, config, str(idx) + '_pred_right')
        # visual_disparity(right_disp, config, str(idx) + '_gt_right')
        res = torch.abs(disp - disp_left)
        res[~mask_left.bool()] = 0
        disp_error_image(config, mask_stage3, str(idx) + '_pred_mask')
        disp_error_image(config, res, str(idx) + '_res')
        # res_right = torch.abs(pred_disp_right - right_disp)
        # res_right[~mask] = 0
        # disp_error_image(config, res_right, str(idx) + '_res_right', mask)

    logger.info("total time: {:.2f}, epe.avg: {:.6}, d3.avg:{}".format(time.time() - tic, epe_epoch.avg, d3_epoch.avg))

if __name__ == '__main__':
    main()
