import os

import torch


class BColors:
    """
    在终端打印时字体的效果
    例：print('{:}{:s}'.format(bcolors.OKGREEN, key))
    """
    HEADER = '\033[95m'  # 紫色
    OKBLUE = '\033[94m'  # 蓝色
    OKGREEN = '\033[92m'  # 绿色
    WARNING = '\033[93m'  # 黄色
    FAIL = '\033[91m'  # 红色
    ENDC = '\033[0m'  # 清除效果
    BOLD = '\033[1m'  # 加粗
    UNDERLINE = '\033[4m'  # 加下划线


# 在训练过程中累计训练结果
class AverageMeter(object):
    """计算并存储 当前值和平均值"""

    def __init__(self):
        self.val = 0  # 当前值
        self.avg = 0  # 平均值
        self.sum = 0  # 累加和
        self.count = 0  # 累计数量
        self.err_rate = []
        self.err_rate_sum = 0
        self.err_rate_avg = 0

    def update(self, val, n=1):
        self.err_rate_sum += val
        self.err_rate.append(val)
        self.err_rate_avg = self.err_rate_sum / len(self.err_rate)
        self.val = val  # val是 n 个sample的平均值
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


# 载入预训练模型
def load_checkpoint(logger, config, model, optimizer, scheduler, is_continue=False):
    logger.info("=> loading checkpoint '{}'".format(config.load_path))

    checkpoint = torch.load(config.load_path, map_location=torch.device('cpu'))
    # for k, v in checkpoint.items():
    #     print(k)
    # model.module.load_state_dict(checkpoint)
    model.module.load_state_dict(checkpoint['model'])

    if is_continue:
        config.start_epoch = checkpoint['epoch'] + 1
        optimizer.load_state_dict(checkpoint['optimizer'])
        scheduler.load_state_dict(checkpoint['scheduler'])

    logger.info("=> loaded successfully '{}' (epoch {})".format(config.load_path, checkpoint['epoch']))

    del checkpoint
    torch.cuda.empty_cache()


# 保存模型
def save_checkpoint(logger, config, epoch, model, optimizer, scheduler, idx=-1):
    logger.info('==> Saving checkpoint...')
    state = {
        'config': config,
        'epoch': epoch,
        'idx': idx,
        'model': model.module.state_dict(),
        'optimizer': optimizer.state_dict(),
        'scheduler': scheduler.state_dict(),
    }

    # torch.save(state, os.path.join(config.log_dir, 'current.tar'))
    if idx == -1:
        torch.save(state, os.path.join(config.log_dir, 'ckpt_epoch_{}.tar'.format(epoch)))
        logger.info("Saved in {}".format(os.path.join(config.log_dir, 'ckpt_epoch_{}.tar'.format(epoch))))
    else:
        torch.save(state, os.path.join(config.log_dir, 'ckpt_epoch_{}_idx{}.tar'.format(epoch, idx)))
        logger.info("Saved in {}".format(os.path.join(config.log_dir, 'ckpt_epoch_{}_{}.tar'.format(epoch, idx))))

