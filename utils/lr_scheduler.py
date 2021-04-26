from torch.optim.lr_scheduler import _LRScheduler, MultiStepLR, CosineAnnealingLR, ExponentialLR


class GradualWarmupScheduler(_LRScheduler):
    """
    逐步增加优化器中的学习率，论文《Accurate, Large Minibatch SGD: Training ImageNet in 1 Hour》
    """

    def __init__(self, optimizer, multiplier, warmup_epoch, after_scheduler, last_epoch=-1):
        """
        Parameters
        ----------
        optimizer : 优化器
        multiplier : warmup 的起点学习率   base lr / multiplier
        warmup_epoch : 从epoch 0 开始逐渐增加学习率，到warmop_epoch是达到目标学习率
        after_scheduler : 在warmup_epoch之后采用的调整方式
        last_epoch : 默认值
        """
        self.multiplier = multiplier
        if self.multiplier <= 1.:
            raise ValueError('multiplier should be greater than 1.')
        self.warmup_epoch = warmup_epoch
        self.after_scheduler = after_scheduler
        self.finished = False
        super().__init__(optimizer, last_epoch=last_epoch)

    def get_lr(self):
        if self.last_epoch > self.warmup_epoch:
            return self.after_scheduler.get_lr()
        else:
            return [base_lr / self.multiplier * ((self.multiplier - 1.) * self.last_epoch / self.warmup_epoch + 1.)
                    for base_lr in self.base_lrs]

    def step(self, epoch=None):
        if epoch is None:
            epoch = self.last_epoch + 1    # 不明确指定epoch值时，自动累加1
        self.last_epoch = epoch
        if epoch > self.warmup_epoch:
            self.after_scheduler.step(epoch)
        else:
            super(GradualWarmupScheduler, self).step(epoch)

    def state_dict(self):
        """返回scheduler，保存在checkpoints中
        """

        state = {key: value for key, value in self.__dict__.items() if key != 'optimizer' and key != 'after_scheduler'}
        state['after_scheduler'] = self.after_scheduler.state_dict()
        return state

    def load_state_dict(self, state_dict):
        """载入训练checkpoint时，同步更新scheduler
        scheduler.load_state_dict(checkpoint['scheduler'])
        """

        after_scheduler_state = state_dict.pop('after_scheduler')
        self.__dict__.update(state_dict)
        self.after_scheduler.load_state_dict(after_scheduler_state)


def get_scheduler(optimizer, config):
    """
    optimizer : 使用的优化器
    config : 参数字典
    Returns : 选择出的scheduler
    """
    if "cosine" in config.lr_scheduler:
        # 余弦退火
        scheduler = CosineAnnealingLR(
            optimizer=optimizer,
            eta_min=0.000001,  # 最小学习率
            T_max=config.epochs - config.warmup_epoch)  # 一次学习率周期的epoch数，T_max个epoch后重新设置学习率
    elif "step" in config.lr_scheduler:
        # 计算更改学习率时的epoch，list数组
        #lr_decay_epochs = [config.lr_decay_steps * i for i in range(1, config.epochs // config.lr_decay_steps)]
        lr_decay_epochs = config.lr_decay_steps
        scheduler = MultiStepLR(
            optimizer=optimizer,
            gamma=config.lr_decay_rate,  # 学习率衰减系数
            milestones=lr_decay_epochs)
    elif 'ExponentialLR' in config.lr_scheduler:
        scheduler = ExponentialLR(optimizer, gamma=config.lr_decay_rate)
    else:
        raise NotImplementedError("scheduler {} not supported".format(config.lr_scheduler))

    scheduler = GradualWarmupScheduler(
        optimizer,
        multiplier=config.warmup_multiplier,
        after_scheduler=scheduler,
        warmup_epoch=config.warmup_epoch)
    return scheduler
