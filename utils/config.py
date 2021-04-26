import yaml  # conda install pyyaml
from easydict import EasyDict as edict

config = edict()  # 使得可以像访问属性一样访问字典里的元素

###########################################################
#                    输入输出与保存                          #
###########################################################

config.load_path = ''  # 载入 checkpoint 路径
config.print_freq = 30  # 每几个 batch 打印一次
config.save_freq = 10  # 每几个 epoch 保存一次模型
config.val_freq = 10  # 每几个 epoch 进行一次验证
config.log_dir = 'logs'  # 训练日志保存目录
config.output_dir = 'outputs'  # 网络预测结果保存路径
config.local_rank = 0
config.amp_opt_level = ''
config.rng_seed = 203203  # 随机种子

###########################################################
#                      训练部分                             #
###########################################################

config.start_epoch = 1  # 用于加载断点继续训练
config.epochs = 2000  # 训练轮次
config.train_steps = 600  # 自定义每个epoch有多少个 batch
config.learning_rate = 0.01  # 初始学习率
config.optimizer = 'sgd'  # 优化器，['sgd','adam','adamW']
config.lr_scheduler = 'step'  # 学习率调整方式, ['step','cosine']
config.warmup_epoch = 5  # warmup 轮次，-1 表示没有warmup
config.warmup_multiplier = 100  # warmup 倍率
config.lr_decay_steps = [10, 20, 30]  # 每几个 epoch 调整一次学习率
config.lr_decay_rate = 0.98  # 学习率调整系数
config.weight_decay = 0  # 优化器使用参数
config.momentum = 0.9  # 优化器使用参数
config.grad_clip_norm = 100.0  # 梯度裁剪参数
config.is_conitunue = False
config.self_supervised = False
###########################################################
#                       数据部分                            #
###########################################################

config.data_name = 'sceneflow'  # 数据库名称
config.task = 'stereo_depth'  # 任务名称
config.data_root = ''  # 数据根目录
config.batch_size = 32  # batch size
config.num_workers = 4  # workers
config.max_disp = 192  # 最大视差范围
config.height = 256  # 训练时输入图像高
config.width = 512  # 训练时输入图像宽
config.augment_scale_size = 0  # 数据增强: 缩放尺度
config.augment_input_size = 0  # 数据增强: 输入图像大小
config.augment_brightness = 0.4  # 数据增强: 亮度
config.augment_contrast = 0.4  # 数据增强: 对比度
config.augment_saturation = 0.4  # 数据增强: 饱和度
config.augment_lighting = 0.1  # 数据增强: 光照度
config.kitti_mix = True
###########################################################
#                       模型部分                            #
###########################################################
config.use_batch_norm = True
config.batch_norm_momentum = 0.1
config.model_name = 'psm_stackhourglass'
config.uncertainty = False
###########################################################
##transformer#################################3
########################################################
config.channel_dim = 128
config.position_encoding = 'sine1d_rel'
config.nheads = 8
config.num_attn_layers = 6
config.context_adjustment_layer = 'none'
config.cal_num_blocks = 8
config.cal_feat_dim = 16
config.cal_expansion_ratio = 4
config.regression_head = 'ot'


# 将 .yaml中的数据读入，并更新config
def update_config(config_file):
    with open(config_file) as f:
        exp_config = edict(yaml.load(f, Loader=yaml.FullLoader))
        for k, v in exp_config.items():
            # print(k)
            if k in config:
                # print(True)
                if isinstance(v, dict):
                    for vk, vv in v.items():
                        config[k][vk] = vv
                else:
                    config[k] = v
            else:
                raise ValueError(f"{k} key must exist in config.py")
