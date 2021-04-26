import os

import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import cv2
import torch.nn.functional as F
from sklearn import metrics


def disp_to_color(disp, max_disp=192):
    """
    disp: numpy float32 array of dimension
    """
    height, width = disp.shape

    # if max_disp < 0:
    #     max_disp = np.max(disp)  # 当前视差图的最大值
    disp = disp / max_disp  # 归一化
    disp = disp.reshape((disp.size, 1))  # disp.size 指所有的像素数量
    # print(disp)

    colormap = np.array([
        [0, 0, 0, 114],
        [0, 0, 1, 185],
        [1, 0, 0, 114],
        [1, 0, 1, 174],
        [0, 1, 0, 114],
        [0, 1, 1, 185],
        [1, 1, 0, 114],
        [1, 1, 1, 0]
    ])

    # [114., 185., 114., 174., 114., 185., 114.]
    bins = colormap[0:colormap.shape[0] - 1, colormap.shape[1] - 1].astype(float)  # map 中的最后一列
    # [[114.], [185.], [114.], [174.], [114.], [185.], [114.]]
    bins = bins.reshape((bins.shape[0], 1))  # (8,1)
    # [ 114.,  299.,  413.,  587.,  701.,  886., 1000.]
    cbins = np.cumsum(bins)  # 累加和
    # [[0.114],[0.185],[0.114],[0.174],[0.114],[0.185],[0.114]]
    bins = bins / cbins[cbins.shape[0] - 1]  # 归一化
    # [0.114, 0.299, 0.413, 0.587, 0.701, 0.886]
    cbins = cbins[0:cbins.shape[0] - 1] / cbins[cbins.shape[0] - 1]
    # [[0.114], [0.299], [0.413], [0.587], [0.701], [0.886]]
    cbins = cbins.reshape((cbins.shape[0], 1))  # (6,1)

    ind = np.tile(disp.T, (6, 1))  # (6, disp.size) disp 拉成一行，复制了6行
    tmp = np.tile(cbins, (1, disp.size))  # (6, disp.size)
    b = (ind > tmp).astype(int)  # 0/1 mask,
    s = np.sum(b, axis=0)  # 每个像素的disp值，大于cbins中的几个数, 表示视察所处的bin
    bins = 1 / bins  # [[8.77],[5.40],[8.77],[5.74],[8.77],[5.40],[8.77]]
    t = cbins
    cbins = np.zeros((cbins.size + 1, 1))  # (7,1) 一共7个bin,
    cbins[1:] = t  # [[0],[0.114],[0.299],[0.413],[0.587],[0.701],[0.886]]
    disp = (disp - cbins[s]) * bins[s]
    disp = colormap[s, 0:3] * np.tile(1 - disp, (1, 3)) + colormap[s + 1, 0:3] * np.tile(disp, (1, 3))

    disp = disp.reshape((height, width, 3))
    disp = (disp * 255).astype('uint8')
    # print(disp)
    return disp


def visual_img(img, config, img_name):
    path = os.path.join(config.output_dir, 'visualization')
    if not os.path.exists(path):
        os.mkdir(path)

    img = img * 255
    img = img.squeeze()
    img = img.data.cpu().numpy().astype('uint8')
    img = np.transpose(img, (1, 2, 0))
    img = Image.fromarray(img, 'RGB')
    img.save(os.path.join(path, img_name + '.png'))


def visual_disparity(disp, config, name, mask=None):
    path = os.path.join(config.output_dir, 'visualization')
    if not os.path.exists(path):
        os.mkdir(path)

    if mask is not None:
        disp[~mask] = 0  # 这里的mask为1时表示有ground truth值，0表示没有ground truth值
    disp = disp.squeeze()  # 去掉维度为1的维度
    disp = disp.data.cpu().numpy()  # 把数据拷贝到CPU上来
    disp = disp_to_color(disp, 192)  # 将视差图转换成RGB图像，RGB值和视差大小相关
    disp = Image.fromarray(disp, 'RGB')
    # print(os.path.join(path, name+'.png'))

    disp.save(os.path.join(path, name+'.png'))


def visual_confidence(conf_map, config, name):
    path = os.path.join(config.output_dir, 'visualization')
    if not os.path.exists(path):
        os.mkdir(path)

    conf_map = conf_map.squeeze()
    conf_map = conf_map.data.cpu().numpy()
    # 这里操作是基于估计出来confidence值为0~255之间的浮点数值
    conf_map = conf_map.astype('uint8')  # TODO：0/1的 confidence map

    conf_map = Image.fromarray(conf_map)
    conf_map.save(os.path.join(path, 'confidence_{}.png'.format(name)))


def plot_list(config, inds, list, name):
    path = os.path.join(config.log_dir, 'plot_losslist')
    if not os.path.exists(path):
        os.mkdir(path)

    plt.figure()
    plt.xlabel('epoch')
    plt.ylabel('err')
    plt.plot(inds, list)

    plt.savefig(os.path.join(path, 'lossplot_{}.png'.format(name)))


def err_unc_plot(config, err_unc, name):
    path = os.path.join(config.output_dir, 'plot')
    if not os.path.exists(path):
        os.mkdir(path)

    err_unc = np.array(err_unc)  # N行，表示N个点，2列，第一列为误差，第二列为不确定性
    np.save(os.path.join(path, name), err_unc)

    err_random = err_unc.copy()
    err_unc = err_unc[np.argsort(-err_unc[:, 1])]  # 按照不确定性从高到低重新排列数组
    err_err = err_unc[np.argsort(-err_unc[:, 0])]
    np.random.shuffle(err_random)

    err = err_unc[:, 0]  # 误差列
    err_e = err_err[:, 0]
    err_r = err_random[:, 0]
    percentiles = np.arange(100) / 100.
    cutoff_inds = (percentiles * err_unc.shape[0]).astype(int)  # 每个节点对应的点的个数
    cutoff_inds_err = (percentiles * err_e.shape[0]).astype(int)  # 每个节点对应的点的个数
    cutoff_inds_random = (percentiles * err_r.shape[0]).astype(int)  # 每个节点对应的点的个数

    pix_error = [err[cutoff:].mean() for cutoff in cutoff_inds]  # 每个节点之前的所有误差的平均值
    pix_error_e = [err_e[cutoff:].mean() for cutoff in cutoff_inds_err]  # 每个节点之前的所有误差的平均值
    pix_error_random = [err_r[cutoff:].mean() for cutoff in cutoff_inds_random]  # 每个节点之前的所有误差的平均值
    # pix_error = [metrics.mean_squared_error(err[cutoff:], np.zeros_like(err[cutoff:])) for cutoff in cutoff_inds]  # 每个节点之前的所有误差的平均值

    plt.figure()
    plt.xlabel('drop_percentile')
    plt.ylabel('err')
    plt.plot(percentiles, pix_error, color='green', label='uncertainty')
    plt.plot(percentiles, pix_error_e, color='aqua', label='oracle')
    plt.plot(percentiles, pix_error_random, color='red', label='random')
    plt.legend()
    plt.savefig(os.path.join(path, name + '.png'))


def pixerr_unc_plot(config, err_unc, threshold=3, name='test'):
    path = os.path.join(config.output_dir, 'plot')
    if not os.path.exists(path):
        os.mkdir(path)

    err_unc = np.array(err_unc)  # N行，表示N个点，2列，第一列为误差，第二列为不确定性
    np.save(os.path.join(path, name), err_unc)

    err_random = err_unc.copy()
    err_unc = err_unc[np.argsort(-err_unc[:, 1])]  # 按照不确定性从高到低重新排列数组
    err_err = err_unc[np.argsort(-err_unc[:, 0])]
    np.random.shuffle(err_random)

    err = err_unc[:, 0]  # 误差列
    err_e = err_err[:, 0]
    err_r = err_random[:, 0]
    percentiles = np.arange(100) / 100.
    cutoff_inds = (percentiles * err_unc.shape[0]).astype(int)  # 每个节点对应的点的个数
    cutoff_inds_err = (percentiles * err_e.shape[0]).astype(int)  # 每个节点对应的点的个数
    cutoff_inds_random = (percentiles * err_r.shape[0]).astype(int)  # 每个节点对应的点的个数

    count = len(err)
    err = (err > threshold).astype(float)
    err_e = (err_e > threshold).astype(float)
    err_r = (err_r > threshold).astype(float)

    pix_error = [(err[cutoff:].sum())/(count-cutoff) for cutoff in cutoff_inds]  # 每个节点之前的所有误差的平均值
    pix_error_e = [err_e[cutoff:].mean() for cutoff in cutoff_inds_err]  # 每个节点之前的所有误差的平均值
    pix_error_random = [err_r[cutoff:].mean() for cutoff in cutoff_inds_random]  # 每个节点之前的所有误差的平均值
    # pix_error = [metrics.mean_squared_error(err[cutoff:], np.zeros_like(err[cutoff:])) for cutoff in cutoff_inds]  # 每个节点之前的所有误差的平均值

    plt.figure()
    plt.xlabel('drop_percentile')
    plt.ylabel('{}pix_err'.format(threshold))
    plt.plot(percentiles, pix_error, color='green', label='uncertainty')
    plt.plot(percentiles, pix_error_e, color='aqua', label='oracle')
    plt.plot(percentiles, pix_error_random, color='red', label='random')
    plt.legend()
    plt.savefig(os.path.join(path, name + '.png'))


def visual_uncertainty(config, img, name, mode=cv2.COLORMAP_BONE, norm=False, percent=1, threshold=-1):
    path = os.path.join(config.output_dir, 'visualization')
    if not os.path.exists(path):
        os.mkdir(path)

    img = img.data.cpu().numpy()

    # 大于阈值的地方，值设为255，小于阈值的地方，值设为0
    if threshold != -1:
        mask = img > threshold
        img[mask] = 255
        img[~mask] = 0

    if percent != 1:
        amount = img.size
        amount = int(amount * (1 - percent))

        index_zero = np.array(np.argsort(img.ravel()))[:amount]
        index_one = np.array(np.argsort(img.ravel()))[amount:]

        mask_zero = np.unravel_index(index_zero, img.shape)
        mask_one = np.unravel_index(index_one, img.shape)
        img[mask_zero] = 0
        img[mask_one] = 255


    img = img.astype(np.uint8)
    img = img.transpose((1, 2, 0))

    img = cv2.applyColorMap(img, mode)
    cv2.imwrite(os.path.join(path, '{}.jpg'.format(name)), img)


def visual_mask(disp, config, name, mask=None, max_disp=-1):
    if mask != None:
        disp[~mask] = 0
    disp = disp.squeeze()  # 去掉维度为1的维度
    disp = disp.data.cpu().numpy()  # 把数据拷贝到CPU上来
    disp = disp_to_color(disp, max_disp)  # 将视差图转换成RGB图像，RGB值和视差大小相关
    disp = Image.fromarray(disp, 'RGB')
    # path = os.path.join(config.output_dir,'visualization', 'mask')
    path = os.path.join(config.output_dir, 'visualization', 'mask')
    if not os.path.exists(path):
        os.mkdir(path)
    disp.save(os.path.join(path, name+'.png'))

def visual_uncertainty_(disp, config, name, mask=None, max_disp=-1):
    if mask != None:
        disp[~mask] = 0
    disp = disp.squeeze()  # 去掉维度为1的维度
    disp = disp.data.cpu().numpy()  # 把数据拷贝到CPU上来
    disp = disp_to_color(disp, max_disp)  # 将视差图转换成RGB图像，RGB值和视差大小相关
    disp = Image.fromarray(disp, 'RGB')
    # path = os.path.join(config.output_dir,'visualization', 'mask')
    path = os.path.join(config.output_dir, 'visualization')
    if not os.path.exists(path):
        os.mkdir(path)
    disp.save(os.path.join(path, name+'.png'))


def err_to_color(errmap):

    cols = np.array([
        [0/3.0, 0.1875/3.0, 49, 54, 149],
        [0.1875/3.0, 0.375/3.0, 69, 117, 180],
        [0.375/3.0, 0.75/3.0, 116, 173, 209],
        [0.75/3.0, 1.5/3.0, 171, 217, 233],
        [1.5/3.0, 3/3.0, 224, 243, 248],
        [3/3.0, 6/3.0, 254, 224, 144],
        [6/3.0, 12/3.0, 253, 174, 97],
        [12/3.0, 24/3.0, 244, 109, 67],
        [24/3.0, 48/3.0, 215, 48, 39],
        [48/3.0, 255, 165, 0, 38]
    ])

    # print(errmap.shape)

    img = np.zeros((3, errmap.shape[0], errmap.shape[1]))
    # print(img.shape)


    for i in range(cols.shape[0]):

        v,u = np.where((errmap>cols[i][0])&(errmap<=cols[i][1]))
        # print(v)
        # print(u)
        img[0, v, u] = cols[i][2]
        img[1, v, u] = cols[i][3]
        img[2, v, u] = cols[i][4]

    img = img.transpose((1, 2, 0)).astype('uint8')
    return img


def disp_error_image(config, errmap, name, mask=None):
    path = os.path.join(config.output_dir, 'visualization')
    if not os.path.exists(path):
        os.mkdir(path)

    if mask != None:
        errmap[~mask] = 0
    errmap = errmap.squeeze()  # 去掉维度为1的维度
    errmap = errmap.data.cpu().numpy()  # 把数据拷贝到CPU上来
    errmap = err_to_color(errmap)  # 将视差图转换成RGB图像，RGB值和视差大小相关
    errmap = Image.fromarray(errmap, 'RGB')

    errmap.save(os.path.join(path, name + '.png'))