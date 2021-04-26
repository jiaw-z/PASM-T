import numpy as np
import torch
import torch.nn as nn
import copy


# 将 uncertainty 和 像素误差拼成 N * 2 的列表
def img_to_pix(pred, gt, unc, mask):
    # pred = torch.mul(pred, mask.float())  # KITTI 只计算有ground truth的位置
    # gt = torch.mul(gt, mask.float())
    # unc = torch.mul(unc, mask.float())

    pred = pred[mask]  # KITTI 只计算有ground truth的位置
    gt = gt[mask]
    unc = unc[mask]

    res = torch.abs(pred - gt)
    res = np.array(res.flatten().cpu())
    unc = np.array(unc.flatten().cpu())
    err_unc = list(map(list, zip(res, unc)))

    return err_unc


class NestedTensor(object):
    def __init__(self, left, right, disp=None, disp_right=None, sampled_cols=None, sampled_rows=None, occ_mask=None,
                 occ_mask_right=None):
        self.left = left
        self.right = right
        self.disp = disp
        self.disp_right = disp_right
        self.occ_mask = occ_mask
        self.occ_mask_right = occ_mask_right
        self.sampled_cols = sampled_cols
        self.sampled_rows = sampled_rows


def center_crop(layer, max_height, max_width):
    _, _, h, w = layer.size()
    xy1 = (w - max_width) // 2
    xy2 = (h - max_height) // 2
    return layer[:, :, xy2:(xy2 + max_height), xy1:(xy1 + max_width)]


def batched_index_select(source, dim, index):
    views = [source.shape[0]] + [1 if i != dim else -1 for i in range(1, len(source.shape))]
    expanse = list(source.shape)
    expanse[0] = -1
    expanse[dim] = -1
    index = index.view(views).expand(expanse)
    return torch.gather(source, dim, index)


def torch_1d_sample(source, sample_points, mode='linear'):
    """
    linearly sample source tensor along the last dimension
    input:
        source [N,D1,D2,D3...,Dn]
        sample_points [N,D1,D2,....,Dn-1,1]
    output:
        [N,D1,D2...,Dn-1]
    """
    idx_l = torch.floor(sample_points).long().clamp(0, source.size(-1) - 1)
    idx_r = torch.ceil(sample_points).long().clamp(0, source.size(-1) - 1)

    if mode == 'linear':
        weight_r = sample_points - idx_l
        weight_l = 1 - weight_r
    elif mode == 'sum':
        weight_r = (idx_r != idx_l).int()  # we only sum places of non-integer locations
        weight_l = 1
    else:
        raise Exception('mode not recognized')

    out = torch.gather(source, -1, idx_l) * weight_l + torch.gather(source, -1, idx_r) * weight_r
    return out.squeeze(-1)


def get_clones(module, N):
    return nn.ModuleList([copy.deepcopy(module) for i in range(N)])
