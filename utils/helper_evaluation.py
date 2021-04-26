import torch

# 计算1像素，3像素，5像素误差
def err_pix(input, gt, mask, threshold=1):
    count = torch.sum(mask)
    err_abs = torch.abs(input[mask] - gt[mask])
    err = torch.sum(torch.gt(err_abs, threshold)).float() / count

    return err, count
