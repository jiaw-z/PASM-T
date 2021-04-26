import random

import numpy as np
import torch
import torch.nn.functional as F

imagenet_stats = {'mean': [0.485, 0.456, 0.406],
                  'std': [0.229, 0.224, 0.225]}

imagenet_pca = {
    'eigval': torch.Tensor([0.2175, 0.0188, 0.0045]),
    'eigvec': torch.Tensor([
        [-0.5675, 0.7192, 0.4009],
        [-0.5808, -0.0045, -0.8140],
        [-0.5836, -0.6948, 0.4203],
    ])
}


class Compose(object):
    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, left_img, right_img, left_disp):
        for t in self.transforms:
            left_img, right_img, left_disp = t(left_img, right_img, left_disp)
        return left_img, right_img, left_disp


class RandomCrop(object):

    def __init__(self, height, width):
        self.height = height  # 裁剪后图像的高
        self.width = width  # 裁剪后图像的宽

    def __call__(self, left_img, right_img, left_disp=None):
        in_h, in_w, in_c = left_img.shape  # H,W,C

        h = np.random.randint(0, in_h - self.height)
        w = np.random.randint(0, in_w - self.width)

        left_img_crop = left_img[h:h + self.height, w:w + self.width, :]
        right_img_crop = right_img[h:h + self.height, w:w + self.width, :]
        if left_disp is not None:
            left_disp_crop = left_disp[h:h + self.height, w:w + self.width]
        else:
            left_img_crop = None

        return left_img_crop, right_img_crop, left_disp_crop


# 固定裁剪右下角的一个sub-image
class FixCrop(object):

    def __init__(self, height=368, width=1232):
        self.height = height  # 裁剪后图像的高
        self.width = width  # 裁剪后图像的宽

    def __call__(self, left_img, right_img, left_disp=None):
        in_h, in_w, in_c = left_img.shape  # H,W,C

        left_img_crop = left_img[in_h - self.height:, in_w - self.width:, :]
        right_img_crop = right_img[in_h - self.height:, in_w - self.width:, :]
        if left_disp is not None:
            left_disp_crop = left_disp[in_h - self.height:, in_w - self.width:]
        else:
            left_img_crop = None

        return left_img_crop, right_img_crop, left_disp_crop


class ArrayToTensor(object):

    def __call__(self, left_img, right_img, left_disp=None):
        left_img = np.transpose(left_img, (2, 0, 1))  # H,W,C => C,H,W
        right_img = np.transpose(right_img, (2, 0, 1))  # H,W,C => C,H,W

        left_img = torch.from_numpy(left_img).float() / 255  # (0,255) => (0,1)
        right_img = torch.from_numpy(right_img).float() / 255
        if left_disp is not None:
            left_disp = torch.from_numpy(left_disp).float()
        else:
            left_disp = None

        #mean = [0.485, 0.456, 0.406]
        #std = [0.229, 0.224, 0.225]

        #left_img = F.normalize(left_img, mean, std)
        #right_img = F.normalize(right_img, mean, std)

        return left_img, right_img, left_disp


# TODO: 下面的还需要重新改一下，改成输入左右目+视差图的
class Grayscale(object):
    def __call__(self, img):
        gs = img.clone()
        gs[0].mul_(0.299).add_(0.587, gs[1]).add_(0.114, gs[2])
        gs[1].copy_(gs[0])
        gs[2].copy_(gs[0])
        return gs


class Brightness(object):
    def __init__(self, var):
        self.var = var

    def __call__(self, img):
        gs = img.new().resize_as_(img).zero_()
        alpha = random.uniform(0, self.var)
        return img.lerp(gs, alpha)


class Contrast(object):
    def __init__(self, var):
        self.var = var

    def __call__(self, img):
        gs = Grayscale()(img)
        gs.fill_(gs.mean())
        alpha = random.uniform(0, self.var)
        return img.lerp(gs, alpha)


class Saturation(object):

    def __init__(self, var):
        self.var = var

    def __call__(self, img):
        gs = Grayscale()(img)
        alpha = random.uniform(0, self.var)
        return img.lerp(gs, alpha)


class ColorJitter(object):
    def __init__(self, brightness=0.4, contrast=0.4, saturation=0.4):
        self.transforms = []
        if brightness != 0:
            self.transforms.append(Brightness(brightness))
        if contrast != 0:
            self.transforms.append(Contrast(contrast))
        if saturation != 0:
            self.transforms.append(Saturation(saturation))

    def __call__(self, img):
        if self.transforms is None:
            return img
        order = torch.randperm(len(self.transforms))
        for i in order:
            img = self.transforms[i](img)
        return img


class Lighting(object):
    """Lighting noise(AlexNet - style PCA - based noise)"""

    def __init__(self, alphastd, eigval, eigvec):
        self.alphastd = alphastd
        self.eigval = eigval
        self.eigvec = eigvec

    def __call__(self, img):
        if self.alphastd == 0:
            return img

        alpha = img.new().resize_(3).normal_(0, self.alphastd)
        rgb = self.eigvec.type_as(img).clone() \
            .mul(alpha.view(1, 3).expand(3, 3)) \
            .mul(self.eigval.view(1, 3).expand(3, 3)) \
            .sum(1).squeeze()

        return img.add(rgb.view(3, 1, 1).expand_as(img))
