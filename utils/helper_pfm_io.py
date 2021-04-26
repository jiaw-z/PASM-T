import re
import sys

import numpy as np
import chardet


def readPFM(file):
    file = open(file, 'rb')

    # 第一行: RGB 还是灰度图
    header = file.readline().rstrip()
    encode_type = chardet.detect(header)['encoding']  # 检测编码方式，有encoding, confidence,language三个key
    header = header.decode(encode_type)

    if header == 'PF':
        color = True  # => RGB(3 channels)
    elif header == 'Pf':
        color = False  # => Grayscale(1 channel)
    else:
        raise Exception('Not a PFM file.')

    # 第二行：图像的 width 和 height
    dim_match = re.match(r'^(\d+)\s(\d+)\s$', file.readline().decode(encode_type))

    if dim_match:
        width, height = map(int, dim_match.groups())
    else:
        raise Exception('Malformed PFM header.')

    # 第三行：大小端模式
    scale = float(file.readline().rstrip().decode(encode_type))

    if scale < 0:  # 小端
        endian = '<'
        scale = -scale
    else:
        endian = '>'  # 大端

    data = np.fromfile(file, endian + 'f')
    shape = (height, width, 3) if color else (height, width)

    data = np.reshape(data, shape)
    data = np.flipud(data)
    return data, scale

# TODO：有没有做测试？
def writePFM(file, image, scale=1):
    """
    file: 图像保存路径
    image: 需要保存的 disp 图
    """
    file = open(file, 'wb')

    color = None

    if image.dtype.name != 'float32':
        raise Exception('Image dtype must be float32.')

    image = np.flipud(image)

    if len(image.shape) == 3 and image.shape[2] == 3:  # color image
        color = True
    elif len(image.shape) == 2 or len(image.shape) == 3 and image.shape[2] == 1:  # greyscale
        color = False
    else:
        raise Exception('Image must have H x W x 3, H x W x 1 or H x W dimensions.')

    file.write(b'PF\n' if color else b'Pf\n')
    file.write(b'%d %d\n' % (image.shape[1], image.shape[0]))

    endian = image.dtype.byteorder

    if endian == '<' or endian == '=' and sys.byteorder == 'little':
        scale = -scale

    file.write(b'%f\n' % scale)

    image.tofile(file)
