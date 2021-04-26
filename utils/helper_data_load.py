import os

# 图像文件后缀列表
IMG_EXTENSIONS = ['.jpg', '.JPG', '.jpeg', '.JPEG', '.png', '.PNG', '.ppm', '.PPM', '.bmp', '.BMP']


def is_image_file(filename):
    return any(filename.endswith(extension) for extension in IMG_EXTENSIONS)


def get_file_list_sceneflow(data_root, split, subset, maskroot=None):
    left_img_list = []
    right_img_list = []
    left_disp_list = []
    right_disp_list = []
    mask_list = []

    if split == 'train':
        if 'monkaa' in subset:
            # 读取 monkaa 数据集列表
            monkaa_img_path = os.path.join(data_root, 'monkaa', 'frames_cleanpass')
            monkaa_disp_path = os.path.join(data_root, 'monkaa', 'disparity')
            monkaa_dir = os.listdir(monkaa_img_path)

            for dd in monkaa_dir:
                for im in os.listdir(os.path.join(monkaa_img_path, dd, 'left')):
                    left_img_list.append(os.path.join(monkaa_img_path, dd, 'left', im))
                    right_img_list.append(os.path.join(monkaa_img_path, dd, 'right', im))
                    left_disp_list.append(os.path.join(monkaa_disp_path, dd, 'left', im.split(".")[0] + '.pfm'))
                    if maskroot != None:
                        mask_list.append(os.path.join(maskroot, str(dd) + im.split(".")[0] + '.npy'))
        if 'flyingthings3d' in subset:
            # 读取 flyingthings3d 数据集列表
            flying_img_path = os.path.join(data_root, 'flyingthings3d', 'frames_cleanpass', 'TRAIN')
            flying_disp_path = os.path.join(data_root, 'flyingthings3d', 'disparity', 'TRAIN')
            subdir = ['A', 'B', 'C']

            for ss in subdir:
                flying = os.listdir(os.path.join(flying_img_path, ss))
                for ff in flying:
                    imm_l = os.listdir(os.path.join(flying_img_path, ss, ff, 'left'))
                    for im in imm_l:
                        left_img_list.append(os.path.join(flying_img_path, ss, ff, 'left', im))
                        right_img_list.append(os.path.join(flying_img_path, ss, ff, 'right', im))
                        left_disp_list.append(
                            os.path.join(flying_disp_path, ss, ff, 'left', im.split(".")[0] + '.pfm'))
                        right_disp_list.append(
                            os.path.join(flying_disp_path, ss, ff, 'right', im.split(".")[0] + '.pfm'))
                        if maskroot != None:
                            mask_list.append(
                                os.path.join(maskroot, str(ss) + '_' + str(ff) + '_' + im.split(".")[0] + '.npy'))
        if 'driving' in subset:
            # 读取driving数据集列表
            driving_img_path = os.path.join(data_root, 'driving', 'frames_cleanpass', 'TRAIN')
            driving_disp_path = os.path.join(data_root, 'driving', 'disparity', 'TRAIN')

            subdir1 = ['35mm_focallength', '15mm_focallength']
            subdir2 = ['scene_backwards', 'scene_forwards']
            subdir3 = ['fast', 'slow']
            for i in subdir1:
                for j in subdir2:
                    for k in subdir3:
                        imm_l = os.listdir(os.path.join(driving_img_path, i, j, k, 'left'))
                        for im in imm_l:
                            left_img_list.append(os.path.join(driving_img_path, i, j, k, 'left', im))
                            right_img_list.append(os.path.join(driving_img_path, i, j, k, 'right', im))
                            left_disp_list.append(
                                os.path.join(driving_disp_path, i, j, k, 'left', im.split(".")[0] + '.pfm'))
                            if maskroot != None:
                                mask_list.append(
                                    os.path.join(maskroot, str(i) + str(j) + str(k) + im.split(".")[0] + '.npy'))
    else:
        if 'flyingthings3d' in subset:
            flying_img_path = os.path.join(data_root, 'flyingthings3d', 'frames_cleanpass', 'TEST')
            flying_disp_path = os.path.join(data_root, 'flyingthings3d', 'disparity', 'TEST')
            subdir = ['A', 'B', 'C']

            for ss in subdir:
                flying = os.listdir(os.path.join(flying_img_path, ss))
                for ff in flying:
                    imm_l = os.listdir(os.path.join(flying_img_path, ss, ff, 'left'))
                    for im in imm_l:
                        left_img_list.append(os.path.join(flying_img_path, ss, ff, 'left', im))
                        right_img_list.append(os.path.join(flying_img_path, ss, ff, 'right', im))
                        left_disp_list.append(os.path.join(flying_disp_path, ss, ff, 'left', im.split(".")[0] + '.pfm'))
                        right_disp_list.append(os.path.join(flying_disp_path, ss, ff, 'right', im.split(".")[0] + '.pfm'))
                        if maskroot != None:
                            mask_list.append(
                                os.path.join(maskroot, 'test', str(ss) + '_' + str(ff) + '_' + im.split(".")[0] + '.npy'))

    return left_img_list, right_img_list, left_disp_list, right_disp_list, mask_list


def get_file_list_kitti(data_root, split, subset, split_num=120):
    if 'kitti_2015' in subset:
        left_fold = 'image_2'
        right_fold = 'image_3'
        disp_l = 'disp_occ_0'
    elif 'kitti_2012' in subset:
        left_fold = 'colored_0'
        right_fold = 'colored_1'
        disp_l = 'disp_occ'

    left_img_list = []
    right_img_list = []
    left_disp_list = []

    if split == 'train':
        file_path = os.path.join(data_root, 'training')
        left_img_path = os.path.join(data_root, 'training', left_fold)
        file_list = [img for img in os.listdir(left_img_path) if img.find('_10') > -1]
        file_list = file_list[:split_num]
    elif split == 'train_full':
        file_path = os.path.join(data_root, 'training')
        left_img_path = os.path.join(data_root, 'training', left_fold)
        file_list = [img for img in os.listdir(left_img_path) if img.find('_10') > -1]
        file_list = file_list[:]
    elif split == 'valid':
        file_path = os.path.join(data_root, 'training')
        left_img_path = os.path.join(data_root, 'training', left_fold)
        file_list = [img for img in os.listdir(left_img_path) if img.find('_10') > -1]
        file_list = file_list[split_num:]
    elif split == 'test':
        file_path = os.path.join(data_root, 'testing')
        left_img_path = os.path.join(data_root, 'testing', left_fold)
        file_list = [img for img in os.listdir(left_img_path) if img.find('_10') > -1]

    for img in file_list:
        left_img_list.append(os.path.join(file_path, left_fold, img))
        right_img_list.append(os.path.join(file_path, right_fold, img))
        left_disp_list.append(os.path.join(file_path, disp_l, img))

    return left_img_list, right_img_list, left_disp_list


def write_file_list_sceneflow(data_root, split, save_path='', subset=None):
    file = os.path.join(save_path, split + '_sceneflow.txt')

    with open(file, 'w') as f:

        if split == 'train':
            if 'monkaa' in subset:
                # 读取 monkaa 数据集列表
                monkaa_img_path = os.path.join(data_root, 'monkaa', 'frames_cleanpass')
                monkaa_disp_path = os.path.join(data_root, 'monkaa', 'disparity')
                monkaa_dir = os.listdir(monkaa_img_path)

                for dd in monkaa_dir:
                    for im in os.listdir(os.path.join(monkaa_img_path, dd, 'left')):
                        f.write(os.path.join(monkaa_img_path, dd, 'left', im))
                        f.write(os.path.join(monkaa_img_path, dd, 'right', im))
                        f.write(os.path.join(monkaa_disp_path, dd, 'left', im.split(".")[0] + '.pfm' + '\n'))
            if 'flyingthings3d' in subset:

                # 读取 flyingthings3d 数据集列表
                flying_img_path = os.path.join(data_root, 'flyingthings3d', 'frames_cleanpass', 'TRAIN')
                flying_disp_path = os.path.join(data_root, 'flyingthings3d', 'disparity', 'TRAIN')
                subdir = ['A', 'B', 'C']

                for ss in subdir:
                    flying = os.listdir(os.path.join(flying_img_path, ss))
                    for ff in flying:
                        imm_l = os.listdir(os.path.join(flying_img_path, ss, ff, 'left'))
                        for im in imm_l:
                            f.write(os.path.join(flying_img_path, ss, ff, 'left', im) + ' ')
                            f.write(os.path.join(flying_img_path, ss, ff, 'right', im) + ' ')
                            f.write(os.path.join(flying_disp_path, ss, ff, 'left', im.split('.')[0] + '.pfm' + '\n'))
            if 'driving' in subset:
                # 读取driving数据集列表
                driving_img_path = os.path.join(data_root, 'driving', 'frames_cleanpass', 'TRAIN')
                driving_disp_path = os.path.join(data_root, 'driving', 'disparity', 'TRAIN')

                subdir1 = ['35mm_focallength', '15mm_focallength']
                subdir2 = ['scene_backwards', 'scene_forwards']
                subdir3 = ['fast', 'slow']
                for i in subdir1:
                    for j in subdir2:
                        for k in subdir3:
                            imm_l = os.listdir(os.path.join(driving_img_path, i, j, k, 'left'))
                            for im in imm_l:
                                f.write(os.path.join(driving_img_path, i, j, k, 'left', im))
                                f.write(os.path.join(driving_img_path, i, j, k, 'right', im))
                                f.write(
                                    os.path.join(driving_disp_path, i, j, k, 'left', im.split(".")[0] + '.pfm' + '\n'))
        else:
            if 'flyingthings3d' in subset:

                flying_img_path = os.path.join(data_root, 'flyingthings3d', 'frames_cleanpass', 'TEST')
                flying_disp_path = os.path.join(data_root, 'flyingthings3d', 'disparity', 'TEST')
                subdir = ['A', 'B', 'C']

                for ss in subdir:
                    flying = os.listdir(os.path.join(flying_img_path, ss))
                    for ff in flying:
                        imm_l = os.listdir(os.path.join(flying_img_path, ss, ff, 'left'))
                        for im in imm_l:
                            f.write(os.path.join(flying_img_path, ss, ff, 'left', im) + ' ')
                            f.write(os.path.join(flying_img_path, ss, ff, 'right', im) + ' ')
                            f.write(os.path.join(flying_disp_path, ss, ff, 'left', im.split('.')[0]) + '.pfm' + '\n')


def get_file_list_sceneflow_txt(txt_path):
    lines = read_text_lines(txt_path)

    left_img_list = []
    right_img_list = []
    left_disp_list = []

    for line in lines:
        splits = line.split()
        left_img, right_img, left_disp = splits
        left_img_list.append(left_img)
        right_img_list.append(right_img)
        left_disp_list.append(left_disp)

    return left_img_list, right_img_list, left_disp_list


def read_text_lines(path):
    with open(path, 'r') as f:
        lines = f.readlines()
    lines = [l.rstrip() for l in lines]
    return lines

def read_all_lines(filename):
    with open(filename) as f:
        lines = [line.rstrip() for line in f.readlines()]
    return lines

if __name__ == '__main__':
    # write_file_list_sceneflow('/home/g203-1/PSMNet-master/dataset/SceneFLow', 'train')
    # l1, l2, l3 = get_file_list_sceneflow_txt('/home/g203-1/StereoMatching/StereoMatching/utils/train_sceneflow.txt')
    print(1)
