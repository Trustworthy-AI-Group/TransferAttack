# Courtsy of https://github.com/xinntao/BasicSR

import random
import numpy as np
import cv2
import torch
import torch.utils.data as data
import os

class custom_dataset(data.Dataset):
    def __init__(self, dir_root):
        super(custom_dataset, self).__init__()

        self.paths = get_image_paths(dir_root)

    def __getitem__(self, index):

        # get  image
        path = self.paths[index]
        img = read_img(path)

        # BGR to RGB, HWC to CHW, numpy to tensor
        if img.shape[2] == 3:
            img = img[:, :, [2, 1, 0]]
        img = torch.from_numpy(np.ascontiguousarray(np.transpose(img, (2, 0, 1)))).float()

        if img.size(0) == 1:
            # stack greyscale image
            img = torch.cat((img, img, img), dim=0)
        if img.size(0) == 4:
            # remove alpha channel
            img = img[:3, :, :]

        return img, path.split('/')[-1]

    def __len__(self):
        return len(self.paths)


def _get_paths_from_images(path):
    '''get image path list from image folder'''
    assert os.path.isdir(path), '{:s} is not a valid directory'.format(path)
    images = []
    for dirpath, _, fnames in sorted(os.walk(path)):
        for fname in sorted(fnames):
            img_path = os.path.join(dirpath, fname)
            images.append(img_path)
    assert images, '{:s} has no valid image file'.format(path)
    return images


def get_image_paths(dataroot):
    paths = sorted(_get_paths_from_images(dataroot))
    return paths


def read_img(path, size=None):
    '''read image by cv2 or from lmdb
    return: Numpy float32, HWC, BGR, [0,1]'''
    img = cv2.imread(path, cv2.IMREAD_UNCHANGED)

    # Resizing for natural images
    # img = cv2.resize(img, (256, 256))
    # img = cv2.resize(img, (224, 224))

    img = img.astype(np.float32) / 255.
    if img.ndim == 2:
        img = np.expand_dims(img, axis=2)
    # some images have 4 channels
    if img.shape[2] > 3:
        img = img[:, :, :3]
    return img


def tensor2img(tensor, out_type=np.uint8, min_max=(0, 1)):
    '''
    Converts a torch Tensor into an image Numpy array
    Input: 4D(B,(3/1),H,W), 3D(C,H,W), or 2D(H,W), any range, RGB channel order
    Output: 3D(H,W,C) or 2D(H,W), [0,255], np.uint8 (default)
    '''
    tensor = tensor.squeeze().float().cpu().clamp_(*min_max)  # clamp
    tensor = (tensor - min_max[0]) / (min_max[1] - min_max[0])  # to range [0,1]
    n_dim = tensor.dim()
    if n_dim == 4:
        n_img = len(tensor)
        img_np = make_grid(tensor, nrow=int(math.sqrt(n_img)), normalize=False).numpy()
        img_np = np.transpose(img_np[[2, 1, 0], :, :], (1, 2, 0))  # HWC, BGR
    elif n_dim == 3:
        img_np = tensor.numpy()
        img_np = np.transpose(img_np[[2, 1, 0], :, :], (1, 2, 0))  # HWC, BGR
    elif n_dim == 2:
        img_np = tensor.numpy()
    else:
        raise TypeError(
            'Only support 4D, 3D and 2D tensor. But received with dimension: {:d}'.format(n_dim))
    if out_type == np.uint8:
        img_np = (img_np * 255.0).round()
        # Important. Unlike matlab, numpy.unit8() WILL NOT round by default.
    return img_np.astype(out_type)


def save_img(img, img_path, mode='RGB'):
    cv2.imwrite(img_path, img)
