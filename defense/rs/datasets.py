from torchvision import transforms, datasets
from typing import *
import torch
import os
from torch.utils.data import Dataset

# set this environment variable to the location of your imagenet directory if you want to read ImageNet data.
# make sure your val directory is preprocessed to look like the train directory, e.g. by running this script
# https://raw.githubusercontent.com/soumith/imagenetloader.torch/master/valprep.sh
IMAGENET_LOC_ENV = "IMAGENET_DIR"

# list of all datasets
DATASETS = ["imagenet", "cifar10"]


def get_dataset(dataset: str, split: str) -> Dataset:
    """Return the dataset as a PyTorch Dataset object"""
    if dataset == "imagenet":
        return _imagenet(split)
    elif dataset == "cifar10":
        return _cifar10(split)


def get_num_classes(dataset: str):
    """Return the number of classes in the dataset. """
    if dataset == "imagenet":
        return 1000
    elif dataset == "cifar10":
        return 10


def get_normalize_layer(dataset: str) -> torch.nn.Module:
    """Return the dataset's normalization layer"""
    if dataset == "imagenet":
        return NormalizeLayer(_IMAGENET_MEAN, _IMAGENET_STDDEV)
    elif dataset == "cifar10":
        return NormalizeLayer(_CIFAR10_MEAN, _CIFAR10_STDDEV)


_IMAGENET_MEAN = [0.485, 0.456, 0.406]
_IMAGENET_STDDEV = [0.229, 0.224, 0.225]

_CIFAR10_MEAN = [0.4914, 0.4822, 0.4465]
_CIFAR10_STDDEV = [0.2023, 0.1994, 0.2010]


def _cifar10(split: str) -> Dataset:
    if split == "train":
        return datasets.CIFAR10("./dataset_cache", train=True, download=True, transform=transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor()
        ]))
    elif split == "test":
        return datasets.CIFAR10("./dataset_cache", train=False, download=True, transform=transforms.ToTensor())


def _imagenet(split: str) -> Dataset:
    if not IMAGENET_LOC_ENV in os.environ:
        raise RuntimeError("environment variable for ImageNet directory not set")

    dir = os.environ[IMAGENET_LOC_ENV]
    if split == "train":
        subdir = os.path.join(dir, "train")
        transform = transforms.Compose([
            transforms.RandomSizedCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor()
        ])
    elif split == "test":
        subdir = os.path.join(dir, "val")
        transform = transforms.Compose([
            transforms.Scale(256),
            transforms.CenterCrop(224),
            transforms.ToTensor()
        ])
    return datasets.ImageFolder(subdir, transform)

import torch
import torchvision.models as models
import torchvision.transforms as transforms
from PIL import Image
import numpy as np
import pandas as pd

import os

img_height, img_width = 224, 224
img_max, img_min = 1., 0

mean = np.array([0.485, 0.456, 0.406])
std = np.array([0.229, 0.224, 0.225])

def load_images(input_dir, batch_size):
    images = [] 
    filenames = []
    idx = 0
    for filepath in os.listdir(input_dir):
        image = Image.open(os.path.join(input_dir,filepath))
        image = image.resize((img_height, img_width)).convert('RGB')
        # Images for inception classifier are normalized to be in [-1, 1] interval.
        images.append(np.array(image).astype(np.float32)/255)
        filenames.append(os.path.basename(filepath))
        idx += 1
        if idx == batch_size:
            images = torch.from_numpy(np.array(images)).permute(0,3,1,2)
            yield filenames, images
            filenames = []
            images = []
            idx = 0
    if idx > 0:
        images = torch.from_numpy(np.array(images)).permute(0,3,1,2)
        yield filenames, images

def get_labels(names, f2l):
    labels = []
    for name in names:
        labels.append(f2l[name]-1)
    return torch.from_numpy(np.array(labels, dtype=np.int64))

def load_labels(file_name, targeted=False):
    dev = pd.read_csv(file_name)
    if targeted:
        dev = dev.iloc[:,[0,2]]
    else:
        dev = dev.iloc[:,[0,1]]
    dev.columns = ['filename', 'label']
    f2l = {dev.iloc[i]['filename']: dev.iloc[i]['label'] for i in range(len(dev))}
    return f2l

def save_images(output_dir, adversaries, filenames):
    adversaries = (adversaries.detach().permute((0,2,3,1)).cpu().numpy() * 255).astype(np.uint8)
    for i, filename in enumerate(filenames):
        Image.fromarray(adversaries[i]).save(os.path.join(output_dir, filename))

def clamp(x, x_min, x_max):
    return torch.min(torch.max(x, x_min), x_max)

def save_img(output_path,img):
    img = (img.detach().permute((0,2,3,1)).cpu().numpy() * 255).astype(np.uint8)
    Image.fromarray(img).save(output_path)


    




class NormalizeLayer(torch.nn.Module):
    """Standardize the channels of a batch of images by subtracting the dataset mean
      and dividing by the dataset standard deviation.

      In order to certify radii in original coordinates rather than standardized coordinates, we
      add the Gaussian noise _before_ standardizing, which is why we have standardization be the first
      layer of the classifier rather than as a part of preprocessing as is typical.
      """

    def __init__(self, means: List[float], sds: List[float]):
        """
        :param means: the channel means
        :param sds: the channel standard deviations
        """
        super(NormalizeLayer, self).__init__()
        self.means = torch.tensor(means).cuda()
        self.sds = torch.tensor(sds).cuda()

    def forward(self, input: torch.tensor):
        (batch_size, num_channels, height, width) = input.shape
        means = self.means.repeat((batch_size, height, width, 1)).permute(0, 3, 1, 2)
        sds = self.sds.repeat((batch_size, height, width, 1)).permute(0, 3, 1, 2)
        return (input - means) / sds
