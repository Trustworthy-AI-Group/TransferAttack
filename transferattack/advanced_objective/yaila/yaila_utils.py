import torch
import torch.nn.functional as F
import torch.nn as nn
import torchvision
import numpy as np
from torch.utils.data import Dataset
import csv
import PIL.Image as Image
import os
import torchvision.transforms as T
import pickle


# Selected imagenet. The .csv file format:
# class_index, class, image_name
# 0,n01440764,ILSVRC2012_val_00002138.JPEG
# 2,n01484850,ILSVRC2012_val_00004329.JPEG
# ...
class SelectedImagenet(Dataset):
    def __init__(self, imagenet_val_dir, selected_images_csv, transform=None):
        super(SelectedImagenet, self).__init__()
        self.imagenet_val_dir = imagenet_val_dir
        self.selected_images_csv = selected_images_csv
        self.transform = transform
        self._load_csv()
    def _load_csv(self):
        reader = csv.reader(open(self.selected_images_csv, 'r'))
        next(reader)
        self.selected_list = list(reader)
    def __getitem__(self, item):
        target, target_name, image_name = self.selected_list[item]
        image = Image.open(os.path.join(self.imagenet_val_dir, target_name, image_name))
        if image.mode != 'RGB':
            image = image.convert('RGB')
        if self.transform is not None:
            image = self.transform(image)
        return image, int(target)
    def __len__(self):
        return len(self.selected_list)

# Selected cifar-100. The .csv file format:
# class_index,data_index
# 49,0
# 33,1
# ...
class SelectedCifar100(Dataset):
    def __init__(self, cifar100_dir, selected_images_csv, transform=None):
        super(SelectedCifar100, self).__init__()
        self.cifar100_dir = cifar100_dir
        self.data = []
        self.targets = []
        file_path = os.path.join(cifar100_dir, 'test')
        with open(file_path, 'rb') as f:
            entry = pickle.load(f, encoding='latin1')
            self.data.append(entry['data'])
            self.targets.extend(entry['fine_labels'])
        self.data = np.vstack(self.data).reshape(-1, 3, 32, 32)
        self.data = self.data.transpose((0, 2, 3, 1))  # convert to HWC
        self.transform = transform
        self.selected_images_csv = selected_images_csv
        self._load_csv()
    def _load_csv(self):
        reader = csv.reader(open(self.selected_images_csv, 'r'))
        next(reader)
        self.selected_list = list(reader)
    def __getitem__(self, item):
        t_class, t_ind = map(int, self.selected_list[item])
        assert self.targets[t_ind] == t_class, 'Wrong targets in csv file.(line {})'.format(item+1)
        img, target = self.data[int(self.selected_list[item][1])], self.targets[t_ind]
        # doing this so that it is consistent with all other datasets
        # to return a PIL Image
        img = Image.fromarray(img)
        if self.transform is not None:
            img = self.transform(img)
        return img, target
    def __len__(self):
        return len(self.selected_list)

class Normalize(nn.Module):
    def __init__(self,dataset_name):
        super(Normalize, self).__init__()
        assert dataset_name in ['imagenet', 'cifar100'], 'check dataset_name'
        if dataset_name == 'imagenet':
            self.normalize = [(0.485, 0.456, 0.406), (0.229, 0.224, 0.225)]
        elif dataset_name == 'cifar100':
            self.normalize = [(0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)]
    def forward(self, input):
        x = input.clone()
        for i in range(x.shape[1]):
            x[:,i] = (x[:,i] - self.normalize[0][i]) / self.normalize[1][i]
        return x


def resnet50_forward(ila, model, x, mid_layer_index, tap):
    assert ila != tap or (ila == False and tap == False)
    def block_forward(x, block, block_index, mid_layer_index = mid_layer_index, ila = ila):
        y = None
        for ind, unit in enumerate(block):
            x = unit(x)
            if '{}_{}'.format(block_index, ind) == mid_layer_index:
                if ila:
                    return x, None
                else:
                    y = x.clone()
        return y, x
    if tap:
        mid_output = [[],] # [[outputs of blocks] , intermediate layer output]
    x = model[0](x)
    x = model[1].conv1(x)
    x = model[1].bn1(x)
    x = model[1].relu(x)
    x = model[1].maxpool(x)
    for block_ind in range(4):
        mid_feats, x = block_forward(x, eval('model[1].layer{}'.format(block_ind + 1)), block_ind + 1)
        if tap:
            mid_output[0].append(x.clone())
        if int(mid_layer_index.split('_')[0]) == block_ind + 1:
            if ila:
                return mid_feats, None
            elif tap:
                mid_output.append(mid_feats)
            else:
                mid_output = mid_feats
    x = model[1].avgpool(x)
    x = torch.flatten(x, 1)
    x = model[1].fc(x)
    return mid_output, x


def matrix_mul(x):
    n_dims = x.shape[2]
    n_parts = int(np.ceil(n_dims ** 0.5))
    x_parts = np.array_split(x, n_parts, axis=2)
    y = 0
    for x_ in x_parts:
        y += np.matmul(x_,x_.transpose((0, 2, 1)))
    return y


def calculate_w(H, r, lam, normalize_H):
    if normalize_H:
        H = H / np.linalg.norm(H, ord=None, axis=2, keepdims=True)#H.norm(p=2, dim=2, keepdim = True)
    if lam == 'inf':
        return (H * r).mean(axis=1)
    n_imgs, n_iters, n_dims = H.shape
    Ht_r = np.einsum('ijk,ikl -> ijl', H.transpose(0, 2, 1), r)
    del r
    H_Ht = matrix_mul(H)
    H_Ht_lamI = H_Ht + lam * np.repeat(np.eye(n_iters)[None, ...], n_imgs, axis=0)
    del H_Ht
    inversed_H_Ht_lamI = np.linalg.inv(H_Ht_lamI)
    H_Ht_r = np.einsum('ijk,ikl -> ijl', H, Ht_r)
    inversed_H_Ht_lamI_H_Ht_r = np.einsum('ijk,ikl -> ijl', inversed_H_Ht_lamI, H_Ht_r)
    del inversed_H_Ht_lamI
    Ht_inversed_H_Ht_lamI_H_Ht_r = np.einsum('ijk,ikl -> ijl', H.transpose((0, 2, 1)), inversed_H_Ht_lamI_H_Ht_r)
    del inversed_H_Ht_lamI_H_Ht_r, H
    new_mid_feat = Ht_r - Ht_inversed_H_Ht_lamI_H_Ht_r
    del Ht_inversed_H_Ht_lamI_H_Ht_r, Ht_r
    return np.squeeze(new_mid_feat, axis=-1)


def attack(ILA:bool, w, ori_img, label, device, niters, baseline_method, epsilon, model, mid_layer_index, batch_size, lr):
    ori_img, label = ori_img.to(device), label.to(device)
    img = ori_img.clone()
    loss_ls = []
    for i in range(niters + 1):
        if baseline_method == 'pgd' and not ILA:
            # In our implementation of PGD, we incorporate randomness at each iteration to further enhance the transferability
            img_x = img + img.new(img.size()).uniform_(-epsilon, epsilon).to(device)
        else:
            img_x = img
        img_x.requires_grad_(True)
        att_h_feats, att_out = resnet50_forward(ila=True if ILA else False,
                                                model=model, x=img_x, mid_layer_index=mid_layer_index,
                                                tap=True if baseline_method == 'tap' and not ILA else False)

        if ILA:
            if i == 0:
                ilaloss = Proj_Loss(ori_h_feats=att_h_feats.data, guide_feats=w.to(device))
            loss = ilaloss(att_h_feats)
            print('ILA iter {}, loss {:0.4f}'.format(i, loss.item()))
        else:
            if baseline_method =='tap':
                if i == 0:
                    tap_ori_h_feats = [t.data for t in att_h_feats[0]]
                    ori_h_feats = att_h_feats[1].data.cpu().view(img_x.size(0), -1).clone()
                    h_feats = torch.zeros(batch_size, niters, int(ori_h_feats.numel() / ori_h_feats.shape[0]))
                loss, ce = Transferable_Adversarial_Perturbations_Loss()(
                    ori_img,
                    img,
                    tap_ori_h_feats,
                    att_h_feats[0],
                    label,
                    att_out,
                    lam=0.005,
                    alpha=0.5,
                    s=3,
                    yita=0.01
                )
            else:
                if i == 0:
                    ori_h_feats = att_h_feats.data.cpu().view(img.size(0), -1).clone()
                    h_feats = torch.zeros(batch_size, niters, int(ori_h_feats.numel() / ori_h_feats.shape[0]))
                ce = nn.CrossEntropyLoss(reduction='none')(att_out, label)
                loss = ce.mean()
            pred = torch.argmax(att_out, dim=1)
            print('iter {}, loss {:0.4f}, success {}'.format(i, loss.item(), (pred.squeeze(0) != label).sum()))

        model.zero_grad()
        loss.backward()

        if not ILA:
            if baseline_method =='tap':
                att_h_feats = att_h_feats[1].data.cpu().view(img.size(0), -1)
            else:
                att_h_feats = att_h_feats.data.cpu().view(img.size(0), -1)
            ce = ce.data.cpu()
            if i != 0:
                h_feats[:, i - 1, :] = att_h_feats - ori_h_feats
                loss_ls.append(ce.unsqueeze(0))
            del att_h_feats, att_out, ce

        input_grad = img_x.grad.data

        if 'mifgsm' in baseline_method and not ILA:
            if i == 0:
                last_grad = 0
                momentum = float(baseline_method.split('_')[-1])
            input_grad = img.grad.data
            input_grad = momentum * last_grad + input_grad / torch.norm(input_grad, dim=(1, 2, 3), p=1, keepdim=True)
            last_grad = input_grad.clone()

        img = img.data + lr * torch.sign(input_grad)
        img = torch.where(img > ori_img + epsilon, ori_img + epsilon, img)
        img = torch.where(img < ori_img - epsilon, ori_img - epsilon, img)
        img = torch.clamp(img, min=0, max=1)
    if ILA:
        return img.data
    else:
        h_feats = h_feats.numpy()
        loss_ls = torch.cat(loss_ls, dim=0).permute(1, 0).view(batch_size, niters, 1).numpy()
        return h_feats, loss_ls

# ILA loss
class Proj_Loss(torch.nn.Module):
    def __init__(self,ori_h_feats, guide_feats):
        super(Proj_Loss, self).__init__()
        n_imgs = ori_h_feats.size(0)
        self.n_imgs = n_imgs
        self.ori_h_feats = ori_h_feats.view(n_imgs, -1)
        guide_feats = guide_feats.view(n_imgs, -1)
        self.guide_feats = guide_feats / guide_feats.norm(p=2, dim=1, keepdim = True)
    def forward(self, att_h_feats):
        att_h_feats = att_h_feats.view(self.n_imgs, -1)
        loss = ((att_h_feats - self.ori_h_feats) * self.guide_feats).sum() / self.n_imgs
        return loss

# TAP (transferable adversairal perturbation ECCV 2018)
# copied from https://github.com/CUVL/Intermediate-Level-Attack
class Transferable_Adversarial_Perturbations_Loss(torch.nn.Module):
    def __init__(self):
        super(Transferable_Adversarial_Perturbations_Loss, self).__init__()
    def forward(
            self,
            X,
            X_pert,
            original_mids,
            new_mids,
            y,
            output_perturbed,
            lam,
            alpha,
            s,
            yita,
    ):
        l1 = nn.CrossEntropyLoss(reduction='none')(output_perturbed, y)
        l2 = 0
        for i, new_mid in enumerate(new_mids):
            a = torch.sign(original_mids[i]) * torch.pow(
                torch.abs(original_mids[i]), alpha
            )
            b = torch.sign(new_mid) * torch.pow(torch.abs(new_mid), alpha)
            l2 += (lam * (torch.norm(a - b, p = 2, dim = (1,2,3)) ** 2).sum()) / X.shape[0]
        l3 = yita * torch.abs(nn.AvgPool2d(s)(X - X_pert)).sum() / X.shape[0]
        return l1.mean() + l2 + l3, l1