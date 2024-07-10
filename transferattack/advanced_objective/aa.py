import torch
import json
import random

from ..utils import *
from ..attack import Attack
import torch.nn as nn

class AA(Attack):
    """
    Activation Attack
    'Feature Space Perturbations Yield More Transferable Adversarial Examples (CVPR 2019) (https://openaccess.thecvf.com/content_CVPR_2019/papers/Inkawhich_Feature_Space_Perturbations_Yield_More_Transferable_Adversarial_Examples_CVPR_2019_paper.pdf)'

    Arguments:
        model_name (str): the name of surrogate model for attack.
        epsilon (float): the perturbation budget.
        alpha (float): the step size.
        epoch (int): the number of iterations.
        decay (float): the decay factor for momentum calculation.
        targeted (bool): targeted/untargeted attack.
        random_start (bool): whether using random initialization for delta.
        norm (str): the norm of perturbation, l2/linfty.
        loss (str): the loss function.
        device (torch.device): the device for data. If it is None, the device would be same as model
        feature_layer: feature layer to launch the attack.

    Official arguments:
        epsilon=0.07, alpha=epsilon/epoch=0.007, epoch=10, decay=1.
    """

    def __init__(self, model_name, epsilon=16/255, alpha=2/255, epoch=300, decay=1., targeted=True, random_start=False, 
                norm='linfty', loss='crossentropy', layer_name='layer2', device=None, attack='AA', **kwargs):
        super().__init__(attack, model_name, epsilon, targeted, random_start, norm, loss, device)
        self.alpha = alpha
        self.epoch = epoch
        self.decay = decay
        self.feature_layer = self.find_layer(layer_name)
        self.l2f = self.get_l2f(file_name='labels.csv')

    def get_l2f(self, file_name):
        dev = pd.read_csv(os.path.join('./data', file_name))
        l2f = {dev.iloc[i]['label']: dev.iloc[i]['filename'] for i in range(len(dev))}
        return l2f

    def get_tar_data(self, tar_label):
        base_path = './data/images'
        tar_data = []

        for label in tar_label:
            filename = self.l2f[label.item()]
            filepath = os.path.join(base_path, filename)
            image = Image.open(filepath)
            image = image.resize((img_height, img_width)).convert('RGB')
            # Images for inception classifier are normalized to be in [-1, 1] interval.
            image = np.array(image).astype(np.float32)/255
            image = torch.from_numpy(image).permute(2, 0, 1)
            tar_data.append(image)

        return torch.stack(tar_data)

    def find_layer(self,layer_name):
        parser = layer_name.split(' ')
        m = self.model[1]
        for layer in parser:
            if layer not in m._modules.keys():
                print("Selected layer is not in Model")
                exit() 
            else:
                m = m._modules.get(layer)
        return m

    def get_loss(self, mid_t_fmap, mid_s_fmap):
        # Calculate l2 norm of the difference between the feature maps of the target and source images in the batch, and return the mean value of it
        loss = (mid_t_fmap - mid_s_fmap).reshape(mid_t_fmap.shape[0], -1).norm(p=2, dim=1).mean()
        return -loss if self.targeted else loss
            
    def forward(self, data, label, **kwargs):
        """
        The Activation attack procedure

        Arguments:
            data: (N, C, H, W) tensor for input images
            labels: (N,) tensor for ground-truth labels if untargetd, otherwise targeted labels
        """
        if self.targeted:
            assert len(label) == 2
            ori_label = label[0]
            tar_label = label[1] # the second element is the targeted label tensor
        data = data.clone().detach().to(self.device)
        tar_data = self.get_tar_data(tar_label).clone().detach().to(self.device)
        ori_label = ori_label.clone().detach().to(self.device)
        tar_label = tar_label.clone().detach().to(self.device)

        # Initialize adversarial perturbation
        delta = self.init_delta(data)

        def get_mid_output(model, input, output):
            global mid_output
            mid_output = output

        h = self.feature_layer.register_forward_hook(get_mid_output)

        # store target image feature map at Layer L
        with torch.no_grad():
            logits = self.get_logits(self.transform(tar_data))
            # Copy the mid_output to mid_t_fmap
            mid_t_fmap = torch.zeros_like(mid_output).to(self.device)
            mid_t_fmap.copy_(mid_output)

        momentum = 0
        for _ in range(self.epoch):
            # 获取source的feature mpa并保存
            logits = self.get_logits(self.transform(data + delta))

            # Calculate the loss
            loss = self.get_loss(mid_t_fmap, mid_output)

            # Calculate the gradients
            grad = self.get_grad(loss, delta)

            # Calculate the momentum
            momentum = self.get_momentum(grad, momentum)

            # Update adversarial perturbation
            delta = self.update_delta(delta, data, momentum, self.alpha)

        h.remove()
        return delta.detach()