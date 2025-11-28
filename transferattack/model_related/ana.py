import torch
import torch.nn.functional as F
import torchvision
from ..utils import *
from ..gradient.mifgsm import MIFGSM


class AlignmentNetCNN(nn.Module):
    def __init__(self, channels: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(channels, channels, kernel_size=3, padding=1, stride=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels, channels, kernel_size=3, padding=1, stride=1),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.net(x)


class ResNet50WithAlign(nn.Module):
    def __init__(self, pretrained: bool = True, device='cuda', alignment_path=None):
        super().__init__()
        base = torchvision.models.resnet50(pretrained=pretrained)
        self.stem = nn.Sequential(base.conv1, base.bn1, base.relu, base.maxpool,
                                  base.layer1, base.layer2, base.layer3)
        self.alignment = AlignmentNetCNN(channels=1024)
        if alignment_path is not None:
            self.alignment.load_state_dict(torch.load(alignment_path)['alignment_state_dict'])
        self.layer4 = base.layer4
        self.avgpool = base.avgpool
        self.fc = base.fc
        self.device = device

        for p in self.stem.parameters():
            p.requires_grad = False
        for p in self.layer4.parameters():
            p.requires_grad = False
        for p in self.fc.parameters():
            p.requires_grad = False

        mean = [0.485, 0.456, 0.406]
        std = [0.229, 0.224, 0.225]
        Resize = 224
        self.preprocess_model = PreprocessingModel(Resize, mean, std)

    def forward_features(self, x):
        return self.stem(x)

    def forward_from_layer3(self, feat_layer3):
        x = self.layer4(feat_layer3)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        logits = self.fc(x)
        return logits, x

    def forward(self, x_orig, x_masked):
        feat3_orig = self.forward_features(self.preprocess_model(x_orig))
        feat3_mask = self.forward_features(self.preprocess_model(x_masked))
        feat3_mask_aligned = self.alignment(feat3_mask)
        logits_orig, pooled_orig = self.forward_from_layer3(feat3_orig)
        logits_mask_aligned, pooled_mask_aligned = self.forward_from_layer3(feat3_mask_aligned)
        return logits_orig, pooled_orig, pooled_mask_aligned


class ANA(MIFGSM):
    """
    ANA Attack
    Enhancing Adversarial Transferability With Alignment Network (TIFS 2025) (https://ieeexplore.ieee.org/document/11018095)

    Arguments:
        model_name (str): the name of surrogate model for attack.
        epsilon (float): the perturbation budget.
        featur_layer (str): the feature layer name
        alpha (float): the step size.
        epoch (int): the number of iterations.
        decay (float): the decay factor for momentum calculation.
        num_ens (int): the number of gradients to aggregate
        targeted (bool): targeted/untargeted attack.
        random_start (bool): whether using random initialization for delta.
        norm (str): the norm of perturbation, l2/linfty.
        loss (str): the loss function.
        device (torch.device): the device for data. If it is None, the device would be same as model
        feature_layer: feature layer to launch the attack
        drop_rate : probability to drop random pixel

    Official arguments:
        epsilon=16/255, alpha=epsilon/epoch=1.6/255, epoch=10, decay=1., num_scale=10

    Example script:
        python main.py --input_dir ./path/to/data --output_dir adv_data/ana/resnet50 --attack ana --model=resnet50
        python main.py --input_dir ./path/to/data --output_dir adv_data/ana/resnet50 --eval
    """

    def __init__(self, model_name, epsilon=16 / 255, alpha=1.6 / 255, epoch=10, decay=1., num_ens=30,
                 targeted=False, random_start=False, feature_layer='1.layer2',
                 norm='linfty', loss='crossentropy', device=None, attack='P2FA', eta=28.0, gamma=2.0, checkpoint_path='./path/to/checkpoints/', **kwargs):
        self.checkpoint_path = checkpoint_path
        super().__init__(model_name, epsilon, alpha, epoch, decay, targeted, random_start, norm, loss, device, attack)
        self.ensemble_number = num_ens
        self.eta = eta
        self.gamma = gamma

    def load_model(self, model_name):
        alignment_path = os.path.join(self.checkpoint_path, 'aligned_res50_v3.pth')

        if not os.path.exists(alignment_path):
            raise ValueError("Please download checkpoints from 'https://drive.google.com/file/d/1UDqKV3zIATfcdSLionDvYmr6dvUHmz49/view?usp=drive_link', and put them into the path './path/to/checkpoints'.")
        
        model = ResNet50WithAlign(alignment_path=alignment_path)
        print(f"Loading alignment model: {alignment_path}")
        return model.eval().cuda()

    def get_grad(self, loss, delta, **kwargs):
        return torch.autograd.grad(loss, delta, retain_graph=kwargs.get('retain_graph'), create_graph=False)[0]

    def forward(self, data, label, **kwargs):
        data = data.clone().detach().to(self.device)
        x = data.clone().detach().to(self.device)
        y = label.clone().detach().to(self.device)
        delta = self.init_delta(data)
        with torch.no_grad():
            _, feat_orig_flat, feat_mask_aligned_ref = self.model(x, x)
            ref_pre = feat_orig_flat.detach()
            ref_post = feat_mask_aligned_ref.detach()
        momentum = 0
        for step in range(self.epoch):
            x_adv = (x + delta).requires_grad_(True)
            logits_adv, feat_adv_pre, feat_adv_post_aligned = self.model(x_adv, x_adv)

            cls_loss = F.cross_entropy(logits_adv, y)
            D1 = F.mse_loss(feat_adv_pre, ref_pre, reduction='sum')
            D2 = F.mse_loss(feat_adv_post_aligned, ref_post, reduction='sum')

            grads = []
            self.model.zero_grad()
            g_cls = self.get_grad(cls_loss, delta, retain_graph=True)
            grads.append(g_cls)

            g_d1 = self.get_grad(self.gamma * D1, delta, retain_graph=True)
            grads.append(g_d1)

            g_d2 = self.get_grad(self.gamma * D2, delta, retain_graph=True)
            grads.append(g_d2)

            def norm_grad(g):
                g_flat = g.view(g.size(0), -1)
                denom = torch.norm(g_flat, p=2, dim=1).view(-1, 1, 1, 1) + 1e-10
                return g / denom

            g_sum = norm_grad(grads[0]) + norm_grad(grads[1]) + norm_grad(grads[2])
            momentum = self.get_momentum(g_sum, momentum)
            delta = self.update_delta(delta, data, momentum, self.alpha)
        return delta