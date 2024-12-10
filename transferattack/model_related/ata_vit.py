import torch
import torchvision.transforms.functional as TF
import cv2
import random
import tqdm
from ..utils import *
from ..attack import Attack
from .ata_vit_utils.Transformer_Explainability.samples.CLS2IDX import CLS2IDX
from .ata_vit_utils.Transformer_Explainability.baselines.ViT.ViT_LRP import *
from .ata_vit_utils.Transformer_Explainability.baselines.ViT.ViT_explanation_generator import LRP


class ATA_ViT(Attack):
    """
    ATA Attack
    'Generating Transferable Adversarial Examples against Vision Transformers (ACM-MM 2022)'(https://dl.acm.org/doi/abs/10.1145/3503161.3547989)

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
        device (torch.device): the device for data. If it is None, the device would be same as model.

    Official arguments:
        epoch=250, LR=1.0, model_name='deit_tiny_patch16_224

    Example script:
        python main.py --input_dir ./path/to/data --output_dir adv_data/ata_vit/deit_tiny_patch16_224 --attack ma --model deit_tiny_patch16_224
        python main.py --input_dir ./path/to/data --output_dir adv_data/ata_vit/deit_tiny_patch16_224 --eval
    """
    def __init__(self, model_name, epsilon=16/255, alpha=1.6/255, epoch=250, decay=1., targeted=False, random_start=False,
                norm='linfty', loss='crossentropy', device=None, attack='ATA_ViT', **kwargs):
        super().__init__(attack, model_name, epsilon, targeted, random_start, norm, loss, device)
        self.alpha = alpha
        self.epoch = epoch
        self.decay = decay
        self.LR = 1.0
        self.model_name = model_name
        self.data_preprocess = DataPreprocess(self.model_name)
        self.patch_path, self.mask_path = self.data_preprocess()
        
    def forward(self, data, label, **kwargs):
        """
        The attack procedure for ATA_ViT

        Arguments:
            data (N, C, H, W): tensor for input images
            labels (N,): tensor for ground-truth labels if untargetd
            labels (2,N): tensor for [ground-truth, targeted labels] if targeted
        """
        if self.targeted:
            assert len(label) == 2
            label = label[1] # the second element is the targeted label tensor

        data = data.clone().detach().to(self.device)
        labels = label.clone().detach().to(self.device)

        # Obtain the image filenames
        filenames = kwargs.get('filenames', None)
        if filenames is None:
            raise ValueError('The filenames should be provided for ATA_ViT attack.')
        
        # Load stored patchs and masks
        patchs = []
        masks = []
        for filename in filenames:
            patch = np.load(os.path.join(self.patch_path, filename[:-5]+'.npy'))
            mask = np.load(os.path.join(self.mask_path, filename[:-5]+'.npy'))
            patchs.append(patch)
            masks.append(mask)
        
        # Convert patchs and masks to tensor; Patchs have been normalized, we need to denormalize it to match the input data
        patchs = torch.from_numpy(np.array(patchs))
        if 'vit' in self.model_name:
            patchs = patchs * 0.5 + 0.5
        else:
            patchs = TF.normalize(patchs, mean=[-0.485/0.229, -0.456/0.224, -0.406/0.225], std=[1/0.229, 1/0.224, 1/0.225])
        
        masks = torch.from_numpy(np.array(masks))

        # Compute the patchs and masks for each image
        patchs_pad = patchs * masks
        masks_pad = masks
        patchs_pad = patchs_pad.to(self.device)
        masks_pad = masks_pad.to(self.device)
        patchs_pad.requires_grad = True

        # Initialize the optimizer and scheduler
        optimizer = torch.optim.Adam([patchs_pad], lr=self.LR)# , weight_decay=1e-4)
        scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[50, 150], gamma=1/3)
        
        for _ in range(self.epoch):
            # Obtain the inputs
            inputs = data * (1 - masks_pad) + patchs_pad * masks_pad

            inputs.clamp_(0, 1)

            # Obtain the logits
            outputs = self.model(inputs)

            # Calculate the loss
            probs = torch.nn.functional.softmax(outputs, dim=1)
            probs = probs.index_select(1, labels)
            loss =  (-torch.log(1-probs+1e-10) * torch.eye(data.shape[0]).cuda()).sum() / data.shape[0]

            # Update the patchs
            optimizer.zero_grad()
            loss.backward(retain_graph=True)
            optimizer.step()
            scheduler.step()

            patchs_pad.data.clamp_(0, 1)

        # Obtain the adversarial examples
        inputs = data * (1 - masks_pad) + patchs_pad * masks_pad

        inputs.clamp_(0, 1)

        # Compute the perturbation
        delta = inputs - data
        
        return delta.detach()

class ATTENTION_RIGION(torch.nn.Module):
    """
    Compute the attention region for ATA_ViT

    Arguments:
        model_name (str): the name of the model

    Returns:
        region_path (str): the path for storing the attention region

    NOTE:
        The code is referenced from https://github.com/nlsde-safety-team/ATA
    """
    def __init__(self, model_name):
        super(ATTENTION_RIGION, self).__init__()
        self.model_name = model_name
        self.model = self.get_model(model_name)
        self.attribution_generator = LRP(self.model)

    def get_model(self, model_name):
        model = eval(model_name)(pretrained=True).cuda()
        model.eval()
        return model

    def generate_visualization(self, original_image, class_index=None):
        _, rollout = self.attribution_generator.generate_LRP(original_image.unsqueeze(0).cuda(), method="transformer_attribution", index=class_index)
        return rollout.detach()
    
    def load_labels(self, file_name):
        dev = pd.read_csv(file_name)
        f2l = {dev.iloc[i]['filename']: dev.iloc[i]['label'] for i in range(len(dev))}
        return f2l
    
    def forward(self, datapath='./data/images', labelpath='./data/labels.csv'):
        """
        Obtain the attention region for ATA_ViT
        """
        region_path = 'attention_region_' + self.model_name
        os.makedirs(os.path.join(region_path), exist_ok=True)
        f2l = self.load_labels(labelpath)

        if 'vit' in self.model_name:
            test_transforms = transforms.Compose([
                transforms.Resize((224,224)),
                transforms.ToTensor(),
                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
            ])
        else:
            test_transforms = transforms.Compose([
                transforms.Resize((224,224)),
                transforms.ToTensor(),
                transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
            ])

        tqdm_bar = tqdm.tqdm(f2l.keys())
        for i, file in enumerate(tqdm_bar):
            image_file = os.path.join(datapath, file)
            image = Image.open(image_file).convert('RGB').resize((224, 224))
            image_torch = test_transforms(image)
            image_torch = image_torch.cuda()

            category_index = f2l[file]
            
            rollout = self.generate_visualization(image_torch, class_index=category_index)
            rollout = rollout.data.cpu().numpy()
            
            np.save(os.path.join(region_path, file[:-5]+'.npy'), rollout)

        return region_path
            

class EMBED_POSITION(torch.nn.Module):
    """
    Compute the embedding position for ATA_ViT

    Arguments:
        model_name (str): the name of the model

    Returns:
        output_path (str): the path for storing the embedding position

    NOTE:
        The code is referenced from https://github.com/nlsde-safety-team/ATA
    """
    def __init__(self, model_name):
        super(EMBED_POSITION, self).__init__()
        self.model_name = model_name
        self.model = self.get_model(model_name)
    
    def get_model(self, model_name):
        model = timm.create_model(model_name, pretrained=True).eval().cuda()
        return model
    
    def load_labels(self, file_name):
        dev = pd.read_csv(file_name)
        f2l = {dev.iloc[i]['filename']: dev.iloc[i]['label'] for i in range(len(dev))}
        return f2l

    def forward(self, datapath='./data/images', labelpath='./data/labels.csv'):
        """
        Obtain the embedding position for ATA_ViT
        """
        output_path = f'./embed_position_{self.model_name}'
        os.makedirs(output_path, exist_ok=True)

        if 'vit' in self.model_name:
            test_transforms = transforms.Compose([
                transforms.Resize((224,224)),
                transforms.ToTensor(),
                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
            ])
        else:
            test_transforms = transforms.Compose([
                transforms.Resize((224,224)),
                transforms.ToTensor(),
                transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
            ])

        f2l = self.load_labels(labelpath)

        for file in tqdm.tqdm(f2l.keys()):
            image_file = os.path.join(datapath, file)
            image = Image.open(image_file).convert('RGB').resize((224, 224))
            image_torch = test_transforms(image).unsqueeze(0)
            image_torch = image_torch.cuda()
            
            origin_embed = self.model.patch_embed(image_torch)
            
            embed_mask_2 = torch.zeros((224, 224))
            
            for i in range(16):
                for j in range(16):
                    new_image_torch = image_torch.clone()
                    for x in range(14):
                        for y in range(14):
                            new_image_torch[:, :, x*16+i, y*16+j] = 0
                        
                    new_embed = self.model.patch_embed(new_image_torch)
                    
                    diff = origin_embed - new_embed

                    diff = (diff ** 2) ** 0.5
                    
                    for x in range(14):
                        for y in range(14):
                            embed_mask_2[x*16+i, y*16+j] = diff[:, x*14+y, :].sum().data.cpu()
                    
            
            embed_mask_2 = embed_mask_2.data.cpu().numpy()
            embed_mask_2 -= np.min(embed_mask_2)
            embed_mask_2 /= np.max(embed_mask_2)

            np.save(os.path.join(output_path, file[:-5]+'.npy'), embed_mask_2)

        return output_path


class DataPreprocess(torch.nn.Module):
    """
    Data preprocess for ATA_ViT
    Compute the attention region and embedding position for ATA_ViT, and generate the patchs and masks

    Arguments:
        model_name (str): the name of the model

    Returns:
        PATCH_PATH (str): the path for storing the patchs
        MASK_PATH (str): the path for storing the masks

    NOTE:
        The code is referenced from https://github.com/nlsde-safety-team/ATA
    """

    def __init__(self, model_name):
        super(DataPreprocess, self).__init__()
        self.model_name = model_name # the name of the model
        self.attention_region = ATTENTION_RIGION(model_name) # the path for storing the attention region
        self.embed_position = EMBED_POSITION(model_name) # the path for storing the embedding position
        self.compute_attn = True # whether to compute the attention region
        self.compute_embed = True # whether to compute the embedding position

        if 'vit' in model_name:
            self.transform = transforms.Compose([
                transforms.Resize((224,224)),
                transforms.ToTensor(),
                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
            ])
        else:
            self.transform = transforms.Compose([
                transforms.Resize((224,224)),
                transforms.ToTensor(),
                transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
            ])
    
    def make_mask_embed(self, shape, num_pixel, embed):
        mask = torch.zeros(shape)
        embed_reshape = embed.reshape((256))
        sort_arg = np.argsort(-embed_reshape)
        for i in range(int(num_pixel)):
            x = sort_arg[i] // 16
            y = sort_arg[i] % 16
            assert embed_reshape[sort_arg[i]] == embed[x, y]
            mask[x, y] = 1
        return mask

    def forward(self, datapath='./data/images', labelpath='./data/labels.csv'):
        """
        The data preprocess for ATA_ViT
        """

        if self.compute_attn and self.compute_embed:
            ATTN_PATH = self.attention_region(datapath=datapath, labelpath=labelpath)
            EMBED_PATH = self.embed_position(datapath=datapath, labelpath=labelpath)
        else:
            ATTN_PATH = f'./attention_region_{self.model_name}'
            EMBED_PATH = f'./embed_position_{self.model_name}'

        PATCH_PATH = f'./patchs_{self.model_name}'
        MASK_PATH = f'./masks_{self.model_name}'

        os.makedirs(PATCH_PATH, exist_ok=True)
        os.makedirs(MASK_PATH, exist_ok=True)

        f2l = self.attention_region.load_labels(labelpath)

        for image_file in tqdm.tqdm(f2l.keys()):
            image = Image.open(os.path.join(datapath, image_file)).convert('RGB').resize((224, 224))
            image = self.transform(image)
            
            embed_image = np.load(os.path.join(EMBED_PATH, image_file[:-5]+'.npy'))
            rollout  = np.load(os.path.join(ATTN_PATH, image_file[:-5]+'.npy'))

            grad_token = np.zeros((196))
            cls_sum = rollout[0,0,1:].sum()

            for i in range(1,197):
                rollout[0,0,i] /= cls_sum
                for j in range(1,197):
                    if i != j:
                        grad_token[i-1] -= rollout[0,j,i]*np.log2(rollout[0,j,i])

            grad_token = np.reshape(grad_token,(14,14))
            grad_token /= grad_token.sum()
            grad_token = np.floor(grad_token* 1024)
            
            
            for i in range(14):
                for j in range(14):
                    while grad_token[i, j] > 255:
                        grad_token[i, j] -= 1
            
            for i in range(1024,int(grad_token.sum())):
                x = random.randint(0, 13)
                y = random.randint(0, 13)
                while grad_token[x, y] <= 100:
                    x = random.randint(0, 13)
                    y = random.randint(0, 13)
                grad_token[x, y] -= 1
            
            for i in range(int(grad_token.sum()), 1024):
                x = random.randint(0, 13)
                y = random.randint(0, 13)
                while grad_token[x, y] >= 250:
                    x = random.randint(0, 13)
                    y = random.randint(0, 13)
                grad_token[x, y] += 1


            patch = torch.randn_like(image)
            total = 0.
            count = 0
            mask = torch.zeros_like(image)
            for i in range(0, 224, 16):
                for j in range(0, 224, 16):
                    total += (32*32) / (14*14)
                    diff = (total - count) // 1 + 1

                    _mask = self.make_mask_embed((16, 16), grad_token[i//16, j//16], embed_image[i:i+16, j:j+16])
                    
                    mask[:, i:i+16, j:j+16] = _mask
                    
                    count += diff
            
            assert mask.sum() == 3 * 32 * 32

            patch = patch.data.cpu().numpy()
            mask = mask.data.cpu().numpy()

            np.save(os.path.join(PATCH_PATH, image_file[:-5]+'.npy'), patch)
            np.save(os.path.join(MASK_PATH, image_file[:-5]+'.npy'), mask)

        return PATCH_PATH, MASK_PATH
