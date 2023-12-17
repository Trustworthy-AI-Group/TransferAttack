import torch

from ..utils import *
from ..attack import Attack

class VAIFGSM(Attack):
    """
    VA-I-FGSM Attack
    'Improving Transferability of Adversarial Examples with Virtual Step and Auxiliary Gradients (IJCAI 2022)'(https://www.ijcai.org/proceedings/2022/0227.pdf)

    Arguments:
        model_name (str): the name of surrogate model for attack.
        epsilon (float): the perturbation budget.
        alpha (float): the step size.
        epoch (int): the number of iterations.
        aux_num (int): the number of auxiliary labels.
        targeted (bool): targeted/untargeted attack
        random_start (bool): whether using random initialization for delta.
        norm (str): the norm of perturbation, l2/linfty.
        loss (str): the loss function.
        device (torch.device): the device for data. If it is None, the device would be same as model

    Official arguments:
        epsilon=16/255, alpha=0.007, epoch=20, aux_num=3

    Example script:
        python main.py --attack vaifgsm --output_dir adv_data/vaifgsm/resnet18
    """

    def __init__(self, model_name, epsilon=16/255, alpha=0.007, epoch=20, aux_num=3, targeted=False, random_start=False, 
                norm='linfty', loss='crossentropy', device=None, attack='VA-I-FGSM', **kwargs):
        super().__init__(attack, model_name, epsilon, targeted, random_start, norm, loss, device)
        self.alpha = alpha
        self.epoch = epoch
        self.aux_num = aux_num
        self.num_classes = 1000

    def get_aux_labels(self, label):   
        """
        Generate auxiliary label for a batch of images
        Arguments:
            label: the ground-truth label
        """
        aux_labels = []   
        for i in range(label.shape[0]):
            # Shuffle label list consists of all classes
            aux_label = torch.randperm(self.num_classes).tolist()

            # Remove ground truth label from the list
            aux_label.remove(label[i].item())

            # Select auxiliary labels from the list
            aux_label = aux_label[:self.aux_num]

            # Store auxiliary labels
            aux_labels.append(aux_label)

        # Reshape from [batch_size, aux_num] to [aux_num, batch_size]
        aux_labels = np.transpose(np.array(aux_labels, dtype=np.int64),(1,0))
        
        aux_labels_ = []
        for i in range(aux_labels.shape[0]):
            aux_labels_.append(torch.from_numpy(aux_labels[i]).detach().to(self.device))

        return aux_labels_

    def update_delta(self, delta, data, grad, alpha, **kwargs):
        if self.norm == 'linfty':
            delta = delta + alpha * grad.sign()
        else:
            grad_norm = torch.norm(grad.view(grad.size(0), -1), dim=1).view(-1, 1, 1, 1)
            scaled_grad = grad / (grad_norm + 1e-20)
            delta = (delta + scaled_grad * alpha).view(delta.size(0), -1).renorm(p=2, dim=0).view_as(delta)
        delta = clamp(delta, img_min-data, img_max-data)
        return delta

    def forward(self, data, label, **kwargs):
        """
        The VA-I-FGSM attack procedure

        Arguments:
            data: (N, C, H, W) tensor for input images
            labels: (N,) tensor for ground-truth labels if untargetd, otherwise targeted labels
        """
        if self.targeted:
            assert len(label) == 2
            label = label[1] # the second element is the targeted label tensor
        data = data.clone().detach().to(self.device)
        label = label.clone().detach().to(self.device)
        
        # Initialize adversarial perturbation
        delta = self.init_delta(data)
        
        for _ in range(self.epoch):
            grads = []
            losses = []
            # Obtain the output
            logits = self.get_logits(self.transform(data+delta))

            # Calculate the loss
            loss = self.get_loss(logits, label)
            losses.append(loss)

            # Generate auxiliary label
            aux_labels = self.get_aux_labels(label)

            # Calculate auxiliary loss
            for aux_label in aux_labels:
                aux_loss = self.get_loss(logits, aux_label)
                losses.append(-aux_loss)

            # Calculate the auxiliary gradient
            for loss in losses:
                grad = torch.autograd.grad(loss, delta, retain_graph=True, create_graph=False)[0]
                grads.append(grad)
            
            # Update adversarial perturbation
            for grad in grads:
                delta = self.update_delta(delta, data, grad, self.alpha)
        
        # Clamp delta into the range of [-epsilon, epsilon]
        delta = torch.clamp(delta, -self.epsilon, self.epsilon)
        return delta.detach()

    
