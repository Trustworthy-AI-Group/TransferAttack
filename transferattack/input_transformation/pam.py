import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init
import torch.optim as optim
from tqdm import tqdm


from ..utils import *
from ..gradient.mifgsm import MIFGSM

class PAM(MIFGSM):
    """
    PAM Attack
    'Improving the Transferability of Adversarial Samples by Path-Augmented Method (CVPR 2023)'(https://arxiv.org/abs/2303.15735)

    Arguments:
        model_name (str): the name of surrogate model for attack.
        epsilon (float): the perturbation budget.
        alpha (float): the step size.
        epoch (int): the number of iterations.
        decay (float): the decay factor for momentum calculation.
        num_aug_path (int): the number of augmentation paths.
        num_scale (int): the number of scaled copies in each iteration.
        train_epoch (int): the number of epoches for training the semantic predictor.
        targeted (bool): targeted/untargeted attack.
        random_start (bool): whether using random initialization for delta.
        norm (str): the norm of perturbation, l2/linfty.
        loss (str): the loss function.
        device (torch.device): the device for data. If it is None, the device would be same as model

    Official arguments:
        epsilon=16/255, alpha=epsilon/epoch=1.6/255, epoch=10, decay=1., num_aug_path=8, num_scale=4, train_epoch=15
    """

    def __init__(self, model_name, epsilon=16/255, alpha=1.6/255, epoch=10, decay=1., num_aug_path=8, num_scale=4, train_epoch=15, targeted=False, random_start=False, norm='linfty', loss='crossentropy', device=None, attack='PAM', **kwargs):
        super().__init__(model_name, epsilon, alpha, epoch, decay, targeted, random_start, norm, loss, device, attack)
        self.num_aug_path = num_aug_path
        self.num_scale = num_scale
        self.train_epoch = train_epoch
        self.sp_model = self.train_SP(is_training=True)
        
    def create_x_base(self, batch_size, ratios):
        x_base_0 = torch.tensor([[0.0,0.0,0.0]] * batch_size).to(self.device) * ratios[:,0].view(batch_size, 1) # shape=(batch_size, 3)
        x_base_1 = torch.tensor([[0.5,0.5,0.5]] * batch_size).to(self.device) * ratios[:,1].view(batch_size, 1)
        x_base_2 = torch.tensor([[1.0,1.0,1.0]] * batch_size).to(self.device) * ratios[:,2].view(batch_size, 1)
        x_base_3 = torch.tensor([[0.5,0.5,0.0]] * batch_size).to(self.device) * ratios[:,3].view(batch_size, 1)
        x_base_4 = torch.tensor([[1.0,1.0,0.5]] * batch_size).to(self.device) * ratios[:,4].view(batch_size, 1)
        x_base_5 = torch.tensor([[1.0,0.5,1.0]] * batch_size).to(self.device) * ratios[:,5].view(batch_size, 1)
        x_base_6 = torch.tensor([[0.5,1.0,1.0]] * batch_size).to(self.device) * ratios[:,6].view(batch_size, 1)
        x_base_7 = torch.tensor([[0.0,0.5,0.5]] * batch_size).to(self.device) * ratios[:,7].view(batch_size, 1)

        x_base_0_concat = torch.cat([x_base_0 * (1-1/2), x_base_0 * (1-1/4), x_base_0 * (1-1/8), x_base_0 * (1-1/16)], dim=0) # shape=(batch_size*4, 3)
        x_base_1_concat = torch.cat([x_base_1 * (1-1/2), x_base_1 * (1-1/4), x_base_1 * (1-1/8), x_base_1 * (1-1/16)], dim=0)
        x_base_2_concat = torch.cat([x_base_2 * (1-1/2), x_base_2 * (1-1/4), x_base_2 * (1-1/8), x_base_2 * (1-1/16)], dim=0)
        x_base_3_concat = torch.cat([x_base_3 * (1-1/2), x_base_3 * (1-1/4), x_base_3 * (1-1/8), x_base_3 * (1-1/16)], dim=0)
        x_base_4_concat = torch.cat([x_base_4 * (1-1/2), x_base_4 * (1-1/4), x_base_4 * (1-1/8), x_base_4 * (1-1/16)], dim=0)
        x_base_5_concat = torch.cat([x_base_5 * (1-1/2), x_base_5 * (1-1/4), x_base_5 * (1-1/8), x_base_5 * (1-1/16)], dim=0)
        x_base_6_concat = torch.cat([x_base_6 * (1-1/2), x_base_6 * (1-1/4), x_base_6 * (1-1/8), x_base_6 * (1-1/16)], dim=0)
        x_base_7_concat = torch.cat([x_base_7 * (1-1/2), x_base_7 * (1-1/4), x_base_7 * (1-1/8), x_base_7 * (1-1/16)], dim=0)
        x_base = torch.cat([x_base_0_concat, x_base_1_concat, x_base_2_concat, x_base_3_concat, x_base_4_concat, x_base_5_concat, x_base_6_concat, x_base_7_concat], dim=0) # shape=(batch_size*32, 3)
        x_base = x_base.view(-1, 3, 1, 1)
        return x_base

        
    def transform(self, x, **kwargs):
        ratios = kwargs['ratios'] # shape=(batch_size, 8)
        batch_size = x.shape[0]

        x_base = self.create_x_base(batch_size, ratios)

        x_in_concat = torch.cat([x * (1/2**i) for i in range(1, 5)], dim=0)
        x_in = torch.cat([x_in_concat] * 8, dim=0)
        x_aug = x_in + x_base

        return x_aug

    def get_loss(self, logits, label):
        """
        Calculate the loss
        """
        return -self.loss(logits, label.repeat(self.num_scale*self.num_aug_path)) if self.targeted else self.loss(logits, label.repeat(self.num_scale*self.num_aug_path))

    def forward(self, data, label, **kwargs):
        """
        The general attack procedure

        Arguments:
            data (N, C, H, W): tensor for input images
            labels (N,): tensor for ground-truth labels if untargetd
            labels (2,N): tensor for [ground-truth, targeted labels] if targeted
        """
        if self.targeted:
            assert len(label) == 2
            label = label[1] # the second element is the targeted label tensor
        data = data.clone().detach().to(self.device)
        label = label.clone().detach().to(self.device)

        # Initialize adversarial perturbation
        delta = self.init_delta(data)

        # Calculate the scaling factor
        ratios = self.sp_model(data)

        # ratios = torch.ones(data.shape[0], self.num_aug_path).to(self.device)

        momentum = 0
        for _ in range(self.epoch):
            logits = self.get_logits(data+delta)

            loss = self.loss(logits, label)

            grad = self.get_grad(loss, delta)

            # Obtain the output
            logits = self.get_logits(self.transform(data+delta, ratios=ratios))

            # Calculate the loss
            loss = self.get_loss(logits, label)

            # Calculate the gradients
            grad += self.get_grad(loss, delta) * 32

            # Calculate the momentum
            momentum = self.get_momentum(grad, momentum)

            # Update adversarial perturbation
            delta = self.update_delta(delta, data, momentum, self.alpha)

        return delta.detach()
        

    def train_SP(self, input_dir = './data', checkpoint_dir = './checkpoints', batch_size=1, is_training=True, **kwargs):
        # Check the checkpoint directory
        if not os.path.exists(checkpoint_dir):
            os.makedirs(checkpoint_dir)

        # Check if the is_training is False
        if is_training == False:
            # Check if the semantic_predictor.pth exists
            if not os.path.exists(os.path.join(checkpoint_dir, 'semantic_predictor.pth')):
                raise FileNotFoundError('semantic_predictor.pth not found')

            # Load the semantic_predictor.pth
            predictor = SemanticPredictor().to(self.device)
            predictor.load_state_dict(torch.load(os.path.join(checkpoint_dir, 'semantic_predictor.pth')))
            return predictor
        
        # Load the dataset
        dataset = AdvDataset(input_dir=input_dir, targeted=self.targeted, eval=False)
        dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=4)

        # Initialize the semantic predictor, optimizer and criterion
        predictor = SemanticPredictor().to(self.device)
        optimizer = optim.Adam(predictor.parameters(), lr=0.0001)
        criterion = SPLoss()

        baseline = torch.tensor([[0.0, 0.0, 0.0],
                         [0.5, 0.5, 0.5],
                         [1.0, 1.0, 1.0],
                         [0.5, 0.5, 0.0],
                         [1.0, 1.0, 0.5],
                         [1.0, 0.5, 1.0],
                         [0.5, 1.0, 1.0],
                         [0.0, 0.5, 0.5]]).to(self.device)
        baselines = baseline.unsqueeze(-1).unsqueeze(-1)  # Add dimensions for broadcasting, shape=(8, 3, 1, 1)
        
        # Start training
        print('Start training the Semantic Predictor')
        for epoch in tqdm(range(self.train_epoch), desc="Epoch progress"):
            avg_cost = 0.
            total_batch = 1000

            for batch_idx, [images, labels, _] in tqdm(enumerate(dataloader), desc="Batch progress", leave=False):
                if self.targeted:
                    assert len(labels) == 2
                    labels = labels[1] # the second element is the targeted label tensor
                # Put the data and label to the device
                x = images.clone().detach().to(self.device) # shape=(batch_size, 3, 224, 224)
                y = labels.clone().detach().to(self.device) # shape=(batch_size,)

                # Zero the parameter gradients
                optimizer.zero_grad()

                pred = predictor(x)[0].unsqueeze(-1).unsqueeze(-1).unsqueeze(-1) # shape=(8, 1, 1, 1)
                x_in = torch.concat([x] * 8, dim=0) # shape=(8, 3, 224, 224)
                x_aug = x_in * (1 - pred) + baselines * pred # shape=(8, 3, 224, 224)
                logits = self.model(x_aug)

                # Calculate the loss
                loss = criterion(logits, y.repeat(self.num_aug_path))

                # Obtain the gradients
                loss.backward()

                # Update the weights
                optimizer.step()

                avg_cost += loss.item() / total_batch
            
            print(f"Epoch: {epoch}, Average cost: {avg_cost}")

        print('Finished Training')

        # Save the model
        torch.save(predictor.state_dict(), os.path.join(checkpoint_dir, 'semantic_predictor.pth'))
        print('Semantic_Predictor model saved')
        return predictor


class SemanticPredictor(nn.Module):
    def __init__(self):
        super(SemanticPredictor, self).__init__()
        self.conv1 = nn.Conv2d(3, 3, kernel_size=5, padding='same')
        self.pool1 = nn.AvgPool2d(4, stride=4)
        self.conv2 = nn.Conv2d(3, 1, kernel_size=5, padding='same')
        self.pool2 = nn.AvgPool2d(4, stride=4)
        self.fc = nn.Linear(1*14*14, 8)

        # Initialize the weights and biases in fc layer
        init.normal_(self.fc.weight)
        init.normal_(self.fc.bias)


    def forward(self, x):
        # First layer
        x = F.relu(self.conv1(x))
        x = self.pool1(x)

        # Second layer
        x = F.relu(self.conv2(x))
        x = self.pool2(x)

        # Flatten the tensor
        x = torch.flatten(x, 1)

        # Output layer with linear activation
        x = self.fc(x)
        x = torch.sigmoid(x) * 0.1 + 0.9
        return x

class SPLoss(nn.Module):
    def __init__(self):
        super(SPLoss, self).__init__()
        
    def forward(self, logits, labels):
        # x shape: batch_size x num_classes, y shape: batch_size
        # Get the confidence score of the target class
        true_scores = logits.gather(1, labels.view(-1, 1)).squeeze()

        # Get the confidence score of the second highest class
        second_highest_score = logits.scatter(1, labels.view(-1, 1), float('-inf')).max(dim=1).values

        # Calculate the loss
        loss = torch.sum((true_scores - second_highest_score) ** 2)

        return loss