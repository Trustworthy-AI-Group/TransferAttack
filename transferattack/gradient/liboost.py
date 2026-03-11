import torch

from ..utils import *
from ..attack import Attack
from scipy.optimize import brentq
class LIBoost(Attack):
    """
    LI-Boost Attack (eg.,LI-Boost-MI)
    'Boosting the Local Invariance for Better Adversarial Transferability '(https://arxiv.org/abs/2503.06140)
    
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
        N (int): the number of sampled perturbations. 
        k (int): the upper bound of translated pixels.  

    Official arguments:
        epsilon=16/255, alpha=epsilon/epoch=1.6/255, epoch=10, decay=1, N=30, k=6.

    Example script:
        python main.py --input_dir ./path/to/data --output_dir adv_data/li-boost-mi/resnet50 --attack liboost --model=resnet50
        python main.py --input_dir ./path/to/data --output_dir adv_data/li-boost-mi/resnet50 --eval
    """

    def __init__(self, model_name, epsilon=16/255, alpha=1.6/255, epoch=10, decay=1., targeted=False, random_start=False,
                norm='linfty', N=30, k=6, loss='crossentropy', device=None, attack='LIBoost', **kwargs):
        super().__init__(attack, model_name, epsilon, targeted, random_start, norm, loss, device)
        self.alpha = alpha
        self.epoch = epoch
        self.decay = decay
        self.N = N
        self.k = k
    
    
    def theoretical_pdf(self,dx, m):
        if dx <= 0 or dx > 224 * m:
            return 0.0
        return (1 / (224 * m)) * np.log((224 * m) / dx)

    def theoretical_cdf(self,dx, m):
        if dx <= 0:
            return 0.0
        elif dx >= 224 * m:
            return 1.0
        return (1 / (224 * m)) * (dx * np.log(224 * m / dx) + dx)

    def inverse_cdf(self,u, m):
        return brentq(lambda dx: self.theoretical_cdf(dx, m) - u, 1e-10, 224 * m)
    
    def generate_shift(self, k, num_samples=1):
        m = (k + 1) / 224
        u = np.random.uniform(0, 1, num_samples)
        dx_samples = np.array([self.inverse_cdf(ui, m) for ui in u])
        integer_shifts = np.floor(dx_samples).astype(int)
        return integer_shifts
    
    def move_log_distribute(self, delta, k, **kwargs):
        dx = int(self.generate_shift(k))
        dy = int(self.generate_shift(k))
        dx *= np.random.choice([-1,1])
        dy *= np.random.choice([-1,1])
        
        shifted_delta = torch.roll(delta, shifts=(dx, dy), dims=(2, 3))
        if dx > 0: 
            shifted_delta[:, :, :dx, :] = 0
        elif dx < 0:
            shifted_delta[:, :, dx:, :] = 0
        if dy > 0: 
            shifted_delta[:, :, :, :dy] = 0
        elif dy < 0:
            shifted_delta[:, :, :, dy:] = 0
        return shifted_delta   
    
    def forward(self, data, label, **kwargs):
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        if self.targeted:
            assert len(label) == 2
            label = label[1] 
            
        data = data.clone().detach().to(device)
        label = label.clone().detach().to(device)
        delta = self.init_delta(data)
        
        momentum = 0
        
        for _ in range(self.epoch): 
            grads = 0
            for _ in range(self.N):
                re_delta = self.move_log_distribute(delta, k=self.k)
                logits = self.get_logits(data+re_delta, momentum=momentum)
                loss = self.get_loss(logits, label)
                grads += self.get_grad(loss, delta)
            grads /= self.N
            momentum = self.get_momentum(grads, momentum)
            delta = self.update_delta(delta, data, momentum, self.alpha)
        return delta.detach()
