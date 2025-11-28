import torch
import torch.nn as nn
import torch.nn.functional as F
import time

from ..utils import *
from ..attack import Attack

class Foolmix(Attack):
    """
    Foolmix Attack Algorithm
    'Foolmix: Strengthen the Transferability of Adversarial Examples by Dual-Blending and Direction Update Strategy. (TIFS 2024)' (https://ieeexplore.ieee.org/document/10508615)
    Arguments:
        model_name (str): the name of surrogate model for attack.
        epsilon (float): the perturbation budget (default: 16/255).
        alpha (float): the step size (default: 1.6/255).
        epoch (int): the number of iterations (default: 15).
        decay (float): the decay factor for momentum calculation (default: 1.0).
        targeted (bool): targeted/untargeted attack.
        random_start (bool): whether using random initialization for delta.
        norm (str): the norm of perturbation, l2/linfty.
        loss (str): the loss function.
        device (torch.device): the device for data. If it is None, the device would be same as model
        m (int): the number of scale transformations (default: 5).
        n (int): the number of random pixel-blocks to generate (default: 3).
        z (int): the number of random other-class labels (default: 1).
        k (int): the number of top classifications to consider (default: 5).
        zeta (float): the strength parameter for random pixel-blocks (default: 0.2).
        beta (float): the strength parameter for random other-class labels (default: 1.0).
        gamma (float): the perturbation direction update parameter (default: 0.1).
        print_timing (bool): whether to print detailed timing statistics (default: True).

    Example script:
        python main.py --input_dir ./path/to/data --output_dir adv_data/foolmix/resnet50 --attack foolmix --model=resnet50
        python main.py --input_dir ./path/to/data --output_dir adv_data/foolmix/resnet50 --eval
    """

    def __init__(self, model_name, epsilon=16/255, alpha=1.6/255, epoch=10, decay=1., targeted=False, 
                 random_start=False, norm='linfty', loss='crossentropy', device=None, attack='Foolmix',
                 m=5, n=3, z=1, k=5, zeta=0.2, beta=1.0, gamma=0.1,  print_timing=True, 
                 use_amp=True, use_cache=False, grad_chunk_size=16, **kwargs):
        super().__init__(attack, model_name, epsilon, targeted, random_start, norm, loss, device)
        # super().__init__(attack, model_name, epsilon, targeted, random_start, norm, loss, device)
        self.alpha = alpha
        self.epoch = epoch
        self.decay = decay
        self.m = m
        self.n = n
        self.z = z
        self.k = k
        self.zeta = zeta
        self.beta = beta
        self.gamma = gamma
        self.print_timing = print_timing
        self.use_amp = use_amp
        self.use_cache = use_cache
        self.grad_chunk_size = grad_chunk_size
        
        # Initialize gradient cache
        if self.use_cache:
            self.gradient_cache = {}
        
        # Initialize mixed precision scaler
        if self.use_amp and torch.cuda.is_available():
            self.scaler = torch.cuda.amp.GradScaler()
        else:
            self.scaler = None

    def forward(self, data, label, **kwargs):
        """
        The optimized Foolmix attack procedure
        """
        if self.targeted:
            assert len(label) == 2
            label = label[1]
        
        data = data.clone().detach().to(self.device)
        label = label.clone().detach().to(self.device)

        # Initialize timing statistics
        total_start_time = time.time()
        timing_stats = {
            'total_time': 0,
            'init_time': 0,
            'iterations': []
        }

        # Initialize adversarial perturbation
        start_time = time.time()
        delta = self.init_delta(data)
        momentum = torch.zeros_like(delta).to(self.device)
        alpha = self.alpha
        timing_stats['init_time'] = time.time() - start_time

        for t in range(self.epoch):
            iter_start_time = time.time()
            iter_timing = {
                'iteration': t,
                'top_k_time': 0,
                'misclassified_adjust_time': 0,
                'pixel_blocks_time': 0,
                'other_labels_time': 0,
                'integrated_gradient_time': 0,
                'blended_gradient_time': 0,
                'momentum_update_time': 0,
                'delta_update_time': 0,
                'total_iter_time': 0
            }
            
            # Get top-k classifications
            start_time = time.time()
            with torch.no_grad():
                logits = self.model(data + delta)
            top_k_indices = torch.topk(logits, self.k + 1, dim=1)[1]
            iter_timing['top_k_time'] = time.time() - start_time
            
            # Check if misclassified and adjust
            if not self.targeted:
                misclassified = ~torch.any(top_k_indices == label.unsqueeze(1), dim=1)
                
                start_time = time.time()
                for i in range(data.size(0)):
                    if misclassified[i]:
                        f_topk = self.get_integrated_logits(data[i:i+1] + delta[i:i+1], top_k_indices[i])
                        omega_y = self.get_class_gradient(data[i:i+1] + delta[i:i+1], label[i:i+1])
                        omega_topk = self.get_integrated_gradient(data[i:i+1] + delta[i:i+1], top_k_indices[i])
                        d_direction = self.get_update_direction(f_topk, omega_y, omega_topk, label[i:i+1], data[i:i+1] + delta[i:i+1])
                        adjusted_delta = self.adjust_adversarial_example(delta[i:i+1], data[i:i+1], d_direction, alpha)
                        delta = torch.cat([delta[:i], adjusted_delta, delta[i+1:]], dim=0)
                iter_timing['misclassified_adjust_time'] = time.time() - start_time
            
            # Generate random pixel-blocks and labels
            start_time = time.time()
            P = self.generate_random_pixel_blocks(data)
            iter_timing['pixel_blocks_time'] = time.time() - start_time
            
            start_time = time.time()
            L = self.generate_random_other_class_labels(data, P)
            iter_timing['other_labels_time'] = time.time() - start_time
            
            # Calculate integrated gradient with batch processing
            start_time = time.time()
            g_lens = self.calculate_integrated_gradient_batch(data + delta, P, L)
            iter_timing['integrated_gradient_time'] = time.time() - start_time
            
            # Calculate blended gradient with batch processing
            start_time = time.time()
            g_mix = self.calculate_average_blended_gradient_batch(data + delta, P, g_lens, label)
            iter_timing['blended_gradient_time'] = time.time() - start_time
            
            # Update momentum and delta
            start_time = time.time()
            g_mix_norm = torch.norm(g_mix, p=1, dim=(1, 2, 3), keepdim=True)
            momentum = self.decay * momentum + g_mix / (g_mix_norm + 1e-8)
            iter_timing['momentum_update_time'] = time.time() - start_time
            
            start_time = time.time()
            delta = self.update_delta(delta, data, momentum, alpha)
            iter_timing['delta_update_time'] = time.time() - start_time
            
            iter_timing['total_iter_time'] = time.time() - iter_start_time
            timing_stats['iterations'].append(iter_timing)

        timing_stats['total_time'] = time.time() - total_start_time
        
        if self.print_timing:
            self.print_timing_stats(timing_stats)
        
        return delta.detach()

    def calculate_integrated_gradient_batch(self, x, P, L):
        """
        Optimized integrated gradient calculation using batch processing
        """
        batch_size, channels, height, width = x.shape
        g_lens = torch.zeros_like(x)
        
        # Prepare batch inputs for gradient computation
        batch_inputs = []
        batch_labels = []
        
        for i in range(batch_size):
            for j in range(self.n):
                for k in range(self.z):
                    scale_factor = 1.0 / (2 ** k)
                    perturbed_x = scale_factor * (x[i:i+1] + self.zeta * P[i, j])
                    batch_inputs.append(perturbed_x)
                    batch_labels.append(L[i, j, k:k+1])
        
        if batch_inputs:
            # Compute gradients in batch
            batch_inputs = torch.cat(batch_inputs, dim=0)
            batch_labels = torch.cat(batch_labels, dim=0)
            batch_grads = self.calculate_gradient_batch(batch_inputs, batch_labels)
            
            # Distribute gradients back
            idx = 0
            for i in range(batch_size):
                for j in range(self.n):
                    for k in range(self.z):
                        g_lens[i] = g_lens[i] + batch_grads[idx]
                        idx += 1
        
        # Normalize
        Z = self.n * self.z
        g_lens = g_lens / Z
        
        return g_lens

    def calculate_average_blended_gradient_batch(self, x, P, g_lens, label):
        """
        Optimized blended gradient calculation using batch processing
        """
        batch_size, channels, height, width = x.shape
        g_mix = torch.zeros_like(x)
        
        # Prepare batch inputs for gradient computation
        batch_inputs = []
        batch_labels = []
        
        for i in range(batch_size):
            for j in range(self.n):
                for k in range(self.m):
                    scale_factor = 1.0 / (2 ** k)
                    perturbed_x = scale_factor * (x[i:i+1] + self.zeta * P[i, j])
                    batch_inputs.append(perturbed_x)
                    batch_labels.append(label[i:i+1])
        
        if batch_inputs:
            # Compute gradients in batch
            batch_inputs = torch.cat(batch_inputs, dim=0)
            batch_labels = torch.cat(batch_labels, dim=0)
            batch_grads = self.calculate_gradient_batch(batch_inputs, batch_labels)
            
            # Distribute gradients back
            idx = 0
            for i in range(batch_size):
                for j in range(self.n):
                    for k in range(self.m):
                        g_mix[i] = g_mix[i] + batch_grads[idx] - self.beta * g_lens[i]
                        idx += 1
        
        # Normalize
        g_mix = g_mix / (self.n * self.m)
        
        return g_mix

    def calculate_gradient_batch(self, x, label):
        """
        Optimized gradient calculation with caching, mixed precision and chunking to avoid OOM
        """
        # Optional cache (disabled by default)
        cache_key = None
        if self.use_cache:
            try:
                cache_key = hash((float(x.sum().detach().cpu()), float(label.sum().detach().cpu())))
                if cache_key in self.gradient_cache:
                    return self.gradient_cache[cache_key]
            except Exception:
                cache_key = None

        total = x.size(0)
        chunk = max(1, min(getattr(self, 'grad_chunk_size', 16), total))
        grads = []

        while True:
            grads.clear()
            oom = False
            try:
                for start in range(0, total, chunk):
                    end = min(total, start + chunk)
                    x_chunk = x[start:end].clone().detach().requires_grad_(True)
                    label_chunk = label[start:end]
                    if self.use_amp and self.scaler is not None:
                        with torch.cuda.amp.autocast():
                            logits = self.model(x_chunk)
                            loss = self.get_loss(logits, label_chunk)
                    else:
                        logits = self.model(x_chunk)
                        loss = self.get_loss(logits, label_chunk)
                    grad_chunk = torch.autograd.grad(loss, x_chunk, retain_graph=False, create_graph=False)[0]
                    grads.append(grad_chunk.detach())
                    del x_chunk, label_chunk, logits, loss, grad_chunk
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
            except RuntimeError as e:
                if 'out of memory' in str(e).lower() and chunk > 1:
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
                    chunk = max(1, chunk // 2)
                    oom = True
                else:
                    raise
            if not oom:
                break

        grad = torch.cat(grads, dim=0)
        if cache_key is not None and grad.numel() <= 1_000_000:
            self.gradient_cache[cache_key] = grad
        return grad

    def get_integrated_logits(self, x, top_k_indices):
        """Calculate integrated logits output"""
        with torch.no_grad():
            logits = self.model(x)
        if top_k_indices.dim() == 1:
            top_k_indices = top_k_indices.unsqueeze(0)
        top_k_logits = torch.gather(logits, 1, top_k_indices)
        return torch.mean(top_k_logits, dim=1, keepdim=True)

    def get_class_gradient(self, x, label):
        """Calculate class gradient"""
        x_clone = x.clone().detach().requires_grad_(True)
        logits = self.model(x_clone)
        class_logits = torch.gather(logits, 1, label.unsqueeze(1))
        grad = torch.autograd.grad(class_logits.sum(), x_clone, retain_graph=False, create_graph=False)[0]
        return grad

    def get_integrated_gradient(self, x, top_k_indices):
        """Calculate integrated gradient for top-k classes"""
        x_clone = x.clone().detach().requires_grad_(True)
        logits = self.model(x_clone)
        
        if top_k_indices.dim() == 1:
            top_k_indices = top_k_indices.unsqueeze(0)
        
        top_k_logits = torch.gather(logits, 1, top_k_indices)
        integrated_logits = torch.mean(top_k_logits, dim=1, keepdim=True)
        grad = torch.autograd.grad(integrated_logits.sum(), x_clone, retain_graph=False, create_graph=False)[0]
        return grad

    def get_update_direction(self, f_topk, omega_y, omega_topk, label, x_adv):
        """Calculate update direction"""
        x = omega_y - omega_topk
        with torch.no_grad():
            logits = self.model(x_adv)
        f_y = torch.gather(logits, 1, label.unsqueeze(1))
        numerator = torch.abs(f_y - f_topk)
        denominator = torch.norm(x, p=1, dim=(1, 2, 3), keepdim=True)
        d_direction = (numerator / (denominator + 1e-8)) * torch.sign(x)
        return d_direction

    def adjust_adversarial_example(self, delta, data, d_direction, alpha):
        """Adjust adversarial example"""
        d_mean = torch.mean(torch.abs(d_direction))
        M = torch.ones_like(d_direction)
        scaling_factor = (alpha * M / (d_mean + 1e-8))
        delta = delta - self.gamma * d_direction * scaling_factor
        return delta

    def generate_random_pixel_blocks(self, data):
        """Generate random pixel-blocks"""
        batch_size, channels, height, width = data.shape
        P = torch.randn(batch_size, self.n, channels, height, width, device=self.device) * 0.1
        return P

    def generate_random_other_class_labels(self, data, P):
        """Generate random other-class labels"""
        batch_size = data.shape[0]
        with torch.no_grad():
            dummy_input = torch.zeros(1, *data.shape[1:], device=self.device)
            dummy_output = self.model(dummy_input)
            num_classes = dummy_output.shape[1]
        
        L = torch.randint(0, num_classes, (batch_size, self.n, self.z), device=self.device)
        return L

    def update_delta(self, delta, data, grad, alpha):
        """Update adversarial perturbation"""
        if self.norm == 'linfty':
            delta = torch.clamp(delta + alpha * grad.sign(), -self.epsilon, self.epsilon)
        else:
            grad_norm = torch.norm(grad.view(grad.size(0), -1), dim=1).view(-1, 1, 1, 1)
            scaled_grad = grad / (grad_norm + 1e-20)
            delta = (delta + scaled_grad * alpha).view(delta.size(0), -1).renorm(p=2, dim=0, maxnorm=self.epsilon).view_as(delta)
        
        delta = clamp(delta, img_min-data, img_max-data)
        return delta.detach().requires_grad_(True)

    def print_timing_stats(self, timing_stats):
        """Print timing statistics"""
        print("\n" + "="*60)
        print("FOOLMIX FAST ATTACK TIMING STATISTICS")
        print("="*60)
        
        print(f"Initialization time: {timing_stats['init_time']:.4f} seconds")
        print(f"\nIteration Statistics (Total: {len(timing_stats['iterations'])} iterations):")
        print("-" * 60)
        
        avg_times = {
            'top_k_time': 0, 'misclassified_adjust_time': 0, 'pixel_blocks_time': 0,
            'other_labels_time': 0, 'integrated_gradient_time': 0, 'blended_gradient_time': 0,
            'momentum_update_time': 0, 'delta_update_time': 0, 'total_iter_time': 0
        }
        
        for iter_timing in timing_stats['iterations']:
            for key in avg_times:
                avg_times[key] += iter_timing[key]
        
        for key in avg_times:
            avg_times[key] /= len(timing_stats['iterations'])
        
        print(f"Average Top-K classification time: {avg_times['top_k_time']:.4f} seconds")
        print(f"Average Misclassified adjustment time: {avg_times['misclassified_adjust_time']:.4f} seconds")
        print(f"Average Pixel blocks generation time: {avg_times['pixel_blocks_time']:.4f} seconds")
        print(f"Average Other labels generation time: {avg_times['other_labels_time']:.4f} seconds")
        print(f"Average Integrated gradient calculation time: {avg_times['integrated_gradient_time']:.4f} seconds")
        print(f"Average Blended gradient calculation time: {avg_times['blended_gradient_time']:.4f} seconds")
        print(f"Average Momentum update time: {avg_times['momentum_update_time']:.4f} seconds")
        print(f"Average Delta update time: {avg_times['delta_update_time']:.4f} seconds")
        print(f"Average Total iteration time: {avg_times['total_iter_time']:.4f} seconds")
        
        print(f"\nTotal attack time: {timing_stats['total_time']:.4f} seconds")
        
        # Calculate inference count
        total_inference = len(timing_stats['iterations']) * (
            1 +  # Top-K classification
            3 +  # Misclassified adjustment (estimated)
            4 +  # Integrated gradient (batch_size * n * z = 2 * 2 * 1)
            12   # Blended gradient (batch_size * n * m = 2 * 2 * 3)
        )
        
        print(f"Estimated total inference count: {total_inference}")
        print(f"Average inference per iteration: {total_inference / len(timing_stats['iterations']):.1f}")
        
        print("="*60)
        
        self.timing_stats = timing_stats

    def get_timing_stats(self):
        """Get timing statistics"""
        if hasattr(self, 'timing_stats'):
            return self.timing_stats
        else:
            return None

    def clear_cache(self):
        """Clear gradient cache"""
        if hasattr(self, 'gradient_cache'):
            self.gradient_cache.clear()
