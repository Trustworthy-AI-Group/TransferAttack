import math

import numpy as np
import torch
import torch.nn.functional as F
from timm.models.vision_transformer import (
    Attention,
    Mlp,
    VisionTransformer,
)

from ..attack import Attack

# Hyperparameter mappings for different model types
hyper_param_map = {
    "type_list": ["vit"],
    "sparse_p_map": {"vit": 0.4},  # Sparsity probability for attention masking
    "shuffle_head_prob_ratio_map": {  # (shuffle_head_prob, shuffle_head_ratio)
        "vit": (0.5, 0.45),  # Probability and ratio for head shuffling
    },
    "moe_param_map": {  # (N, prob) - Mixture of Experts parameters
        "vit": (5, 0.3),  # Maximum number of experts and dropout probability
    },
}


class GlobalState:
    """Global state management class for attack operations"""

    def __init__(self):
        # Hyperparameters for different attack strategies
        self.rest_p = 0.3  # REST token selection probability
        self.sparse_p = None  # Sparsity probability for attention masking
        self.shuffle_head_prob = None  # Probability of shuffling attention heads
        self.shuffle_head_ratio = None  # Ratio of heads to shuffle
        self.moe_N = None  # Maximum number of MoE experts
        self.moe_prob = None  # MoE dropout probability

        # Global states for REST attack - storing query, key, value tokens
        self.q_rest = None  # REST query tokens
        self.k_rest = None  # REST key tokens
        self.v_rest = None  # REST value tokens
        self.robust_tokens = None  # Robust tokens for robust attacks

    def reset_states(self):
        """Reset all stored states to None"""
        self.q_rest = None
        self.k_rest = None
        self.v_rest = None
        self.robust_tokens = None

    def set_rest_tokens(self, q_rest, k_rest, v_rest):
        """Store REST tokens for query, key, and value"""
        self.q_rest = q_rest
        self.k_rest = k_rest
        self.v_rest = v_rest

    def set_robust_tokens(self, robust_tokens):
        """Set robust tokens for attack"""
        self.robust_tokens = robust_tokens

    def setup_params(self, hyper_param_type: str):
        """Setup hyperparameters based on model type"""
        assert hyper_param_type in hyper_param_map["type_list"], (
            f"Only {hyper_param_map['type_list']} are supported"
        )
        self.sparse_p = hyper_param_map["sparse_p_map"][hyper_param_type]
        self.shuffle_head_prob, self.shuffle_head_ratio = hyper_param_map[
            "shuffle_head_prob_ratio_map"
        ][hyper_param_type]
        self.moe_N, self.moe_prob = hyper_param_map["moe_param_map"][hyper_param_type]


# Global state instance
global_state = GlobalState()


def select_op(op_params, num_ops):
    """Select operations based on learned probabilities"""
    prob = F.softmax(op_params, dim=-1)
    op_ids = torch.multinomial(prob, num_ops, replacement=True).tolist()
    return op_ids


def trace_prob(op_params, op_ids):
    """Calculate the probability trace for selected operations"""
    probs = F.softmax(op_params, dim=-1)  # shape: (n_layers, n_ops)
    layer_indices = torch.arange(len(op_ids))[:, None]  # shape: (n_layers, 1)
    selected_probs = probs[layer_indices, op_ids]  # shape: (n_layers, n_sampled_ops)
    tp = torch.prod(selected_probs)
    return tp


class RWAug_Search:
    """class for operation selection"""

    def __init__(self, n, idxs, op_list):
        self.n = n  # Number of operations
        self.idxs = idxs  # Operation indices
        self.op_list = op_list  # List of available operations


def Wrapped_Attention_forward_REST_Attack(self, x: torch.Tensor) -> torch.Tensor:
    """
    REST Attack implementation:
    This attack concatenates benign image features with adversarial examples.
    """
    B, N, C = x.shape
    # Compute query, key, value projections
    qkv = (
        self.qkv(x)
        .reshape(B, N, 3, self.num_heads, self.head_dim)
        .permute(2, 0, 3, 1, 4)
    )
    q, k, v = qkv.unbind(0)
    num_tokens = q.shape[2]
    filling = False

    # Check if REST tokens are already stored
    if global_state.q_rest is not None:
        # Concatenate stored REST tokens with current tokens
        filling = True
        q = torch.cat([q, global_state.q_rest], dim=2)
        k = torch.cat([k, global_state.k_rest], dim=2)
        v = torch.cat([v, global_state.v_rest], dim=2)
    else:
        # Sample tokens for REST storage on first pass
        sample_num_tokens = int(global_state.rest_p * num_tokens)
        num_heads = q.shape[1]

        # Randomly select tokens for each head (excluding CLS token at index 0)
        selected_token_ids = [
            torch.from_numpy(
                np.random.choice(
                    torch.arange(1, num_tokens), sample_num_tokens, replace=False
                )
            )
            for _ in range(num_heads)
        ]
        selected_token_ids = (
            torch.stack(selected_token_ids, dim=0).unsqueeze(0).expand(B, -1, -1)
        )

        # Create batch and head indices for token selection
        batch_indices = (
            torch.arange(B).view(B, 1, 1).expand(-1, num_heads, sample_num_tokens)
        )
        head_indices = (
            torch.arange(num_heads)
            .view(1, num_heads, 1)
            .expand(B, -1, sample_num_tokens)
        )

        # Store selected tokens globally for reuse
        global_state.set_rest_tokens(
            q[batch_indices, head_indices, selected_token_ids],
            k[batch_indices, head_indices, selected_token_ids],
            v[batch_indices, head_indices, selected_token_ids],
        )

    # Apply normalization and scaling
    q, k = self.q_norm(q), self.k_norm(k)
    q = q * self.scale

    # Compute attention weights and apply dropout
    attn = q @ k.transpose(-2, -1)
    attn = attn.softmax(dim=-1)
    attn = self.attn_drop(attn)

    # Apply attention to values
    x = attn @ v

    # If we added REST tokens, remove them from output
    if filling:
        x = x[:, :, :num_tokens]

    # Apply output projection and dropout
    x = x.transpose(1, 2).reshape(B, N, C)
    x = self.proj(x)
    x = self.proj_drop(x)
    return x


def Wrapped_Attention_forward_Sparse_Attack(self, x: torch.Tensor) -> torch.Tensor:
    """
    Sparse Attack implementation
    This attack randomly masks attention weights to disrupt attention patterns
    """
    B, N, C = x.shape
    # Compute query, key, value projections
    qkv = (
        self.qkv(x)
        .reshape(B, N, 3, self.num_heads, self.head_dim)
        .permute(2, 0, 3, 1, 4)
    )
    q, k, v = qkv.unbind(0)

    # Apply normalization and scaling
    q, k = self.q_norm(q), self.k_norm(k)
    q = q * self.scale

    # Compute attention weights
    attn = q @ k.transpose(-2, -1)
    attn = attn.softmax(dim=-1)
    attn = self.attn_drop(attn)

    # Apply sparse masking: randomly zero out attention weights
    attn = attn * (torch.rand_like(attn) > global_state.sparse_p).float()

    # Apply attention to values and project output
    x = attn @ v
    x = x.transpose(1, 2).reshape(B, N, C)
    x = self.proj(x)
    x = self.proj_drop(x)
    return x


def Wrapped_Attention_forward_Shuffle_Attack(self, x: torch.Tensor) -> torch.Tensor:
    """
    Shuffle Attack implementation
    This attack randomly shuffles attention weights across different heads
    """
    B, N, C = x.shape
    # Compute query, key, value projections
    qkv = (
        self.qkv(x)
        .reshape(B, N, 3, self.num_heads, self.head_dim)
        .permute(2, 0, 3, 1, 4)
    )
    q, k, v = qkv.unbind(0)

    # Apply normalization and scaling
    q, k = self.q_norm(q), self.k_norm(k)
    q = q * self.scale

    # Compute attention weights
    attn = q @ k.transpose(-2, -1)
    attn = attn.softmax(dim=-1)
    attn = self.attn_drop(attn)

    # Randomly decide whether to shuffle heads
    if torch.rand(1) < global_state.shuffle_head_prob:
        # Shuffle attention weights across different heads
        num_heads = attn.shape[1]
        shuffled_num_heads = int(num_heads * global_state.shuffle_head_ratio)
        head_indices = torch.randperm(num_heads)[:shuffled_num_heads]
        ordered_head_indices = torch.sort(head_indices)[0]

        # Create a copy and shuffle the selected heads
        copy_attn = attn.clone()
        copy_attn[:, head_indices, :] = attn[:, ordered_head_indices, :]
        attn = copy_attn.clone()

    # Apply attention to values and project output
    x = attn @ v
    x = x.transpose(1, 2).reshape(B, N, C)
    x = self.proj(x)
    x = self.proj_drop(x)
    return x


def Wrapper_FFN_forward_MoE_Attack(self, input):
    """
    Mixture of Experts (MoE) Attack implementation
    This attack simulates multiple expert networks with random dropout
    """
    output = 0.0
    # Randomly select number of experts to use
    current_N = np.random.randint(2, global_state.moe_N + 1)

    # Apply multiple expert transformations
    for n in range(current_N):
        # Standard FFN forward pass
        x = self.fc1(input)  # First linear layer
        x = self.act(x)  # Activation function
        # Apply random dropout based on MoE probability
        x = x * (torch.rand_like(x) > global_state.moe_prob).float()
        x = self.fc2(x)  # Second linear layer
        output += x

    # Average the outputs from all experts
    output = output / current_N
    return output


def vit_forward_features(self, x):
    """
    Modified Vision Transformer forward_features method
    Adds support for robust tokens injection
    """
    # Apply patch embedding
    x = self.patch_embed(x)
    x = self._pos_embed(x)

    # Inject robust tokens if available
    if global_state.robust_tokens is not None:
        x = torch.cat([x, global_state.robust_tokens], dim=1)
    else:
        x = x

    # Apply pre-normalization and transformer blocks
    x = self.norm_pre(x)
    x = self.blocks(x)
    x = self.norm(x)
    return x


class ModelWrapper:
    """
    Model wrapper context manager for applying different attack operations
    This class manages the temporary modification of model forward methods
    """

    def __init__(
        self, attention_modules, ffn_modules, num_layers, model, selected_op_idx_list
    ):
        self.attention_modules = attention_modules  # List of attention modules
        self.ffn_modules = ffn_modules  # List of FFN modules
        self.num_layers = num_layers  # Number of transformer layers
        self.model = model  # The model to wrap
        self.selected_op_idx_list = (
            selected_op_idx_list  # Selected operations per layer
        )
        self._wrapped_ops = None  # Track wrapped operations

    def wrap_attention(self):
        """Apply attack operations to attention and FFN modules"""
        self._wrapped_ops = self.selected_op_idx_list

        for layer_idx in range(self.num_layers):
            selected_op = op_list[self.selected_op_idx_list[layer_idx]]

            # Apply FFN-based attacks
            if selected_op in [Wrapper_FFN_forward_MoE_Attack]:
                self.ffn_modules[layer_idx][1].forward = selected_op.__get__(
                    self.ffn_modules[layer_idx][1]
                )
            # Apply attention-based attacks
            elif selected_op in [
                Wrapped_Attention_forward_REST_Attack,
                Wrapped_Attention_forward_Sparse_Attack,
                Wrapped_Attention_forward_Shuffle_Attack,
            ]:
                self.attention_modules[layer_idx][1].forward = selected_op.__get__(
                    self.attention_modules[layer_idx][1]
                )
            else:
                raise ValueError(f"Unsupported operation: {selected_op}")

    def __enter__(self):
        """Context manager entry: automatically apply wrapper"""
        self.wrap_attention()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit: automatically cleanup"""
        self.cleanup()
        return False  # Don't suppress exceptions

    def cleanup(self):
        """Restore original forward methods"""
        if self._wrapped_ops is not None:
            for layer_idx in range(self.num_layers):
                # Restore original attention forward method
                self.attention_modules[layer_idx][1].forward = self.attention_modules[
                    layer_idx
                ][1].original_forward
                # Restore original FFN forward method
                self.ffn_modules[layer_idx][1].forward = self.ffn_modules[layer_idx][
                    1
                ].original_forward
            self._wrapped_ops = None


def wrap_vit_forward_features(model):
    """
    Replace the forward_features method of VisionTransformer with our custom version
    This enables injection of robust tokens during forward pass
    """
    for name, module in model.named_modules():
        if isinstance(module, VisionTransformer):
            module.forward_features = vit_forward_features.__get__(module)
            return model
    raise Exception("The model does not contain VisionTransformer module")


op_list = [
    Wrapped_Attention_forward_REST_Attack,  # REST token attack
    Wrapped_Attention_forward_Sparse_Attack,  # Sparse attention attack
    Wrapped_Attention_forward_Shuffle_Attack,  # Head shuffling attack
    Wrapper_FFN_forward_MoE_Attack,  # MoE FFN attack
]


class LL2S(Attack):
    """
    LL2S Attack
    Based on 'Harnessing the Computation Redundancy in ViTs to Boost Adversarial Transferability'
    (https://arxiv.org/abs/2504.10804)

    This attack, motivated by computational redundancy in ViTs, incorporates a suite of techniques,
    including attention sparsity manipulation, attention head permutation, clean token regularization,
    ghost MoE diversification, and learning to robustify before the attack.
    A dynamic online learning strategy is also proposed to fully leverage these operations to enhance
    the adversarial transferability.

    Arguments:
        model_name (str): Name of the surrogate model for attack
        epsilon (float): Perturbation budget (L∞ norm)
        alpha (float): Step size for gradient updates
        epoch (int): Number of attack iterations
        decay (float): Momentum decay factor
        targeted (bool): Whether to perform targeted attack
        random_start (bool): Whether to use random initialization for perturbations
        norm (str): Perturbation norm constraint (l2/linfty)
        loss (str): Loss function to use
        device (torch.device): Computation device
        robust_tokens_type (str): Type of robust tokens ('dynamic', 'global', 'none')
        num_robust_tokens (int): Number of robust tokens to use

    Official recommended arguments:
        epsilon=16/255, alpha=1.6/255, epoch=10, decay=1

    Robust-token usage note:
    - `robust_tokens_type="global"` uses precomputed tokens (shape: 400x768) for faster attacks.
    - Users may provide their own robust tokens of shape (num_tokens, token_dim).
    - Our global tokens were produced by ensembling dynamic robust tokens learned over a large image set.
      Download: https://drive.google.com/file/d/1IOtBzdeTA_SABXlyW3f-4ckGQem8sDHs/view?usp=sharing

    Example usage:
        python main.py --input_dir ./path/to/data --output_dir adv_data/l2t/vit_base_patch16_224 --attack ll2s --model=vit_base_patch16_224 --batchsize 1
        python main.py --input_dir ./path/to/data --output_dir adv_data/l2t/vit_base_patch16_224 --eval
    """

    def __init__(
        self,
        model_name,
        epsilon=16 / 255,
        alpha=1.6 / 255,
        epoch=10,
        decay=1.0,
        targeted=False,
        random_start=False,
        norm="linfty",
        loss="crossentropy",
        device=None,
        attack="LL2S",
        robust_tokens_type="global",
        num_robust_tokens=400,
        **kwargs,
    ):
        super().__init__(
            attack, model_name, epsilon, targeted, random_start, norm, loss, device
        )
        self.alpha = alpha
        self.epoch = epoch
        self.decay = decay
        self.num_scale = 10
        self.model_name = model_name

        # Setup model-specific parameters
        if "vit" in model_name:
            self.hyper_param_type = "vit"
            attention_modules, ffn_modules = self.enumerate_module(self.model)
            self.model = wrap_vit_forward_features(self.model)
            self.token_dim = 768  # ViT token dimension
            self.model_op_list = op_list
        else:
            raise ValueError(f"Model {model_name} is not supported yet")

        # Attack configuration
        self.num_tokens = num_robust_tokens
        self.ops_num = 2  # Number of operations to select per iteration
        self.ops_learning_rate = 0.01  # Learning rate for operation parameters

        # Store module references
        self.attention_modules = attention_modules
        self.ffn_modules = ffn_modules
        self.num_attention = len(attention_modules)
        self.num_ffn = len(ffn_modules)

        assert self.num_attention == self.num_ffn, (
            "Number of attention and FFN modules must match"
        )
        self.num_layers = self.num_attention

        self._model_name_ = model_name

        # Robust tokens configuration
        self.robust_tokens_type = robust_tokens_type
        assert self.robust_tokens_type in ["dynamic", "global", "none"], (
            f"robust_tokens_type {robust_tokens_type} is not supported"
        )
        self.prompt_learning_alpha = 1e-2  # Learning rate for dynamic robust tokens

        # Setup global state parameters
        global_state.setup_params(self.hyper_param_type)

    def init_robust_tokens(self, N):
        """Initialize robust token perturbations"""
        delta_shape = (N, self.num_tokens, self.token_dim)
        delta = torch.randn(delta_shape).to(self.device) * 10
        delta.requires_grad = True
        return delta

    def update_robust_tokens(self, delta, grad, **kwargs):
        """Update robust tokens using gradient information"""
        delta = delta - grad.sign() * self.prompt_learning_alpha
        return delta.detach().requires_grad_(True)

    def get_robust_momentum(self, grad, momentum, **kwargs):
        """Calculate momentum for robust token updates"""
        return momentum * self.decay + grad

    def get_loss(self, logits, label, num_copy):
        """
        Calculate loss for the attack

        Args:
            logits: Model output logits
            label: Ground truth labels
            num_copy: Number of copies (for handling multiple scales)
        """
        return (
            -self.loss(logits, label.repeat(num_copy))
            if self.targeted
            else self.loss(logits, label.repeat(num_copy))
        )

    def get_grad(self, loss, delta, **kwargs):
        """Calculate gradients with respect to perturbations"""
        return torch.autograd.grad(loss, delta, retain_graph=False, create_graph=False)[
            0
        ]

    def enumerate_module(self, model):
        """
        Enumerate and store references to attention and FFN modules
        Also stores original forward methods for later restoration
        """
        ffn_modules = []
        attention_modules = []

        for name, module in model.named_modules():
            if isinstance(module, Attention):
                module.original_forward = module.forward  # Store original method
                attention_modules.append((name, module))
            elif isinstance(module, Mlp):
                module.original_forward = module.forward  # Store original method
                ffn_modules.append((name, module))

        return attention_modules, ffn_modules

    def forward(self, data, label, **kwargs):
        """
        Main attack procedure that learns to dynamically select transformations

        Arguments:
            data (N, C, H, W): Input images tensor
            label (N,): Ground-truth labels for untargeted attack
                   or (2, N): [ground-truth, target] labels for targeted attack

        Returns:
            delta: Adversarial perturbations
        """
        # Handle targeted attack labels
        if self.targeted:
            assert len(label) == 2
            label = label[1]  # Use target labels

        # Move data to device
        data = data.clone().detach().to(self.device)
        label = label.clone().detach().to(self.device)

        # Initialize learnable operation parameters
        aug_length = len(self.model_op_list)
        aug_param = torch.nn.Parameter(
            torch.zeros(self.num_layers, aug_length, requires_grad=True),
            requires_grad=True,
        )

        # Initialize adversarial perturbation
        delta = self.init_delta(data)
        momentum = 0

        # Reset global state for new attack
        global_state.reset_states()

        # Initialize robust tokens based on configuration
        if self.robust_tokens_type == "global":
            # Use pre-computed global robust tokens
            if "vit" in self.model_name:
                tensor_filepath = "path/to/vit/global/robust/tokens"
                robust_tokens = (
                    torch.load(tensor_filepath)
                    .to(self.device)
                    .unsqueeze(0)
                    .repeat([data.shape[0], 1, 1])
                    .clone()
                )
            else:
                raise ValueError("No global tokens for model: " + self.model_name)
        elif self.robust_tokens_type == "dynamic":
            # Use dynamically learned robust tokens
            momentum_robust = 0.0
            robust_tokens = self.init_robust_tokens(len(data)).to(self.device)
        else:
            # No robust tokens
            assert self.robust_tokens_type == "none"
            robust_tokens = None

        # Main attack loop
        for epoch_idx in range(self.epoch):
            # Set current robust tokens in global state
            global_state.set_robust_tokens(
                robust_tokens.clone().detach() if robust_tokens is not None else None
            )

            # Initialize containers for probabilities and losses
            aug_probs = []
            losses = []

            # Generate multiple transformation scales
            for scale_idx in range(self.num_scale):
                # Create random walk search instance
                rw_search = RWAug_Search(self.ops_num, [0, 0], self.model_op_list)

                # Select operations based on learned parameters
                augtype = (self.ops_num, np.array(select_op(aug_param, self.ops_num)))

                # Calculate probabilities for selected operations
                for ops_index in range(self.ops_num):
                    aug_prob = trace_prob(aug_param, augtype[1][:, ops_index])
                    aug_probs.append(aug_prob)

                # Update search parameters
                rw_search.n = augtype[0]
                rw_search.idxs = augtype[1]

                # Apply selected operations and compute losses
                for ops_index in range(self.ops_num):
                    selected_ops = rw_search.idxs[:, ops_index]

                    # Use context manager to temporarily modify model
                    with ModelWrapper(
                        self.attention_modules,
                        self.ffn_modules,
                        self.num_layers,
                        self.model,
                        selected_ops,
                    ):
                        # Forward pass with adversarial examples
                        logits = self.get_logits(self.transform(data + delta))

                        # Calculate loss for this transformation
                        losses.append(
                            self.get_loss(
                                logits,
                                label,
                                math.floor((len(logits) + 0.01) / len(label)),
                            ).reshape(1)
                        )

            # Compute average loss across all scales
            loss = torch.sum(torch.cat(losses)) / self.num_scale

            # Calculate gradients for adversarial perturbations
            grad = self.get_grad(loss, delta)

            # Calculate weighted loss for operation parameter updates
            aug_losses = torch.cat(
                [aug_probs[i] * losses[i].reshape(1) for i in range(self.num_scale)]
            )
            aug_loss = torch.sum(aug_losses) / self.num_scale

            # Update operation parameters
            aug_grad = torch.autograd.grad(
                aug_loss, aug_param, retain_graph=False, create_graph=False
            )[0]
            aug_param = aug_param + self.ops_learning_rate * aug_grad

            # Update momentum and adversarial perturbations
            momentum = self.get_momentum(grad, momentum)
            delta = self.update_delta(delta, data, momentum, self.alpha)

            # Update dynamic robust tokens if enabled
            if self.robust_tokens_type == "dynamic":
                # Update robust tokens after updating the adversarial perturbations
                # to ensure they are optimized for the latest adversarial examples
                global_state.set_robust_tokens(robust_tokens)
                robust_logits = self.get_logits(self.transform(data + delta))
                robust_loss = self.get_loss(
                    logits=robust_logits, label=label, num_copy=1
                )
                robust_grad = self.get_grad(robust_loss, robust_tokens)
                momentum_robust = self.get_robust_momentum(
                    robust_grad, momentum=momentum_robust
                )
                robust_tokens = self.update_robust_tokens(
                    robust_tokens, momentum_robust
                )

        # Clean up GPU memory
        torch.cuda.empty_cache()
        return delta.detach()
