import torch
import torch.nn as nn
import torch.nn.functional as F
from ..utils import *
from ..attack import Attack

from typing import Union, Tuple
import torch
import abc

from typing import Optional
import numpy as np
import torch
from PIL import Image
from tqdm import tqdm
from torch import optim

import numpy as np
import torch
from PIL import Image
from typing import Tuple
import numpy as np
import warnings

import torch
try:
    from diffusers import StableDiffusionPipeline, DDIMScheduler
except ImportError as e:
    raise ImportError(
        "The 'diffusers' package is required for DiffAttack. "
        "Please install it with 'pip install diffusers>=0.30.3'."
    ) from e
from PIL import Image
import numpy as np

warnings.filterwarnings("ignore")

class DiffAttack(Attack):
    """
    DiffAttack
    'Diffusion Models for Imperceptible and Transferable Adversarial Attack (TPAMI 2024)'(https://ieeexplore.ieee.org/abstract/document/10716799)

    Arguments:
        model_name (str): the name of surrogate model for attack.
        targeted (bool): targeted/untargeted attack.
        device (torch.device): the device for data. If it is None, the device would be same as model

    Example script:
        python main.py --input_dir ./path/to/data --output_dir adv_data/diffattack/generation --attack diffattack --batchsize=1 --model=resnet50
        python main.py --input_dir ./path/to/data --output_dir adv_data/diffattack/generation --eval
    """

    def __init__(self, model_name, targeted=False, device=None, attack='DiffAttack', checkpoint_path='./path/to/checkpoints', **kwargs):
        self.checkpoint_path = checkpoint_path
        super().__init__(attack, model_name, None, targeted, None, 'l2', 'crossentropy', device)

        print("\033[96m[DiffAttack Notice]\033[0m This implementation is adapted for TransferAttack. Some functions are simplified. For full features, see: https://github.com/WindVChen/DiffAttack/")

        print("\033[91m[IMPORTANT]\033[0m TransferAttack loads torchvision models with IMAGENET1K_V2 weights. Original DiffAttack uses IMAGENET1K_V1. This will cause performance differences from the paper. For original results, use 'models.__dict__[model_name](weights=\"IMAGENET1K_V1\")' in transferattack/utils.py and transferattack/attack.py or use the original DiffAttack repo.")

        print("\033[93m[Usage Reminder]\033[0m DiffAttack ignores epsilon, alpha, epoch, and decay. Resolution is fixed at 224. Batch size must be 1 to avoid OOM errors.")

        pretrained_diffusion_path = os.path.join(self.checkpoint_path, 'stable-diffusion-2-base')
        
        if not os.path.exists(pretrained_diffusion_path):
            raise ValueError("Please download the checkpoint of the 'stable-diffusion-2-base' from https://huggingface.co/, and put it into the path '{}'.".format(self.checkpoint_path))
        
        self.ldm_stable = StableDiffusionPipeline.from_pretrained(pretrained_diffusion_path).to('cuda:0')
        self.ldm_stable.scheduler = DDIMScheduler.from_config(self.ldm_stable.scheduler.config)
        self.diffusion_steps = kwargs.get('diffusion_steps', 20)
        self.start_step = kwargs.get('start_step', 15)
        self.iterations = kwargs.get('iterations', 30)
        self.res = kwargs.get('res', 224)
        self.guidance = kwargs.get('guidance', 2.5)
        self.attack_loss_weight = kwargs.get('attack_loss_weight', 10)
        self.cross_attn_loss_weight = kwargs.get('cross_attn_loss_weight', 10000)
        self.self_attn_loss_weight = kwargs.get('self_attn_loss_weight', 100)
    
    def forward(self, data, label, **kwargs):
        """
        The general attack procedure

        Arguments:
            data: (N, C, H, W) tensor for input images
            labels: (N,) tensor for ground-truth labels if untargetd, otherwise targeted labels
        """
        assert data.shape[0] == 1, "Batch size must be 1"
        assert data.shape[2] == data.shape[3] == self.res, "Input image must be 224x224"

        data = data.clone().detach().to(self.device)

        controller = AttentionControlEdit(self.diffusion_steps, 1., self.res)

        image = Image.fromarray((data[0].cpu().numpy().transpose(1, 2, 0) * 255.).clip(0, 255).astype(np.uint8), 'RGB')

        perturbed = self.diffattack(self.ldm_stable, np.array([int(label[0])]), controller,
                                            num_inference_steps=self.diffusion_steps,
                                            guidance_scale=self.guidance,
                                            image=image,
                                            res=self.res,
                                            start_step=self.start_step,
                                            iterations=self.iterations)
        return torch.from_numpy(perturbed).unsqueeze(0).permute(0, 3, 1, 2).to(self.device) - data

    @torch.enable_grad()
    def diffattack(
            self,
            model,
            label,
            controller,
            num_inference_steps: int = 20,
            guidance_scale: float = 2.5,
            image=None,
            res=224,
            start_step=15,
            iterations=30,
            verbose=True,
            topN=1,
    ):
        label = torch.from_numpy(label).long().cuda()

        model.vae.requires_grad_(False)
        model.text_encoder.requires_grad_(False)
        model.unet.requires_grad_(False)

        classifier = self.model.eval()
        classifier.requires_grad_(False)

        height = width = res

        test_image = image.resize((height, height), resample=Image.LANCZOS)
        test_image = np.float32(test_image) / 255.0
        test_image = test_image[:, :, :3]
        test_image = test_image.transpose((2, 0, 1))
        test_image = torch.from_numpy(test_image).unsqueeze(0)

        pred = classifier(test_image.cuda())
        pred_accuracy_clean = (torch.argmax(pred, 1).detach() == label).sum().item() / len(label)
        print("\nAccuracy on benign examples: {}%".format(pred_accuracy_clean * 100))

        logit = torch.nn.Softmax()(pred)
        print("gt_label:", label[0].item(), "pred_label:", torch.argmax(pred, 1).detach().item(), "pred_clean_logit",
            logit[0, label[0]].item())

        _, pred_labels = pred.topk(topN, largest=True, sorted=True)

        target_prompt = " ".join([TextLabel().refined_Label[label.item()] for i in range(1, topN)])
        prompt = [TextLabel().refined_Label[label.item()] + " " + target_prompt] * 2
        print("prompt generate: ", prompt[0], "\tlabels: ", pred_labels.cpu().numpy().tolist())

        true_label = model.tokenizer.encode(TextLabel().refined_Label[label.item()])
        target_label = model.tokenizer.encode(target_prompt)
        print("decoder: ", true_label, target_label)

        """
                ==========================================
                ============ DDIM Inversion ==============
                === Details please refer to Appendix B ===
                ==========================================
        """
        latent, inversion_latents = ddim_reverse_sample(image, prompt, model,
                                                        num_inference_steps,
                                                        0, res=height)
        inversion_latents = inversion_latents[::-1]

        init_prompt = [prompt[0]]
        batch_size = len(init_prompt)
        latent = inversion_latents[start_step - 1]

        """
                ===============================================================================
                === Good initial reconstruction by optimizing the unconditional embeddings ====
                ======================= Details please refer to Section 3.4 ===================
                ===============================================================================
        """
        max_length = 77
        uncond_input = model.tokenizer(
            [""] * batch_size, padding="max_length", max_length=max_length, return_tensors="pt"
        )

        uncond_embeddings = model.text_encoder(uncond_input.input_ids.to(model.device))[0]

        text_input = model.tokenizer(
            init_prompt,
            padding="max_length",
            max_length=model.tokenizer.model_max_length,
            truncation=True,
            return_tensors="pt",
        )
        text_embeddings = model.text_encoder(text_input.input_ids.to(model.device))[0]

        all_uncond_emb = []
        latent, latents = init_latent(latent, model, height, width, batch_size)

        uncond_embeddings.requires_grad_(True)
        optimizer = optim.AdamW([uncond_embeddings], lr=1e-1)
        loss_func = torch.nn.MSELoss()

        context = torch.cat([uncond_embeddings, text_embeddings])

        #  The DDIM should begin from 1, as the inversion cannot access X_T but only X_{T-1}
        for ind, t in enumerate(tqdm(model.scheduler.timesteps[1 + start_step - 1:], desc="Optimize_uncond_embed")):
            for _ in range(10 + 2 * ind):
                out_latents = diffusion_step(model, latents, context, t, guidance_scale)
                optimizer.zero_grad()
                loss = loss_func(out_latents, inversion_latents[start_step - 1 + ind + 1])
                loss.backward()
                optimizer.step()

                context = [uncond_embeddings, text_embeddings]
                context = torch.cat(context)

            with torch.no_grad():
                latents = diffusion_step(model, latents, context, t, guidance_scale).detach()
                all_uncond_emb.append(uncond_embeddings.detach().clone())

        """
                ==========================================
                ============ Latents Attack ==============
                ==== Details please refer to Section 3 ===
                ==========================================
        """

        uncond_embeddings.requires_grad_(False)

        register_attention_control(model, controller)

        batch_size = len(prompt)

        text_input = model.tokenizer(
            prompt,
            padding="max_length",
            max_length=model.tokenizer.model_max_length,
            truncation=True,
            return_tensors="pt",
        )
        text_embeddings = model.text_encoder(text_input.input_ids.to(model.device))[0]

        context = [[torch.cat([all_uncond_emb[i]] * batch_size), text_embeddings] for i in range(len(all_uncond_emb))]
        context = [torch.cat(i) for i in context]

        original_latent = latent.clone()

        latent.requires_grad_(True)

        optimizer = optim.AdamW([latent], lr=1e-2)
        cross_entro = torch.nn.CrossEntropyLoss()
        init_image = preprocess(image, res)

        init_mask = torch.ones([1, 1, *init_image.shape[-2:]]).cuda()

        pbar = tqdm(range(iterations), desc="Iterations")
        for _, _ in enumerate(pbar):
            controller.loss = 0

            #  The DDIM should begin from 1, as the inversion cannot access X_T but only X_{T-1}
            controller.reset()
            latents = torch.cat([original_latent, latent])


            for ind, t in enumerate(model.scheduler.timesteps[1 + start_step - 1:]):
                latents = diffusion_step(model, latents, context[ind], t, guidance_scale)

            before_attention_map = aggregate_attention(prompt, controller, self.res // 32, ("up", "down"), True, 0, is_cpu=False)
            after_attention_map = aggregate_attention(prompt, controller, self.res // 32, ("up", "down"), True, 1, is_cpu=False)

            before_true_label_attention_map = before_attention_map[:, :, 1: len(true_label) - 1]

            after_true_label_attention_map = after_attention_map[:, :, 1: len(true_label) - 1]

            init_out_image = model.vae.decode(1 / 0.18215 * latents)['sample'][1:] * init_mask + (
                    1 - init_mask) * init_image

            out_image = (init_out_image / 2 + 0.5).clamp(0, 1)

            # For datasets like CUB, Standford Car, the logit should be divided by 10, or there will be gradient Vanishing.
            pred = classifier(out_image)

            attack_loss = - cross_entro(pred, label) * self.attack_loss_weight

            # “Deceive” Strong Diffusion Model. Details please refer to Section 3.3
            variance_cross_attn_loss = after_true_label_attention_map.var() * self.cross_attn_loss_weight

            # Preserve Content Structure. Details please refer to Section 3.4
            self_attn_loss = controller.loss * self.self_attn_loss_weight

            loss = self_attn_loss + attack_loss + variance_cross_attn_loss

            if verbose:
                pbar.set_postfix_str(
                    f"attack_loss: {attack_loss.item():.5f} "
                    f"variance_cross_attn_loss: {variance_cross_attn_loss.item():.5f} "
                    f"self_attn_loss: {self_attn_loss.item():.5f} "
                    f"loss: {loss.item():.5f}")

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        with torch.no_grad():
            controller.loss = 0
            controller.reset()

            latents = torch.cat([original_latent, latent])

            for ind, t in enumerate(model.scheduler.timesteps[1 + start_step - 1:]):
                latents = diffusion_step(model, latents, context[ind], t, guidance_scale)

        out_image = model.vae.decode(1 / 0.18215 * latents.detach())['sample'][1:] * init_mask + (
                1 - init_mask) * init_image
        out_image = (out_image / 2 + 0.5).clamp(0, 1)

        pred = classifier(out_image)
        pred_label = torch.argmax(pred, 1).detach()
        pred_accuracy = (torch.argmax(pred, 1).detach() == label).sum().item() / len(label)
        print("Accuracy on adversarial examples: {}%".format(pred_accuracy * 100))

        logit = torch.nn.Softmax()(pred)
        print("after_pred:", pred_label, logit[0, pred_label[0]])
        print("after_true:", label, logit[0, label[0]])

        """
                ==========================================
                ============= Visualization ==============
                ==========================================
        """

        image = latent2image(model.vae, latents.detach())

        real = (init_image / 2 + 0.5).clamp(0, 1).permute(0, 2, 3, 1).cpu().numpy()
        perturbed = image[1:].astype(np.float32) / 255 * init_mask.squeeze().unsqueeze(-1).cpu().numpy() + (
                1 - init_mask.squeeze().unsqueeze(-1).cpu().numpy()) * real
        
        reset_attention_control(model)

        return perturbed[0]


def aggregate_attention(prompts, attention_store, res: int, from_where, is_cross: bool, select: int, is_cpu=True):
    out = []
    attention_maps = attention_store.get_average_attention()
    num_pixels = res ** 2
    for location in from_where:
        for item in attention_maps[f"{location}_{'cross' if is_cross else 'self'}"]:
            if item.shape[1] == num_pixels:
                cross_maps = item.reshape(len(prompts), -1, res, res, item.shape[-1])[select]
                out.append(cross_maps)
    out = torch.cat(out, dim=0)
    out = out.sum(0) / out.shape[0]
    return out.cpu() if is_cpu else out

def preprocess(image, res=512):
    image = image.resize((res, res), resample=Image.LANCZOS)
    image = np.array(image).astype(np.float32) / 255.0
    image = image[None].transpose(0, 3, 1, 2)
    image = torch.from_numpy(image)[:, :3, :, :].cuda()
    return 2.0 * image - 1.0


def encoder(image, model, res=512):
    generator = torch.Generator().manual_seed(8888)
    image = preprocess(image, res)
    gpu_generator = torch.Generator(device=image.device)
    gpu_generator.manual_seed(generator.initial_seed())
    return 0.18215 * model.vae.encode(image).latent_dist.sample(generator=gpu_generator)


@torch.no_grad()
def ddim_reverse_sample(image, prompt, model, num_inference_steps: int = 20, guidance_scale: float = 2.5,
                        res=512):
    """
            ==========================================
            ============ DDIM Inversion ==============
            ==========================================
    """
    batch_size = 1

    max_length = 77
    uncond_input = model.tokenizer(
        [""] * batch_size, padding="max_length", max_length=max_length, return_tensors="pt"
    )
    uncond_embeddings = model.text_encoder(uncond_input.input_ids.to(model.device))[0]

    text_input = model.tokenizer(
        prompt[0],
        padding="max_length",
        max_length=model.tokenizer.model_max_length,
        truncation=True,
        return_tensors="pt",
    )
    text_embeddings = model.text_encoder(text_input.input_ids.to(model.device))[0]

    context = [uncond_embeddings, text_embeddings]
    context = torch.cat(context)

    model.scheduler.set_timesteps(num_inference_steps)

    latents = encoder(image, model, res=res)
    timesteps = model.scheduler.timesteps.flip(0)

    all_latents = [latents]

    #  Not inverse the last step, as the alpha_bar_next will be set to 0 which is not aligned to its real value (~0.003)
    #  and this will lead to a bad result.
    for t in tqdm(timesteps[:-1], desc="DDIM_inverse"):
        latents_input = torch.cat([latents] * 2)
        noise_pred = model.unet(latents_input, t, encoder_hidden_states=context)["sample"]

        noise_pred_uncond, noise_prediction_text = noise_pred.chunk(2)
        noise_pred = noise_pred_uncond + guidance_scale * (noise_prediction_text - noise_pred_uncond)

        next_timestep = t + model.scheduler.config.num_train_timesteps // model.scheduler.num_inference_steps
        alpha_bar_next = model.scheduler.alphas_cumprod[next_timestep] \
            if next_timestep <= model.scheduler.config.num_train_timesteps else torch.tensor(0.0)

        "leverage reversed_x0"
        reverse_x0 = (1 / torch.sqrt(model.scheduler.alphas_cumprod[t]) * (
                latents - noise_pred * torch.sqrt(1 - model.scheduler.alphas_cumprod[t])))

        latents = reverse_x0 * torch.sqrt(alpha_bar_next) + torch.sqrt(1 - alpha_bar_next) * noise_pred

        all_latents.append(latents)

    #  all_latents[N] -> N: DDIM steps  (X_{T-1} ~ X_0)
    return latents, all_latents


def register_attention_control(model, controller):
    def ca_forward(self, place_in_unet):
        def forward(
            hidden_states: torch.FloatTensor,
            encoder_hidden_states: Optional[torch.FloatTensor] = None,
            attention_mask: Optional[torch.FloatTensor] = None,
            temb: Optional[torch.FloatTensor] = None,
            # scale: float = 1.0,
        ):
            if self.spatial_norm is not None:
                hidden_states = self.spatial_norm(hidden_states, temb)

            batch_size, sequence_length, _ = (
                hidden_states.shape
                if encoder_hidden_states is None
                else encoder_hidden_states.shape
            )

            if attention_mask is not None:
                attention_mask = self.prepare_attention_mask(
                    attention_mask, sequence_length, batch_size
                )
                # scaled_dot_product_attention expects attention_mask shape to be
                # (batch, heads, source_length, target_length)
                attention_mask = attention_mask.view(
                    batch_size, self.heads, -1, attention_mask.shape[-1]
                )  # type: ignore

            if self.group_norm is not None:
                hidden_states = self.group_norm(
                    hidden_states.transpose(1, 2)
                ).transpose(1, 2)

            query = self.to_q(hidden_states)

            is_cross = encoder_hidden_states is not None
            if encoder_hidden_states is None:
                encoder_hidden_states = hidden_states
            elif self.norm_cross:
                encoder_hidden_states = self.norm_encoder_hidden_states(
                    encoder_hidden_states
                )
            key = self.to_k(encoder_hidden_states)
            value = self.to_v(encoder_hidden_states)

            def reshape_heads_to_batch_dim(tensor):
                batch_size, seq_len, dim = tensor.shape
                head_size = self.heads
                tensor = tensor.reshape(
                    batch_size, seq_len, head_size, dim // head_size
                )
                tensor = tensor.permute(0, 2, 1, 3).reshape(
                    batch_size * head_size, seq_len, dim // head_size
                )
                return tensor

            query = reshape_heads_to_batch_dim(query)
            key = reshape_heads_to_batch_dim(key)
            value = reshape_heads_to_batch_dim(value)

            sim = torch.einsum("b i d, b j d -> b i j", query, key) * self.scale
            attn = sim.softmax(dim=-1)
            attn = controller(attn, is_cross, place_in_unet)
            out = torch.einsum("b i j, b j d -> b i d", attn, value)

            def reshape_batch_dim_to_heads(tensor):
                batch_size, seq_len, dim = tensor.shape
                head_size = self.heads
                tensor = tensor.reshape(
                    batch_size // head_size, head_size, seq_len, dim
                )
                tensor = tensor.permute(0, 2, 1, 3).reshape(
                    batch_size // head_size, seq_len, dim * head_size
                )
                return tensor

            out = reshape_batch_dim_to_heads(out)
            out = self.to_out[0](out)
            out = self.to_out[1](out)

            out = out / self.rescale_output_factor
            return out

        return forward

    def register_recr(net_, count, place_in_unet):
        if net_.__class__.__name__ == "Attention":
            net_.forward = ca_forward(net_, place_in_unet)
            return count + 1
        elif hasattr(net_, "children"):
            for net__ in net_.children():
                count = register_recr(net__, count, place_in_unet)
        return count

    cross_att_count = 0
    sub_nets = model.unet.named_children()
    for net in sub_nets:
        if "down" in net[0]:
            cross_att_count += register_recr(net[1], 0, "down")
        elif "up" in net[0]:
            cross_att_count += register_recr(net[1], 0, "up")
        elif "mid" in net[0]:
            cross_att_count += register_recr(net[1], 0, "mid")
    controller.num_att_layers = cross_att_count


def reset_attention_control(model):
    def ca_forward(self):
        def forward(
            hidden_states: torch.FloatTensor,
            encoder_hidden_states: Optional[torch.FloatTensor] = None,
            attention_mask: Optional[torch.FloatTensor] = None,
            temb: Optional[torch.FloatTensor] = None,
            # scale: float = 1.0,
        ):
            if self.spatial_norm is not None:
                hidden_states = self.spatial_norm(hidden_states, temb)

            batch_size, sequence_length, _ = (
                hidden_states.shape
                if encoder_hidden_states is None
                else encoder_hidden_states.shape
            )

            if attention_mask is not None:
                attention_mask = self.prepare_attention_mask(
                    attention_mask, sequence_length, batch_size
                )
                # scaled_dot_product_attention expects attention_mask shape to be
                # (batch, heads, source_length, target_length)
                attention_mask = attention_mask.view(
                    batch_size, self.heads, -1, attention_mask.shape[-1]
                )  # type: ignore

            if self.group_norm is not None:
                hidden_states = self.group_norm(
                    hidden_states.transpose(1, 2)
                ).transpose(1, 2)

            query = self.to_q(hidden_states)
            if encoder_hidden_states is None:
                encoder_hidden_states = hidden_states
            elif self.norm_cross:
                encoder_hidden_states = self.norm_encoder_hidden_states(
                    encoder_hidden_states
                )

            key = self.to_k(encoder_hidden_states)
            value = self.to_v(encoder_hidden_states)

            def reshape_heads_to_batch_dim(tensor):
                batch_size, seq_len, dim = tensor.shape
                head_size = self.heads
                tensor = tensor.reshape(
                    batch_size, seq_len, head_size, dim // head_size
                )
                tensor = tensor.permute(0, 2, 1, 3).reshape(
                    batch_size * head_size, seq_len, dim // head_size
                )
                return tensor

            query = reshape_heads_to_batch_dim(query)
            key = reshape_heads_to_batch_dim(key)
            value = reshape_heads_to_batch_dim(value)

            sim = torch.einsum("b i d, b j d -> b i j", query, key) * self.scale

            attn = sim.softmax(dim=-1)
            out = torch.einsum("b i j, b j d -> b i d", attn, value)

            def reshape_batch_dim_to_heads(tensor):
                batch_size, seq_len, dim = tensor.shape
                head_size = self.heads
                tensor = tensor.reshape(
                    batch_size // head_size, head_size, seq_len, dim
                )
                tensor = tensor.permute(0, 2, 1, 3).reshape(
                    batch_size // head_size, seq_len, dim * head_size
                )
                return tensor

            out = reshape_batch_dim_to_heads(out)
            out = self.to_out[0](out)
            out = self.to_out[1](out)

            out = out / self.rescale_output_factor

            return out

        return forward

    def register_recr(net_):
        if net_.__class__.__name__ == "Attention":
            net_.forward = ca_forward(net_)
        elif hasattr(net_, "children"):
            for net__ in net_.children():
                register_recr(net__)

    sub_nets = model.unet.named_children()
    for net in sub_nets:
        if "down" in net[0]:
            register_recr(net[1])
        elif "up" in net[0]:
            register_recr(net[1])
        elif "mid" in net[0]:
            register_recr(net[1])


def init_latent(latent, model, height, width, batch_size):
    latents = latent.expand(batch_size, model.unet.in_channels, height // 8, width // 8).to(model.device)
    return latent, latents


def diffusion_step(model, latents, context, t, guidance_scale):
    latents_input = torch.cat([latents] * 2)
    noise_pred = model.unet(latents_input, t, encoder_hidden_states=context)["sample"]
    noise_pred_uncond, noise_prediction_text = noise_pred.chunk(2)
    noise_pred = noise_pred_uncond + guidance_scale * (noise_prediction_text - noise_pred_uncond)
    latents = model.scheduler.step(noise_pred, t, latents)["prev_sample"]
    return latents


def latent2image(vae, latents):
    latents = 1 / 0.18215 * latents
    image = vae.decode(latents)['sample']
    image = (image / 2 + 0.5).clamp(0, 1)
    image = image.cpu().permute(0, 2, 3, 1).numpy()
    image = (image * 255).astype(np.uint8)
    return image


class AttentionControl(abc.ABC):
    def __init__(self):
        self.cur_step = 0
        self.num_att_layers = -1
        self.cur_att_layer = 0

    def between_steps(self):
        return

    def reset(self):
        self.cur_step = 0
        self.cur_att_layer = 0

    @abc.abstractmethod
    def forward(self, attn, is_cross: bool, place_in_unet: str):
        raise NotImplementedError

    def __call__(self, attn, is_cross: bool, place_in_unet: str):
        if self.cur_att_layer >= 0:
            h = attn.shape[0]
            self.forward(attn[h // 2:], is_cross, place_in_unet)

        self.cur_att_layer += 1
        if self.cur_att_layer == self.num_att_layers:
            self.cur_att_layer = 0
            self.cur_step += 1
            self.between_steps()
        return attn


class AttentionStore(AttentionControl):
    def __init__(self, res):
        super(AttentionStore, self).__init__()
        self.step_store = self.get_empty_store()
        self.attention_store = {}
        self.res = res

    @staticmethod
    def get_empty_store():
        return {"down_cross": [], "mid_cross": [], "up_cross": [],
                "down_self": [], "mid_self": [], "up_self": []}

    def forward(self, attn, is_cross: bool, place_in_unet: str):
        key = f"{place_in_unet}_{'cross' if is_cross else 'self'}"
        if attn.shape[1] <= (self.res // 16) ** 2:  # avoid memory overhead
            self.step_store[key].append(attn)
        return attn

    def between_steps(self):
        if len(self.attention_store) == 0:
            self.attention_store = self.step_store
        else:
            for key in self.attention_store:
                for i in range(len(self.attention_store[key])):
                    self.attention_store[key][i] = self.step_store[key][i] + self.attention_store[key][i]
        self.step_store = self.get_empty_store()

    def get_average_attention(self):
        average_attention = {key: [item / self.cur_step for item in self.attention_store[key]] for key in
                             self.attention_store}
        return average_attention

    def reset(self):
        super(AttentionStore, self).reset()
        self.step_store = self.get_empty_store()
        self.attention_store = {}


class AttentionControlEdit(AttentionStore, abc.ABC):
    def __init__(self, num_steps: int,
                 self_replace_steps: Union[float, Tuple[float, float]], res):
        super(AttentionControlEdit, self).__init__(res)
        self.batch_size = 2
        if type(self_replace_steps) is float:
            self_replace_steps = 0, self_replace_steps
        self.num_self_replace = int(num_steps * self_replace_steps[0]), int(num_steps * self_replace_steps[1])
        self.loss = 0
        self.criterion = torch.nn.MSELoss()

    def forward(self, attn, is_cross: bool, place_in_unet: str):
        super(AttentionControlEdit, self).forward(attn, is_cross, place_in_unet)
        if is_cross or (self.num_self_replace[0] <= self.cur_step < self.num_self_replace[1]):
            h = attn.shape[0] // (self.batch_size)
            attn = attn.reshape(self.batch_size, h, *attn.shape[1:])
            attn_base, attn_repalce = attn[0], attn[1:]
            if not is_cross:
                """
                        ==========================================
                        ========= Self Attention Control =========
                        === Details please refer to Section 3.4 ==
                        ==========================================
                """
                self.loss += self.criterion(attn[1:], self.replace_self_attention(attn_base, attn_repalce))
            attn = attn.reshape(self.batch_size * h, *attn.shape[2:])
        return attn

    def replace_self_attention(self, attn_base, att_replace):
        return attn_base.unsqueeze(0).expand(att_replace.shape[0], *attn_base.shape)


class TextLabel:
    Label = {0: 'tench, Tinca tinca',
         1: 'goldfish, Carassius auratus',
         2: 'great white shark, white shark, man-eater, man-eating shark, Carcharodon carcharias',
         3: 'tiger shark, Galeocerdo cuvieri',
         4: 'hammerhead, hammerhead shark',
         5: 'electric ray, crampfish, numbfish, torpedo',
         6: 'stingray',
         7: 'cock',
         8: 'hen',
         9: 'ostrich, Struthio camelus',
         10: 'brambling, Fringilla montifringilla',
         11: 'goldfinch, Carduelis carduelis',
         12: 'house finch, linnet, Carpodacus mexicanus',
         13: 'junco, snowbird',
         14: 'indigo bunting, indigo finch, indigo bird, Passerina cyanea',
         15: 'robin, American robin, Turdus migratorius',
         16: 'bulbul',
         17: 'jay',
         18: 'magpie',
         19: 'chickadee',
         20: 'water ouzel, dipper',
         21: 'kite',
         22: 'bald eagle, American eagle, Haliaeetus leucocephalus',
         23: 'vulture',
         24: 'great grey owl, great gray owl, Strix nebulosa',
         25: 'European fire salamander, Salamandra salamandra',
         26: 'common newt, Triturus vulgaris',
         27: 'eft',
         28: 'spotted salamander, Ambystoma maculatum',
         29: 'axolotl, mud puppy, Ambystoma mexicanum',
         30: 'bullfrog, Rana catesbeiana',
         31: 'tree frog, tree-frog',
         32: 'tailed frog, bell toad, ribbed toad, tailed toad, Ascaphus trui',
         33: 'loggerhead, loggerhead turtle, Caretta caretta',
         34: 'leatherback turtle, leatherback, leathery turtle, Dermochelys coriacea',
         35: 'mud turtle',
         36: 'terrapin',
         37: 'box turtle, box tortoise',
         38: 'banded gecko',
         39: 'common iguana, iguana, Iguana iguana',
         40: 'American chameleon, anole, Anolis carolinensis',
         41: 'whiptail, whiptail lizard',
         42: 'agama',
         43: 'frilled lizard, Chlamydosaurus kingi',
         44: 'alligator lizard',
         45: 'Gila monster, Heloderma suspectum',
         46: 'green lizard, Lacerta viridis',
         47: 'African chameleon, Chamaeleo chamaeleon',
         48: 'Komodo dragon, Komodo lizard, dragon lizard, giant lizard, Varanus komodoensis',
         49: 'African crocodile, Nile crocodile, Crocodylus niloticus',
         50: 'American alligator, Alligator mississipiensis',
         51: 'triceratops',
         52: 'thunder snake, worm snake, Carphophis amoenus',
         53: 'ringneck snake, ring-necked snake, ring snake',
         54: 'hognose snake, puff adder, sand viper',
         55: 'green snake, grass snake',
         56: 'king snake, kingsnake',
         57: 'garter snake, grass snake',
         58: 'water snake',
         59: 'vine snake',
         60: 'night snake, Hypsiglena torquata',
         61: 'boa constrictor, Constrictor constrictor',
         62: 'rock python, rock snake, Python sebae',
         63: 'Indian cobra, Naja naja',
         64: 'green mamba',
         65: 'sea snake',
         66: 'horned viper, cerastes, sand viper, horned asp, Cerastes cornutus',
         67: 'diamondback, diamondback rattlesnake, Crotalus adamanteus',
         68: 'sidewinder, horned rattlesnake, Crotalus cerastes',
         69: 'trilobite',
         70: 'harvestman, daddy longlegs, Phalangium opilio',
         71: 'scorpion',
         72: 'black and gold garden spider, Argiope aurantia',
         73: 'barn spider, Araneus cavaticus',
         74: 'garden spider, Aranea diademata',
         75: 'black widow, Latrodectus mactans',
         76: 'tarantula',
         77: 'wolf spider, hunting spider',
         78: 'tick',
         79: 'centipede',
         80: 'black grouse',
         81: 'ptarmigan',
         82: 'ruffed grouse, partridge, Bonasa umbellus',
         83: 'prairie chicken, prairie grouse, prairie fowl',
         84: 'peacock',
         85: 'quail',
         86: 'partridge',
         87: 'African grey, African gray, Psittacus erithacus',
         88: 'macaw',
         89: 'sulphur-crested cockatoo, Kakatoe galerita, Cacatua galerita',
         90: 'lorikeet',
         91: 'coucal',
         92: 'bee eater',
         93: 'hornbill',
         94: 'hummingbird',
         95: 'jacamar',
         96: 'toucan',
         97: 'drake',
         98: 'red-breasted merganser, Mergus serrator',
         99: 'goose',
         100: 'black swan, Cygnus atratus',
         101: 'tusker',
         102: 'echidna, spiny anteater, anteater',
         103: 'platypus, duckbill, duckbilled platypus, duck-billed platypus, Ornithorhynchus anatinus',
         104: 'wallaby, brush kangaroo',
         105: 'koala, koala bear, kangaroo bear, native bear, Phascolarctos cinereus',
         106: 'wombat',
         107: 'jellyfish',
         108: 'sea anemone, anemone',
         109: 'brain coral',
         110: 'flatworm, platyhelminth',
         111: 'nematode, nematode worm, roundworm',
         112: 'conch',
         113: 'snail',
         114: 'slug',
         115: 'sea slug, nudibranch',
         116: 'chiton, coat-of-mail shell, sea cradle, polyplacophore',
         117: 'chambered nautilus, pearly nautilus, nautilus',
         118: 'Dungeness crab, Cancer magister',
         119: 'rock crab, Cancer irroratus',
         120: 'fiddler crab',
         121: 'king crab, Alaska crab, Alaskan king crab, Alaska king crab, Paralithodes camtschatica',
         122: 'American lobster, Northern lobster, Maine lobster, Homarus americanus',
         123: 'spiny lobster, langouste, rock lobster, crawfish, crayfish, sea crawfish',
         124: 'crayfish, crawfish, crawdad, crawdaddy',
         125: 'hermit crab',
         126: 'isopod',
         127: 'white stork, Ciconia ciconia',
         128: 'black stork, Ciconia nigra',
         129: 'spoonbill',
         130: 'flamingo',
         131: 'little blue heron, Egretta caerulea',
         132: 'American egret, great white heron, Egretta albus',
         133: 'bittern',
         134: 'crane',
         135: 'limpkin, Aramus pictus',
         136: 'European gallinule, Porphyrio porphyrio',
         137: 'American coot, marsh hen, mud hen, water hen, Fulica americana',
         138: 'bustard',
         139: 'ruddy turnstone, Arenaria interpres',
         140: 'red-backed sandpiper, dunlin, Erolia alpina',
         141: 'redshank, Tringa totanus',
         142: 'dowitcher',
         143: 'oystercatcher, oyster catcher',
         144: 'pelican',
         145: 'king penguin, Aptenodytes patagonica',
         146: 'albatross, mollymawk',
         147: 'grey whale, gray whale, devilfish, Eschrichtius gibbosus, Eschrichtius robustus',
         148: 'killer whale, killer, orca, grampus, sea wolf, Orcinus orca',
         149: 'dugong, Dugong dugon',
         150: 'sea lion',
         151: 'Chihuahua',
         152: 'Japanese spaniel',
         153: 'Maltese dog, Maltese terrier, Maltese',
         154: 'Pekinese, Pekingese, Peke',
         155: 'Shih-Tzu',
         156: 'Blenheim spaniel',
         157: 'papillon',
         158: 'toy terrier',
         159: 'Rhodesian ridgeback',
         160: 'Afghan hound, Afghan',
         161: 'basset, basset hound',
         162: 'beagle',
         163: 'bloodhound, sleuthhound',
         164: 'bluetick',
         165: 'black-and-tan coonhound',
         166: 'Walker hound, Walker foxhound',
         167: 'English foxhound',
         168: 'redbone',
         169: 'borzoi, Russian wolfhound',
         170: 'Irish wolfhound',
         171: 'Italian greyhound',
         172: 'whippet',
         173: 'Ibizan hound, Ibizan Podenco',
         174: 'Norwegian elkhound, elkhound',
         175: 'otterhound, otter hound',
         176: 'Saluki, gazelle hound',
         177: 'Scottish deerhound, deerhound',
         178: 'Weimaraner',
         179: 'Staffordshire bullterrier, Staffordshire bull terrier',
         180: 'American Staffordshire terrier, Staffordshire terrier, American pit bull terrier, pit bull terrier',
         181: 'Bedlington terrier',
         182: 'Border terrier',
         183: 'Kerry blue terrier',
         184: 'Irish terrier',
         185: 'Norfolk terrier',
         186: 'Norwich terrier',
         187: 'Yorkshire terrier',
         188: 'wire-haired fox terrier',
         189: 'Lakeland terrier',
         190: 'Sealyham terrier, Sealyham',
         191: 'Airedale, Airedale terrier',
         192: 'cairn, cairn terrier',
         193: 'Australian terrier',
         194: 'Dandie Dinmont, Dandie Dinmont terrier',
         195: 'Boston bull, Boston terrier',
         196: 'miniature schnauzer',
         197: 'giant schnauzer',
         198: 'standard schnauzer',
         199: 'Scotch terrier, Scottish terrier, Scottie',
         200: 'Tibetan terrier, chrysanthemum dog',
         201: 'silky terrier, Sydney silky',
         202: 'soft-coated wheaten terrier',
         203: 'West Highland white terrier',
         204: 'Lhasa, Lhasa apso',
         205: 'flat-coated retriever',
         206: 'curly-coated retriever',
         207: 'golden retriever',
         208: 'Labrador retriever',
         209: 'Chesapeake Bay retriever',
         210: 'German short-haired pointer',
         211: 'vizsla, Hungarian pointer',
         212: 'English setter',
         213: 'Irish setter, red setter',
         214: 'Gordon setter',
         215: 'Brittany spaniel',
         216: 'clumber, clumber spaniel',
         217: 'English springer, English springer spaniel',
         218: 'Welsh springer spaniel',
         219: 'cocker spaniel, English cocker spaniel, cocker',
         220: 'Sussex spaniel',
         221: 'Irish water spaniel',
         222: 'kuvasz',
         223: 'schipperke',
         224: 'groenendael',
         225: 'malinois',
         226: 'briard',
         227: 'kelpie',
         228: 'komondor',
         229: 'Old English sheepdog, bobtail',
         230: 'Shetland sheepdog, Shetland sheep dog, Shetland',
         231: 'collie',
         232: 'Border collie',
         233: 'Bouvier des Flandres, Bouviers des Flandres',
         234: 'Rottweiler',
         235: 'German shepherd, German shepherd dog, German police dog, alsatian',
         236: 'Doberman, Doberman pinscher',
         237: 'miniature pinscher',
         238: 'Greater Swiss Mountain dog',
         239: 'Bernese mountain dog',
         240: 'Appenzeller',
         241: 'EntleBucher',
         242: 'boxer',
         243: 'bull mastiff',
         244: 'Tibetan mastiff',
         245: 'French bulldog',
         246: 'Great Dane',
         247: 'Saint Bernard, St Bernard',
         248: 'Eskimo dog, husky',
         249: 'malamute, malemute, Alaskan malamute',
         250: 'Siberian husky',
         251: 'dalmatian, coach dog, carriage dog',
         252: 'affenpinscher, monkey pinscher, monkey dog',
         253: 'basenji',
         254: 'pug, pug-dog',
         255: 'Leonberg',
         256: 'Newfoundland, Newfoundland dog',
         257: 'Great Pyrenees',
         258: 'Samoyed, Samoyede',
         259: 'Pomeranian',
         260: 'chow, chow chow',
         261: 'keeshond',
         262: 'Brabancon griffon',
         263: 'Pembroke, Pembroke Welsh corgi',
         264: 'Cardigan, Cardigan Welsh corgi',
         265: 'toy poodle',
         266: 'miniature poodle',
         267: 'standard poodle',
         268: 'Mexican hairless',
         269: 'timber wolf, grey wolf, gray wolf, Canis lupus',
         270: 'white wolf, Arctic wolf, Canis lupus tundrarum',
         271: 'red wolf, maned wolf, Canis rufus, Canis niger',
         272: 'coyote, prairie wolf, brush wolf, Canis latrans',
         273: 'dingo, warrigal, warragal, Canis dingo',
         274: 'dhole, Cuon alpinus',
         275: 'African hunting dog, hyena dog, Cape hunting dog, Lycaon pictus',
         276: 'hyena, hyaena',
         277: 'red fox, Vulpes vulpes',
         278: 'kit fox, Vulpes macrotis',
         279: 'Arctic fox, white fox, Alopex lagopus',
         280: 'grey fox, gray fox, Urocyon cinereoargenteus',
         281: 'tabby, tabby cat',
         282: 'tiger cat',
         283: 'Persian cat',
         284: 'Siamese cat, Siamese',
         285: 'Egyptian cat',
         286: 'cougar, puma, catamount, mountain lion, painter, panther, Felis concolor',
         287: 'lynx, catamount',
         288: 'leopard, Panthera pardus',
         289: 'snow leopard, ounce, Panthera uncia',
         290: 'jaguar, panther, Panthera onca, Felis onca',
         291: 'lion, king of beasts, Panthera leo',
         292: 'tiger, Panthera tigris',
         293: 'cheetah, chetah, Acinonyx jubatus',
         294: 'brown bear, bruin, Ursus arctos',
         295: 'American black bear, black bear, Ursus americanus, Euarctos americanus',
         296: 'ice bear, polar bear, Ursus Maritimus, Thalarctos maritimus',
         297: 'sloth bear, Melursus ursinus, Ursus ursinus',
         298: 'mongoose',
         299: 'meerkat, mierkat',
         300: 'tiger beetle',
         301: 'ladybug, ladybeetle, lady beetle, ladybird, ladybird beetle',
         302: 'ground beetle, carabid beetle',
         303: 'long-horned beetle, longicorn, longicorn beetle',
         304: 'leaf beetle, chrysomelid',
         305: 'dung beetle',
         306: 'rhinoceros beetle',
         307: 'weevil',
         308: 'fly',
         309: 'bee',
         310: 'ant, emmet, pismire',
         311: 'grasshopper, hopper',
         312: 'cricket',
         313: 'walking stick, walkingstick, stick insect',
         314: 'cockroach, roach',
         315: 'mantis, mantid',
         316: 'cicada, cicala',
         317: 'leafhopper',
         318: 'lacewing, lacewing fly',
         319: "dragonfly, darning needle, devil's darning needle, sewing needle, snake feeder, snake doctor, mosquito hawk, skeeter hawk",
         320: 'damselfly',
         321: 'admiral',
         322: 'ringlet, ringlet butterfly',
         323: 'monarch, monarch butterfly, milkweed butterfly, Danaus plexippus',
         324: 'cabbage butterfly',
         325: 'sulphur butterfly, sulfur butterfly',
         326: 'lycaenid, lycaenid butterfly',
         327: 'starfish, sea star',
         328: 'sea urchin',
         329: 'sea cucumber, holothurian',
         330: 'wood rabbit, cottontail, cottontail rabbit',
         331: 'hare',
         332: 'Angora, Angora rabbit',
         333: 'hamster',
         334: 'porcupine, hedgehog',
         335: 'fox squirrel, eastern fox squirrel, Sciurus niger',
         336: 'marmot',
         337: 'beaver',
         338: 'guinea pig, Cavia cobaya',
         339: 'sorrel',
         340: 'zebra',
         341: 'hog, pig, grunter, squealer, Sus scrofa',
         342: 'wild boar, boar, Sus scrofa',
         343: 'warthog',
         344: 'hippopotamus, hippo, river horse, Hippopotamus amphibius',
         345: 'ox',
         346: 'water buffalo, water ox, Asiatic buffalo, Bubalus bubalis',
         347: 'bison',
         348: 'ram, tup',
         349: 'bighorn, bighorn sheep, cimarron, Rocky Mountain bighorn, Rocky Mountain sheep, Ovis canadensis',
         350: 'ibex, Capra ibex',
         351: 'hartebeest',
         352: 'impala, Aepyceros melampus',
         353: 'gazelle',
         354: 'Arabian camel, dromedary, Camelus dromedarius',
         355: 'llama',
         356: 'weasel',
         357: 'mink',
         358: 'polecat, fitch, foulmart, foumart, Mustela putorius',
         359: 'black-footed ferret, ferret, Mustela nigripes',
         360: 'otter',
         361: 'skunk, polecat, wood pussy',
         362: 'badger',
         363: 'armadillo',
         364: 'three-toed sloth, ai, Bradypus tridactylus',
         365: 'orangutan, orang, orangutang, Pongo pygmaeus',
         366: 'gorilla, Gorilla gorilla',
         367: 'chimpanzee, chimp, Pan troglodytes',
         368: 'gibbon, Hylobates lar',
         369: 'siamang, Hylobates syndactylus, Symphalangus syndactylus',
         370: 'guenon, guenon monkey',
         371: 'patas, hussar monkey, Erythrocebus patas',
         372: 'baboon',
         373: 'macaque',
         374: 'langur',
         375: 'colobus, colobus monkey',
         376: 'proboscis monkey, Nasalis larvatus',
         377: 'marmoset',
         378: 'capuchin, ringtail, Cebus capucinus',
         379: 'howler monkey, howler',
         380: 'titi, titi monkey',
         381: 'spider monkey, Ateles geoffroyi',
         382: 'squirrel monkey, Saimiri sciureus',
         383: 'Madagascar cat, ring-tailed lemur, Lemur catta',
         384: 'indri, indris, Indri indri, Indri brevicaudatus',
         385: 'Indian elephant, Elephas maximus',
         386: 'African elephant, Loxodonta africana',
         387: 'lesser panda, red panda, panda, bear cat, cat bear, Ailurus fulgens',
         388: 'giant panda, panda, panda bear, coon bear, Ailuropoda melanoleuca',
         389: 'barracouta, snoek',
         390: 'eel',
         391: 'coho, cohoe, coho salmon, blue jack, silver salmon, Oncorhynchus kisutch',
         392: 'rock beauty, Holocanthus tricolor',
         393: 'anemone fish',
         394: 'sturgeon',
         395: 'gar, garfish, garpike, billfish, Lepisosteus osseus',
         396: 'lionfish',
         397: 'puffer, pufferfish, blowfish, globefish',
         398: 'abacus',
         399: 'abaya',
         400: "academic gown, academic robe, judge's robe",
         401: 'accordion, piano accordion, squeeze box',
         402: 'acoustic guitar',
         403: 'aircraft carrier, carrier, flattop, attack aircraft carrier',
         404: 'airliner',
         405: 'airship, dirigible',
         406: 'altar',
         407: 'ambulance',
         408: 'amphibian, amphibious vehicle',
         409: 'analog clock',
         410: 'apiary, bee house',
         411: 'apron',
         412: 'ashcan, trash can, garbage can, wastebin, ash bin, ash-bin, ashbin, dustbin, trash barrel, trash bin',
         413: 'assault rifle, assault gun',
         414: 'backpack, back pack, knapsack, packsack, rucksack, haversack',
         415: 'bakery, bakeshop, bakehouse',
         416: 'balance beam, beam',
         417: 'balloon',
         418: 'ballpoint, ballpoint pen, ballpen, Biro',
         419: 'Band Aid',
         420: 'banjo',
         421: 'bannister, banister, balustrade, balusters, handrail',
         422: 'barbell',
         423: 'barber chair',
         424: 'barbershop',
         425: 'barn',
         426: 'barometer',
         427: 'barrel, cask',
         428: 'barrow, garden cart, lawn cart, wheelbarrow',
         429: 'baseball',
         430: 'basketball',
         431: 'bassinet',
         432: 'bassoon',
         433: 'bathing cap, swimming cap',
         434: 'bath towel',
         435: 'bathtub, bathing tub, bath, tub',
         436: 'beach wagon, station wagon, wagon, estate car, beach waggon, station waggon, waggon',
         437: 'beacon, lighthouse, beacon light, pharos',
         438: 'beaker',
         439: 'bearskin, busby, shako',
         440: 'beer bottle',
         441: 'beer glass',
         442: 'bell cote, bell cot',
         443: 'bib',
         444: 'bicycle-built-for-two, tandem bicycle, tandem',
         445: 'bikini, two-piece',
         446: 'binder, ring-binder',
         447: 'binoculars, field glasses, opera glasses',
         448: 'birdhouse',
         449: 'boathouse',
         450: 'bobsled, bobsleigh, bob',
         451: 'bolo tie, bolo, bola tie, bola',
         452: 'bonnet, poke bonnet',
         453: 'bookcase',
         454: 'bookshop, bookstore, bookstall',
         455: 'bottlecap',
         456: 'bow',
         457: 'bow tie, bow-tie, bowtie',
         458: 'brass, memorial tablet, plaque',
         459: 'brassiere, bra, bandeau',
         460: 'breakwater, groin, groyne, mole, bulwark, seawall, jetty',
         461: 'breastplate, aegis, egis',
         462: 'broom',
         463: 'bucket, pail',
         464: 'buckle',
         465: 'bulletproof vest',
         466: 'bullet train, bullet',
         467: 'butcher shop, meat market',
         468: 'cab, hack, taxi, taxicab',
         469: 'caldron, cauldron',
         470: 'candle, taper, wax light',
         471: 'cannon',
         472: 'canoe',
         473: 'can opener, tin opener',
         474: 'cardigan',
         475: 'car mirror',
         476: 'carousel, carrousel, merry-go-round, roundabout, whirligig',
         477: "carpenter's kit, tool kit",
         478: 'carton',
         479: 'car wheel',
         480: 'cash machine, cash dispenser, automated teller machine, automatic teller machine, automated teller, automatic teller, ATM',
         481: 'cassette',
         482: 'cassette player',
         483: 'castle',
         484: 'catamaran',
         485: 'CD player',
         486: 'cello, violoncello',
         487: 'cellular telephone, cellular phone, cellphone, cell, mobile phone',
         488: 'chain',
         489: 'chainlink fence',
         490: 'chain mail, ring mail, mail, chain armor, chain armour, ring armor, ring armour',
         491: 'chain saw, chainsaw',
         492: 'chest',
         493: 'chiffonier, commode',
         494: 'chime, bell, gong',
         495: 'china cabinet, china closet',
         496: 'Christmas stocking',
         497: 'church, church building',
         498: 'cinema, movie theater, movie theatre, movie house, picture palace',
         499: 'cleaver, meat cleaver, chopper',
         500: 'cliff dwelling',
         501: 'cloak',
         502: 'clog, geta, patten, sabot',
         503: 'cocktail shaker',
         504: 'coffee mug',
         505: 'coffeepot',
         506: 'coil, spiral, volute, whorl, helix',
         507: 'combination lock',
         508: 'computer keyboard, keypad',
         509: 'confectionery, confectionary, candy store',
         510: 'container ship, containership, container vessel',
         511: 'convertible',
         512: 'corkscrew, bottle screw',
         513: 'cornet, horn, trumpet, trump',
         514: 'cowboy boot',
         515: 'cowboy hat, ten-gallon hat',
         516: 'cradle',
         517: 'crane',
         518: 'crash helmet',
         519: 'crate',
         520: 'crib, cot',
         521: 'Crock Pot',
         522: 'croquet ball',
         523: 'crutch',
         524: 'cuirass',
         525: 'dam, dike, dyke',
         526: 'desk',
         527: 'desktop computer',
         528: 'dial telephone, dial phone',
         529: 'diaper, nappy, napkin',
         530: 'digital clock',
         531: 'digital watch',
         532: 'dining table, board',
         533: 'dishrag, dishcloth',
         534: 'dishwasher, dish washer, dishwashing machine',
         535: 'disk brake, disc brake',
         536: 'dock, dockage, docking facility',
         537: 'dogsled, dog sled, dog sleigh',
         538: 'dome',
         539: 'doormat, welcome mat',
         540: 'drilling platform, offshore rig',
         541: 'drum, membranophone, tympan',
         542: 'drumstick',
         543: 'dumbbell',
         544: 'Dutch oven',
         545: 'electric fan, blower',
         546: 'electric guitar',
         547: 'electric locomotive',
         548: 'entertainment center',
         549: 'envelope',
         550: 'espresso maker',
         551: 'face powder',
         552: 'feather boa, boa',
         553: 'file, file cabinet, filing cabinet',
         554: 'fireboat',
         555: 'fire engine, fire truck',
         556: 'fire screen, fireguard',
         557: 'flagpole, flagstaff',
         558: 'flute, transverse flute',
         559: 'folding chair',
         560: 'football helmet',
         561: 'forklift',
         562: 'fountain',
         563: 'fountain pen',
         564: 'four-poster',
         565: 'freight car',
         566: 'French horn, horn',
         567: 'frying pan, frypan, skillet',
         568: 'fur coat',
         569: 'garbage truck, dustcart',
         570: 'gasmask, respirator, gas helmet',
         571: 'gas pump, gasoline pump, petrol pump, island dispenser',
         572: 'goblet',
         573: 'go-kart',
         574: 'golf ball',
         575: 'golfcart, golf cart',
         576: 'gondola',
         577: 'gong, tam-tam',
         578: 'gown',
         579: 'grand piano, grand',
         580: 'greenhouse, nursery, glasshouse',
         581: 'grille, radiator grille',
         582: 'grocery store, grocery, food market, market',
         583: 'guillotine',
         584: 'hair slide',
         585: 'hair spray',
         586: 'half track',
         587: 'hammer',
         588: 'hamper',
         589: 'hand blower, blow dryer, blow drier, hair dryer, hair drier',
         590: 'hand-held computer, hand-held microcomputer',
         591: 'handkerchief, hankie, hanky, hankey',
         592: 'hard disc, hard disk, fixed disk',
         593: 'harmonica, mouth organ, harp, mouth harp',
         594: 'harp',
         595: 'harvester, reaper',
         596: 'hatchet',
         597: 'holster',
         598: 'home theater, home theatre',
         599: 'honeycomb',
         600: 'hook, claw',
         601: 'hoopskirt, crinoline',
         602: 'horizontal bar, high bar',
         603: 'horse cart, horse-cart',
         604: 'hourglass',
         605: 'iPod',
         606: 'iron, smoothing iron',
         607: "jack-o'-lantern",
         608: 'jean, blue jean, denim',
         609: 'jeep, landrover',
         610: 'jersey, T-shirt, tee shirt',
         611: 'jigsaw puzzle',
         612: 'jinrikisha, ricksha, rickshaw',
         613: 'joystick',
         614: 'kimono',
         615: 'knee pad',
         616: 'knot',
         617: 'lab coat, laboratory coat',
         618: 'ladle',
         619: 'lampshade, lamp shade',
         620: 'laptop, laptop computer',
         621: 'lawn mower, mower',
         622: 'lens cap, lens cover',
         623: 'letter opener, paper knife, paperknife',
         624: 'library',
         625: 'lifeboat',
         626: 'lighter, light, igniter, ignitor',
         627: 'limousine, limo',
         628: 'liner, ocean liner',
         629: 'lipstick, lip rouge',
         630: 'Loafer',
         631: 'lotion',
         632: 'loudspeaker, speaker, speaker unit, loudspeaker system, speaker system',
         633: "loupe, jeweler's loupe",
         634: 'lumbermill, sawmill',
         635: 'magnetic compass',
         636: 'mailbag, postbag',
         637: 'mailbox, letter box',
         638: 'maillot',
         639: 'maillot, tank suit',
         640: 'manhole cover',
         641: 'maraca',
         642: 'marimba, xylophone',
         643: 'mask',
         644: 'matchstick',
         645: 'maypole',
         646: 'maze, labyrinth',
         647: 'measuring cup',
         648: 'medicine chest, medicine cabinet',
         649: 'megalith, megalithic structure',
         650: 'microphone, mike',
         651: 'microwave, microwave oven',
         652: 'military uniform',
         653: 'milk can',
         654: 'minibus',
         655: 'miniskirt, mini',
         656: 'minivan',
         657: 'missile',
         658: 'mitten',
         659: 'mixing bowl',
         660: 'mobile home, manufactured home',
         661: 'Model T',
         662: 'modem',
         663: 'monastery',
         664: 'monitor',
         665: 'moped',
         666: 'mortar',
         667: 'mortarboard',
         668: 'mosque',
         669: 'mosquito net',
         670: 'motor scooter, scooter',
         671: 'mountain bike, all-terrain bike, off-roader',
         672: 'mountain tent',
         673: 'mouse, computer mouse',
         674: 'mousetrap',
         675: 'moving van',
         676: 'muzzle',
         677: 'nail',
         678: 'neck brace',
         679: 'necklace',
         680: 'nipple',
         681: 'notebook, notebook computer',
         682: 'obelisk',
         683: 'oboe, hautboy, hautbois',
         684: 'ocarina, sweet potato',
         685: 'odometer, hodometer, mileometer, milometer',
         686: 'oil filter',
         687: 'organ, pipe organ',
         688: 'oscilloscope, scope, cathode-ray oscilloscope, CRO',
         689: 'overskirt',
         690: 'oxcart',
         691: 'oxygen mask',
         692: 'packet',
         693: 'paddle, boat paddle',
         694: 'paddlewheel, paddle wheel',
         695: 'padlock',
         696: 'paintbrush',
         697: "pajama, pyjama, pj's, jammies",
         698: 'palace',
         699: 'panpipe, pandean pipe, syrinx',
         700: 'paper towel',
         701: 'parachute, chute',
         702: 'parallel bars, bars',
         703: 'park bench',
         704: 'parking meter',
         705: 'passenger car, coach, carriage',
         706: 'patio, terrace',
         707: 'pay-phone, pay-station',
         708: 'pedestal, plinth, footstall',
         709: 'pencil box, pencil case',
         710: 'pencil sharpener',
         711: 'perfume, essence',
         712: 'Petri dish',
         713: 'photocopier',
         714: 'pick, plectrum, plectron',
         715: 'pickelhaube',
         716: 'picket fence, paling',
         717: 'pickup, pickup truck',
         718: 'pier',
         719: 'piggy bank, penny bank',
         720: 'pill bottle',
         721: 'pillow',
         722: 'ping-pong ball',
         723: 'pinwheel',
         724: 'pirate, pirate ship',
         725: 'pitcher, ewer',
         726: "plane, carpenter's plane, woodworking plane",
         727: 'planetarium',
         728: 'plastic bag',
         729: 'plate rack',
         730: 'plow, plough',
         731: "plunger, plumber's helper",
         732: 'Polaroid camera, Polaroid Land camera',
         733: 'pole',
         734: 'police van, police wagon, paddy wagon, patrol wagon, wagon, black Maria',
         735: 'poncho',
         736: 'pool table, billiard table, snooker table',
         737: 'pop bottle, soda bottle',
         738: 'pot, flowerpot',
         739: "potter's wheel",
         740: 'power drill',
         741: 'prayer rug, prayer mat',
         742: 'printer',
         743: 'prison, prison house',
         744: 'projectile, missile',
         745: 'projector',
         746: 'puck, hockey puck',
         747: 'punching bag, punch bag, punching ball, punchball',
         748: 'purse',
         749: 'quill, quill pen',
         750: 'quilt, comforter, comfort, puff',
         751: 'racer, race car, racing car',
         752: 'racket, racquet',
         753: 'radiator',
         754: 'radio, wireless',
         755: 'radio telescope, radio reflector',
         756: 'rain barrel',
         757: 'recreational vehicle, RV, R.V.',
         758: 'reel',
         759: 'reflex camera',
         760: 'refrigerator, icebox',
         761: 'remote control, remote',
         762: 'restaurant, eating house, eating place, eatery',
         763: 'revolver, six-gun, six-shooter',
         764: 'rifle',
         765: 'rocking chair, rocker',
         766: 'rotisserie',
         767: 'rubber eraser, rubber, pencil eraser',
         768: 'rugby ball',
         769: 'rule, ruler',
         770: 'running shoe',
         771: 'safe',
         772: 'safety pin',
         773: 'saltshaker, salt shaker',
         774: 'sandal',
         775: 'sarong',
         776: 'sax, saxophone',
         777: 'scabbard',
         778: 'scale, weighing machine',
         779: 'school bus',
         780: 'schooner',
         781: 'scoreboard',
         782: 'screen, CRT screen',
         783: 'screw',
         784: 'screwdriver',
         785: 'seat belt, seatbelt',
         786: 'sewing machine',
         787: 'shield, buckler',
         788: 'shoe shop, shoe-shop, shoe store',
         789: 'shoji',
         790: 'shopping basket',
         791: 'shopping cart',
         792: 'shovel',
         793: 'shower cap',
         794: 'shower curtain',
         795: 'ski',
         796: 'ski mask',
         797: 'sleeping bag',
         798: 'slide rule, slipstick',
         799: 'sliding door',
         800: 'slot, one-armed bandit',
         801: 'snorkel',
         802: 'snowmobile',
         803: 'snowplow, snowplough',
         804: 'soap dispenser',
         805: 'soccer ball',
         806: 'sock',
         807: 'solar dish, solar collector, solar furnace',
         808: 'sombrero',
         809: 'soup bowl',
         810: 'space bar',
         811: 'space heater',
         812: 'space shuttle',
         813: 'spatula',
         814: 'speedboat',
         815: "spider web, spider's web",
         816: 'spindle',
         817: 'sports car, sport car',
         818: 'spotlight, spot',
         819: 'stage',
         820: 'steam locomotive',
         821: 'steel arch bridge',
         822: 'steel drum',
         823: 'stethoscope',
         824: 'stole',
         825: 'stone wall',
         826: 'stopwatch, stop watch',
         827: 'stove',
         828: 'strainer',
         829: 'streetcar, tram, tramcar, trolley, trolley car',
         830: 'stretcher',
         831: 'studio couch, day bed',
         832: 'stupa, tope',
         833: 'submarine, pigboat, sub, U-boat',
         834: 'suit, suit of clothes',
         835: 'sundial',
         836: 'sunglass',
         837: 'sunglasses, dark glasses, shades',
         838: 'sunscreen, sunblock, sun blocker',
         839: 'suspension bridge',
         840: 'swab, swob, mop',
         841: 'sweatshirt',
         842: 'swimming trunks, bathing trunks',
         843: 'swing',
         844: 'switch, electric switch, electrical switch',
         845: 'syringe',
         846: 'table lamp',
         847: 'tank, army tank, armored combat vehicle, armoured combat vehicle',
         848: 'tape player',
         849: 'teapot',
         850: 'teddy, teddy bear',
         851: 'television, television system',
         852: 'tennis ball',
         853: 'thatch, thatched roof',
         854: 'theater curtain, theatre curtain',
         855: 'thimble',
         856: 'thresher, thrasher, threshing machine',
         857: 'throne',
         858: 'tile roof',
         859: 'toaster',
         860: 'tobacco shop, tobacconist shop, tobacconist',
         861: 'toilet seat',
         862: 'torch',
         863: 'totem pole',
         864: 'tow truck, tow car, wrecker',
         865: 'toyshop',
         866: 'tractor',
         867: 'trailer truck, tractor trailer, trucking rig, rig, articulated lorry, semi',
         868: 'tray',
         869: 'trench coat',
         870: 'tricycle, trike, velocipede',
         871: 'trimaran',
         872: 'tripod',
         873: 'triumphal arch',
         874: 'trolleybus, trolley coach, trackless trolley',
         875: 'trombone',
         876: 'tub, vat',
         877: 'turnstile',
         878: 'typewriter keyboard',
         879: 'umbrella',
         880: 'unicycle, monocycle',
         881: 'upright, upright piano',
         882: 'vacuum, vacuum cleaner',
         883: 'vase',
         884: 'vault',
         885: 'velvet',
         886: 'vending machine',
         887: 'vestment',
         888: 'viaduct',
         889: 'violin, fiddle',
         890: 'volleyball',
         891: 'waffle iron',
         892: 'wall clock',
         893: 'wallet, billfold, notecase, pocketbook',
         894: 'wardrobe, closet, press',
         895: 'warplane, military plane',
         896: 'washbasin, handbasin, washbowl, lavabo, wash-hand basin',
         897: 'washer, automatic washer, washing machine',
         898: 'water bottle',
         899: 'water jug',
         900: 'water tower',
         901: 'whiskey jug',
         902: 'whistle',
         903: 'wig',
         904: 'window screen',
         905: 'window shade',
         906: 'Windsor tie',
         907: 'wine bottle',
         908: 'wing',
         909: 'wok',
         910: 'wooden spoon',
         911: 'wool, woolen, woollen',
         912: 'worm fence, snake fence, snake-rail fence, Virginia fence',
         913: 'wreck',
         914: 'yawl',
         915: 'yurt',
         916: 'web site, website, internet site, site',
         917: 'comic book',
         918: 'crossword puzzle, crossword',
         919: 'street sign',
         920: 'traffic light, traffic signal, stoplight',
         921: 'book jacket, dust cover, dust jacket, dust wrapper',
         922: 'menu',
         923: 'plate',
         924: 'guacamole',
         925: 'consomme',
         926: 'hot pot, hotpot',
         927: 'trifle',
         928: 'ice cream, icecream',
         929: 'ice lolly, lolly, lollipop, popsicle',
         930: 'French loaf',
         931: 'bagel, beigel',
         932: 'pretzel',
         933: 'cheeseburger',
         934: 'hotdog, hot dog, red hot',
         935: 'mashed potato',
         936: 'head cabbage',
         937: 'broccoli',
         938: 'cauliflower',
         939: 'zucchini, courgette',
         940: 'spaghetti squash',
         941: 'acorn squash',
         942: 'butternut squash',
         943: 'cucumber, cuke',
         944: 'artichoke, globe artichoke',
         945: 'bell pepper',
         946: 'cardoon',
         947: 'mushroom',
         948: 'Granny Smith',
         949: 'strawberry',
         950: 'orange',
         951: 'lemon',
         952: 'fig',
         953: 'pineapple, ananas',
         954: 'banana',
         955: 'jackfruit, jak, jack',
         956: 'custard apple',
         957: 'pomegranate',
         958: 'hay',
         959: 'carbonara',
         960: 'chocolate sauce, chocolate syrup',
         961: 'dough',
         962: 'meat loaf, meatloaf',
         963: 'pizza, pizza pie',
         964: 'potpie',
         965: 'burrito',
         966: 'red wine',
         967: 'espresso',
         968: 'cup',
         969: 'eggnog',
         970: 'alp',
         971: 'bubble',
         972: 'cliff, drop, drop-off',
         973: 'coral reef',
         974: 'geyser',
         975: 'lakeside, lakeshore',
         976: 'promontory, headland, head, foreland',
         977: 'sandbar, sand bar',
         978: 'seashore, coast, seacoast, sea-coast',
         979: 'valley, vale',
         980: 'volcano',
         981: 'ballplayer, baseball player',
         982: 'groom, bridegroom',
         983: 'scuba diver',
         984: 'rapeseed',
         985: 'daisy',
         986: "yellow lady's slipper, yellow lady-slipper, Cypripedium calceolus, Cypripedium parviflorum",
         987: 'corn',
         988: 'acorn',
         989: 'hip, rose hip, rosehip',
         990: 'buckeye, horse chestnut, conker',
         991: 'coral fungus',
         992: 'agaric',
         993: 'gyromitra',
         994: 'stinkhorn, carrion fungus',
         995: 'earthstar',
         996: 'hen-of-the-woods, hen of the woods, Polyporus frondosus, Grifola frondosa',
         997: 'bolete',
         998: 'ear, spike, capitulum',
         999: 'toilet tissue, toilet paper, bathroom tissue'}

    refined_Label = {}
    for k, v in Label.items():
        if len(v.split(",")) == 1:
            select = v
        else:
            select = v.split(",")[0]
        refined_Label.update({k: select})