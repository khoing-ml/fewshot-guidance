import math
from typing import Callable
from pathlib import Path
import random
from tqdm import tqdm

import numpy as np
import torch
from einops import rearrange, repeat
from PIL import Image
from torch import Tensor

from .model import Flux
from .modules.autoencoder import AutoEncoder
from .modules.conditioner import HFEmbedder
from .modules.image_embedders import CannyImageEncoder, DepthImageEncoder, ReduxImageEncoder
from .util import PREFERED_KONTEXT_RESOLUTIONS


# ============================================================================
# Dataset Loading Utilities for Fewshot Guidance
# ============================================================================

def load_dataset_from_directory(root_dir, extensions=['.jpg', '.jpeg', '.png', '.webp']):
    """
    Load image paths from a dataset directory.
    
    Supports two structures:
    1. Class-based: dataset_root/class1/img1.jpg, dataset_root/class2/img2.jpg
    2. Flat: dataset_root/img1.jpg, dataset_root/img2.jpg
    
    Returns:
        tuple: (image_paths, class_labels, class_to_idx)
    """
    root_dir = Path(root_dir)
    
    if not root_dir.exists():
        raise ValueError(f"Dataset directory does not exist: {root_dir}")
    
    image_paths = []
    class_labels = []
    class_to_idx = {}
    
    # Check for class-based directory structure
    subdirs = [d for d in root_dir.iterdir() if d.is_dir()]
    
    if subdirs:
        # Class-based structure
        for class_idx, class_dir in enumerate(sorted(subdirs)):
            class_name = class_dir.name
            class_to_idx[class_name] = class_idx
            
            for ext in extensions:
                for img_path in class_dir.glob(f"*{ext}"):
                    image_paths.append(str(img_path))
                    class_labels.append(class_idx)
    else:
        # Flat structure - all images in one class
        for ext in extensions:
            for img_path in root_dir.glob(f"*{ext}"):
                image_paths.append(str(img_path))
                class_labels.append(0)
        class_to_idx = {'default': 0}
    
    if len(image_paths) == 0:
        raise ValueError(f"No images found in {root_dir} with extensions {extensions}")
    
    return image_paths, class_labels, class_to_idx


def sample_images_from_dataset(image_paths, class_labels, class_to_idx, 
                                num_shots=5, class_name=None, seed=None):
    """
    Sample images from a loaded dataset.
    
    Args:
        image_paths: List of all image paths
        class_labels: List of class labels for each image
        class_to_idx: Dict mapping class names to indices
        num_shots: Number of images to sample
        class_name: Optional class to sample from
        seed: Random seed for reproducibility
    
    Returns:
        List of selected image paths
    """
    if seed is not None:
        random.seed(seed)
    
    if class_name:
        if class_name in class_to_idx:
            class_idx = class_to_idx[class_name]
            indices = [i for i, c in enumerate(class_labels) if c == class_idx]
            if len(indices) < num_shots:
                print(f"Warning: Only {len(indices)} images in class '{class_name}', using all")
                selected_indices = indices
            else:
                selected_indices = random.sample(indices, num_shots)
        else:
            raise ValueError(f"Class '{class_name}' not found. Available: {list(class_to_idx.keys())}")
    else:
        # Random sampling across all images
        selected_indices = random.sample(range(len(image_paths)), min(num_shots, len(image_paths)))
    
    return [image_paths[i] for i in selected_indices]


def load_and_encode_images(image_paths, ae, device, height=1024, width=1024):
    """
    Load and encode images to latent space for fewshot guid ance.
    
    Args:
        image_paths: List of image file paths
        ae: AutoEncoder model
        device: torch device
        height: Target height
        width: Target width
    
    Returns:
        Encoded latents [num_images, seq_len, channels]
    """
    latents = []
    
    for img_path in image_paths:
        # Load and preprocess image
        img = Image.open(img_path).convert("RGB")
        img = img.resize((width, height), Image.Resampling.LANCZOS)
        img = np.array(img).astype(np.float32) / 127.5 - 1.0
        img = torch.from_numpy(img).permute(2, 0, 1).unsqueeze(0).to(device)
        
        # Encode to latent
        with torch.no_grad():
            latent = ae.encode(img.to(torch.bfloat16))
            # Pack the latent
            latent = rearrange(latent, "b c (h ph) (w pw) -> b (h w) (c ph pw)", ph=2, pw=2)
            latents.append(latent)
    
    # Stack all latents [num_images, seq_len, channels]
    fewshot_latents = torch.cat(latents, dim=0)
    return fewshot_latents


# ============================================================================
# Sampling Functions
# ============================================================================

def get_noise(
    num_samples: int,
    height: int,
    width: int,
    device: torch.device,
    dtype: torch.dtype,
    seed: int,
):
    return torch.randn(
        num_samples,
        16,
        # allow for packing
        2 * math.ceil(height / 16),
        2 * math.ceil(width / 16),
        dtype=dtype,
        generator=torch.Generator(device="cpu").manual_seed(seed),
    ).to(device)


def prepare(t5: HFEmbedder, clip: HFEmbedder, img: Tensor, prompt: str | list[str]) -> dict[str, Tensor]:
    bs, c, h, w = img.shape
    if bs == 1 and not isinstance(prompt, str):
        bs = len(prompt)

    img = rearrange(img, "b c (h ph) (w pw) -> b (h w) (c ph pw)", ph=2, pw=2)
    if img.shape[0] == 1 and bs > 1:
        img = repeat(img, "1 ... -> bs ...", bs=bs)

    img_ids = torch.zeros(h // 2, w // 2, 3)
    img_ids[..., 1] = img_ids[..., 1] + torch.arange(h // 2)[:, None]
    img_ids[..., 2] = img_ids[..., 2] + torch.arange(w // 2)[None, :]
    img_ids = repeat(img_ids, "h w c -> b (h w) c", b=bs)

    if isinstance(prompt, str):
        prompt = [prompt]
    txt = t5(prompt)
    if txt.shape[0] == 1 and bs > 1:
        txt = repeat(txt, "1 ... -> bs ...", bs=bs)
    txt_ids = torch.zeros(bs, txt.shape[1], 3)

    vec = clip(prompt)
    if vec.shape[0] == 1 and bs > 1:
        vec = repeat(vec, "1 ... -> bs ...", bs=bs)

    return {
        "img": img,
        "img_ids": img_ids.to(img.device),
        "txt": txt.to(img.device),
        "txt_ids": txt_ids.to(img.device),
        "vec": vec.to(img.device),
    }


def prepare_control(
    t5: HFEmbedder,
    clip: HFEmbedder,
    img: Tensor,
    prompt: str | list[str],
    ae: AutoEncoder,
    encoder: DepthImageEncoder | CannyImageEncoder,
    img_cond_path: str,
) -> dict[str, Tensor]:
    # load and encode the conditioning image
    bs, _, h, w = img.shape
    if bs == 1 and not isinstance(prompt, str):
        bs = len(prompt)

    img_cond = Image.open(img_cond_path).convert("RGB")

    width = w * 8
    height = h * 8
    img_cond = img_cond.resize((width, height), Image.Resampling.LANCZOS)
    img_cond = np.array(img_cond)
    img_cond = torch.from_numpy(img_cond).float() / 127.5 - 1.0
    img_cond = rearrange(img_cond, "h w c -> 1 c h w")

    with torch.no_grad():
        img_cond = encoder(img_cond)
        img_cond = ae.encode(img_cond)

    img_cond = img_cond.to(torch.bfloat16)
    img_cond = rearrange(img_cond, "b c (h ph) (w pw) -> b (h w) (c ph pw)", ph=2, pw=2)
    if img_cond.shape[0] == 1 and bs > 1:
        img_cond = repeat(img_cond, "1 ... -> bs ...", bs=bs)

    return_dict = prepare(t5, clip, img, prompt)
    return_dict["img_cond"] = img_cond
    return return_dict


def prepare_fill(
    t5: HFEmbedder,
    clip: HFEmbedder,
    img: Tensor,
    prompt: str | list[str],
    ae: AutoEncoder,
    img_cond_path: str,
    mask_path: str,
) -> dict[str, Tensor]:
    # load and encode the conditioning image and the mask
    bs, _, _, _ = img.shape
    if bs == 1 and not isinstance(prompt, str):
        bs = len(prompt)

    img_cond = Image.open(img_cond_path).convert("RGB")
    img_cond = np.array(img_cond)
    img_cond = torch.from_numpy(img_cond).float() / 127.5 - 1.0
    img_cond = rearrange(img_cond, "h w c -> 1 c h w")

    mask = Image.open(mask_path).convert("L")
    mask = np.array(mask)
    mask = torch.from_numpy(mask).float() / 255.0
    mask = rearrange(mask, "h w -> 1 1 h w")

    with torch.no_grad():
        img_cond = img_cond.to(img.device)
        mask = mask.to(img.device)
        img_cond = img_cond * (1 - mask)
        img_cond = ae.encode(img_cond)
        mask = mask[:, 0, :, :]
        mask = mask.to(torch.bfloat16)
        mask = rearrange(
            mask,
            "b (h ph) (w pw) -> b (ph pw) h w",
            ph=8,
            pw=8,
        )
        mask = rearrange(mask, "b c (h ph) (w pw) -> b (h w) (c ph pw)", ph=2, pw=2)
        if mask.shape[0] == 1 and bs > 1:
            mask = repeat(mask, "1 ... -> bs ...", bs=bs)

    img_cond = img_cond.to(torch.bfloat16)
    img_cond = rearrange(img_cond, "b c (h ph) (w pw) -> b (h w) (c ph pw)", ph=2, pw=2)
    if img_cond.shape[0] == 1 and bs > 1:
        img_cond = repeat(img_cond, "1 ... -> bs ...", bs=bs)

    img_cond = torch.cat((img_cond, mask), dim=-1)

    return_dict = prepare(t5, clip, img, prompt)
    return_dict["img_cond"] = img_cond.to(img.device)
    return return_dict


def prepare_redux(
    t5: HFEmbedder,
    clip: HFEmbedder,
    img: Tensor,
    prompt: str | list[str],
    encoder: ReduxImageEncoder,
    img_cond_path: str,
) -> dict[str, Tensor]:
    bs, _, h, w = img.shape
    if bs == 1 and not isinstance(prompt, str):
        bs = len(prompt)

    img_cond = Image.open(img_cond_path).convert("RGB")
    with torch.no_grad():
        img_cond = encoder(img_cond)

    img_cond = img_cond.to(torch.bfloat16)
    if img_cond.shape[0] == 1 and bs > 1:
        img_cond = repeat(img_cond, "1 ... -> bs ...", bs=bs)

    img = rearrange(img, "b c (h ph) (w pw) -> b (h w) (c ph pw)", ph=2, pw=2)
    if img.shape[0] == 1 and bs > 1:
        img = repeat(img, "1 ... -> bs ...", bs=bs)

    img_ids = torch.zeros(h // 2, w // 2, 3)
    img_ids[..., 1] = img_ids[..., 1] + torch.arange(h // 2)[:, None]
    img_ids[..., 2] = img_ids[..., 2] + torch.arange(w // 2)[None, :]
    img_ids = repeat(img_ids, "h w c -> b (h w) c", b=bs)

    if isinstance(prompt, str):
        prompt = [prompt]
    txt = t5(prompt)
    txt = torch.cat((txt, img_cond.to(txt)), dim=-2)
    if txt.shape[0] == 1 and bs > 1:
        txt = repeat(txt, "1 ... -> bs ...", bs=bs)
    txt_ids = torch.zeros(bs, txt.shape[1], 3)

    vec = clip(prompt)
    if vec.shape[0] == 1 and bs > 1:
        vec = repeat(vec, "1 ... -> bs ...", bs=bs)

    return {
        "img": img,
        "img_ids": img_ids.to(img.device),
        "txt": txt.to(img.device),
        "txt_ids": txt_ids.to(img.device),
        "vec": vec.to(img.device),
    }


def prepare_kontext(
    t5: HFEmbedder,
    clip: HFEmbedder,
    prompt: str | list[str],
    ae: AutoEncoder,
    img_cond_path: str,
    seed: int,
    device: torch.device,
    target_width: int | None = None,
    target_height: int | None = None,
    bs: int = 1,
) -> tuple[dict[str, Tensor], int, int]:
    # load and encode the conditioning image
    if bs == 1 and not isinstance(prompt, str):
        bs = len(prompt)

    img_cond = Image.open(img_cond_path).convert("RGB")
    width, height = img_cond.size
    aspect_ratio = width / height
    # Kontext is trained on specific resolutions, using one of them is recommended
    _, width, height = min((abs(aspect_ratio - w / h), w, h) for w, h in PREFERED_KONTEXT_RESOLUTIONS)
    width = 2 * int(width / 16)
    height = 2 * int(height / 16)

    img_cond = img_cond.resize((8 * width, 8 * height), Image.Resampling.LANCZOS)
    img_cond = np.array(img_cond)
    img_cond = torch.from_numpy(img_cond).float() / 127.5 - 1.0
    img_cond = rearrange(img_cond, "h w c -> 1 c h w")
    img_cond_orig = img_cond.clone()

    with torch.no_grad():
        img_cond = ae.encode(img_cond.to(device))

    img_cond = img_cond.to(torch.bfloat16)
    img_cond = rearrange(img_cond, "b c (h ph) (w pw) -> b (h w) (c ph pw)", ph=2, pw=2)
    if img_cond.shape[0] == 1 and bs > 1:
        img_cond = repeat(img_cond, "1 ... -> bs ...", bs=bs)

    # image ids are the same as base image with the first dimension set to 1
    # instead of 0
    img_cond_ids = torch.zeros(height // 2, width // 2, 3)
    img_cond_ids[..., 0] = 1
    img_cond_ids[..., 1] = img_cond_ids[..., 1] + torch.arange(height // 2)[:, None]
    img_cond_ids[..., 2] = img_cond_ids[..., 2] + torch.arange(width // 2)[None, :]
    img_cond_ids = repeat(img_cond_ids, "h w c -> b (h w) c", b=bs)

    if target_width is None:
        target_width = 8 * width
    if target_height is None:
        target_height = 8 * height

    img = get_noise(
        1,
        target_height,
        target_width,
        device=device,
        dtype=torch.bfloat16,
        seed=seed,
    )

    return_dict = prepare(t5, clip, img, prompt)
    return_dict["img_cond_seq"] = img_cond
    return_dict["img_cond_seq_ids"] = img_cond_ids.to(device)
    return_dict["img_cond_orig"] = img_cond_orig
    return return_dict, target_height, target_width


def time_shift(mu: float, sigma: float, t: Tensor):
    return math.exp(mu) / (math.exp(mu) + (1 / t - 1) ** sigma)


def get_lin_function(
    x1: float = 256, y1: float = 0.5, x2: float = 4096, y2: float = 1.15
) -> Callable[[float], float]:
    m = (y2 - y1) / (x2 - x1)
    b = y1 - m * x1
    return lambda x: m * x + b


def get_schedule(
    num_steps: int,
    image_seq_len: int,
    base_shift: float = 0.5,
    max_shift: float = 1.15,
    shift: bool = True,
) -> list[float]:
    # extra step for zero
    timesteps = torch.linspace(1, 0, num_steps + 1)

    # shifting the schedule to favor high timesteps for higher signal images
    if shift:
        # estimate mu based on linear estimation between two points
        mu = get_lin_function(y1=base_shift, y2=max_shift)(image_seq_len)
        timesteps = time_shift(mu, 1.0, timesteps)

    return timesteps.tolist()


def denoise(
    model: Flux,
    # model input
    img: Tensor,
    img_ids: Tensor,
    txt: Tensor,
    txt_ids: Tensor,
    vec: Tensor,
    # sampling parameters
    timesteps: list[float],
    guidance: float = 4.0,
    # extra img tokens (channel-wise)
    img_cond: Tensor | None = None,
    # extra img tokens (sequence-wise)
    img_cond_seq: Tensor | None = None,
    img_cond_seq_ids: Tensor | None = None,
):
    # this is ignored for schnell
    guidance_vec = torch.full((img.shape[0],), guidance, device=img.device, dtype=img.dtype)
    for t_curr, t_prev in zip(timesteps[:-1], timesteps[1:]):
        t_vec = torch.full((img.shape[0],), t_curr, dtype=img.dtype, device=img.device)
        img_input = img
        img_input_ids = img_ids
        if img_cond is not None:
            img_input = torch.cat((img, img_cond), dim=-1)
        if img_cond_seq is not None:
            assert (
                img_cond_seq_ids is not None
            ), "You need to provide either both or neither of the sequence conditioning"
            img_input = torch.cat((img_input, img_cond_seq), dim=1)
            img_input_ids = torch.cat((img_input_ids, img_cond_seq_ids), dim=1)
        pred = model(
            img=img_input,
            img_ids=img_input_ids,
            txt=txt,
            txt_ids=txt_ids,
            y=vec,
            timesteps=t_vec,
            guidance=guidance_vec,
        )
        if img_input_ids is not None:
            pred = pred[:, : img.shape[1]]

        img = img + (t_prev - t_curr) * pred

    return img


def denoise_with_guidance(
    model: Flux,
    # model input
    img: Tensor,
    img_ids: Tensor,
    txt: Tensor,
    txt_ids: Tensor,
    vec: Tensor,
    # sampling parameters
    timesteps: list[float],
    guidance: float = 4.0,
    # extra img tokens (channel-wise)
    img_cond: Tensor | None = None,
    # extra img tokens (sequence-wise)
    img_cond_seq: Tensor | None = None,
    img_cond_seq_ids: Tensor | None = None,
    # learned guidance controller (trained online)
    guidance_model: Callable | None = None,
    guidance_optimizer: torch.optim.Optimizer | None = None,
    guidance_loss_fn: Callable | None = None,
    guidance_train_steps: int = 10,
    guidance_scale: float = 1.0,
    # fewshot reference images
    fewshot_latents: Tensor | None = None,
):
    """
    Denoise with online-trained guidance controller c^φ(t, X_t).
    
    The guidance model is trained AT EACH TIMESTEP to minimize your custom loss:
    
    L_RGM(φ) = E[||u_t^θ(t,X_t) + c_t^φ(t,X_t) - (X_1 - X_0)||^2] + λ|c_t^φ(X_1) - c_t^φ(X̂_1)|
    
    Args:
        model: The Flux diffusion model
        img: Initial latent image
        img_ids: Image position IDs
        txt: Text embeddings
        txt_ids: Text position IDs
        vec: CLIP pooled embeddings
        timesteps: Denoising timestep schedule
        guidance: Classifier-free guidance scale
        img_cond: Optional channel-wise image conditioning
        img_cond_seq: Optional sequence-wise image conditioning
        img_cond_seq_ids: Position IDs for sequence conditioning
        guidance_model: Controller c^φ to be trained at each timestep
        guidance_optimizer: Optimizer for guidance_model (e.g., Adam)
        guidance_loss_fn: Custom loss function with signature:
                         (img, pred, guidance_correction, txt, vec, timestep, step_idx, fewshot_latents) -> loss_tensor
                         This should return your L_RGM(φ) or similar cost
        guidance_train_steps: Number of optimization steps per timestep
        guidance_scale: Scaling factor λ for the guidance correction
        fewshot_latents: Fewshot reference image latents [batch, seq_len, channels] or [num_shots, seq_len, channels]
                        Sampled from your fewshot dataset
    
    Returns:
        Denoised latent image
    """
    # this is ignored for schnell
    guidance_vec = torch.full((img.shape[0],), guidance, device=img.device, dtype=img.dtype)
    
    for step_idx, (t_curr, t_prev) in enumerate(zip(timesteps[:-1], timesteps[1:])):
        t_vec = torch.full((img.shape[0],), t_curr, dtype=img.dtype, device=img.device)
        img_input = img
        img_input_ids = img_ids
        if img_cond is not None:
            img_input = torch.cat((img, img_cond), dim=-1)
        if img_cond_seq is not None:
            assert (
                img_cond_seq_ids is not None
            ), "You need to provide either both or neither of the sequence conditioning"
            img_input = torch.cat((img_input, img_cond_seq), dim=1)
            img_input_ids = torch.cat((img_input_ids, img_cond_seq_ids), dim=1)
        
        # Get base model prediction u_t^θ(t, X_t) with no_grad
        with torch.no_grad():
            pred = model(
                img=img_input,
                img_ids=img_input_ids,
                txt=txt,
                txt_ids=txt_ids,
                y=vec,
                timesteps=t_vec,
                guidance=guidance_vec,
            )
            if img_input_ids is not None:
                pred = pred[:, : img.shape[1]]
        
        # Train guidance controller at this timestep if provided
        guidance_correction = None
        if guidance_model is not None and guidance_optimizer is not None and guidance_loss_fn is not None:
            guidance_model.train()
            
            # Optimize guidance model for this timestep
            for train_step in tqdm(range(guidance_train_steps), desc=f"Guidance Training Step {step_idx+1}/{len(timesteps)-1}"):
                guidance_optimizer.zero_grad()
                
                # Get guidance correction c_t^φ(t, X_t) conditioned on fewshot examples
                guidance_correction = guidance_model(
                    img=img,
                    pred=pred,
                    txt=txt,
                    vec=vec,
                    timestep=t_vec,
                    step_idx=step_idx,
                    fewshot_img=fewshot_latents,  # Pass fewshot reference
                )
                perturbed_img = img + torch.randn_like(img) * 0.01  # small perturbation
                perturbed_guidance_correction = guidance_model(
                    img=perturbed_img,
                    pred=pred,
                    txt=txt,
                    vec=vec,
                    timestep=t_vec,
                    step_idx=step_idx,
                    fewshot_img=fewshot_latents,
                )
                # Handle dict return
                if isinstance(guidance_correction, dict):
                    guidance_correction = guidance_correction['guidance']

                # Compute your custom loss L_RGM(φ)
                loss = guidance_loss_fn(
                    img=img,
                    pred=pred,
                    guidance_correction=guidance_correction,
                    perturbed_guidance_correction=perturbed_guidance_correction,
                    txt=txt,
                    vec=vec,
                    timestep=t_vec,
                    step_idx=step_idx,
                    fewshot_latents=fewshot_latents,
                )
                
                print(f"  Step {train_step+1}/{guidance_train_steps}, Loss: {loss.item():.6f}")
                # Backprop and update
                loss.backward()
                guidance_optimizer.step()
            
            # After training, get final correction with no_grad
            guidance_model.eval()
            with torch.no_grad():
                guidance_correction = guidance_model(
                    img=img,
                    pred=pred,
                    txt=txt,
                    vec=vec,
                    timestep=t_vec,
                    step_idx=step_idx,
                    fewshot_img=fewshot_latents,
                )
                if isinstance(guidance_correction, dict):
                    guidance_correction = guidance_correction['guidance']
        
        # Apply guidance if it was computed
        if guidance_correction is not None:
            # Apply scaled guidance: pred_final = u_t^θ + λ * c_t^φ
            pred = pred + guidance_scale * guidance_correction.detach()

        # Euler integration step: X_{t-Δt} = X_t + Δt * pred_final
        img = img + (t_prev - t_curr) * pred

    return img



def unpack(x: Tensor, height: int, width: int) -> Tensor:
    return rearrange(
        x,
        "b (h w) (c ph pw) -> b c (h ph) (w pw)",
        h=math.ceil(height / 16),
        w=math.ceil(width / 16),
        ph=2,
        pw=2,
    )

def pack_x(x: Tensor, height: int, width: int) -> Tensor:
    return rearrange(
        x,
        "b c (h ph) (w pw) -> b (h w) (c ph pw)",
        h=16,
        w=16,
        ph=math.ceil(height / 16),
        pw=math.ceil(width / 16),
    )