#!/usr/bin/env python3
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import os
import torch
import numpy as np
from dotenv import load_dotenv

from flux.sampling import get_noise, prepare, get_schedule, denoise, unpack
from flux.util import load_flow_model, load_t5, load_clip


load_dotenv(".env")


def main():
    device = torch.device("cuda")
    print("Using device:", device)
    dtype = torch.float32
    height, width = 64, 64
    seed = 42

    # Load the FLUX model
    model = load_flow_model("flux-dev", device=device)
    noise = get_noise(num_samples=1, height=height, width=width, device=device, dtype=dtype, seed=seed)
    prompt = ""
    t5_model = load_t5(device=device, dtype=dtype)
    clip_model = load_clip(device=device, dtype=dtype)
    # Prepare unconditional inputs
    inp = prepare(noise, prompt, t5_model, clip_model, device=device, dtype=dtype)
    img = inp["img"]
    img_ids = inp["img_ids"]
    txt = inp["txt"]
    txt_ids = inp["txt_ids"]
    vec = inp["vec"]

    # Simple timestep schedule
    timesteps = get_schedule(num_steps=10, image_seq_len=img.shape[1], shift=False)


    # move inputs to the same device/dtype as model parameters
    model_dtype = next(model.parameters()).dtype
    img = img.to(device=device, dtype=model_dtype)
    img_ids = img_ids.to(device=device, dtype=model_dtype)
    txt = txt.to(device=device, dtype=model_dtype)
    txt_ids = txt_ids.to(device=device, dtype=model_dtype)
    vec = vec.to(device=device, dtype=model_dtype)

    # Run denoising loop unconditionally
    out = denoise(
        model=model,
        img=img,
        img_ids=img_ids,
        txt=txt,
        txt_ids=txt_ids,
        vec=vec,
        timesteps=timesteps,
        guidance=1.0, # ignored for unconditional
    )

    # Unpack to spatial latent shape
    latents = unpack(out, height=height, width=width)

    print("Output latents shape:", out.shape)
    print("Unpacked image shape:", latents.shape)
    print("Latents stats: min, max, mean:", float(latents.min()), float(latents.max()), float(latents.mean()))

    # Save to .npy for inspection
    np.save("unconditional_sample.npy", latents.numpy())
    print("Saved unconditional_sample.npy")


if __name__ == "__main__":
    main()
