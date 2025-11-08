#!/usr/bin/env python3
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import torch
import numpy as np

from flux.sampling import get_noise, prepare_unconditional, get_schedule, denoise, unpack


def main():
    device = torch.device("cpu")
    dtype = torch.float32
    height, width = 64, 64
    seed = 42

    # Create initial noise
    noise = get_noise(num_samples=1, height=height, width=width, device=device, dtype=dtype, seed=seed)

    # Prepare unconditional inputs
    inp = prepare_unconditional(noise)
    img = inp["img"]
    img_ids = inp["img_ids"]
    txt = inp["txt"]
    txt_ids = inp["txt_ids"]
    vec = inp["vec"]

    # Simple timestep schedule
    timesteps = get_schedule(num_steps=10, image_seq_len=img.shape[1], shift=False)

    # Dummy model compatible with denoise() call signature
    class DummyFlux(torch.nn.Module):
        def forward(self, img, img_ids, txt, txt_ids, y, timesteps, guidance, unconditional=False):
            # return a small random correction prediction
            return torch.randn_like(img) * 0.05

    model = DummyFlux()

    # Run denoising loop unconditionally
    out = denoise(
        model=model,
        img=img,
        img_ids=img_ids,
        txt=txt,
        txt_ids=txt_ids,
        vec=vec,
        timesteps=timesteps,
        guidance=1.0,
        unconditional=True,
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
