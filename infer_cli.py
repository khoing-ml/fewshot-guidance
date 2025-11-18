#!/usr/bin/env python3
import os
import time
import uuid
import argparse

import torch
from einops import rearrange
from PIL import Image
from transformers import pipeline

from flux.sampling import denoise, get_noise, get_schedule, prepare, unpack
from flux.util import load_ae, load_clip, load_flow_model, load_t5, save_image, configs


@torch.inference_mode()
def generate(
    name: str,
    prompt: str,
    width: int,
    height: int,
    num_steps: int | None,
    guidance: float,
    seed: int | None,
    device: str,
    offload: bool,
    output_dir: str,
    add_sampling_metadata: bool,
    track_usage: bool,
):
    if name not in configs:
        raise ValueError(f"Unknown model name {name}, choose from: {list(configs.keys())}")

    torch_device = torch.device(device)
    is_schnell = name == "flux-schnell"

    if num_steps is None:
        num_steps = 4 if is_schnell else 50

    # load models (allow offload to CPU)
    t5 = load_t5(torch_device, max_length=256 if is_schnell else 512)
    clip = load_clip(torch_device)
    model = load_flow_model(name, device="cpu" if offload else torch_device)
    ae = load_ae(name, device="cpu" if offload else torch_device)

    # NSFW classifier (optional) - use CPU for the classifier if device is cpu
    try:
        nsfw_device = -1 if torch_device.type == "cpu" else 0
        nsfw_classifier = pipeline("image-classification", model="Falconsai/nsfw_image_detection", device=nsfw_device)
    except Exception:
        nsfw_classifier = None

    # prepare RNG and options
    rng = torch.Generator(device="cpu")
    if seed is None:
        seed = rng.seed()

    # ensure output dir
    os.makedirs(output_dir, exist_ok=True)
    output_name = os.path.join(output_dir, "img_{idx}.jpg")
    idx = 0

    print(f"Generating with seed {seed}: {prompt}")
    t0 = time.perf_counter()

    # choose dtype: prefer bfloat16 on CUDA, float32 on CPU
    dtype = torch.bfloat16 if torch_device.type == "cuda" and torch.cuda.is_available() else torch.float32

    x = get_noise(
        1,
        height,
        width,
        device=torch_device,
        dtype=dtype,
        seed=int(seed),
    )

    timesteps = get_schedule(num_steps, x.shape[1], shift=(not is_schnell))

    if offload:
        ae = ae.cpu()
        model = model.to(torch_device)

    inp = prepare(t5, clip, x, prompt=prompt)

    if offload:
        model = model.to(torch_device)

    x = denoise(model, **inp, timesteps=timesteps, guidance=guidance)

    if offload:
        model.cpu()
        torch.cuda.empty_cache()
        ae.decoder.to(x.device)

    x = unpack(x.float(), height, width)
    with torch.autocast(device_type=torch_device.type, dtype=torch.bfloat16 if torch_device.type == "cuda" else torch.float32):
        x = ae.decode(x)

    t1 = time.perf_counter()
    print(f"Done in {t1 - t0:.1f}s. Saving image...")

    # reuse utility that handles NSFW checking and metadata
    idx = save_image(
        nsfw_classifier, name, output_name, idx, x, add_sampling_metadata, prompt, track_usage=track_usage
    )

    saved = output_name.format(idx=idx - 1) if idx > 0 else None
    if saved:
        print(f"Saved to {saved}")
    else:
        print("Image not saved (possibly flagged NSFW)")


def main():
    parser = argparse.ArgumentParser(description="Simple Flux CLI (no Gradio). Generates one image and exits.")
    parser.add_argument("--name", type=str, default="flux-schnell", choices=list(configs.keys()))
    parser.add_argument("--prompt", type=str, default="")
    parser.add_argument("--width", type=int, default=256)
    parser.add_argument("--height", type=int, default=256)
    parser.add_argument("--num_steps", type=int, default=50)
    parser.add_argument("--guidance", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--device", type=str, default=("cuda" if torch.cuda.is_available() else "cpu"))
    parser.add_argument("--offload", action="store_true", help="Offload model to CPU when not in use")
    parser.add_argument("--output_dir", type=str, default="output")
    parser.add_argument("--no_metadata", dest="add_sampling_metadata", action="store_false")
    parser.add_argument("--track_usage", action="store_true")

    args = parser.parse_args()

    generate(
        name=args.name,
        prompt=args.prompt,
        width=args.width,
        height=args.height,
        num_steps=args.num_steps,
        guidance=args.guidance,
        seed=args.seed,
        device=args.device,
        offload=args.offload,
        output_dir=args.output_dir,
        add_sampling_metadata=args.add_sampling_metadata,
        track_usage=args.track_usage,
    )


if __name__ == "__main__":
    main()
