#!/usr/bin/env python3
"""Decode an unconditional latent .npy to an image.

This script will:
- load a .npy latent saved as (B, C, H, W) or (C, H, W)
- try to find a matching AutoEncoder config in `flux.util.configs` based on spatial size
- load the AE (may download checkpoint) and decode the latent
- if no matching AE is found, try upsampling the latent to a likely AE shape and decode
- fallback: create a quick channel-mapped preview PNG

Usage:
  PYTHONPATH=./src python3 examples/decode_unconditional_auto.py unconditional_sample.npy
"""
import sys
from pathlib import Path
import numpy as np
from PIL import Image

try:
    # allow running from repo root
    sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
    from flux.util import configs, load_ae
    import torch
    has_flux = True
except Exception:
    has_flux = False
    import torch


def quick_map_channels(arr):
    # arr: numpy C,H,W or H,W,C -> return H,W,3 uint8
    if arr.ndim == 3 and arr.shape[0] <= arr.shape[2]:
        # likely H,W,C already
        h, w, c = arr.shape
        arr_hwc = arr
    else:
        # assume C,H,W
        arr_hwc = arr.transpose(1, 2, 0)
        h, w, c = arr_hwc.shape

    if c >= 3:
        rgb = arr_hwc[:, :, :3]
    else:
        rgb = np.repeat(arr_hwc[:, :, :1], 3, axis=2)

    out = np.zeros_like(rgb, dtype="uint8")
    for i in range(3):
        ch = rgb[:, :, i]
        mn, mx = float(ch.min()), float(ch.max())
        if mx - mn < 1e-6:
            out[:, :, i] = (ch * 0).astype("uint8")
        else:
            out[:, :, i] = ((ch - mn) / (mx - mn) * 255.0).astype("uint8")
    return out


def find_matching_config(ch, h, w):
    # return key of configs whose AE would expect z shape (ch, h, w)
    for name, spec in configs.items():
        ae = spec.ae_params
        # compute latent spatial size expected by AE: curr_res = resolution // 2**(num_resolutions-1)
        num_res = len(ae.ch_mult)
        curr_res = ae.resolution // (2 ** (num_res - 1))
        if ae.z_channels == ch and curr_res == h and curr_res == w:
            return name
    return None


def save_png_from_tensor(img_tensor, out_path):
    # img_tensor: torch.Tensor HWC or CHW, float in range [-1,1] or [0,1]
    if isinstance(img_tensor, torch.Tensor):
        img = img_tensor.detach().cpu().numpy()
    else:
        img = np.array(img_tensor)
    # If CHW -> HWC
    if img.ndim == 3 and img.shape[0] <= 4:
        if img.shape[0] in (1, 3, 4):
            img = img.transpose(1, 2, 0)
    mn, mx = img.min(), img.max()
    if mn < -0.5 and mx <= 2.0:
        img = (127.5 * (img + 1.0)).round().astype("uint8")
    elif mx <= 1.0 and mn >= 0.0:
        img = (255.0 * img).round().astype("uint8")
    else:
        img = (255.0 * (img - mn) / (mx - mn + 1e-12)).round().astype("uint8")

    if img.ndim == 3 and img.shape[2] == 1:
        img = img[:, :, 0]
    Image.fromarray(img).save(out_path)
    print(f"Saved {out_path}")


def main(path_in: str, out_base: str = "unconditional_preview"):
    arr = np.load(path_in)
    print(f"Loaded {path_in} shape={arr.shape} dtype={arr.dtype}")

    if arr.ndim == 4:
        b, c, h, w = arr.shape
        arr0 = arr[0]
    elif arr.ndim == 3:
        c, h, w = arr.shape
        arr0 = arr
        b = 1
    else:
        raise RuntimeError(f"Unsupported array shape: {arr.shape} - expected 3 or 4 dims")

    # Try to find a matching AE config
    if has_flux:
        match = find_matching_config(c, h, w)
        if match is not None:
            print(f"Found matching AE config: {match} - attempting decode")
            try:
                ae = load_ae(match, device="cpu")
                ae.eval()
                z = torch.from_numpy(arr0[None]).to(dtype=next(ae.parameters()).dtype, device="cpu")
                with torch.no_grad():
                    img = ae.decode(z)
                img = img[0].permute(1, 2, 0).cpu()
                save_png_from_tensor(img, out_base + "_decoded.png")
                return
            except Exception as e:
                print("AE decode failed:", e)

        # No exact match — try common fallback: try upsampling to a larger AE if available
        # pick first config whose z_channels == c
        for name, spec in configs.items():
            if spec.ae_params.z_channels == c:
                print(f"No exact spatial match; trying AE '{name}' by upsampling latent to expected spatial size")
                try:
                    ae = load_ae(name, device="cpu")
                    ae.eval()
                    num_res = len(spec.ae_params.ch_mult)
                    expected = spec.ae_params.resolution // (2 ** (num_res - 1))
                    # upsample arr0 from (c,h,w) to (c,expected,expected)
                    z = torch.from_numpy(arr0[None]).float()
                    z_ups = torch.nn.functional.interpolate(z, size=(expected, expected), mode="bicubic", align_corners=False)
                    z_ups = z_ups.to(dtype=next(ae.parameters()).dtype)
                    with torch.no_grad():
                        img = ae.decode(z_ups)
                    img = img[0].permute(1, 2, 0).cpu()
                    save_png_from_tensor(img, out_base + f"_decoded_up_{name}.png")
                    return
                except Exception as e:
                    print("Upsample+decode failed:", e)
                    continue

    # Final fallback: quick channel mapping preview
    print("Falling back to quick channel-mapping preview PNG")
    mapped = quick_map_channels(arr0)
    Image.fromarray(mapped).save(out_base + "_quick.png")
    print("Saved fallback preview:", out_base + "_quick.png")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python examples/decode_unconditional_auto.py <path_to_npy>")
        sys.exit(1)
    main(sys.argv[1])
