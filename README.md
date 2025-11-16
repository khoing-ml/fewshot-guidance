# Installation
```bash
git clone https://github.com/khoing-ml/fewshot-guidance.git
cd fewshot-guidance
conda create -n khoi
conda activate khoi
conda install pip
pip install .
### monitor gpu usage 
watch -n 1 nvidia-smi
```

# Sampling from Original Model
## 1. Basic Text-to-Image Sampling
```bash
python infer_cli.py --prompt "a beautiful sunset over mountains" --width 768 --height 768 --num_steps 50
```

## 2. Programmatic Sampling (Python)
```python
import torch
from flux.sampling import denoise, get_noise, get_schedule, prepare, unpack
from flux.util import load_ae, load_clip, load_flow_model, load_t5, save_image

# Setup
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model_name = "flux-dev"  # or "flux-schnell"

# Load models
t5 = load_t5(device, max_length=512)
clip = load_clip(device)
model = load_flow_model(model_name, device=device)
ae = load_ae(model_name, device=device)

# Image parameters
width, height = 1024, 1024
num_steps = 50 if model_name == "flux-dev" else 4
guidance = 3.5  # guidance value (ignored for schnell)
seed = 42

# Generate noise
x = get_noise(
    num_samples=1,
    height=height,
    width=width,
    device=device,
    dtype=torch.bfloat16,
    seed=seed,
)

# Prepare conditioning (text embeddings, latents, etc.)
prompt = "a beautiful sunset over mountains"
inp = prepare(t5, clip, x, prompt=prompt)

# Get timestep schedule
timesteps = get_schedule(num_steps, inp["img"].shape[1], shift=(model_name != "flux-schnell"))

# Denoise (main sampling loop)
with torch.inference_mode():
    x = denoise(model, **inp, timesteps=timesteps, guidance=guidance)

# Decode latents to image
x = unpack(x.float(), height, width)
with torch.autocast(device_type=device.type, dtype=torch.bfloat16):
    x = ae.decode(x)

# Save or process image
# x is now a tensor of shape [1, 3, height, width]
from torchvision.utils import save_image
save_image(x, "output.jpg")
```