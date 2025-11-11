#!/bin/bash

GUIDANCE=1.0
SEED=42
NAME="flux-dev"
SIZE=256
NUM_STEPS=50
PROMPT="a cat" # leave it "" for unconditional generation

python infer_cli.py \
    --name=$NAME \
    --prompt=$PROMPT \
    --width=$SIZE \
    --height=$SIZE \
    --num_steps=$NUM_STEPS \
    --guidance=$GUIDANCE \
    --seed=$SEED \
    --device="cuda" \
    --offload # reduce VRAM usage

