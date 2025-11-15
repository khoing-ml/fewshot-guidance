#!/bin/bash

GUIDANCE=1.5
SEED=41
NAME="flux-dev"
SIZE=512
NUM_STEPS=50
PROMPT="a kitty with yellow fur sleeping , white background" # leave it "" for unconditional generation

python infer_cli.py \
    --name=$NAME \
    --prompt="$PROMPT" \
    --width=$SIZE \
    --height=$SIZE \
    --num_steps=$NUM_STEPS \
    --guidance=$GUIDANCE \
    --seed=$SEED \
    --device="cuda" \
    --offload # reduce VRAM usage

