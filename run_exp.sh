#!/usr/bin/env bash

python3 lm_pretraining.py --outdir baseline

python3 lm_pretraining.py --weight_sharing --outdir weight_sharing

python3 lm_pretraining.py --fused_qkv --outdir fused_qkv

