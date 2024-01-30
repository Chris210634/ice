#!/bin/bash

# Usage:
# sh scripts/run_ice.sh coca_ViT-L-14 laion2B-s13B-b90k 768 0.08 COCA
# sh scripts/run_ice.sh coca_ViT-L-14 laion2B-s13B-b90k 768 0.08 BLIP
# sh scripts/run_ice.sh coca_ViT-L-14 laion2B-s13B-b90k 768 0.08 LLAVA

# Example:
# python main_ice.py \
#  --modelname coca_ViT-L-14 --pretrained laion2B-s13B-b90k --d 768 \
# --ice --ice_k 5 --ice_lambda 0.08 --captioner COCA

python main_ice.py \
 --modelname $1 --pretrained $2 --d $3 \
--ice --ice_k 5 --ice_lambda $4 --captioner $5