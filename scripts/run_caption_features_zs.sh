#!/bin/bash
# zero-shot without the image features component

python main_ice.py \
 --modelname $1 --pretrained $2 --d $3 \
--ice --ice_k 5 --captioner $4 --use_fixed_lambda --use_all_logits --ice_lambda 1.0