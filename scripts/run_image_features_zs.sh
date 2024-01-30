#!/bin/bash
# zero-shot without the caption component

python main_ice.py \
 --modelname $1 --pretrained $2 --d $3 \
--ice --ice_k 5 --use_fixed_lambda --use_all_logits --ice_lambda 0.0 --v 1