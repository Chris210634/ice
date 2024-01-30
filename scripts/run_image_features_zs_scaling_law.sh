#!/bin/bash

sh scripts/run_image_features_zs.sh coca_ViT-L-14 laion2B-s13B-b90k 768
sh scripts/run_image_features_zs.sh ViT-B-16 laion2b_s34b_b88k 512
sh scripts/run_image_features_zs.sh ViT-B-32 openai 512
sh scripts/run_image_features_zs.sh ViT-L-14 laion2b_s32b_b82k 768
sh scripts/run_image_features_zs.sh ViT-L-14 openai 768
sh scripts/run_image_features_zs.sh ViT-H-14 laion2b_s32b_b79k 1024
sh scripts/run_image_features_zs.sh ViT-g-14 laion2b_s34b_b88k 1024
sh scripts/run_image_features_zs.sh ViT-L-14 datacomp_xl_s13b_b90k 768
sh scripts/run_image_features_zs.sh ViT-L-14-CLIPA datacomp1b 768
sh scripts/run_image_features_zs.sh ViT-H-14-CLIPA datacomp1b 1024