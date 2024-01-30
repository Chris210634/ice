#!/bin/bash

sh scripts/run_ice.sh coca_ViT-L-14 laion2B-s13B-b90k 768 0.08 COCA
sh scripts/run_ice.sh coca_ViT-L-14 laion2B-s13B-b90k 768 0.08 BLIP
sh scripts/run_ice.sh coca_ViT-L-14 laion2B-s13B-b90k 768 0.08 LLAVA

sh scripts/run_ice.sh ViT-B-16 laion2b_s34b_b88k 512 0.08 COCA
sh scripts/run_ice.sh ViT-B-16 laion2b_s34b_b88k 512 0.08 BLIP
sh scripts/run_ice.sh ViT-B-16 laion2b_s34b_b88k 512 0.08 LLAVA

sh scripts/run_ice.sh ViT-B-32 openai 512 0.08 COCA
sh scripts/run_ice.sh ViT-B-32 openai 512 0.08 BLIP
sh scripts/run_ice.sh ViT-B-32 openai 512 0.08 LLAVA

sh scripts/run_ice.sh ViT-L-14 laion2b_s32b_b82k 768 0.08 COCA
sh scripts/run_ice.sh ViT-L-14 laion2b_s32b_b82k 768 0.08 BLIP
sh scripts/run_ice.sh ViT-L-14 laion2b_s32b_b82k 768 0.08 LLAVA

sh scripts/run_ice.sh ViT-L-14 openai 768 0.08 COCA
sh scripts/run_ice.sh ViT-L-14 openai 768 0.08 BLIP
sh scripts/run_ice.sh ViT-L-14 openai 768 0.08 LLAVA

sh scripts/run_ice.sh ViT-H-14 laion2b_s32b_b79k 1024 0.08 COCA
sh scripts/run_ice.sh ViT-H-14 laion2b_s32b_b79k 1024 0.08 BLIP
sh scripts/run_ice.sh ViT-H-14 laion2b_s32b_b79k 1024 0.08 LLAVA

sh scripts/run_ice.sh ViT-g-14 laion2b_s34b_b88k 1024 0.08 COCA
sh scripts/run_ice.sh ViT-g-14 laion2b_s34b_b88k 1024 0.08 BLIP
sh scripts/run_ice.sh ViT-g-14 laion2b_s34b_b88k 1024 0.08 LLAVA

sh scripts/run_ice.sh ViT-L-14 datacomp_xl_s13b_b90k 768 0.08 COCA
sh scripts/run_ice.sh ViT-L-14 datacomp_xl_s13b_b90k 768 0.08 BLIP
sh scripts/run_ice.sh ViT-L-14 datacomp_xl_s13b_b90k 768 0.08 LLAVA

sh scripts/run_ice.sh ViT-L-14-CLIPA datacomp1b 768 0.08 COCA
sh scripts/run_ice.sh ViT-L-14-CLIPA datacomp1b 768 0.08 BLIP
sh scripts/run_ice.sh ViT-L-14-CLIPA datacomp1b 768 0.08 LLAVA

sh scripts/run_ice.sh ViT-H-14-CLIPA datacomp1b 1024 0.08 COCA
sh scripts/run_ice.sh ViT-H-14-CLIPA datacomp1b 1024 0.08 BLIP
sh scripts/run_ice.sh ViT-H-14-CLIPA datacomp1b 1024 0.08 LLAVA