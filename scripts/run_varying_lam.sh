#!/bin/bash

echo start > blip_lams.o
for ((i=0; i<=10; i++)); do
    current_value=$(echo "scale=1; $i/10" | bc)
    echo "Iteration $i: lam=$current_value"
    sh scripts/run_ice.sh coca_ViT-L-14 laion2B-s13B-b90k 768 $current_value BLIP >> blip_lams.o
done

echo start > llava_lams.o
for ((i=0; i<=10; i++)); do
    current_value=$(echo "scale=1; $i/10" | bc)
    echo "Iteration $i: lam=$current_value"
    sh scripts/run_ice.sh coca_ViT-L-14 laion2B-s13B-b90k 768 $current_value LLAVA >> llava_lams.o
done

echo start > coca_lams.o
for ((i=0; i<=10; i++)); do
    current_value=$(echo "scale=1; $i/10" | bc)
    echo "Iteration $i: lam=$current_value"
    sh scripts/run_ice.sh coca_ViT-L-14 laion2B-s13B-b90k 768 $current_value COCA >> coca_lams.o
done