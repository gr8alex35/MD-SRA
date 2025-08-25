#!/bin/bash
# run_all_embedding.sh
# Embed all three modalities

# KoCLIP
CUDA_VISIBLE_DEVICES=0 nohup python utils/tokenize_start.py --model koclip \
    > nohup_0_start_koclip.txt 2>&1 &
CUDA_VISIBLE_DEVICES=0 nohup python utils/tokenize_end.py --model koclip \
    > nohup_0_end_koclip.txt 2>&1 &
CUDA_VISIBLE_DEVICES=0 nohup python utils/tokenize_img.py --encoder koclip \
    > nohup_0_img_koclip.txt 2>&1 &

# KoBERT
CUDA_VISIBLE_DEVICES=1 nohup python utils/tokenize_start.py --model kcbert --max_len 300\
    > nohup_1_start_kcbert.txt 2>&1 &

CUDA_VISIBLE_DEVICES=1 nohup python utils/tokenize_end.py --model kcbert --max_len 300\
    > nohup_1_end_kcbert.txt 2>&1 &

# KoELECTRA
CUDA_VISIBLE_DEVICES=1 nohup python utils/tokenize_start.py --model koelectra \
    > nohup_2_start_koelectra.txt 2>&1 &
CUDA_VISIBLE_DEVICES=1 nohup python utils/tokenize_end.py --model koelectra \
    > nohup_2_end_koelectra.txt 2>&1 &


# ViT
CUDA_VISIBLE_DEVICES=0 nohup python utils/tokenize_img.py --encoder vit \
    > nohup_3_img_vit.txt 2>&1 &

# ResNet
CUDA_VISIBLE_DEVICES=0 nohup python utils/tokenize_img.py --encoder resnet \
    > nohup_3_img_resnet.txt 2>&1 &


