#!/usr/bin/env bash
# run_all.sh - 15 experiments (nohup parallel with pt_data dirs)

PT=./pt_data


# KoCLIP (text) + KoCLIP (image)
CUDA_VISIBLE_DEVICES=1 nohup python main.py \
    --start_dir ${PT}/koclip_pt_start --end_dir ${PT}/koclip_pt_end --image_dir ${PT}/koclip_pt_image \
    --use_start --use_end --use_image --log_path logs/data_all_koclip_SE_koclip_I.log \
    > logs/data_all_koclip_SE_koclip_I.log 2>&1 &

# KcBERT (text) + ViT (image)
CUDA_VISIBLE_DEVICES=1 nohup python main.py \
    --start_dir ${PT}/kcbert_pt_start --end_dir ${PT}/kcbert_pt_end --image_dir ${PT}/vit_pt_image \
    --use_start --use_end --use_image --log_path logs/data_all_kcbert_SE_vit_I.log \
    > logs/data_all_kcbert_SE_vit_I.log 2>&1 &

# KcBERT (text) + Resnet (image)
CUDA_VISIBLE_DEVICES=1 nohup python main.py \
    --start_dir ${PT}/kcbert_pt_start --end_dir ${PT}/kcbert_pt_end --image_dir ${PT}/resnet_pt_image \
    --use_start --use_end --use_image --log_path logs/data_all_kcbert_SE_resnet_I.log \
    > logs/data_all_kcbert_SE_resnet_I.log 2>&1 &

# KoELECTRA (text) + ViT (image)
CUDA_VISIBLE_DEVICES=1 nohup python main.py \
    --start_dir ${PT}/koelectra_pt_start --end_dir ${PT}/koelectra_pt_end --image_dir ${PT}/vit_pt_image \
    --use_start --use_end --use_image --log_path logs/data_all_koelectra_SE_vit_I.log \
    > logs/data_all_koelectra_SE_vit_I.log 2>&1 &

# KoELECTRA (text) + Resnet (image)
CUDA_VISIBLE_DEVICES=1 nohup python main.py \
    --start_dir ${PT}/koelectra_pt_start --end_dir ${PT}/koelectra_pt_end --image_dir ${PT}/resnet_pt_image \
    --use_start --use_end --use_image --log_path logs/data_all_koelectra_SE_resnet_I.log \
    > logs/data_all_koelectra_SE_resnet_I.log 2>&1 &

# ##############
# # S + I 조합 #
# ##############

# KoCLIP (start) + KoCLIP (image)
CUDA_VISIBLE_DEVICES=1 nohup python main.py \
  --start_dir ${PT}/koclip_pt_start --image_dir ${PT}/koclip_pt_image \
  --use_start --use_image --log_path logs/data_all_koclip_S_koclip_I.log \
  > logs/data_all_koclip_S_koclip_I.log 2>&1 &

# KcBERT (start) + ViT (image)
CUDA_VISIBLE_DEVICES=1 nohup python main.py \
  --start_dir ${PT}/kcbert_pt_start --image_dir ${PT}/vit_pt_image \
  --use_start --use_image --log_path logs/data_all_kcbert_S_vit_I.log \
  > logs/data_all_kcbert_S_vit_I.log 2>&1 &

# KcBERT (start) + ResNet (image)
CUDA_VISIBLE_DEVICES=1 nohup python main.py \
  --start_dir ${PT}/kcbert_pt_start --image_dir ${PT}/resnet_pt_image \
  --use_start --use_image --log_path logs/data_all_kcbert_S_resnet_I.log \
  > logs/data_all_kcbert_S_resnet_I.log 2>&1 &

# KoELECTRA (start) + ViT (image)
CUDA_VISIBLE_DEVICES=1 nohup python main.py \
  --start_dir ${PT}/koelectra_pt_start --image_dir ${PT}/vit_pt_image \
  --use_start --use_image --log_path logs/data_all_koelectra_S_vit_I.log \
  > logs/data_all_koelectra_S_vit_I.log 2>&1 &

# KoELECTRA (start) + ResNet (image)
CUDA_VISIBLE_DEVICES=1 nohup python main.py \
  --start_dir ${PT}/koelectra_pt_start --image_dir ${PT}/resnet_pt_image \
  --use_start --use_image --log_path logs/data_all_koelectra_S_resnet_I.log \
  > logs/data_all_koelectra_S_resnet_I.log 2>&1 &


# ##############
# # E + I 조합 #
# ##############

# KoCLIP (end) + KoCLIP (image)
CUDA_VISIBLE_DEVICES=0 nohup python main.py \
  --end_dir ${PT}/koclip_pt_end --image_dir ${PT}/koclip_pt_image \
  --use_end --use_image --log_path logs/data_all_koclip_E_koclip_I.log \
  > logs/data_all_koclip_E_koclip_I.log 2>&1 &

# KcBERT (end) + ViT (image)
CUDA_VISIBLE_DEVICES=0 nohup python main.py \
  --end_dir ${PT}/kcbert_pt_end --image_dir ${PT}/vit_pt_image \
  --use_end --use_image --log_path logs/data_all_kcbert_E_vit_I.log \
  > logs/data_all_kcbert_E_vit_I.log 2>&1 &

# KcBERT (end) + ResNet (image)
CUDA_VISIBLE_DEVICES=0 nohup python main.py \
  --end_dir ${PT}/kcbert_pt_end --image_dir ${PT}/resnet_pt_image \
  --use_end --use_image --log_path logs/data_all_kcbert_E_resnet_I.log \
  > logs/data_all_kcbert_E_resnet_I.log 2>&1 &

# KoELECTRA (end) + ViT (image)
CUDA_VISIBLE_DEVICES=0 nohup python main.py \
  --end_dir ${PT}/koelectra_pt_end --image_dir ${PT}/vit_pt_image \
  --use_end --use_image --log_path logs/data_all_koelectra_E_vit_I.log \
  > logs/data_all_koelectra_E_vit_I.log 2>&1 &

# KoELECTRA (end) + ResNet (image)
CUDA_VISIBLE_DEVICES=0 nohup python main.py \
  --end_dir ${PT}/koelectra_pt_end --image_dir ${PT}/resnet_pt_image \
  --use_end --use_image --log_path logs/data_all_koelectra_E_resnet_I.log \
  > logs/data_all_koelectra_E_resnet_I.log 2>&1 &

echo ">>> All 15 jobs launched. Check logs/data_all_*.log for outputs."

