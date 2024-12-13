#!/bin/bash

torchrun --standalone --nproc_per_node=1 train_gpt2.py \
    --input_bin "$HOME/fineweb10B/fineweb_train_*.bin" \
    --input_val_bin "$HOME/fineweb10B/fineweb_val_*.bin" \
    --val_loss_every 8 \
    --hellaswag_every 8 \
    --sample_every 8 \
    --save_every 8 \
    --output_dir pylog_gpt2_124M \
    --model d12 \
    --batch_size 64 \
    --sequence_length 1024 \
    --total_batch_size 262144 \
    --dtype bfloat16 \
    --compile 1 \
    --tensorcores 1 \
    --flash 1 \
    --num_iterations 18865 \
    --weight_decay 0.1 \
    --zero_stage 1 \
    --learning_rate 0.0015 \
    --warmup_iters 256 \
    --learning_rate_decay_frac 0.1 \
    --overfit_single_batch 0
