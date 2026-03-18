#!/bin/bash
# DO NOT use GPTQ/AWQ model in FSDP+QLoRA

#CUDA_VISIBLE_DEVICES=0,1 accelerate launch --config_file config/fsdp_config.yaml src/train.py config/qwen3vl_lora_sft.yaml 
#> ./logs/qwen3vl-4b-think.out 2>&1 &

CUDA_VISIBLE_DEVICES=0,1 llamafactory-cli export config/qwen3vl_merge_lora.yaml
#CUDA_VISIBLE_DEVICES=0,1 llamafactory-cli export config/qwen3vl_gptq.yaml
