#!/bin/bash

#SBATCH -o myjob.out
#SBATCH -e myjob.err

LABELJSON="labels.json"
MODEL_PATH="pretrained_6mer/"

DATA_PATH="../data/"
OUTPATH="../output/"
PICKLE="../supervised_dataset.p"
NUM_GPUS=4

# Command to be executed with the --normal flag
    # Add your normal command here
source myconda
mamba activate learning
LOCAL_RANK=$(seq 0 $((NUM_GPUS - 1))) CUDA_VISIBLE_DEVICE=$(seq 0 $((NUM_GPUS - 1))) \
torchrun --nproc_per_node $NUM_GPUS model_src/train.py \
        --model_config "dna6" \
        --model_name_or_path $MODEL_PATH \
        --data_path  $DATA_PATH \
        --kmer 6 \
        --data_pickle $PICKLE \
        --label_json $LABELJSON \
        --run_name dnabert-enhancer \
        --model_max_length 512 \
        --use_lora \
        --lora_target_modules 'query,value,key,dense' \
        --per_device_train_batch_size 16 \
        --per_device_eval_batch_size 16 \
        --gradient_accumulation_steps 2 \
        --learning_rate 2e-4 \
        --num_train_epochs 15 \
        --fp16 True \
        --save_steps 400 \
        --output_dir $OUTPATH \
        --evaluation_strategy steps \
        --eval_steps 400 \
        --warmup_steps 200 \
        --logging_steps 100000 \
        --overwrite_output_dir True \
        --log_level info \
        --find_unused_parameters False \

