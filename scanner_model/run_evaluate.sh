#!/bin/bash

#SBATCH -o myjob.out
#SBATCH -e myjob.err

LABELJSON="labels.json"
MODEL_PATH="pretrained_6mer/"
PEFT_PATH="../output/best_berten_718/"

DATA_PATH="../data/"
OUTPATH="../output/positive_evaulation/"
PICKLE="../data/positive.p"

# Command to be executed with the --normal flag
    # Add your normal command here
#source myconda
#mamba activate learning
python3 model_src/evaluate_model.py \
        --model_config "dna6" \
        --dnabert_path $MODEL_PATH \
        --peft_path $PEFT_PATH \
        --label_json $LABELJSON \
        --kmer 6 \
        --data_pickle $PICKLE \
        --run_name dnabert-enhancer \
        --model_max_length 512 \
        --per_device_eval_batch_size 16 \
        --output_dir $OUTPATH \
        --overwrite_output_dir True \
        --log_level info \

