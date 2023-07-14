DATA_PATH="/data/Dcode/pranav/genoscanner/data/"
MODEL_PATH="/data/Dcode/pranav/genoscanner/scanner_model/pretrained_6mer/"
OUTPATH="/data/Dcode/pranav/genoscanner/output/"
PICKLE="/data/Dcode/pranav/genoscanner/scanner_model/supervised_dataset.p"

# Command to be executed with the --accelerate or -a flag
accelerate_command() {
    echo "Running accelerate command."
    accelerate launch train.py \
            --model_config "dna6" \
            --model_name_or_path $MODEL_PATH \
            --data_path  $DATA_PATH \
            --data_pickle $PICKLE \
            --kmer 6 \
            --run_name dnabert-enhancer \
            --model_max_length 512 \
            --per_device_train_batch_size 32 \
            --per_device_eval_batch_size 32 \
            --gradient_accumulation_steps 1 \
            --learning_rate 2e-4 \
            --num_train_epochs 8 \
            --fp16 True \
            --save_steps 200 \
            --output_dir $OUTPATH \
            --evaluation_strategy steps \
            --eval_steps 200 \
            --warmup_steps 50 \
            --logging_steps 100000 \
            --overwrite_output_dir True \
            --log_level info \
            --find_unused_parameters False \

}

# Command to be executed with the --normal flag
normal_command() {
    echo "Running normal command."
    # Add your normal command here
    python3 train.py \
            --model_config "dna6" \
            --model_name_or_path $MODEL_PATH \
            --data_path  $DATA_PATH \
            --kmer 6 \
            --data_pickle $PICKLE \
            --run_name dnabert-enhancer \
            --model_max_length 512 \
            --per_device_train_batch_size 16 \
            --per_device_eval_batch_size 16 \
            --gradient_accumulation_steps 1 \
            --learning_rate 2e-4 \
            --num_train_epochs 8 \
            --fp16 True \
            --save_steps 200 \
            --output_dir $OUTPATH \
            --evaluation_strategy steps \
            --eval_steps 200 \
            --warmup_steps 50 \
            --logging_steps 100000 \
            --overwrite_output_dir True \
            --log_level info \
            --find_unused_parameters False \

}

if [[ $# -eq 0 ]]; then
    # No flag provided, run the default command
    normal_command
    exit 0
fi

# Process command-line arguments
while [[ $# -gt 0 ]]; do
    case "$1" in
        --accelerate|-a)
            accelerate_command
            ;;
        --normal|-n)
            normal_command
            ;;
        *)
            echo "Invalid flag: $1"
            exit 1
            ;;
    esac
    shift
done
