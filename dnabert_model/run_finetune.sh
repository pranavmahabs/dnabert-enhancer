DATA_PATH="/Users/pranavmahableshwarkar/CS/NIH/DNATransformerClassifier/data/"
MODEL_PATH="/Users/pranavmahableshwarkar/CS/NIH/DNATransformerClassifier/dnabert_model/pretrained_6mer/"
CONFIG_FILE="/Users/pranavmahableshwarkar/CS/NIH/DNATransformerClassifier/dnabert_model/pretrained_6mer/config.json"

# accelerate launch train.py \
#             --model_config "dna6" \
#             --model_name_or_path $MODEL_PATH \
#             --data_path  $DATA_PATH \
#             --kmer 6 \
#             --run_name dnabert-enhancer \
#             --model_max_length 512 \
#             --per_device_train_batch_size 32 \
#             --per_device_eval_batch_size 32 \
#             --gradient_accumulation_steps 1 \
#             --learning_rate ${lr} \
#             --num_train_epochs 8 \
#             --fp16 True \
#             --save_steps 200 \
#             --output_dir output/dnabert \
#             --evaluation_strategy steps \
#             --eval_steps 200 \
#             --warmup_steps 50 \
#             --logging_steps 100000 \
#             --overwrite_output_dir True \
#             --log_level info \
#             --find_unused_parameters False \
#             --use_mps_device

python3 train.py \
            --model_config "dna6" \
            --model_name_or_path $MODEL_PATH \
            --data_path  $DATA_PATH \
            --kmer 6 \
            --run_name dnabert-enhancer \
            --model_max_length 512 \
            --per_device_train_batch_size 32 \
            --per_device_eval_batch_size 32 \
            --gradient_accumulation_steps 1 \
            --learning_rate 2e-4 \
            --num_train_epochs 8 \
            --save_steps 200 \
            --output_dir output/dnabert/ \
            --evaluation_strategy steps \
            --eval_steps 200 \
            --warmup_steps 50 \
            --logging_steps 100000 \
            --overwrite_output_dir True \
            --log_level info \
            --find_unused_parameters False \
            --use_mps_device
            # --fp16 True \
