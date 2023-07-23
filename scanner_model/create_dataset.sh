#SBATCH
#SBATCH
#SBATCH
#SBATCH


DATA_PATH=""
KMER=6
CONFIG="dna6" # This must MATCH the KMER. 
HG_FASTA=""

## 1. Activate a conda/mamba environment
# source myconda
# mamba activate <>

## 2. First Generate the TSV files using custom_process.py

python3 utils_dir/custom_preprocess.py \
    --positive-file "../data/positive.bed" \
    --negative-file "../data/negative.bed" \
    --generate-single-tsv \
    --single-bed-file "../data/positive.bed" \
    --fast-file $HG_FASTA \
    --k $KMER \
    --results-folder $DATA_PATH \

# 3. Generate the Pickle Files that Contain the Dataset

python3 model_src/data_dnabert.py $CONFIG $DATA_PATH

