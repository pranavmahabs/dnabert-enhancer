# Author: Pranav Mahableshwarkar
# Last Modified: 08-07-2021
# Description: This script is used to generate smaller datasets from single BED files.

#SBATCH
#SBATCH
#SBATCH
#SBATCH

DATA_PATH=""
KMER=6
CONFIG="dna6" # This must MATCH the KMER. 
HG_FASTA=""

# 1. Activate a conda/mamba environment
# source myconda
# mamba activate <>

# 2. First Generate the TSV files using custom_process.py
python3 utils_dir/custom_preprocess.py \
    --generate-single-tsv \
    --single-bed-file "../data/sample.bed" \
    --fast-file $HG_FASTA \
    --k $KMER \
    --results-folder $DATA_PATH \

# 3. Generate the Pickle Files that Contain the Dataset
python3 model_src/data_dnabert.py \
    --single_file "../data/sample.bed" \
    --single_name "sample" \
    --config $CONFIG \
    --file_base $DATA_PATH \
