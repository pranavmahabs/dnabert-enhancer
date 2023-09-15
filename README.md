# DNABERT-Enhancer

An interface to fine-tune DNABERT on a multi-label dataset including a downstream pipeline to extract/visualize attention scores, visualize classification results, and more. 

## Setup

Training DNABERT-Enhancer requires access to at least 1 GPU with significant memory (150+ GB) due to the significant amount of data required to train the model. 

In addition you will need to have a conda/mamba/python environment of some form that contains all the necessary packages. These can be found and installed from `requirements.txt`. Here is how to do this using a conda environment - though a python venv would also work very well. 

```bash
$ conda create -n dnabert-enhancer
$ conda activate dnabert-enhancer
$ pip install -r requirements.txt
```

### Installing DNABERT-6
DNABERT-Enhancer is fine-tuned model of [DNABERT](https://github.com/jerryji1993/DNABERT/tree/master). DNABERT supports four different tokenizations of DNA (DNABERT-3,4,5 and 6). This model uses DNABERT-6.

You can download the pre-trained model [here](https://drive.google.com/file/d/1BJjqb5Dl2lNMg2warsFQ0-Xvn1xxfFXC/view).

Once you make this download, you will see 5 files present. All you need is `pytorch_model.bin`. Please drag and drop this file into `model/pretrained_6mer` as the other four (much smaller files) are already present in that directory. 

## Dataset

In my training of DNABERT-Enhancer, I utilized around 360000 negative samples (noise from the genome) and around 12000 samples for each of my positive classes. This extreme imbalance is necessary for the model to learn what sequences are truly unique to the enhancers. (Note: the loss function accounts for class imbalance, see `CustomTrainer` in `model/transformer_src/train.py`).

You will need:

* `positive.bed`: A BED file containing all the BED-formatted information for positive samples. Depending on your preprocessing, the fourth, fifth, etc. columns should allow you to determine the label.
* `negative.bed`: A BED file containing all the BED-formatted information for negative samples. 
* `genome.fa`: You will need to a provide a genome fa and fai file (i.e. hg38.fa and hg38.fa.fai)

It is highly recommended that your dataset for positive and negative samples for the entire genome as the data is set up to be split based on chromosome. If you want to a different split that can be done below.

### Custom Preprocessing

**Step One: Setup Python Scripts**

You will need to update `custom_preprocess.py` and `data_dnabert.py` in `model/transformer_src/` to work with your labeling scheme - hence the name custom_preprocess. In particular, I use information in each row of my BED file to assign the label to that sample. To do this I use a labeling function that uses the values in the BED File. This is `get_positive_labels` for my dataset but you can fill in the `custom_label_function` and pass it in to the files that either create a dataset or a single TSV file. If you are using binary classification then the positive label can always be 1. You will notice that negative samples are automatically set with a label of 0.

Please be aware of **lines 87-89** in `data_dnabert.py` where I have included a label transformation specific to my model. Feel free to transform your labels here as well - though if none of your labels are set to -1, this should not be an issue.

**Step Two: `create_dataset.sh`**

Open this Shell script in `model/` and enter the appropriate information (location of your data directory and the path for your genome FASTA file). 

```bash
$ sbatch create_dataset.sh --cpus-per-task=24 --time=08:00:00 --mem=40g
```

These are the expected outputs:

* **TSV Files:** K-merized sequence files that contain a sequence column and label column. There will be four of these files: train.tsv, val.tsv, test.tsv, and positive.tsv. `positive.tsv` contains all of the samples in positive.bed with their labels. 
* **supervised_dataset.p**: The pickle file for *fine-tuning*. This contains the Supervised Datasets from the training set, the validation set, and the testing set.
* **evaluation.p:** The pickle file for *evaluation*. This contains the Supervised Datasets generated from the testing set and the positive set. The positive set is useful for evaluation.

### Create Your Own Evaluation Dataset

If you want to run a random BED file for evaluation on the model that is separate from the positive/dataset pickle files, you can do the following. Go to `create_miniset.sh` and edit the respective information. An example of the key area would look something like this:

```bash
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

```

## Fine-Tuning
**Step One: Customize Metrics**

Currently the training script calculates AUC scores for classes 0 and 2 during validation steps. Once training is completed, the model also calculates AUC scores and FPR logit thresholds (for FPRs of 0.1, 0.05, 0.03, and 0.01) for classes 0 and 2. Feel free to modify which classes these values are calculated for. To modify which metrics are calculated, go to **line 169** which includes `calculate_metric_with_sklearn` which is the metrics function used in validation and `compute_auc_fpr_thresholds` which is the metrics function used in final model evaluation on the test set.

**Step Two: `labels.json`**

Fill out the labels.json file following the same format as the template. Make sure that this is accurate.

```json
{
    "metadata": {
        "num_labels" : 3
    },
    "label2id": {
        "Poised Enhancer" : 0,
        "Noise" : 1,
        "Active Enhancer" : 2
    }
}
```

**Step Three: `run_finetune.sh`**

Unlike classic fine-tuning, DNABERT-Enhancer used LoRA (Low Rank Adaptation for Language Models). Instead of training all 90 million model parameters in the pre-trained model, this strategy uses Peft to place matrices between transformer layers and trains those instead. In my case, I had less than 5 million trainable parameters using this approach. 

Open the fine-tuning script and enter the data-paths. Additionally, fell free to change any of the hyperparameters that are currently sit to fit your model needs. Enter your desired output directory. Make sure to choose a new directory as this will overwrite results from a previous successful run. **Note on GPUS**: When I trained the model, I used 4 NVIDIA TESLA P100 GPUs allocating 120GB memory for each. Please keep this in mind when setting your batch_size for training and evaluation. 

```bash
$ sbatch run_finetune.sh --gres=gpu:p100:4 --time=32:00:00 --mem=120g --cpus-per-task=16
```

The following, amongst other outputs, will be produced in the output directory. There will be THREE (can be changed) models saved, each with separate outputs of the following. It is up to you to choose which one you think is the best. 

* **`eval_results.json`**: Contains all of your metrics chosen for model evaluation. Only will be produced for the model with the lowest loss (will not be computed for each of the three best models).
* **`trainer_state.json`**: Contains all of the validation loss information at each step of the model in addition to metrics calculated at each step. 
* **Other Files**: The other files that are output can be used to perform the evaluation step. 

If you use the same parameters that are currently in the files, the training with 400000 samples took around 8 hours. While it is certainly possible to not use LoRA, it will make training the model infeasible with the sheer number of samples necessary to train the model. 


## Evaluation and Prediction
Prediction is handled in DNABERT-Enhancer using `run_evaluate.sh` in the same model directory. Unlike training, it is only run on **one** GPU. To run the prediction, you will need to fill in the following:

* `MODEL_PATH`: The path to your DNABERT-6.
* `PEFT_PATH`: The path to your fine-tuned model - this will be somewhere in your output directory. Feel free to see what this will look like in `output/best_berten_718/` (though some outputs have been removed there).
* `PICKLE`: The path to your `evaluate.p` file. 

This script is configured to run on the `evaluate.p` file. Internally, the pickle file stores a dictionary that contains the datasets for testing and the positive  dataset. If you want to run evaluation again, make sure to include the following flag: `--re_eval True` to your shell script. **However**, if you only want to run on a single file and get attention scores/logits, do **not** include this flag and provide the path to your desired pickle file. Running the evaluation script should take around 15-20 minutes on around 30000 samples.

```bash
$ sbatch --partition=gpu --gres=gpu:p100:1 --cpus-per-task=16 --mem=120g --time=08:00:00 run_evaluate.sh
```

These are the following outputs:

* `eval_results.json`: It will rerun the evaluation that is performed at the end of the train function (contains AUCs, FPR Thresholds).
* `atten.npy`: The normalized average attention scores for every single provided sequence. This is a numpy array of size (pos_data_len, seq_len)
* `unnorm_atten.npy`: The unnormalized average attention scores for every single provided positive sequence.
* `head_atten.npy`: This is the normalized average attention score for every single for every sequence. This is a numpy array with dims (pos_data_len x 12 x seq_len)
* `pred_results.npy`: The logit scores for each of the classes - **not** the predicted labels. This is a numpy array of size (pos_data_len, num_labels).

These outputs can be fed into downstream analysis tasks.

### Running Evaluation on Custom Sets

If you generated a single tsv in the preprocess pipeline, then that pickle file - which stores a dictionary - would not have a key-value pair for 'test'. The file should automatically detect this. It is important to also mark the flag `--re_eval False`. Then, all you have to do is provide the file-path and output path on the top of evaluate script and you are good to go.

## Downstream Analysis
### Enhancer Explore
The `enhancer_explore.ipynb` in the `analysis` folder provides a sample analysis pipeline on how to process and analyze the aforementioned numpy arrays that are produced in evaluation. Please note that some utility functions are hidden in `explore_utils.py` to decrease the size of the notebook. In particular, some of the plotting functions for the bar graphs are included there. 

Note that a lot of my analysis involves specific comparison/BED files that are included in the Git Repo.

### Attention Explore
This is certainly the trickier explore notebook - and leaves more preference to the user. When you run an evaluation/prediction with `run_evaluate.sh`, there are series of resulting numpy arrays that are produced. The attention scores and the ground labels are read in. 

Ultimately, the attention analysis pipeline is up to your own desired analysis goals. In the current version, the following is present:

* Average Attention plots for sequences in each category of enhancers
* Random plotting of specific samples to visualize varying attention plots
* Sequence extraction of high attention regions from the genome (single and multi-head available). These sequences can be fed into FIMO to perform motif enrichment (as seen in `enhancer_explore.ipynb`).

**Motif Specific Enrichment Plots**

An understandable application of these attention plots is visualizing attention scores in motif regions and proximal areas. Right now, `generate_story` does this alongside `intersect.sh` in `analysis/results/bed_files/`. Note how indices are used throughout BOTH of these exploration notebooks to ensure that the proper sequence and regions are analyzed.
