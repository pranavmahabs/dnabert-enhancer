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

## Dataset

In my training of DNABERT-Enhancer, I utilized around 360000 negative samples (noise from the genome) and around 12000 samples for each of my positive classes. This extreme imbalance is necessary for the model to learn what sequences are truly unique to the enhancers. (Note: the loss function accounts for class imbalance, see `CustomTrainer` in `model/transformer_src/train.py`).

You will need:

* `positive.bed`: A BED file containing all the BED-formatted information for positive samples. Depending on your preprocessing, the fourth, fifth, etc. columns should allow you to determine the label.
* `negative.bed`: A BED file containing all the BED-formatted information for negative samples. 
* `genome.ga`: You will need to a provide a genome fa and fai file (i.e. hg38.fa and hg38.fa.fai)

It is highly recommended that your dataset for positive and negative samples for the entire genome as the data is set up to be split based on chromosome. If you want to a different split that can be done below.

### Custom Preprocessing

**Step One: Setup Python Scripts**

You will need to update `custom_preprocess.py` and `data_dnabert.py` in `model/transformer_src/` to work with your labeling scheme - hence the name custom_preprocess. Lines 126-130 are how I handle labeling for my positive class. In particular, I use information in each row of my BED file to assign the label to that sample. If you are using binary classification then the positive label can always be 1. You will notice that negative samples are automatically set with a label of 0.

Please be aware of **lines 87-89** where I have included a label transformation specific to my model. Feel free to transform your labels here as well - though if none of your labels are set to -1, this should not be an issue.

**Step Two: `create_dataset.sh`**

Open this Shell script in `model/` and enter the appropriate information (location of your data directory and the path for your genome FASTA file). These are the expected outputs:

* **TSV Files:** K-merized sequence files that contain a sequence column and label column. There will be four of these files: train.tsv, val.tsv, test.tsv, and positive.tsv. `positive.tsv` contains all of the samples in positive.bed with their labels. 
* **supervised_dataset.p**: The pickle file for *fine-tuning*. This contains the Supervised Datasets from the training set, the validation set, and the testing set.
* **evaluation.p:** The pickle file for *evaluation*. This contains the Supervised Datasets generated from the testing set and the positive set. The positive set is useful for evaluation.


## Fine-Tuning
**Step One: Customize Metrics**

Currently the training script calculates AUC scores for classes 0 and 2 during validation steps. Once training is completed, the model also calculates AUC scores and FPR logit thresholds (for FPRs of 0.1, 0.05, 0.03, and 0.01) for classes 0 and 2. Feel free to modify which classes these values are calculated for. To modify which metrics are calculated, go to **line 169** which includes `calculate_metric_with_sklearn` which is the metrics function used in validation and `compute_auc_fpr_thresholds` which is the metrics function used in final model evaluation on the test set.

**Step Two: `labels.json`**

Fill out the labels.json file following the same format as the template. Make sure that this is accurate.

**Step Three: `run_finetune.sh`**

Unlike classic fine-tuning, DNABERT-Enhancer used LoRA (Low Rank Adaptation for Language Models). Instead of training all 90 million model parameters in the pre-trained model, this strategy uses Peft to place matrices between transformer layers and trains those instead. In my case, I had less than 5 million trainable parameters using this approach. 

Note on GPUS: When I trained the model, I used 4 P100 NVIDIA GPUs allocating 120GB memory for each. Please keep this in mind when setting your batch_size for training and evaluation. 
