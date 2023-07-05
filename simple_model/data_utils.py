import torch
import random
import numpy as np
from torch.utils.data import Dataset, DataLoader

window_size = 250
nucleotides = ["A", "T", "C", "G", "N"]
special_tokens = ["<pad>", "<mask>","<unk>","<cls>","<sep>",]
token_len = 1


def generate_kmers(file, letters, prefix, k):
    if k == 0:
        file.write(prefix + "\n")
        return

    for letter in letters:
        generate_kmers(file, letters, prefix + letter, k - 1)


def build_vocab(k, letters=nucleotides, vocab_file="vocab"):
    """
    There will be 4^4 possible 4-mers as the vocabulary with <pad> in
    case a shorter sequence is provided to the model.
    """
    if k <= 0:
        raise ValueError("k must be a positive integer. (i.e. 4)")
    with open(vocab_file, "w") as file:
        file.write("<pad>\n")
        file.write("<mask>\n")
        file.write("<unk>\n")
        file.write("<cls>\n")
        file.write("<sep>\n")
        generate_kmers(file, letters, "", k)


class tokenizer(object):
    def __init__(self, k, lang_file="vocab"):
        self.k = k
        self.kmer2idx = {}
        self.lang_file = lang_file
        self.vocab_size = 0

        # Needs to load the language.
        print("Loading and tokenizing vocabulary.")
        with open(lang_file, "r") as file:
            for line in file:
                word = line.strip()
                if len(word) != k and word not in special_tokens:
                    print(
                        f"Uncompatible word {word} must be of length in {k}.")
                    sys.exit()
                if word not in self.kmer2idx:
                    self.kmer2idx[word] = self.vocab_size
                    self.vocab_size += 1

        self.idx2kmer = {v: k for k, v in self.kmer2idx.items()}

    def ktokenize(self, sequence):
        sequence = sequence[0]
        # print(len(sequence))
        chopped = len(sequence) % self.k
        sequence = sequence[: len(sequence) - chopped]

        split_site = [sequence[i: i + self.k]
                      for i in range(0, len(sequence), self.k)]
        # if chopped > 0:
            # print(f"{chopped} nucleotides were chopped from the sequence.")
        return split_site

    def pad(self, sequence, window_size):
        seq_len = len(sequence)
        if seq_len < window_size:
            for i in range(window_size - seq_len):
                sequence.append("<pad>")
        return sequence

    def convert(self, token):
        if token not in self.kmer2idx.keys():
            return self.kmer2idx['<unk>']
        else:
            return self.kmer2idx[token]

    def toindex(self, tokenized):
        converter = np.vectorize(self.convert)
        return converter(tokenized)

    def randommask(self, sequence, proportion=0.15):
        N = int(proportion * len(sequence))
        to_mask = random.sample(range(len(sequence)), N)
        for index in to_mask:
            sequence[index] = "<mask>"

    def sentencetrain():
        pass


class SequenceDataset(Dataset):
    """Dataset Class Extension for Loader"""

    def __init__(self, sequences, masks, labels, device='cuda', num_classes=3):
        self.sequences = torch.from_numpy(sequences)
        self.labels = torch.from_numpy(labels)
        self.masks = torch.from_numpy(masks)

        if -1 in self.labels:
            self.labels += 1

        # Scatter the Labels
        labels_one_hot = torch.zeros(self.labels.size(0), num_classes)
        self.labels = labels_one_hot.scatter_(1, self.labels.unsqueeze(1), 1)
        
        if device != "None":
            self.labels = labels_one_hot.to(device) 
            self.sequences = self.sequences.to(device)
            # self.masks = (self.masks.double()).to(device)


    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        label = self.labels[idx]
        sequences = self.sequences[idx]
        # masks = self.masks[idx]
        sample = {"Sequence": sequences,
                #   "Mask": masks,
                  "Class": label}
        return sample
