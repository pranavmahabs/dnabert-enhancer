wndow_size = 250
nucleotides = ["A", "T", "C", "G"]
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
        file.write("<pad>")
        file.write("<mask>")
        file.write("<unk>")
        file.write("<cls>")
        file.write("<sep>")
        generate_kmers(file, letters, "", k)


class tokenizer(object):
    def __init__(self, k, lang_file="vocab"):
        self.k = k
        self.kmer2idx = {}
        self.lang_file = lang_file
        self.vocab_size = 0

        ## Needs to load the language.
        print("Loading and tokenizing vocabulary.")
        with open(lang_file, "r") as file:
            for line in file:
                word = line.strip()
                if len(word) != k and word != "<pad>":
                    print(f"Uncompatible word {word} must be of length in {k}.")
                    exit
                if word not in self.kmer2idx:
                    self.kmer2idx[self.vocab_size] = vocab_size
                    vocab_size += 1

        self.idx2kmer = {v: k for k, v in self.kmer2idx.items()}

    def ktokenize(self, sequence):
        chopped = len(sequence) % self.k
        sequence = sequence[: len(sequence) - chopped]

        split_site = [sequence[i : i + self.k] for i in range(0, len(sequence), self.k)]
        print(f"{chopped} nucleotides were chopped from the sequence.")
        return split_site

    def pad(self, sequence, window_size):
        seq_len = len(sequence)
        if seq_len < window_size:
            for i in range(window_size - seq_len):
                sequence.append("<pad>")
        return sequence

    def toindex(self, tokenized):
        for index, kmer in enumerate(tokenized):
            tokenized[index] = self.kmer2idx[kmer]

    def randommask(self, sequence, proportion=0.15):
        N = int(proportion * len(sequence))
        to_mask = random.sample(range(len(sequence)), N)
        for index in to_mask:
            sequence[index] = "<mask>"

    def sentencetrain():
        pass


