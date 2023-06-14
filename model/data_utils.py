wndow_size = 250
nucleotides = ['A', 'T', 'C', 'G']
token_len = 1

def generate_kmers(file, letters, prefix, k):
    if k == 0:
        file.write(prefix + "\n")
        return
    
    for letter in letters:
        generate_kmers(file, letters, prefix + letter, k - 1)

def build_vocab(letters=nucleotides, k):
    """
    There will be 4^4 possible 4-mers as the vocabulary with <pad> in
    case a shorter sequence is provided to the model. 
    """
    if k <= 0:
        raise ValueError("k must be a positive integer. (i.e. 4)")
    with open("vocab", "w") as file:
        generate_kmers(file, letters, "", k)
        file.write('<pad>')

# build_vocab(nucleotides, token_len)

class tokenizer(object):
    def __init__(self, k, lang_file="vocab")
        self.k = k
        self.kmer2idx = {}
        self.vocab_size = 0

        ## Needs to load the language.
        with open(lang_file, "r") as file:
            for line in file:
                word = line.strip()
                if word not in kmer2idx:
                    kmer2idx[self.vocab_size] = vocab_size
                    vocab_size += 1:q
                    :q

        
        self.idx2kmer = {v:k for k,v in self.kmer2idx.items()}

    def ktokenize(self, sequence, k):
        chopped = len(sequence) % self.k
        sequence = sequence[:len(sequence) - chopped]

        split_site = ' '.join([sequence[i:i+self.k] for i in range(0, len(sequence), self.k)])
        print(f'{chopped} nucleotides were chopped from the sequence.')
        return split_site

    def pad(self, sequence, window_size):
        seq_len = len(sequence)
        if seq_len < window_size:
            for i in range(window_size - seq_len):
                sequence += " <pad>"
        return sequence

    def toindex(self, tokenized):
        for index, kmer in enumerate(tokenized):
            tokenized[index] = self.kmer2idx[kmer]
