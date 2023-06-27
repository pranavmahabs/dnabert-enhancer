import pandas as pd
import numpy as np
from Bio import SeqIO
from pybedtools import BedTool

## From DNABERT paper...
def seq2kmer(seq, k):
    """ss
    
    Arguments:
    seq -- str, original sequence.
    k -- int, kmer of length k specified.
    
    Returns:
    kmers -- str, kmers separated by space

    """
    kmer = [seq[x:x+k] for x in range(len(seq)+1-k)]
    kmers = " ".join(kmer)
    return kmers