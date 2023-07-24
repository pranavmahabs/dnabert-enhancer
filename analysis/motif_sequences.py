"""
Extractions Regions of High Attention and Exports to Fasta File.
This FASTA file will be analyzed FIMO and other motif exploration tools.
"""

import sys
import os

current_dir = os.path.dirname(os.path.abspath(__file__))
main_dir = os.path.dirname(current_dir)

utils_path = os.path.join(main_dir, "model", "utils_dir")
sys.path.append(utils_path)

# Import Helper Functions from utils_code
import motif_utils as utils
