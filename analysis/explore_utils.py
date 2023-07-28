import numpy as np
import matplotlib.pyplot as plt
import pandas as pd


def process_intersection(bedlist: list):
    total_intersections = len(bedlist)
    tf_map = {}
    for bed in bedlist:
        tf = bed[3]
        if tf not in tf_map:
            tf_map[tf] = 1
        else:
            tf_map[tf] += 1
    return tf_map, total_intersections


def determine_fold_change(base, experimental, base_name, exp_name, sizes):
    tsv = f"base_{base_name}_exp_{exp_name}_FC.tsv"
    tsv_content = "tf\tbase_occur\texp_occur\tfoldchange\n"
    fold_map = {}

    num_b = sizes[base_name]
    num_e = sizes[exp_name]

    all_tfs = list(experimental.keys()) + list(base.keys())
    for tf in all_tfs:
        if tf in base and tf in experimental:
            b_occur = base[tf]
            e_occur = experimental[tf]
            # fold_change = (e_occur / num_e) / (b_occur / num_b)
            fold_change = ((e_occur / num_e) - (b_occur / num_b)) / (b_occur / num_b)
            tsv_content += f"{tf}\t{b_occur}\t{e_occur}\t{fold_change}\n"
            fold_map[tf] = fold_change
        elif tf in base and tf not in experimental:
            b_occur = base[tf]
            e_occur = 0
            fold_change = "nan"
            tsv_content += f"{tf}\t{b_occur}\t{e_occur}\t{fold_change}\n"
        elif tf not in base and tf in experimental:
            b_occur = 0
            e_occur = experimental[tf]
            fold_change = "nan"
            tsv_content += f"{tf}\t{b_occur}\t{e_occur}\t{fold_change}\n"

    with open(tsv, "w") as tsv_file:
        tsv_file.write(tsv_content)
    return fold_map


def plot_histogram(fold_map, base, experimental, exclude):
    # Sort the dictionary by fold change values
    sorted_tf_fold_changes = sorted(fold_map.items(), key=lambda x: x[1])

    # Separate positive and negative fold changes
    positive_fold_changes = [fc for fc in fold_map.values() if fc > 0]
    negative_fold_changes = [fc for fc in fold_map.values() if fc < 0]

    # Calculate mean and standard deviation for positive fold changes
    positive_mean = np.mean(positive_fold_changes)
    positive_std = np.std(positive_fold_changes)

    # Exclude positive fold change values more than three standard deviations away from the mean
    excluded_positive_tfs = [
        (tf, fc)
        for tf, fc in fold_map.items()
        if fc > 0 and fc > positive_mean + exclude * positive_std
    ]

    # Calculate mean and standard deviation for negative fold changes
    negative_mean = np.mean(negative_fold_changes)
    negative_std = np.std(negative_fold_changes)

    # Exclude negative fold change values more than three standard deviations away from the mean
    excluded_negative_tfs = [
        (tf, fc)
        for tf, fc in fold_map.items()
        if fc < 0 and fc < negative_mean - exclude * negative_std
    ]

    # Combine the excluded TFs and their fold changes
    excluded_tfs = excluded_positive_tfs + excluded_negative_tfs

    # Remove the excluded TFs from the original dictionary
    for tf, fc in excluded_tfs:
        print(f"Excluded TF: {tf}, Fold Change: {fc}")
        del fold_map[tf]

    # Sort the remaining TFs by their fold change values
    sorted_tf_fold_changes = sorted(fold_map.items(), key=lambda x: x[1])
    sorted_tf_fold_changes.reverse()

    # Extract TF names and fold changes
    tf_names = [tf[0] for tf in sorted_tf_fold_changes]
    fold_changes = [tf[1] for tf in sorted_tf_fold_changes]

    # Define colors for positive and negative fold changes
    colors = ["blue" if fc < 0 else "red" for fc in fold_changes]

    # Create the bar plot
    plt.figure(figsize=(12, 6))  # Adjust the figure size as needed
    plt.bar(tf_names, fold_changes, color=colors)
    plt.xticks(
        rotation=45, ha="right", fontsize=8
    )  # Rotate and align the x-axis labels
    plt.xlabel("Transcription Factor")
    plt.ylabel("Fold Change")
    plt.title(f"Transcription Factor Fold Changes {experimental} Compared to {base}")
    plt.tight_layout()  # Ensure labels fit within the figure

    # Show the plot
    plt.show()
