import pickle
import os

with open("../data/full_data_tsv/supervised_dataset.p", "rb") as handle:
    dataset = pickle.load(handle)

with open("../data/positive.p", "rb") as handle2:
    pos_dataset = pickle.load(handle2)

pos_dataset["train"] = dataset["train"]
print(pos_dataset.keys())

with open("../../data/full_data_tsv/evaluate.p", "wb") as handle:
    pickle.dump(pos_dataset, handle, protocol=pickle.HIGHEST_PROTOCOL)
