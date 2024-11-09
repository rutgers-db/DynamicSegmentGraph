import pickle

class CompressedPoint:
    def __init__(self, external_id, ll, lr, rl, rr):
        self.external_id = external_id
        self.ll = ll
        self.lr = lr
        self.rl = rl
        self.rr = rr

    def __repr__(self):
        return f"CompressedPoint(external_id={self.external_id}, ll={self.ll}, lr={self.lr}, rl={self.rl}, rr={self.rr})"

def load_pickle_file(pickle_filename):
    with open(pickle_filename, 'rb') as pickle_file:
        data_chunks = pickle.load(pickle_file)
    return data_chunks

# Usage Example
pickle_filename = "../sample_data/path_nns_deep_96_sampled_5pct.pkl"
data_chunks = load_pickle_file(pickle_filename)

import matplotlib.pyplot as plt
from collections import defaultdict

# Group search path lengths by query range
search_path_lengths_by_range = defaultdict(list)

for query_range, search_path in data_chunks:
    # Calculate the length of each search_path and store it
    path_lengths = [len(nns) for _, nns in search_path]
    search_path_lengths_by_range[query_range].extend(path_lengths)

# Plot and save the distribution for each query range
for query_range, path_lengths in search_path_lengths_by_range.items():
    plt.figure()
    plt.hist(path_lengths, bins=10, edgecolor='black')
    plt.title(f"Distribution of Search Path Lengths for Query Range {query_range}")
    plt.xlabel("Search Path Length")
    plt.ylabel("Frequency")

    # Save the figure
    filename = f"./search_path_length_distribution_range_{query_range}.png"
    plt.savefig(filename)
    plt.close()  # Close the figure to free memory
