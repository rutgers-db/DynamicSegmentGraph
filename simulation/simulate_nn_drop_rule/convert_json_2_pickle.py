import json
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

def convert_json_to_pickle(json_filename, pickle_filename):
    with open(json_filename, 'r') as json_file:
        data = json.load(json_file)

    # Convert JSON data into Python objects
    data_chunks = []
    for item in data:
        query_range = item["query_range"]
        search_path = []
        
        for path in item["search_path"]:
            current_node_id = path["current_node_id"]
            nns = [CompressedPoint(**point) for point in path["nns"]]
            search_path.append((current_node_id, nns))
        
        data_chunks.append((query_range, search_path))

    # Save the structured data to a pickle file
    with open(pickle_filename, 'wb') as pickle_file:
        pickle.dump(data_chunks, pickle_file)

# Usage Example
json_filename = "../sample_data/path_nns_deep_96_sampled_5pct.json"
pickle_filename = "../sample_data/path_nns_deep_96_sampled_5pct.pkl"
convert_json_to_pickle(json_filename, pickle_filename)
