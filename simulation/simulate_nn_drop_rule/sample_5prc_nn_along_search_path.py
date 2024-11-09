import struct
import random
import json

class CompressedPoint:
    def __init__(self, external_id, ll, lr, rl, rr):
        self.external_id = external_id
        self.ll = ll
        self.lr = lr
        self.rl = rl
        self.rr = rr

    def __repr__(self):
        return f"CompressedPoint(external_id={self.external_id}, ll={self.ll}, lr={self.lr}, rl={self.rl}, rr={self.rr})"

    def to_dict(self):
        return {
            'external_id': self.external_id,
            'll': self.ll,
            'lr': self.lr,
            'rl': self.rl,
            'rr': self.rr
        }


def read_bin_file(filename):
    data_chunks = []
    marker = 0x7fffffff
    marker_format = "i"  # Format for the marker integer (4 bytes)
    query_range_format = "I"  # Format for the query range (unsigned int, 4 bytes)
    node_id_format = "i"  # Format for current_node_id (4 bytes)
    size_format = "Q"  # Format for the vector size (unsigned long long, 8 bytes)
    point_format = "5I"  # Format for each CompressedPoint (5 unsigned integers, each 4 bytes)

    with open(filename, "rb") as f:
        while True:
            # Read the query range
            query_range_data = f.read(struct.calcsize(query_range_format))
            if not query_range_data:
                break  # End of file
            query_range = struct.unpack(query_range_format, query_range_data)[0]
            # Prepare to collect data for this query range
            search_path = []

            while True:
                # Read current_node_id
                node_id_data = f.read(struct.calcsize(node_id_format))
                if not node_id_data:
                    break  # End of file
                current_node_id = struct.unpack(node_id_format, node_id_data)[0]

                # If we encounter the marker, break out of the inner loop
                if current_node_id == marker:
                    break
                
                # Read the vector size
                size_data = f.read(struct.calcsize(size_format))
                size = struct.unpack(size_format, size_data)[0]

                # Read each CompressedPoint
                nns = []
                for _ in range(size):
                    point_data = f.read(struct.calcsize(point_format))
                    if not point_data:
                        break  # End of file unexpectedly
                    external_id, ll, lr, rl, rr = struct.unpack(point_format, point_data)
                    nns.append(CompressedPoint(external_id, ll, lr, rl, rr))

                # Append the data chunk for this current_node_id
                search_path.append((current_node_id, nns))

            # Append the entire query range section to the main list
            data_chunks.append((query_range, search_path))

    return data_chunks

def process_data_to_json(data_chunks, output_filename, sample_percentage=0.05):
    import json
    grouped_data = {}
    avg_path_sizes = {}

    # Group by query range
    for query_range, search_path in data_chunks:
        if query_range not in grouped_data:
            grouped_data[query_range] = []
        grouped_data[query_range].append(search_path)

    output_data = []

    for query_range, paths in grouped_data.items():
        # Calculate average search path size
        total_size = sum(len(path) for path in paths)
        avg_path_size = total_size / len(paths) if paths else 0
        avg_path_sizes[query_range] = avg_path_size

        # Randomly sample 5% of paths
        sample_size = max(1, int(len(paths) * sample_percentage))
        sampled_paths = random.sample(paths, sample_size)

        # Prepare data for JSON serialization
        for search_path in sampled_paths:
            search_path_data = []
            for current_node_id, nns in search_path:
                nns_data = [point.to_dict() for point in nns]
                search_path_data.append({
                    'current_node_id': current_node_id,
                    'nns': nns_data
                })
            output_data.append({
                'query_range': query_range,
                'search_path': search_path_data
            })

    # Write the output data to a JSON file
    with open(output_filename, 'w') as json_file:
        json.dump(output_data, json_file, indent=2)

    return avg_path_sizes

# Usage Example
filename = "../sample_data/path_nns_deep_96.bin"
output_filename = "../sample_data/path_nns_deep_96_sampled_5pct.json"
data = read_bin_file(filename)
avg_path_sizes = process_data_to_json(data, output_filename)

# Print the average search path sizes
for query_range, avg_size in avg_path_sizes.items():
    print(f"Query Range: {query_range}, Average Search Path Size: {avg_size}")