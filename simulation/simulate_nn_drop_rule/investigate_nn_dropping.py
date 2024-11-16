import pickle
from collections import defaultdict
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
max_elements = 100000

# Group search path lengths by query range
search_path_lengths_by_range = defaultdict(list)
for query_range, search_path in data_chunks:
    search_path_lengths_by_range[query_range].append(search_path)

def drop_useless_points(search_path, max_length=128):
    for point in search_path:
        if point.rr > max_elements:
            # print(point.rr)
            point.rr = max_elements
            
        # Calculate left and right ranges
        left_range = point.lr - point.ll + 1
        right_range = point.rr - point.rl + 1
        
        # Calculate gap between lr and rl
        gap = max(0, point.rl - point.lr - 1)
        
        # Calculate total covered range
        # covered_range = left_range + right_range
        covered_range = left_range * right_range
        # Calculate total span from ll to rr
        # total_span = point.rr - point.ll + 1
        total_span = (point.rr - point.external_id + 1) * (point.external_id - point.ll  + 1)

        # Calculate coverage ratio
        coverage_ratio = covered_range / total_span if total_span > 0 else 0
        
        # Compute usefulness score
        # You can adjust the formula based on your priority
        point.usefulness_score = coverage_ratio # * total_span

    # Sort points by usefulness score in descending order
    search_path_sorted = sorted(search_path, key=lambda p: p.usefulness_score, reverse=True)

    # Select top `max_length` points
    search_path_selected = search_path_sorted[:max_length]

    return search_path_selected

#  A function check if there are some boundary not occurred in external ids and current node id 
def check_any_boundary_lost_in_external_id(search_path, current_node_id):
    # Collect all external IDs in the search path
    external_ids = {point.external_id for point in search_path}
    external_ids.add(current_node_id)

    # List to store unique points
    missing_values = set()

    # Check each point to see if any of its ll, lr, rl, or rr values are in the external_ids
    for point in search_path:
        # If none of the point's ll, lr, rl, rr values are in external_ids, add it to unique_points
        if point.ll != 0 and point.ll - 1 not in external_ids:
            missing_values.add(point.ll- 1 )
        if point.lr not in external_ids:
            missing_values.add(point.lr)
        if point.rl not in external_ids:
            missing_values.add(point.rl)
        if point.rr != max_elements and point.rr + 1 not in external_ids:
        # if point.rr != 200000 and point.rr + 1 not in external_ids:
            missing_values.add(point.rr + 1)

    return missing_values

# Set the maximum allowed length of each search path
max_length = 128
total_unexplorable_paths_count = 0
total_unexplorable_points_count = 0

for query_range, paths in search_path_lengths_by_range.items():
    # Initialize counters for the metrics
    total_dropped_points = 0
    total_points = 0
    dropped_nns_count = 0
    total_nns_amount = 0
    unexplorable_paths_count = 0
    unexplorable_points_count = 0

    for path in paths:
        original_node_ids = {current_node_id for current_node_id, _ in path}  # Collect original current_node_ids
        all_retained_external_ids = set()  # To store external_ids after dropping

        for i, (current_node_id, nns) in enumerate(path):
            original_length = len(nns)
            total_points += original_length
            total_nns_amount += 1
              
            # Drop useless points if needed
            if original_length > max_length:
                dropped_nns_count += 1
                # Apply the dropping rule
                nn_selected = drop_useless_points(nns, max_length=max_length)
                
                lost_points =check_any_boundary_lost_in_external_id(nn_selected, current_node_id)       
                if lost_points:
                    print("Lost points:", lost_points)  

                # Calculate number of points dropped
                dropped_points = original_length - len(nn_selected)
                total_dropped_points += dropped_points
                
                # Update the `nns` with the reduced points
                path[i] = (current_node_id, nn_selected)
                
                # Collect `external_id`s from retained points after dropping
                all_retained_external_ids.update(point.external_id for point in nn_selected)
            else:
                # No dropping needed, retain original `nns`
                path[i] = (current_node_id, nns)
                all_retained_external_ids.update(point.external_id for point in nns)

        # Check for unexplorable current_node_ids
        unexplorable_node_ids = original_node_ids - all_retained_external_ids
        if unexplorable_node_ids:
            unexplorable_paths_count += 1  # Increment if there are any unexplorable IDs
            # print(unexplorable_node_ids)
            unexplorable_points_count += len(unexplorable_node_ids)

    # Calculate the average number of points dropped per path for this query range
    average_dropped_points = total_dropped_points / dropped_nns_count if dropped_nns_count > 0 else 0
    total_unexplorable_paths_count += unexplorable_paths_count
    total_unexplorable_points_count += unexplorable_points_count
    # Output the metrics for this query range
    print(f"Query Range: {query_range}")
    print(f"Total Points Before Dropping: {total_points}")
    print(f"Total Points Dropped: {total_dropped_points}")
    print(f"Average Dropped Points Per NNS (for NNS that required dropping): {average_dropped_points:.2f}")
    print(f"Total NNS Amount: {total_nns_amount}")
    print(f"NNS That Required Dropping: {dropped_nns_count}")
    print(f"Unexplorable Points {unexplorable_points_count} in Unexplorable Paths Count: {unexplorable_paths_count} in {len(paths)} Paths")
    print("----")

print(f"Total unexporable Paths Count: {total_unexplorable_paths_count}")
print(f"Total unexporable Points Count: {total_unexplorable_points_count}")