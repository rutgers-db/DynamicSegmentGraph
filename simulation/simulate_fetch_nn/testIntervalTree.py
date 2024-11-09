import random
import time
from intervaltree import Interval, IntervalTree

# Number of tuples and queries
NUM_TUPLES = 10000
NUM_QUERIES = 1000
PIVOT = 100000 / 2

# Generate random 5-tuples satisfying the constraints
def generate_tuples(num_tuples, p):
    tuples = []
    while len(tuples) < num_tuples:
        # Generate ll, lr, rl, rr satisfying ll <= lr <= pivot <= rl <= rr
        num_1 = random.randint(0, int(p))
        num_2 = random.randint(0, int(p))
        ll = min(num_1, num_2)
        lr = max(num_1, num_2)
        num_1 = random.randint(int(p), 100000)
        num_2 = random.randint(int(p), 100000)
        rl = min(num_1, num_2)
        rr = max(num_1, num_2)

        # Generate a unique p value if necessary
        p_value = random.randint(0, 1000000)
        tuples.append((p_value, ll, lr, rl, rr))
    return tuples

# Build Interval Trees
def build_interval_trees(tuples):
    x_interval_tree = IntervalTree()
    y_interval_tree = IntervalTree()

    for idx, (p, ll, lr, rl, rr) in enumerate(tuples):
        # Add intervals to the interval trees
        x_interval_tree[ll:lr + 1] = idx  # IntervalTree uses half-open intervals
        y_interval_tree[rl:rr + 1] = idx
    return x_interval_tree, y_interval_tree

# Perform queries using Interval Trees and compute average ratio
def interval_tree_query(x_tree, y_tree, tuples, queries):
    results = []
    total_x_hits = 0
    total_y_hits = 0
    total_common_hits = 0
    total_ratio = 0.0

    for L, R in queries:
        # Query x_tree with L
        x_hits = x_tree.at(L)
        # Query y_tree with R
        y_hits = y_tree.at(R)
        # Get the indices from the intervals
        x_indices = set(iv.data for iv in x_hits)
        y_indices = set(iv.data for iv in y_hits)
        common_indices = x_indices & y_indices

        # Record sizes
        size_x_hits = len(x_indices)
        size_y_hits = len(y_indices)
        size_common = len(common_indices)

        # Compute ratio
        if size_x_hits + size_y_hits > 0:
            ratio = size_common / (size_x_hits + size_y_hits)
        else:
            ratio = 0.0  # Avoid division by zero

        # Accumulate totals
        total_x_hits += size_x_hits
        total_y_hits += size_y_hits
        total_common_hits += size_common
        total_ratio += ratio

        # Retrieve the matching tuples
        matching = [tuples[idx] for idx in common_indices]
        results.append(matching)

    # Compute averages
    num_queries = len(queries)
    avg_x_hits = total_x_hits / num_queries
    avg_y_hits = total_y_hits / num_queries
    avg_common_hits = total_common_hits / num_queries
    avg_ratio = total_ratio / num_queries

    print(f"Average size of x_hits: {avg_x_hits:.2f}")
    print(f"Average size of y_hits: {avg_y_hits:.2f}")
    print(f"Average size of common_indices: {avg_common_hits:.2f}")
    print(f"Average ratio of common_indices over sum of x_hits and y_hits: {avg_ratio:.4f}")

    return results

# Perform queries using brute-force scanning
def brute_force_query(tuples, queries):
    results = []
    for L, R in queries:
        matching = []
        for idx, (p, ll, lr, rl, rr) in enumerate(tuples):
            if ll <= L <= lr and rl <= R <= rr:
                matching.append((p, ll, lr, rl, rr))
        results.append(matching)
    return results

# Generate random queries where L <= pivot <= R
def generate_queries(num_queries, pivot):
    queries = []
    for _ in range(num_queries):
        L = random.randint(0, int(pivot))
        R = random.randint(int(pivot), 100000)
        queries.append((L, R))
    return queries

def main():
    # Step 1: Generate tuples
    print("Generating tuples...")
    tuples = generate_tuples(NUM_TUPLES, PIVOT)
    
    # Step 2: Build Interval Trees
    print("Building Interval Trees...")
    x_tree, y_tree = build_interval_trees(tuples)
    
    # Step 3: Generate queries
    print("Generating queries...")
    queries = generate_queries(NUM_QUERIES, PIVOT)
    
    # Step 4: Query using Interval Trees
    print("Querying using Interval Trees...")
    start_time = time.time()
    interval_tree_results = interval_tree_query(x_tree, y_tree, tuples, queries)
    interval_tree_time = time.time() - start_time
    print(f"Interval Trees query time: {interval_tree_time:.4f} seconds")
    
    # Step 5: Query using brute-force
    print("Querying using brute-force scanning...")
    start_time = time.time()
    brute_results = brute_force_query(tuples, queries)
    brute_time = time.time() - start_time
    print(f"Brute-force query time: {brute_time:.4f} seconds")
    
    # Verification and collecting statistics
    total_matches = 0
    min_matches = float('inf')
    max_matches = 0
    
    for i in range(NUM_QUERIES):
        interval_set = set(interval_tree_results[i])
        brute_set = set(brute_results[i])
        assert interval_set == brute_set, f"Mismatch in query {i}"
        num_matches = len(interval_set)
        total_matches += num_matches
        min_matches = min(min_matches, num_matches)
        max_matches = max(max_matches, num_matches)
    
    average_matches = total_matches / NUM_QUERIES
    
    print("Verification passed: Both methods return the same results.")
    print(f"Total number of matching tuples across all queries: {total_matches}")
    print(f"Average number of matching tuples per query: {average_matches:.2f}")
    print(f"Minimum number of matches in a query: {min_matches}")
    print(f"Maximum number of matches in a query: {max_matches}")

if __name__ == "__main__":
    main()
