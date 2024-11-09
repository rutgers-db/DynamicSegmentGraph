import random
import time
from rtree import index
import numpy as np

# Number of tuples and queries
NUM_TUPLES = 10000
NUM_QUERIES = 1000
PIVOT = 100000/2
# Generate random 5-tuples satisfying the constraints
def generate_tuples(num_tuples, p):
    tuples = []
    p_values = set()
    while len(tuples) < num_tuples:
        # Generate ll, lr, rl, rr satisfying ll <= lr <= pivot <= rl <= rr
        num_1 = random.randint(0, p)
        num_2 = random.randint(0, p)
        ll = min(num_1, num_2)
        lr = max(num_1, num_2)
        num_1 = random.randint(p, 100000)
        num_2 = random.randint(p, 100000)
        rl = min(num_1, num_2)
        rr = max(num_1, num_2)
        
        tuples.append((p, ll, lr, rl, rr))
    return tuples

# Build R-tree index
def build_rtree(tuples):
    p_idx = index.Property()
    p_idx.dimension = 2  # 2D R-tree
    p_idx.variant = index.RT_Star  # Use R*-tree variant
    rtree_idx = index.Index(properties=p_idx)
    
    for idx, (p, ll, lr, rl, rr) in enumerate(tuples):
        # The rectangle is defined by [ll, lr] and [rl, rr]
        rtree_idx.insert(idx, (ll, rl, lr, rr), obj=(p, ll, lr, rl, rr))
    return rtree_idx

# Perform queries using R-tree
def rtree_query(rtree_idx, queries):
    results = []
    for L, R in queries:
        hits = list(rtree_idx.intersection((L, R, L, R), objects=True))
        results.append([hit.object for hit in hits])
    return results

# Perform queries using brute-force scanning
def brute_force_query(tuples, queries):
    results = []
    for L, R in queries:
        matching = []
        for (p, ll, lr, rl, rr) in tuples:
            if ll <= L <= lr and rl <= R <= rr:
                matching.append((p, ll, lr, rl, rr))
        results.append(matching)
    return results

# Generate random queries where L <= pivot <= R
def generate_queries(num_queries, pivot):
    queries = []
    for _ in range(num_queries):
        L = random.randint(0, pivot)
        R = random.randint(pivot, 100000)
        queries.append((L, R))
    return queries

def main():
    # Step 1: Generate tuples
    print("Generating tuples...")
    tuples = generate_tuples(NUM_TUPLES, PIVOT)
    
    # Step 2: Build R-tree index
    print("Building R-tree index...")
    rtree_idx = build_rtree(tuples)
    
    # Step 3: Generate queries
    print("Generating queries...")
    queries = generate_queries(NUM_QUERIES, PIVOT)
    
    # Step 5: Query using brute-force
    print("Querying using brute-force scanning...")
    start_time = time.time()
    brute_results = brute_force_query(tuples, queries)
    brute_time = time.time() - start_time
    print(f"Brute-force query time: {brute_time:.4f} seconds")

    # Step 4: Query using R-tree
    print("Querying using R-tree...")
    start_time = time.time()
    rtree_results = rtree_query(rtree_idx, queries)
    rtree_time = time.time() - start_time
    print(f"R-tree query time: {rtree_time:.4f} seconds")
    
    # Optional: Verify that both methods return the same results
    # Initialize a variable to keep track of the total number of matching tuples
    # Initialize variables to collect statistics
    total_matches = 0
    min_matches = float('inf')
    max_matches = 0

    # Verification and collecting statistics
    for i in range(NUM_QUERIES):
        rtree_set = set(rtree_results[i])
        brute_set = set(brute_results[i])
        assert rtree_set == brute_set, f"Mismatch in query {i}"
        num_matches = len(rtree_set)
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
