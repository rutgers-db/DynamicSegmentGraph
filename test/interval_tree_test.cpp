#include <iostream>
#include <vector>
#include <cstdlib>
#include <ctime>
#include <set>
#include <unordered_set>
#include <algorithm>
#include <chrono>

// Include the IntervalTree header
#include "IntervalTree.h"

using namespace std;

// Define the tuple structure
struct Tuple {
    int p;
    int ll;
    int lr;
    int rl;
    int rr;
};

// Function to generate random tuples
vector<Tuple> generate_tuples(int num_tuples, int pivot) {
    vector<Tuple> tuples;
    tuples.reserve(num_tuples);
    srand(time(NULL));
    for (int i = 0; i < num_tuples; ++i) {
        // Generate ll, lr, rl, rr satisfying ll <= lr <= pivot <= rl <= rr
        int num_1 = rand() % (pivot + 1);
        int num_2 = rand() % (pivot + 1);
        int ll = min(num_1, num_2);
        int lr = max(num_1, num_2);

        num_1 = rand() % (100000 - pivot + 1) + pivot;
        num_2 = rand() % (100000 - pivot + 1) + pivot;
        int rl = min(num_1, num_2);
        int rr = max(num_1, num_2);

        // Generate a p value (can be any value; indices will be used for uniqueness)
        int p_value = rand();
        tuples.push_back({p_value, ll, lr, rl, rr});
    }
    return tuples;
}

// Function to build interval trees
void build_interval_trees(const vector<Tuple>& tuples,
                          IntervalTree<int, int>& x_tree,
                          IntervalTree<int, int>& y_tree) {
    vector<Interval<int, int>> x_intervals;
    vector<Interval<int, int>> y_intervals;
    for (size_t idx = 0; idx < tuples.size(); ++idx) {
        const Tuple& t = tuples[idx];
        // IntervalTree uses half-open intervals [start, stop)
        x_intervals.push_back(Interval<int, int>(t.ll, t.lr + 1, idx));
        y_intervals.push_back(Interval<int, int>(t.rl, t.rr + 1, idx));
    }
    x_tree = IntervalTree<int, int>(std::move(x_intervals));
    y_tree = IntervalTree<int, int>(std::move(y_intervals));
}

// Function to perform queries using interval trees
vector<vector<int>> interval_tree_query(const IntervalTree<int, int>& x_tree,
                                        const IntervalTree<int, int>& y_tree,
                                        const vector<Tuple>& tuples,
                                        const vector<pair<int, int>>& queries) {
    vector<vector<int>> results;
    results.reserve(queries.size());

    size_t total_x_hits = 0;
    size_t total_y_hits = 0;
    size_t total_common_hits = 0;
    double total_ratio = 0.0;

    for (const auto& q : queries) {
        int L = q.first;
        int R = q.second;

        // Query x_tree with L
        std::vector<Interval<int, int>> x_hits = x_tree.findOverlapping(L, L + 1);

        // Query y_tree with R
        std::vector<Interval<int, int>> y_hits = y_tree.findOverlapping(R, R + 1);

        // Get the indices from the intervals
        unordered_set<int> x_indices;
        for (const auto& iv : x_hits) {
            x_indices.insert(iv.value);
        }
        unordered_set<int> y_indices;
        for (const auto& iv : y_hits) {
            y_indices.insert(iv.value);
        }

        // Compute the intersection
        vector<int> common_indices;
        for (const auto& idx : x_indices) {
            if (y_indices.find(idx) != y_indices.end()) {
                common_indices.push_back(idx);
            }
        }

        // Record sizes
        size_t size_x_hits = x_indices.size();
        size_t size_y_hits = y_indices.size();
        size_t size_common = common_indices.size();

        // Compute ratio
        double ratio = 0.0;
        if (size_x_hits + size_y_hits > 0) {
            ratio = static_cast<double>(size_common) / (size_x_hits + size_y_hits);
        }

        // Accumulate totals
        total_x_hits += size_x_hits;
        total_y_hits += size_y_hits;
        total_common_hits += size_common;
        total_ratio += ratio;

        // Append the matching indices to results
        results.push_back(common_indices);
    }

    // Compute averages
    size_t num_queries = queries.size();
    double avg_x_hits = static_cast<double>(total_x_hits) / num_queries;
    double avg_y_hits = static_cast<double>(total_y_hits) / num_queries;
    double avg_common_hits = static_cast<double>(total_common_hits) / num_queries;
    double avg_ratio = total_ratio / num_queries;

    cout << "Average size of x_hits: " << avg_x_hits << endl;
    cout << "Average size of y_hits: " << avg_y_hits << endl;
    cout << "Average size of common_indices: " << avg_common_hits << endl;
    cout << "Average ratio of common_indices over sum of x_hits and y_hits: " << avg_ratio << endl;

    return results;
}

// Function to perform queries using brute-force scanning
vector<vector<int>> brute_force_query(const vector<Tuple>& tuples,
                                      const vector<pair<int, int>>& queries) {
    vector<vector<int>> results;
    results.reserve(queries.size());
    for (const auto& q : queries) {
        int L = q.first;
        int R = q.second;
        vector<int> matching;
        for (size_t idx = 0; idx < tuples.size(); ++idx) {
            const Tuple& t = tuples[idx];
            if (t.ll <= L && L <= t.lr && t.rl <= R && R <= t.rr) {
                matching.push_back(idx);
            }
        }
        results.push_back(matching);
    }
    return results;
}

// Function to generate queries
vector<pair<int, int>> generate_queries(int num_queries, int pivot) {
    vector<pair<int, int>> queries;
    queries.reserve(num_queries);
    for (int i = 0; i < num_queries; ++i) {
        int L = rand() % (pivot + 1);
        int R = rand() % (100000 - pivot + 1) + pivot;
        queries.emplace_back(L, R);
    }
    return queries;
}

int main() {
    // Number of tuples and queries
    const int NUM_TUPLES = 10000;
    const int NUM_QUERIES = 1000;
    const int PIVOT = 100000 / 2;

    // Step 1: Generate tuples
    cout << "Generating tuples..." << endl;
    vector<Tuple> tuples = generate_tuples(NUM_TUPLES, PIVOT);

    // Step 2: Build Interval Trees
    cout << "Building Interval Trees..." << endl;
    IntervalTree<int, int> x_tree;
    IntervalTree<int, int> y_tree;
    build_interval_trees(tuples, x_tree, y_tree);

    // Step 3: Generate queries
    cout << "Generating queries..." << endl;
    vector<pair<int, int>> queries = generate_queries(NUM_QUERIES, PIVOT);

    // Step 4: Query using Interval Trees
    cout << "Querying using Interval Trees..." << endl;
    auto start = chrono::high_resolution_clock::now();
    vector<vector<int>> interval_tree_results = interval_tree_query(x_tree, y_tree, tuples, queries);
    auto end = chrono::high_resolution_clock::now();
    double interval_tree_time = chrono::duration<double>(end - start).count();
    cout << "Interval Trees query time: " << interval_tree_time << " seconds" << endl;

    // Step 5: Query using brute-force scanning
    cout << "Querying using brute-force scanning..." << endl;
    start = chrono::high_resolution_clock::now();
    vector<vector<int>> brute_results = brute_force_query(tuples, queries);
    end = chrono::high_resolution_clock::now();
    double brute_time = chrono::duration<double>(end - start).count();
    cout << "Brute-force query time: " << brute_time << " seconds" << endl;

    // Verification and collecting statistics
    size_t total_matches = 0;
    size_t min_matches = numeric_limits<size_t>::max();
    size_t max_matches = 0;

    for (size_t i = 0; i < NUM_QUERIES; ++i) {
        // Convert vectors to sets for comparison
        set<int> interval_set(interval_tree_results[i].begin(), interval_tree_results[i].end());
        set<int> brute_set(brute_results[i].begin(), brute_results[i].end());

        if (interval_set != brute_set) {
            cerr << "Mismatch in query " << i << endl;
            return 1;
        }
        size_t num_matches = interval_set.size();
        total_matches += num_matches;
        min_matches = min(min_matches, num_matches);
        max_matches = max(max_matches, num_matches);
    }

    double average_matches = static_cast<double>(total_matches) / NUM_QUERIES;

    cout << "Verification passed: Both methods return the same results." << endl;
    cout << "Total number of matching tuples across all queries: " << total_matches << endl;
    cout << "Average number of matching tuples per query: " << average_matches << endl;
    cout << "Minimum number of matches in a query: " << min_matches << endl;
    cout << "Maximum number of matches in a query: " << max_matches << endl;

    return 0;
}
