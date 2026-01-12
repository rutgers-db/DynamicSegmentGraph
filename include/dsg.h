// Author: Zhencan Peng, 2025/11/30
/**
 * @file dsg.h
 * @brief Dynamic Segment Graph (DSG) header. Rebuilt from compact_graph.h
 * @details This header declares the Dynamic Segment Graph, an implementation that first builds a temporary HNSW,
 *          runs ef_max-sized neighbor searches for every node, applies a DFS-based compression over the neighbors,
 *          and stores only forward segment edges (reverse edges and insertions will be implemented later).
 *          The class inherits `BaseIndex` so it can plug into the existing indexing/search pipeline while exposing
 *          range-filtering queries over compressed segment neighbors. The file introduces the basic data structures
 *          (segment edges, per-node containers, DFS scratch buffers) together with the public API needed for building,
 *          querying, and persisting the DSG index. Insert/update workflows are intentionally omitted in this version
 *          and will be added after the core rebuild is complete.
 */

#pragma once

#include <cstdint>
#include <memory>
#include <queue>
#include <utility>
#include <vector>

#include "base_hnsw/hnswalg.h"
#include "data_wrapper.h"
#include "base_index.h"

namespace dsg {

using hnswlib::tableint;

/**
 * @brief Scratchpad buffers reused during DFS.
 */
struct DfsScratch {
    /// Sorted (id, distance) pairs produced by the temporary HNSW search.
    std::vector<std::pair<unsigned, float>> ordered_candidates;
    /// Current DFS prefix indexes into ordered_candidates.
    std::vector<unsigned> prefix;
    /// Marks whether a candidate has already been accepted as a neighbor.
    std::vector<bool> is_neighbor;
    /// Per-candidate left-lower boundary.
    std::vector<unsigned> left_lower;
    /// Per-candidate left-upper boundary.
    std::vector<unsigned> left_upper;
    /// Per-candidate right-lower boundary.
    std::vector<unsigned> right_lower;
    /// Per-candidate right-upper boundary.
    std::vector<unsigned> right_upper;
    /// Cache of domination checks between candidate pairs.
    /// Uses a flattened 2D grid (row-major): index = prev_idx * num_candidates + curr_idx.
    /// 0: unknown, 1: dominated, 2: not dominated.
    std::vector<uint8_t> domination_grid;
};

/**
 * @brief Dynamic Segment Graph that relies on a temporary HNSW during build.
 */
class DynamicSegmentGraph : public BaseIndex {
public:
    using DistType = float;
    using HnswType = hnswlib::HierarchicalNSW<DistType>;

    struct InsertionStats {
        std::uint64_t inserted = 0;
        std::uint64_t recompress_calls = 0;
        double range_search_seconds = 0.0;
        double dfs_seconds = 0.0;
        double add_reverse_seconds = 0.0;
        double recompress_seconds = 0.0;
    };

    /**
     * @brief Construct an empty DSG with the given data wrapper and L2 space.
     */
    DynamicSegmentGraph(hnswlib::SpaceInterface<DistType> *space,
                        const DataWrapper *data);
    /**
     * @brief Release the temporary HNSW and any scratch buffers.
     */
    DynamicSegmentGraph() = default;
    ~DynamicSegmentGraph() override;

    /**
     * @brief Build the DSG over a subset of labels.
     */
    void build(const std::vector<unsigned> &labels) override;
    /**
     * @brief Execute a range query inside [query_bound.first, query_bound.second].
     *        Results are written into `returned_nns`.
     */
    void rangeSearch(const float *query,
                     const std::pair<int, int> query_bound) override;

    /**
     * @brief Persist the compressed forward edges to disk.
     */
    void save(const std::string &file_path) override;
    /**
     * @brief Load a previously saved DSG from disk.
     */
    void load(const std::string &file_path) override;

    /**
     * @brief Configure extra per-row slack when loading.
     *
     * @details If `frac > 0`, `load()` will expand each row capacity by
     *          ceil(deg.fwd * frac) and reset deg.rev to 0, enabling dynamic
     *          reverse-edge insertion after loading. If `frac == 0`, `load()`
     *          keeps a tight CSR layout (static).
     */
    void setLoadSlackFraction(double frac) { load_slack_fraction_ = frac; }

    /**
     * @brief Pre-reserve memory for graph storage (insertion optimization).
     *
     * @details DSG stores the graph in global SoA vectors. Repeated insertions can trigger
     *          costly reallocations when these buffers grow. If you can estimate the final
     *          number of rows (nodes) and total edge capacity (including slack), call this
     *          before inserting to reduce reallocations.
     *
     * @param total_rows_capacity Expected final number of rows (nodes) in the index.
     * @param total_edge_capacity Expected final total edge capacity across all rows.
     */
    void reserveGraphStorage(std::size_t total_rows_capacity,
                             std::size_t total_edge_capacity);

    /**
     * @brief Insert a new node into the graph.
     * @param label The external label (ID) of the node to insert. Must be unique.
     */
    void insert(unsigned label);

    void resetInsertionStats() noexcept { insertion_stats_ = InsertionStats{}; }
    InsertionStats insertionStats() const noexcept { return insertion_stats_; }

    /**
     * @brief Report index statistics to stdout.
     */
    void getStats();
    /// Last query hop count.
    std::size_t last_hop_count() const noexcept { return last_hop_count_; }
    /// Last query distance evaluation count.
    std::size_t last_distance_eval_count() const noexcept { return last_distance_eval_count_; }

private:
    /// @brief Scratchpad buffers reused during per-node level-0 KNN candidate generation.
    /// @details We implement a heap-based version of HNSW base-layer traversal in DSG
    ///          and reuse these buffers across labels to avoid repeated allocations.
    struct KnnScratch {
        std::vector<std::pair<DistType, tableint>> top_candidates_heap;
        std::vector<std::pair<DistType, tableint>> candidate_set_heap;
        std::vector<std::pair<DistType, tableint>> removed_candidates;
    };

    /// Allocate and build the temporary HNSW used for candidate generation.
    void initializeTemporaryHnsw(size_t ef_limit);
    /// Run an ef_max-sized search in the temporary HNSW for the target label.
    void runKnnForLabel(unsigned label,
                        size_t ef_limit,
                        std::vector<std::pair<unsigned, DistType>> &candidates);
    /// Apply DFS-based dominance pruning and produce segment ranges.
    void applyDfsCompression(unsigned center_label,
                             std::vector<std::pair<unsigned, DistType>> &candidates);

    /// Insertion-only candidate generator on the current DSG graph.
    /// Returns candidates in nearest-first order with ReturnAll(top-candidates-ever) semantics.
    void insertionInnerSearch(const float *query,
                              const std::pair<int, int> query_bound,
                              std::size_t ef_limit,
                              std::vector<std::pair<unsigned, DistType>> &out);
    /// Move the compressed neighbors from scratch buffers into forward_edges_.
    // Note: The signature of storeForwardEdges might change in implementation to adapt to SoA,
    // or we might accumulate in a temporary buffer first.
    // Since we are refactoring to SoA, we'll use a temporary structure in build() 
    // and then populate the member vectors.
    // We keep this declaration as a helper if needed, or remove it if the logic moves to build().
    // For now, let's assume we'll handle storage logic inside build() or a helper.
    
private:
    /// Distance space used to construct the temporary HNSW.
    hnswlib::SpaceInterface<DistType> *space_ = nullptr;
    /// Temporary HNSW instance used only during build.
    std::unique_ptr<HnswType> temp_hnsw_;
    /// Distance function supplied by the space.
    hnswlib::DISTFUNC<DistType> dist_func_ = nullptr;
    /// Parameter blob forwarded to the distance function.
    void *dist_func_param_ = nullptr;

    // SoA (Structure of Arrays) storage for the graph.
    // CSR-like structure: row_offset_ points to the start of edges for each node.
    // In dynamic mode, (row_offset_[i+1] - row_offset_[i]) is the allocated capacity.
    std::vector<std::size_t> row_offset_;
    
    // Flat arrays for edge properties.
    std::vector<unsigned> neighbors_;     // external_id(label as the neighbor id, because we assume all labels are unique)
    std::vector<unsigned> left_lower_;
    std::vector<unsigned> left_upper_;
    std::vector<unsigned> right_lower_;
    std::vector<unsigned> right_upper_;

    // Dynamic support
    struct NodeDegree {
        uint16_t fwd = 0; // number of forward (sorted) edges
        uint16_t rev = 0; // number of reverse (unsorted) edges in slack
    };

    bool is_dynamic_ = false;
    // Number of nodes currently inserted/indexed.
    unsigned num_indexed_nodes_ = 0;
    // Per-node forward/reverse degrees for maintaining the dynamic graph.
    std::vector<NodeDegree> node_degrees_;

    // Dense label <-> row mapping for random insertion.
    // label_to_row_[label] == -1 indicates this label has not been inserted yet.
    std::vector<int32_t> label_to_row_;
    std::vector<unsigned> row_to_label_;

    inline bool isInsertedLabel(unsigned label) const noexcept {
        return label < label_to_row_.size() && label_to_row_[label] >= 0;
    }

    void addReverseEdge(unsigned src, unsigned dst, unsigned ll, unsigned lu, unsigned rl, unsigned ru);
    void recompress(unsigned label);

    // Temporary structure used for sorting/merging/scoring edges during build/insert/recompress.
    struct TempEdge {
        unsigned external_id;
        unsigned left_lower;
        unsigned left_upper;
        unsigned right_lower;
        unsigned right_upper;

        bool coversRange(unsigned query_left, unsigned query_right) const noexcept {
            return (left_lower <= query_left && query_left <= left_upper) &&
                   (right_lower <= query_right && query_right <= right_upper);
        }
    };

    // Support-weighted edge pruning shared by build()/insert()/recompress().
    //
    // - Score = EdgeSupport(edge) / (|edge.external_id - center_label| + 1)
    // - Edges within `protect_span` are treated as protected and get max score.
    // - Keeps the Top `keep_total_limit` edges by score (0 means keep all).
    // - Rewrites `edges` in-place and sorts by external_id (forward-region invariant).
    void selectEdgesBySupport(std::vector<TempEdge> &edges,
                              unsigned center_label,
                              std::size_t protect_span,
                              std::size_t keep_total_limit) const;

    /// Reusable buffers for DFS compression.
    DfsScratch dfs_scratch_;
    /// Reusable buffers for base-layer KNN candidate generation during build.
    KnnScratch knn_scratch_;
    /// Pool for visited lists
    hnswlib::VisitedListPool *visited_list_pool_ = nullptr;

    // Load-time CSR expansion ratio for dynamic reverse-edge slack.
    // 0.0 means no expansion (tight/static load).
    double load_slack_fraction_ = 0.0;
    /// Last query hop count.
    std::size_t last_hop_count_ = 0;
    /// Last query distance evaluation count.
    std::size_t last_distance_eval_count_ = 0;

    InsertionStats insertion_stats_{};
};

} // namespace dsg
