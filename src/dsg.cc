// Author: Zhencan Peng, 2025/11/30
/**
 * @brief Dynamic Segment Graph implementation.
 *
 * The DSG first constructs a temporary HNSW, extracts ef_max-sized neighbor
 * lists for every label, compresses them with a DFS-based dominance filter,
 * and finally serves range queries using the stored forward segment edges.
 * This implementation is based on the paper "Dynamic Segment Graph for Approximate Nearest Neighbor Search" by Zhencan Peng et al.
 * And previously we use compact_graph.h for arbitrary query and insertion. But now we restructure the code to be more modular and easy to understand.
 */

#include "dsg.h"

#include <algorithm>
#include <chrono>
#include <cstring>
#include <cstdint>
#include <cstdlib>
#include <fstream>
#include <limits>
#include <type_traits>
#include <unordered_set>
#include <stdexcept>
#include <iostream>
#include <tuple>
#include <xmmintrin.h>
#include <immintrin.h>

namespace dsg {

namespace {
using Clock = std::chrono::steady_clock;
using Candidate = std::pair<DynamicSegmentGraph::DistType, unsigned>;

// Support heuristic for an edge (center -> v) under the full space of query ranges.
// A query range [L, R] can use this edge only if:
//   - L in [ll, lu]
//   - R in [rl, ru]
//   - L <= v <= R   (because adjacency is scanned only for neighbor ids in [L, R])
//
// The support is the number of (L, R) pairs satisfying these constraints:
//   support = |{L}| * |{R}|
static inline std::uint64_t EdgeSupport(unsigned v,
                                       unsigned ll,
                                       unsigned lu,
                                       unsigned rl,
                                       unsigned ru) noexcept {
    const std::uint64_t count_L = static_cast<std::uint64_t>(lu - ll + 1);
    const std::uint64_t count_R = static_cast<std::uint64_t>(ru - rl + 1);
    return count_L * count_R;
}

} // namespace

DynamicSegmentGraph::DynamicSegmentGraph(hnswlib::SpaceInterface<DistType> *space,
                                         const DataWrapper *data) :
    BaseIndex(data),
    space_(space) {
    if (space_ == nullptr) {
        throw std::invalid_argument("DynamicSegmentGraph requires a valid space interface.");
    }
    if (data_wrapper == nullptr) {
        throw std::invalid_argument("DynamicSegmentGraph requires a valid DataWrapper.");
    }
    dist_func_ = space_->get_dist_func();
    dist_func_param_ = space_->get_dist_func_param();
    if (dist_func_ == nullptr) {
        throw std::runtime_error("DynamicSegmentGraph failed to acquire distance function.");
    }
    visited_list_pool_ = new hnswlib::VisitedListPool(1, data_wrapper->data_size);
    returned_nns.resize(query_topK);

    // Dense label <-> row mapping (random insertion support).
    label_to_row_.assign(static_cast<size_t>(data_wrapper->data_size), -1);
    row_to_label_.clear();
}

DynamicSegmentGraph::~DynamicSegmentGraph() {
    temp_hnsw_.reset();
    if (visited_list_pool_) {
        delete visited_list_pool_;
        visited_list_pool_ = nullptr;
    }
}

void DynamicSegmentGraph::reserveGraphStorage(std::size_t total_rows_capacity,
                                             std::size_t total_edge_capacity) {
    // Per-row metadata
    row_to_label_.reserve(total_rows_capacity);
    node_degrees_.reserve(total_rows_capacity);
    row_offset_.reserve(total_rows_capacity + 1);

    // Global SoA edge buffers (capacity across all rows, including slack)
    neighbors_.reserve(total_edge_capacity);
    left_lower_.reserve(total_edge_capacity);
    left_upper_.reserve(total_edge_capacity);
    right_lower_.reserve(total_edge_capacity);
    right_upper_.reserve(total_edge_capacity);
}

/**
 * @brief Select a subset of segment edges by support-weighted score.
 *
 * @details This helper centralizes the pruning policy shared by build()/insert()/recompress().
 *
 * Scoring rule:
 * - For an edge (center_label -> v) with envelope [ll,lu] x [rl,ru], define
 *   support(v) = (lu - ll + 1) * (ru - rl + 1).
 * - We further down-weight edges that are far in label space:
 *
 *   score = support(v) / (|v - center_label| + 1)
 *
 * Protected edges:
 * - If |v - center_label| + 1 <= protect_span, the edge is treated as "protected" and
 *   assigned score = UINT64_MAX, so it is preferentially kept. This is used to avoid
 *   harming very small query ranges where near-label connections are important.
 *
 * Selection policy:
 * - Among prunable edges (non-protected), keep the Top `keep_prunable` by score (nth_element).
 * - Then merge protected + selected-prunable edges.
 * - If `keep_total_limit > 0`, additionally cap the total number of returned edges to
 *   `keep_total_limit` by score (again using nth_element). This is used by recompress()
 *   to keep the stored forward list within the per-row budget (deg.fwd).
 *
 * Output invariant:
 * - Returned edges are sorted by `external_id` so the forward region remains sorted and
 *   supports binary-search scans in `rangeSearch()`.
 *
 */
void DynamicSegmentGraph::selectEdgesBySupport(
    std::vector<TempEdge> &edges,
    unsigned center_label,
    std::size_t protect_span,
    std::size_t keep_total_limit) const {

    if (edges.empty()) {
        return;
    }

    const std::size_t center = static_cast<std::size_t>(center_label);
    // Store (score, index) so we don't copy TempEdge during ranking.
    std::vector<std::pair<std::uint64_t, std::uint32_t>> scored;
    scored.reserve(edges.size());
    for (std::uint32_t i = 0; i < static_cast<std::uint32_t>(edges.size()); ++i) {
        const auto &e = edges[i];
        const std::size_t neighbor = static_cast<std::size_t>(e.external_id);
        const std::size_t diff =
            neighbor > center ? (neighbor - center + 1) : (center - neighbor + 1);

        std::uint64_t score = 0;
        if (diff <= protect_span) {
            score = std::numeric_limits<std::uint64_t>::max();
        } else {
            score = EdgeSupport(e.external_id,
                                e.left_lower,
                                e.left_upper,
                                e.right_lower,
                                e.right_upper);
            score /= static_cast<std::uint64_t>(diff);
        }
        scored.emplace_back(score, i);
    }

    // Keep top-K by score using nth_element (descending).
    // If keep_total_limit == 0, keep all.
    std::size_t k = keep_total_limit;
    if (k == 0 || k > scored.size()) {
        k = scored.size();
    }
    const auto score_desc = [](const auto &a, const auto &b) {
        return a.first > b.first;
    };
    if (k < scored.size()) {
        std::nth_element(scored.begin(), scored.begin() + k, scored.end(), score_desc);
        scored.resize(k);
    }

    std::vector<TempEdge> out;
    out.reserve(scored.size());
    for (const auto &[score, idx] : scored) {
        (void)score;
        out.push_back(edges[static_cast<std::size_t>(idx)]);
    }
    std::sort(out.begin(), out.end(),
              [](const TempEdge &a, const TempEdge &b) {
                  return a.external_id < b.external_id;
              });
    edges.swap(out);
}

/**
 * @brief Build the DSG index over a selected label set.
 *
 * @details Pipeline:
 *  - Build a temporary HNSW containing only the provided labels.
 *  - Run ef_max candidate search for every label in the build set.
 *  - Apply DFS-based dominance compression + support pruning to select segment edges.
 *  - Add reverse edges, merge duplicates, then flatten into CSR-style SoA arrays.
 *
 * @param labels External labels in [0, data_size) to include in the index.
 *
 */
void DynamicSegmentGraph::build(const std::vector<unsigned> &labels) {

    const auto build_start = Clock::now();

    if (data_wrapper == nullptr) {
        throw std::runtime_error("DynamicSegmentGraph::build missing DataWrapper.");
    }

    const std::size_t data_size = static_cast<std::size_t>(data_wrapper->data_size);

    // The caller must provide a sorted, unique build set.
    // This keeps build() simple and avoids hidden reordering costs.
    if (!std::is_sorted(labels.begin(), labels.end())) {
        throw std::runtime_error("DynamicSegmentGraph::build labels must be sorted.");
    }
    for (std::size_t i = 0; i < labels.size(); ++i) {
        const unsigned label = labels[i];
        if (static_cast<std::size_t>(label) >= data_size) {
            throw std::runtime_error("DynamicSegmentGraph::build label out of range: " +
                                     std::to_string(label));
        }
        if (i > 0 && labels[i - 1] == label) {
            throw std::runtime_error("DynamicSegmentGraph::build labels must be unique (duplicate=" +
                                     std::to_string(label) + ").");
        }
    }

    // Reset and initialize dense mapping for random insertion/search.
    label_to_row_.assign(data_size, -1);
    row_to_label_ = labels;
    for (std::size_t row = 0; row < row_to_label_.size(); ++row) {
        label_to_row_[row_to_label_[row]] = static_cast<int32_t>(row);
    }

    const std::size_t num_rows = row_to_label_.size();
    num_indexed_nodes_ = static_cast<unsigned>(num_rows);

    // Use a temporary vector of vectors for building, merging, and sorting edges.
    std::vector<std::vector<TempEdge>> temp_adj(num_rows);

    std::vector<std::vector<std::pair<unsigned, DistType>>> all_candidates(num_rows);
    
    // build temporary HNSW and time it
    auto hnsw_build_start = Clock::now();
    initializeTemporaryHnsw(ef_max);
    for (unsigned label : row_to_label_) {
        temp_hnsw_->addPoint(static_cast<const void *>(data_wrapper->nodes.at(label)),
                             static_cast<hnswlib::labeltype>(label));
    }
    temp_hnsw_->setEf(ef_max);
    auto hnsw_build_end = Clock::now();
    double hnsw_build_time = std::chrono::duration<double>(hnsw_build_end - hnsw_build_start).count();
    std::cout << "[DSG] Temporary HNSW built in " << hnsw_build_time << " seconds." << std::endl;

    // run KNN for each label
    double total_knn_time = 0.0;
    double total_dfs_time = 0.0;
    double total_store_time = 0.0;

    auto t1 = Clock::now();
    for (std::size_t row = 0; row < num_rows; ++row) {
        const unsigned label = row_to_label_[row];
        runKnnForLabel(label, ef_max, all_candidates.at(row));
    }
    auto t2 = Clock::now();
    total_knn_time += std::chrono::duration<double>(t2 - t1).count();

    // Local lambda to store forward edges into temp_adj from dfs_scratch_
    auto store_edges_locally = [&](std::size_t center_row) {
        auto &edges = temp_adj.at(center_row);
        const std::size_t candidate_count = dfs_scratch_.ordered_candidates.size();
        for (std::size_t idx = 0; idx < candidate_count; ++idx) {
            if (!dfs_scratch_.is_neighbor[idx]) {
                continue;
            }
            const unsigned neighbor_label = dfs_scratch_.ordered_candidates[idx].first;
            if (neighbor_label >= label_to_row_.size() || label_to_row_[neighbor_label] < 0) {
                continue;
            }
            const unsigned ll = dfs_scratch_.left_lower[idx];
            const unsigned lu = dfs_scratch_.left_upper[idx];
            const unsigned rl = dfs_scratch_.right_lower[idx];
            const unsigned ru = dfs_scratch_.right_upper[idx];

            edges.push_back(TempEdge{neighbor_label, ll, lu, rl, ru});
        }
    };

    for (std::size_t row = 0; row < num_rows; ++row) {
        const unsigned center_label = row_to_label_[row];
        auto &candidates = all_candidates[row];

        const auto t_dfs_start = Clock::now();
        applyDfsCompression(center_label, candidates);
        const auto t_dfs_end = Clock::now();
        store_edges_locally(row);
        const auto t_store_end = Clock::now();

        total_dfs_time += std::chrono::duration<double>(t_dfs_end - t_dfs_start).count();
        total_store_time += std::chrono::duration<double>(t_store_end - t_dfs_end).count();
    }

    // After all forward edges are stored, we *materialize* the symmetric counterpart
    // for each edge (dst <- src) and merge it into the same adjacency list.
    //
    // Important distinction:
    // - These "reverse edges" exist only as a build-time construction step to make the
    //   final graph more connected (effectively bidirectional).
    // - They are NOT the "future reverse edges" used by dynamic insertion.
    //   Dynamic reverse edges are appended later into a per-row unsorted slack tail
    //   (`node_degrees_[row].rev`) and may trigger `recompress()` when the slack is full.
    //
    // Here, we explicitly MERGE (deduplicate + union of envelopes) and then FLATTEN;
    // after flattening, all stored edges are treated as forward edges (deg.fwd) and
    // `deg.rev` starts at 0.
    std::vector<std::vector<TempEdge>> reverse_edges(num_rows);
    for (std::size_t src_row = 0; src_row < num_rows; ++src_row) {
        const unsigned src_label = row_to_label_[src_row];
        for (const auto &edge : temp_adj[src_row]) {
            const unsigned dst = edge.external_id;
            if (dst >= label_to_row_.size()) {
                continue;
            }
            const int32_t dst_row_i = label_to_row_[dst];
            if (dst_row_i < 0) {
                continue;
            }
            const std::size_t dst_row = static_cast<std::size_t>(dst_row_i);
            reverse_edges[dst_row].push_back(TempEdge{src_label,
                                            edge.left_lower,
                                            edge.left_upper,
                                            edge.right_lower,
                                            edge.right_upper});
        }
    }

    auto merge_edges_func = [](std::vector<TempEdge> &edges) {
        if (edges.empty()) {
            return;
        }
        std::sort(edges.begin(), edges.end(),
                  [](const auto &a, const auto &b) {
                      if (a.external_id == b.external_id) {
                          return std::tie(a.left_lower, a.left_upper, a.right_lower, a.right_upper) <
                                 std::tie(b.left_lower, b.left_upper, b.right_lower, b.right_upper);
                      }
                      return a.external_id < b.external_id;
                  });
        std::vector<TempEdge> merged;
        merged.reserve(edges.size());
        merged.push_back(edges.front());
        for (std::size_t i = 1; i < edges.size(); ++i) {
            auto &last = merged.back();
            const auto &cur = edges[i];
            if (last.external_id == cur.external_id) {
                last.left_lower = std::min(last.left_lower, cur.left_lower);
                last.left_upper = std::max(last.left_upper, cur.left_upper);
                last.right_lower = std::min(last.right_lower, cur.right_lower);
                last.right_upper = std::max(last.right_upper, cur.right_upper);
            } else {
                merged.push_back(cur);
            }
        }
        edges.swap(merged);
    };

    for (std::size_t row = 0; row < num_rows; ++row) {
        auto &edges = temp_adj[row];
        auto &rev = reverse_edges[row];
        edges.insert(edges.end(), rev.begin(), rev.end());
        merge_edges_func(edges);
    }

    // ---------------------------------------------------------------------
    // Build-time support pruning (bottom 10% support per node).
    //
    // Motivation:
    // - Very low-support edges correspond to tiny (L, R) eligibility regions.
    // - These edges rarely contribute across the full query-range space.
    //
    // Small-range guard:
    // - `rangeSearch()` skips envelope checks when range_span < 0.02 * N.
    // - For such tiny ranges, useful edges tend to connect labels close to the
    //   current node (center_label). To avoid harming this regime, we never prune
    //   edges whose neighbor label lies within +/- ceil(0.02 * N) around center_label.
    // ---------------------------------------------------------------------
    const auto prune_start = Clock::now();
    const std::size_t protect_span = (data_size + 49) / 50; // ceil(0.02 * N) = ceil(N / 50)
    // Distance-weighted pruning score to discourage far-in-label edges:
    // score = support / |center_label - neighbor_label|.
    std::cout << "[DSG] Build-time support prune: using distance-weighted score support/|nbr-center|"
              << std::endl;
    std::size_t edges_before_prune = 0;
    std::size_t edges_after_prune = 0;
    std::size_t pruned_edges = 0;
    std::size_t protected_edges = 0;

    for (std::size_t row = 0; row < num_rows; ++row) {
        const std::size_t center_label = static_cast<std::size_t>(row_to_label_[row]);
        auto &edges = temp_adj[row];
        edges_before_prune += edges.size();

        // Count protected edges for logging only.
        for (const auto &edge : edges) {
            const std::size_t neighbor = static_cast<std::size_t>(edge.external_id);
            const std::size_t diff =
                neighbor > center_label ? (neighbor - center_label + 1)
                                        : (center_label - neighbor + 1);
            if (diff <= protect_span) {
                ++protected_edges;
            }
        }

        const std::size_t before = edges.size();
        const std::size_t keep_total_limit = before - (before / 11); // keep top 10/11
        selectEdgesBySupport(edges,
                             static_cast<unsigned>(center_label),
                             protect_span,
                             /*keep_total_limit=*/keep_total_limit);
        edges_after_prune += edges.size();
        pruned_edges += (before - edges.size());
    }

    // If we skipped pruning for some nodes (e.g. prune_count==0), edges_after_prune
    // already includes their original degree. For nodes that were empty, we added 0.
    // For nodes pruned, edges_after_prune was added after pruning.
    if (edges_after_prune == 0) {
        for (const auto &edges : temp_adj) {
            edges_after_prune += edges.size();
        }
    }

    const auto prune_end = Clock::now();
    const double prune_time_s =
        std::chrono::duration<double>(prune_end - prune_start).count();
    const double prune_frac = edges_before_prune == 0
                                  ? 0.0
                                  : static_cast<double>(pruned_edges) /
                                        static_cast<double>(edges_before_prune);

    std::cout << "[DSG] Build-time support prune (per node, bottom 10% of prunable edges): "
              << "pruned=" << pruned_edges << "/" << edges_before_prune
              << " (" << prune_frac * 100.0 << "%), "
              << "protected_edges=" << protected_edges
              << " (|nbr-center| <= " << protect_span << "), "
              << "score=support/|nbr-center|, "
              << "time=" << prune_time_s << " s" << std::endl;

    // Now flatten temp_adj into SoA members
    row_offset_.resize(num_rows + 1);
    std::size_t total_edges = 0;
    for (const auto &edges : temp_adj) {
        total_edges += edges.size();
    }
    neighbors_.resize(total_edges);
    left_lower_.resize(total_edges);
    left_upper_.resize(total_edges);
    right_lower_.resize(total_edges);
    right_upper_.resize(total_edges);

    std::size_t current_offset = 0;
    for (std::size_t i = 0; i < num_rows; ++i) {
        row_offset_[i] = current_offset;
        for (const auto &edge : temp_adj[i]) {
            neighbors_[current_offset] = edge.external_id;
            left_lower_[current_offset] = edge.left_lower;
            left_upper_[current_offset] = edge.left_upper;
            right_lower_[current_offset] = edge.right_lower;
            right_upper_[current_offset] = edge.right_upper;
            current_offset++;
        }
    }
    row_offset_[num_rows] = current_offset;

    // Initialize per-row degrees:
    // - After the explicit merge above, the adjacency stored in SoA is a single list.
    // - We treat the entire list as "forward" (sorted by neighbor id within each row).
    // - We do NOT reserve slack capacity here, so `rev` starts at 0 and capacity == degree.
    //
    // Rationale (build -> save workflow):
    // - The typical workflow is to build a static index and save it immediately.
    // - To minimize memory footprint and disk size, we flatten into a tight CSR layout
    //   with no per-row padding (i.e., no extra space for dynamic reverse neighbors).
    //
    // Dynamic insertion workflow (load-time slack control):
    // - The dynamic mode relies on a per-row "slack" tail for reverse edges (deg.rev),
    //   where capacity = row_offset_[row+1] - row_offset_[row].
    // - If you want insertion support, you can explicitly choose how much slack to
    //   reserve when loading a saved index (e.g., by expanding row capacities and
    //   updating row_offset_ accordingly before setting deg.fwd/deg.rev).
    node_degrees_.resize(num_rows);
    for (std::size_t row = 0; row < num_rows; ++row) {
        const std::size_t deg = row_offset_[row + 1] - row_offset_[row];
        node_degrees_[row].fwd = static_cast<uint16_t>(deg);
        node_degrees_[row].rev = 0;
    }

    const auto build_end = Clock::now();
    index_time = std::chrono::duration<double>(build_end - build_start).count();
    
    std::cout << "[DSG] Detailed breakdown:" << std::endl;
    std::cout << "  KNN Search: " << total_knn_time << " s" << std::endl;
    std::cout << "  DFS Compress: " << total_dfs_time << " s" << std::endl;
    std::cout << "  Store Edges: " << total_store_time << " s" << std::endl;

}

void DynamicSegmentGraph::rangeSearch(const float *query,
                                      const std::pair<int, int> query_bound) {

    const int left = query_bound.first;
    const int right = query_bound.second;

    const unsigned left_u = static_cast<unsigned>(left);
    const unsigned right_u = static_cast<unsigned>(right);

    hnswlib::VisitedList *vl = visited_list_pool_->getFreeVisitedList();
    hnswlib::vl_type *visited_array = vl->mass;
    hnswlib::vl_type visited_array_tag = vl->curV;
    std::size_t hop_counter = 0;
    std::size_t distance_eval_count = 0;

    auto timed_distance = [&](unsigned label) -> DistType {
        ++distance_eval_count;
        const DistType dist =
            dist_func_(query, data_wrapper->nodes[label], dist_func_param_);
        return dist;
    };

    auto cmp = [](const Candidate &lhs, const Candidate &rhs) {
        return lhs.first > rhs.first;
    };
    std::priority_queue<Candidate, std::vector<Candidate>, decltype(cmp)> candidate_set(cmp);
    std::priority_queue<Candidate> top_candidates;
    std::vector<unsigned> fetched_nns;
    fetched_nns.reserve(search_ef);

    // Try to enqueue an inserted label as a new seed.
    // Returns true if we actually pushed a new element into candidate_set.
    auto try_enqueue_seed = [&](unsigned label) -> bool {
        if (!isInsertedLabel(label)) {
            return false;
        }
        if (visited_array[label] == visited_array_tag) {
            return false;
        }
        visited_array[label] = visited_array_tag;
        const DistType dist = timed_distance(label);
        candidate_set.emplace(dist, label);
        return true;
    };

    const unsigned range_span = right_u - left_u + 1;
    constexpr double kSmallRangeFrac = 0.02;
    // Small optimization: if the query span is tiny, skip envelope checks and
    // admit neighbors inside [left_u, right_u] to avoid over-pruning sparse ranges.
    const bool skip_range_envelope =
        static_cast<double>(range_span) < kSmallRangeFrac * static_cast<double>(data_wrapper->data_size);

    // Seeding strategy (important after label<->row decoupling):
    // - In the dynamic setting, only a subset of labels are inserted (label_to_row_[label] != -1).
    // - Naively seeding with fixed anchors (left/mid/quarters) can pick uninserted labels, leaving
    //   the frontier empty and causing the search to return nothing even when valid nodes exist.
    // - To keep search robust with small overhead, we probe a small +/- radius window around each
    //   anchor until we find an inserted label. This is O(kProbeRadius) per anchor in the worst case.
    constexpr unsigned kProbeRadius = 64;
    auto enqueue_nearby_seed = [&](unsigned anchor) {
        if (try_enqueue_seed(anchor)) {
            return;
        }
        auto maxD = std::min(kProbeRadius, right_u - anchor);
        for (unsigned d = 1; d <= maxD; ++d) {
            if (try_enqueue_seed(anchor + d)) {
                return;
            }
        }
    };

    enqueue_nearby_seed(left_u);
    enqueue_nearby_seed(left_u + range_span / 2);
    enqueue_nearby_seed(left_u + range_span / 4);
    enqueue_nearby_seed(left_u + 3 * range_span / 4);

    // If we still cannot find any inserted seed inside the query range, return empty.
    if (candidate_set.empty()) {
        constexpr unsigned kFallbackScan = 4096;
        unsigned scanned = 0;
        for (unsigned probe = left_u; probe <= right_u && scanned < kFallbackScan; ++probe, ++scanned) {
            if (try_enqueue_seed(probe)) {
                break;
            }
        }
    }
    if (candidate_set.empty()) {
        returned_nns.clear();
        returned_nns_with_dist_.clear();
        last_hop_count_ = 0;
        last_distance_eval_count_ = 0;
        visited_list_pool_->releaseVisitedList(vl);
        std::cout << "[DSG] No inserted seed found in the query range" << std::endl;
        return;
    }

    // Current worst (largest) distance among the kept top candidates.
    DistType worst_top_dist = std::numeric_limits<DistType>::max();

    // Prepare SIMD constants
    // Note: using signed comparison because standard _mm_cmple_epi32 is signed. 
    // For unsigned comparison in SSE2/AVX2, we toggle the sign bit (0x80000000).
    // (val ^ 0x80000000) > (bound ^ 0x80000000)
    const __m128i sign_bit = _mm_set1_epi32(0x80000000);
    const __m128i v_left_u = _mm_set1_epi32(static_cast<int>(left_u));
    const __m128i v_right_u = _mm_set1_epi32(static_cast<int>(right_u));
    const __m128i v_left_u_adj = _mm_xor_si128(v_left_u, sign_bit);
    const __m128i v_right_u_adj = _mm_xor_si128(v_right_u, sign_bit);

    // Pointers for SoA arrays
    const unsigned* neighbors_ptr = neighbors_.data();
    const unsigned* ll_ptr = left_lower_.data();
    const unsigned* lu_ptr = left_upper_.data();
    const unsigned* rl_ptr = right_lower_.data();
    const unsigned* ru_ptr = right_upper_.data();

    while (!candidate_set.empty()) {
        const auto [dist, current] = candidate_set.top();
        candidate_set.pop();
        ++hop_counter;

        if (dist > worst_top_dist) {
            break;
        }

        const unsigned current_label = current;
        if (current_label >= label_to_row_.size()) {
            continue;
        }
        const int32_t row_i = label_to_row_[current_label];
        if (row_i < 0) {
            continue;
        }
        const size_t row = static_cast<size_t>(row_i);

        // SoA access by row-id.
        const size_t start_idx = row_offset_[row];

        const auto &deg = node_degrees_[row];
        const size_t sorted_count = static_cast<size_t>(deg.fwd);
        const size_t reverse_count = static_cast<size_t>(deg.rev);
        const size_t sorted_end_idx = start_idx + sorted_count;
        // Dynamic adjacency layout:
        // - [start_idx, sorted_end_idx): forward edges (sorted by neighbor label)
        // - [sorted_end_idx, sorted_end_idx + reverse_count): reverse-edge slack (unsorted)

        fetched_nns.clear();

        // 1. Binary search in SORTED portion
        auto start_it = neighbors_.begin() + start_idx;
        auto sorted_end_it = neighbors_.begin() + sorted_end_idx;
        auto it = std::lower_bound(start_it, sorted_end_it, left_u);
        
        size_t current_scan_idx = std::distance(neighbors_.begin(), it);

        // Scan sorted portion
        if (!skip_range_envelope) {
             // SIMD Loop for sorted part
            for (; current_scan_idx + 4 <= sorted_end_idx; current_scan_idx += 4) {
                 // Prefetch ahead (16 elements ahead = 64 bytes)
                _mm_prefetch(reinterpret_cast<const char*>(neighbors_ptr + current_scan_idx + 16), _MM_HINT_T0);
                
                _mm_prefetch(reinterpret_cast<const char*>(ll_ptr + current_scan_idx + 16), _MM_HINT_T0);
                _mm_prefetch(reinterpret_cast<const char*>(lu_ptr + current_scan_idx + 16), _MM_HINT_T0);
                _mm_prefetch(reinterpret_cast<const char*>(rl_ptr + current_scan_idx + 16), _MM_HINT_T0);
                _mm_prefetch(reinterpret_cast<const char*>(ru_ptr + current_scan_idx + 16), _MM_HINT_T0);

                // Load neighbors
                __m128i v_nbr = _mm_loadu_si128(reinterpret_cast<const __m128i*>(neighbors_ptr + current_scan_idx));

                // Check if any neighbor > right_u (break condition for SORTED part)
                __m128i v_nbr_adj = _mm_xor_si128(v_nbr, sign_bit);
                __m128i v_break_cmp = _mm_cmpgt_epi32(v_nbr_adj, v_right_u_adj);
                
                if (_mm_movemask_ps(_mm_castsi128_ps(v_break_cmp)) != 0) {
                    break; 
                }

                // Load range attributes
                __m128i v_ll = _mm_loadu_si128(reinterpret_cast<const __m128i*>(ll_ptr + current_scan_idx));
                __m128i v_lu = _mm_loadu_si128(reinterpret_cast<const __m128i*>(lu_ptr + current_scan_idx));
                __m128i v_rl = _mm_loadu_si128(reinterpret_cast<const __m128i*>(rl_ptr + current_scan_idx));
                __m128i v_ru = _mm_loadu_si128(reinterpret_cast<const __m128i*>(ru_ptr + current_scan_idx));

                // Adjust for unsigned comparison
                v_ll = _mm_xor_si128(v_ll, sign_bit);
                v_lu = _mm_xor_si128(v_lu, sign_bit);
                v_rl = _mm_xor_si128(v_rl, sign_bit);
                v_ru = _mm_xor_si128(v_ru, sign_bit);

                // Check conditions:
                // 1. neighbor >= left_u is guaranteed by lower_bound

                // 2. left_lower <= left_u  => !(left_lower > left_u) => !(v_ll > v_left_u_adj)
                __m128i c1_fail = _mm_cmpgt_epi32(v_ll, v_left_u_adj);

                // 3. left_u <= left_upper  => !(left_u > left_upper) => !(v_left_u_adj > v_lu)
                __m128i c2_fail = _mm_cmpgt_epi32(v_left_u_adj, v_lu);

                // 4. right_lower <= right_u => !(right_lower > right_u) => !(v_rl > v_right_u_adj)
                __m128i c3_fail = _mm_cmpgt_epi32(v_rl, v_right_u_adj);

                // 5. right_u <= right_upper => !(right_u > right_upper) => !(v_right_u_adj > v_ru)
                __m128i c4_fail = _mm_cmpgt_epi32(v_right_u_adj, v_ru);

                // Combine failures
                __m128i any_fail = _mm_or_si128(c1_fail, c2_fail);
                any_fail = _mm_or_si128(any_fail, _mm_or_si128(c3_fail, c4_fail));

                // mask = 1 where valid (any_fail is 0)
                int fail_mask = _mm_movemask_ps(_mm_castsi128_ps(any_fail));
                int valid_mask = (~fail_mask) & 0xF;

                while (valid_mask) {
                    int bit = __builtin_ctz(valid_mask);
                    unsigned neighbor = neighbors_ptr[current_scan_idx + bit];
                    
                    // Check visited status
                    if (visited_array[neighbor] != visited_array_tag) {
                        fetched_nns.push_back(neighbor);
                        _mm_prefetch(reinterpret_cast<const char*>(data_wrapper->nodes[neighbor]), _MM_HINT_T0);
                    }
                    valid_mask &= (valid_mask - 1);
                }
            }
        }

        // Scalar Loop for remaining sorted items or after break
        for (; current_scan_idx < sorted_end_idx; ++current_scan_idx) {
            const unsigned neighbor = neighbors_ptr[current_scan_idx];
            
            if (neighbor > right_u) {
                break;
            }

            if (!skip_range_envelope) {
                if (!((ll_ptr[current_scan_idx] <= left_u && left_u <= lu_ptr[current_scan_idx]) &&
                      (rl_ptr[current_scan_idx] <= right_u && right_u <= ru_ptr[current_scan_idx]))) {
                    continue;
                }
            }
            
            if (visited_array[neighbor] == visited_array_tag) {
                continue;
            }
            fetched_nns.push_back(neighbor);
            _mm_prefetch(reinterpret_cast<const char*>(data_wrapper->nodes[neighbor]), _MM_HINT_T0);
        }

        // 2. Linear Scan in UNSORTED reverse edges (dynamic mode)
        if (is_dynamic_ && reverse_count > 0) {
            const size_t slack_end_idx = sorted_end_idx + reverse_count;
            for (size_t slack_idx = sorted_end_idx; slack_idx < slack_end_idx; ++slack_idx) {
                const unsigned neighbor = neighbors_ptr[slack_idx];

                // Range filter (id check)
                if (neighbor < left_u || neighbor > right_u) {
                    continue;
                }

                if (!skip_range_envelope) {
                    if (!((ll_ptr[slack_idx] <= left_u && left_u <= lu_ptr[slack_idx]) &&
                          (rl_ptr[slack_idx] <= right_u && right_u <= ru_ptr[slack_idx]))) {
                        continue;
                    }
                }

                if (visited_array[neighbor] == visited_array_tag) {
                    continue;
                }
                fetched_nns.push_back(neighbor);
                _mm_prefetch(reinterpret_cast<const char*>(data_wrapper->nodes[neighbor]), _MM_HINT_T0);
            }
        }

        for (const auto neighbor : fetched_nns) {
            visited_array[neighbor] = visited_array_tag;
            const DistType nbr_dist = timed_distance(neighbor);
            
            if (top_candidates.size() < search_ef) {
                candidate_set.emplace(nbr_dist, neighbor);
                top_candidates.emplace(nbr_dist, neighbor);
                worst_top_dist = top_candidates.top().first;
            } else if (nbr_dist < worst_top_dist) {
                candidate_set.emplace(nbr_dist, neighbor);
                top_candidates.emplace(nbr_dist, neighbor);
                top_candidates.pop();
                worst_top_dist = top_candidates.top().first;
            }
        }
    }

    visited_list_pool_->releaseVisitedList(vl);

    while (top_candidates.size() > query_topK) {
        top_candidates.pop();
    }

    returned_nns.clear();
    returned_nns_with_dist_.clear();
    while (!top_candidates.empty()) {
        const auto [dist, lbl] = top_candidates.top();
        returned_nns.emplace_back(lbl);
        returned_nns_with_dist_.emplace_back(lbl, dist);
        top_candidates.pop();
    }

    last_hop_count_ = hop_counter;
    last_distance_eval_count_ = distance_eval_count;
}

void DynamicSegmentGraph::save(const std::string &file_path) {
    std::ofstream out(file_path, std::ios::binary);
    if (!out) {
        throw std::runtime_error("DynamicSegmentGraph::save failed to open file: " + file_path);
    }

    static_assert(sizeof(unsigned) == 4, "DSG save/load assumes 32-bit unsigned.");

    // WARNING: The v3 format below writes several std::vector<T> buffers directly
    // (row_to_label_, node_degrees_, row_offset_) using their in-memory layout.
    // This is NOT portable across architectures/compilers/ABIs. We assume save/load
    // happen on the same machine with the same build configuration.
    constexpr char kMagic[8] = {'D', 'S', 'G', 'I', 'D', 'X', '3', '\0'};
    constexpr std::uint32_t kVersion = 3;

    const std::uint64_t data_size =
        data_wrapper == nullptr ? 0ULL : static_cast<std::uint64_t>(data_wrapper->data_size);
    const std::uint64_t num_rows = static_cast<std::uint64_t>(row_to_label_.size());

    if (row_offset_.size() != static_cast<std::size_t>(num_rows + 1)) {
        throw std::runtime_error("DynamicSegmentGraph::save row_offset_ size mismatch.");
    }
    if (node_degrees_.size() != static_cast<std::size_t>(num_rows)) {
        throw std::runtime_error("DynamicSegmentGraph::save node_degrees_ size mismatch.");
    }

    const std::uint64_t num_edges_total = static_cast<std::uint64_t>(neighbors_.size());

    out.write(kMagic, sizeof(kMagic));
    out.write(reinterpret_cast<const char *>(&kVersion), sizeof(kVersion));
    out.write(reinterpret_cast<const char *>(&data_size), sizeof(data_size));
    out.write(reinterpret_cast<const char *>(&num_rows), sizeof(num_rows));

    // row_to_label_ (direct dump; same-machine assumption)
    out.write(reinterpret_cast<const char *>(row_to_label_.data()),
              row_to_label_.size() * sizeof(unsigned));

    // node_degrees_ (direct dump; same-machine assumption)
    static_assert(std::is_trivially_copyable<NodeDegree>::value,
                  "NodeDegree must be trivially copyable for direct dump.");
    out.write(reinterpret_cast<const char *>(node_degrees_.data()),
              node_degrees_.size() * sizeof(NodeDegree));

    // row_offset_ (direct dump; same-machine assumption)
    out.write(reinterpret_cast<const char *>(row_offset_.data()),
              row_offset_.size() * sizeof(std::size_t));

    out.write(reinterpret_cast<const char *>(&num_edges_total), sizeof(num_edges_total));

    out.write(reinterpret_cast<const char *>(neighbors_.data()), neighbors_.size() * sizeof(unsigned));
    out.write(reinterpret_cast<const char *>(left_lower_.data()), left_lower_.size() * sizeof(unsigned));
    out.write(reinterpret_cast<const char *>(left_upper_.data()), left_upper_.size() * sizeof(unsigned));
    out.write(reinterpret_cast<const char *>(right_lower_.data()), right_lower_.size() * sizeof(unsigned));
    out.write(reinterpret_cast<const char *>(right_upper_.data()), right_upper_.size() * sizeof(unsigned));
}

/**
 * @brief Load a previously saved Dynamic Segment Graph index from a binary file.
 * 
 * @details File format (v3, see save()):
 *  - magic[8], version(u32)
 *  - data_size(u64), num_rows(u64)
 *  - row_to_label_[num_rows] (u32)
 *  - node_degrees_[num_rows] (NodeDegree, direct dump)
 *  - row_offset_[num_rows + 1] (size_t, direct dump)
 *  - num_edges_total(u64)
 *  - neighbors_ + (left_lower_, left_upper_, right_lower_, right_upper_) arrays (u32),
 *    each with length num_edges_total
 *
 * After loading we rebuild `label_to_row_` (dense label -> internal row id), validate
 * CSR invariants, and compute forward/reverse edge statistics from `node_degrees_`.
 * 
 * @param file_path Path to the binary file containing the saved DSG index
 * 
 * @throws std::runtime_error If the file cannot be opened or if the loaded node count
 *                            does not match the dataset size in data_wrapper
 * 
 */
void DynamicSegmentGraph::load(const std::string &file_path) {
    std::ifstream in(file_path, std::ios::binary);
    if (!in) {
        throw std::runtime_error("DynamicSegmentGraph::load failed to open file: " + file_path);
    }

    static_assert(sizeof(unsigned) == 4, "DSG save/load assumes 32-bit unsigned.");

    constexpr char kMagic[8] = {'D', 'S', 'G', 'I', 'D', 'X', '3', '\0'};
    char magic[8]{};
    in.read(magic, sizeof(magic));
    if (std::memcmp(magic, kMagic, sizeof(kMagic)) != 0) {
        throw std::runtime_error("DynamicSegmentGraph::load invalid magic header.");
    }

    std::uint32_t version = 0;
    in.read(reinterpret_cast<char *>(&version), sizeof(version));
    if (version != 3) {
        throw std::runtime_error("DynamicSegmentGraph::load unsupported version: " + std::to_string(version));
    }

    std::uint64_t data_size = 0;
    std::uint64_t num_rows = 0;
    in.read(reinterpret_cast<char *>(&data_size), sizeof(data_size));
    in.read(reinterpret_cast<char *>(&num_rows), sizeof(num_rows));

    if (data_wrapper == nullptr) {
        throw std::runtime_error("DynamicSegmentGraph::load missing DataWrapper.");
    }
    if (data_size != static_cast<std::uint64_t>(data_wrapper->data_size)) {
        throw std::runtime_error("DynamicSegmentGraph::load mismatched data_size.");
    }

    row_to_label_.resize(static_cast<std::size_t>(num_rows));
    in.read(reinterpret_cast<char *>(row_to_label_.data()),
            row_to_label_.size() * sizeof(unsigned));

    node_degrees_.resize(static_cast<std::size_t>(num_rows));
    in.read(reinterpret_cast<char *>(node_degrees_.data()),
            node_degrees_.size() * sizeof(NodeDegree));
    
    row_offset_.resize(static_cast<std::size_t>(num_rows + 1));
    in.read(reinterpret_cast<char *>(row_offset_.data()),
            row_offset_.size() * sizeof(std::size_t));

    std::uint64_t num_edges_total = 0;
    in.read(reinterpret_cast<char *>(&num_edges_total), sizeof(num_edges_total));

    neighbors_.resize(static_cast<std::size_t>(num_edges_total));
    left_lower_.resize(static_cast<std::size_t>(num_edges_total));
    left_upper_.resize(static_cast<std::size_t>(num_edges_total));
    right_lower_.resize(static_cast<std::size_t>(num_edges_total));
    right_upper_.resize(static_cast<std::size_t>(num_edges_total));

    in.read(reinterpret_cast<char *>(neighbors_.data()), neighbors_.size() * sizeof(unsigned));
    in.read(reinterpret_cast<char *>(left_lower_.data()), left_lower_.size() * sizeof(unsigned));
    in.read(reinterpret_cast<char *>(left_upper_.data()), left_upper_.size() * sizeof(unsigned));
    in.read(reinterpret_cast<char *>(right_lower_.data()), right_lower_.size() * sizeof(unsigned));
    in.read(reinterpret_cast<char *>(right_upper_.data()), right_upper_.size() * sizeof(unsigned));

    // Optional load-time CSR expansion to create per-row slack capacity for dynamic reverse edges.
    // Enable by calling setLoadSlackFraction(frac) with frac > 0 before load().
    if (load_slack_fraction_ > 0.0) {
        std::vector<std::size_t> new_row_offset(static_cast<std::size_t>(num_rows + 1), 0);
        for (std::size_t row = 0; row < static_cast<std::size_t>(num_rows); ++row) {
            const std::size_t fwd = static_cast<std::size_t>(node_degrees_[row].fwd);
            const std::size_t slack =
                static_cast<std::size_t>(std::ceil(static_cast<double>(fwd) * load_slack_fraction_));
            new_row_offset[row + 1] = new_row_offset[row] + fwd + slack;
            node_degrees_[row].rev = 0;
        }

        std::vector<unsigned> new_neighbors(new_row_offset.back());
        std::vector<unsigned> new_left_lower(new_row_offset.back());
        std::vector<unsigned> new_left_upper(new_row_offset.back());
        std::vector<unsigned> new_right_lower(new_row_offset.back());
        std::vector<unsigned> new_right_upper(new_row_offset.back());

        for (std::size_t row = 0; row < static_cast<std::size_t>(num_rows); ++row) {
            const std::size_t fwd = static_cast<std::size_t>(node_degrees_[row].fwd);
            const std::size_t old_start = row_offset_[row];
            const std::size_t new_start = new_row_offset[row];

            std::memcpy(new_neighbors.data() + new_start,
                        neighbors_.data() + old_start,
                        fwd * sizeof(unsigned));
            std::memcpy(new_left_lower.data() + new_start,
                        left_lower_.data() + old_start,
                        fwd * sizeof(unsigned));
            std::memcpy(new_left_upper.data() + new_start,
                        left_upper_.data() + old_start,
                        fwd * sizeof(unsigned));
            std::memcpy(new_right_lower.data() + new_start,
                        right_lower_.data() + old_start,
                        fwd * sizeof(unsigned));
            std::memcpy(new_right_upper.data() + new_start,
                        right_upper_.data() + old_start,
                        fwd * sizeof(unsigned));
        }

        row_offset_.swap(new_row_offset);
        neighbors_.swap(new_neighbors);
        left_lower_.swap(new_left_lower);
        left_upper_.swap(new_left_upper);
        right_lower_.swap(new_right_lower);
        right_upper_.swap(new_right_upper);
    }

    // Rebuild dense mapping.
    label_to_row_.assign(static_cast<std::size_t>(data_size), -1);
    for (std::uint64_t row = 0; row < num_rows; ++row) {
        const unsigned lbl = row_to_label_[static_cast<std::size_t>(row)];
        if (lbl >= data_size) {
            throw std::runtime_error("DynamicSegmentGraph::load row_to_label out of range.");
        }
        label_to_row_[lbl] = static_cast<int32_t>(row);
    }

    if (row_offset_.back() != neighbors_.size()) {
        throw std::runtime_error("DynamicSegmentGraph::load row_offset tail mismatch.");
    }

    num_indexed_nodes_ = static_cast<unsigned>(num_rows);
    is_dynamic_ = (load_slack_fraction_ > 0.0);

    // Stats (counts only valid edges, not slack capacity).
    std::uint64_t sum_fwd = 0;
    std::uint64_t sum_rev = 0;
    for (const auto &deg : node_degrees_) {
        sum_fwd += deg.fwd;
        sum_rev += deg.rev;
    }
    edges_amount = static_cast<std::size_t>(sum_fwd + sum_rev);
    avg_forward_nns = num_rows == 0 ? 0.0F : static_cast<float>(sum_fwd) / static_cast<float>(num_rows);
    avg_reverse_nns = num_rows == 0 ? 0.0F : static_cast<float>(sum_rev) / static_cast<float>(num_rows);
}

/**
 * @brief Insert a new label into a *dynamic* DSG
 *
 * @details High-level steps:
 *  - Run a full-range `rangeSearch()` on the current graph to collect candidate labels.
 *  - Apply DFS dominance compression (`applyDfsCompression`) to compute segment envelopes.
 *  - Prune low-support edges (similar heuristic as build()) and keep edges sorted by label id.
 *  - Append a new CSR row with slack capacity (~1.1x of forward degree) for future reverse edges.
 *  - Add reverse edges into neighbors' slack regions (unsorted), triggering per-row recompression
 *    when slack becomes full.
 *
 * Layout invariants (CSR + slack):
 *  - For row r: [row_offset_[r], row_offset_[r+1]) is the allocated capacity.
 *  - The first `node_degrees_[r].fwd` entries are forward edges and kept sorted by neighbor id.
 *  - The next `node_degrees_[r].rev` entries are reverse edges stored in an unsorted "slack" tail.
 *
 * Complexity (rough):
 *  - Candidate search: similar to HNSW best-first traversal, bounded by `search_ef`.
 *  - DFS compression: O(ef^2 * dist_cost) worst-case due to domination checks.
 */
void DynamicSegmentGraph::insert(unsigned label) {
    if (!is_dynamic_) {
        throw std::runtime_error("Cannot insert into static DSG. Call load() first.");
    }
    const unsigned data_size = static_cast<unsigned>(data_wrapper->data_size);
    if (label >= data_size) {
        throw std::runtime_error("Insert label out of range: " + std::to_string(label));
    }
    if (label_to_row_[label] >= 0) {
        throw std::runtime_error("Insert label already exists: " + std::to_string(label));
    }
    
    // 1. Search for candidates in EXISTING graph
    // We use a simplified range search with infinite bounds to get candidates.
    // rangeSearch() already computes distances internally; we reuse them via
    // returned_nns_with_dist_ to avoid recalculating dist_func_.
    
    // Temporarily set ef large enough for quality
    size_t old_ef = search_ef;
    search_ef = ef_max; 
    
    const float* query_vec = data_wrapper->nodes[label];
    
    // Range search over the full label space; expansion naturally skips uninserted labels.
    rangeSearch(query_vec, {0, static_cast<int>(data_size - 1)});
    
    search_ef = old_ef; // Restore
    
    // 2. Prepare candidates
    std::vector<std::pair<unsigned, DistType>> candidates;
    candidates.reserve(returned_nns_with_dist_.size());
    for (const auto &[nbr, dist] : returned_nns_with_dist_) {
        candidates.push_back({nbr, dist});
    }
    
    // Nearest first
    std::sort(candidates.begin(), candidates.end(), [](const auto& a, const auto& b) {
        return a.second < b.second;
    });
    
    // 3. DFS Compression
    applyDfsCompression(label, candidates);
    
    // 4. Support Pruning & Store Forward Edges
    // Reuse the same support-weighted pruning as build().
    auto &scratch = dfs_scratch_;
    const size_t cand_count = scratch.ordered_candidates.size();

    const size_t protect_span = (static_cast<size_t>(data_size) + 49) / 50;

    std::vector<TempEdge> edges;
    edges.reserve(cand_count);
    for (size_t i = 0; i < cand_count; ++i) {
        if (!scratch.is_neighbor[i]) {
            continue;
        }
        edges.push_back(TempEdge{
            scratch.ordered_candidates[i].first,
            scratch.left_lower[i],
            scratch.left_upper[i],
            scratch.right_lower[i],
            scratch.right_upper[i],
        });
    }

    const std::size_t before = edges.size();
    const std::size_t keep_total_limit = before - (before / 11); // keep top 10/11
    selectEdgesBySupport(edges, label, protect_span, keep_total_limit);
    std::vector<TempEdge> &kept_edges = edges;
    
    // 5. Append new row
    size_t count = kept_edges.size();
    // Note: selectEdgesBySupport() already returns edges sorted by external_id.
    size_t capacity = static_cast<size_t>(std::ceil(count * 1.1));
    
    if (row_offset_.empty()) {
        row_offset_.push_back(0);
    }
    // New row id is the current number of rows. row_offset_ always stores one extra
    // tail entry, so its size is (num_rows + 1).
    const size_t row_id = row_offset_.size() - 1;
    size_t start_offset = neighbors_.size();
    size_t new_total_size = start_offset + capacity;
    
    // Resize arrays
    // Performance note:
    // - These SoA vectors are global flat buffers. If `new_total_size` exceeds current capacity,
    //   std::vector will reallocate and move/copy the entire buffer (amortized, but still costly).
    // - If you expect to insert many nodes, consider calling reserve() upfront with an estimated
    //   final edge capacity to reduce reallocations during repeated insertions.
    neighbors_.resize(new_total_size);
    left_lower_.resize(new_total_size);
    left_upper_.resize(new_total_size);
    right_lower_.resize(new_total_size);
    right_upper_.resize(new_total_size);
    
    // Write edges
    for (size_t i = 0; i < count; ++i) {
        neighbors_[start_offset + i] = kept_edges[i].external_id;
        left_lower_[start_offset + i] = kept_edges[i].left_lower;
        left_upper_[start_offset + i] = kept_edges[i].left_upper;
        right_lower_[start_offset + i] = kept_edges[i].right_lower;
        right_upper_[start_offset + i] = kept_edges[i].right_upper;
    }
    
    // Update metadata
    row_to_label_.push_back(label);
    label_to_row_[label] = static_cast<int32_t>(row_id);
    node_degrees_.push_back({static_cast<uint16_t>(count), 0});
    // row_offset_[row_id] was already the old tail; append the new tail end offset.
    row_offset_.push_back(new_total_size);
    num_indexed_nodes_ = static_cast<unsigned>(row_id + 1);
    edges_amount += count;
    
    // 6. Add reverse edges
    for (const auto& edge : kept_edges) {
        addReverseEdge(edge.external_id, label, edge.left_lower, edge.left_upper, edge.right_lower, edge.right_upper);
    }
}

/**
 * @brief Add (src -> dst) as a reverse edge into src's slack region.
 *
 * The "forward region" is sorted and deduplicated via binary search.
 * The "reverse region" is an unsorted tail used as a write-optimized buffer.
 * When slack is full we call `recompress(src)` to rebuild a compact sorted forward region.
 */
void DynamicSegmentGraph::addReverseEdge(unsigned src, unsigned dst, unsigned ll, unsigned lu, unsigned rl, unsigned ru) {
    // Map external label -> internal row-id.
    if (src >= label_to_row_.size()) {
        return;
    }
    const int32_t src_row_i = label_to_row_[src];
    if (src_row_i < 0) {
        return;
    }
    const size_t src_row = static_cast<size_t>(src_row_i);

    const size_t start = row_offset_[src_row];
    const size_t end = row_offset_[src_row + 1];
    const size_t capacity = end - start;
    auto &deg = node_degrees_[src_row];

    size_t fwd = static_cast<size_t>(deg.fwd);
    size_t rev = static_cast<size_t>(deg.rev);

    if (fwd + rev >= capacity) {
        recompress(src);
        fwd = static_cast<size_t>(deg.fwd);
        rev = static_cast<size_t>(deg.rev);
        if (fwd + rev >= capacity) {
            return;
        }
    }

    // Append into slack (unsorted).
    // We assume `dst` is a newly inserted label, so it does not already exist in src's adjacency.
    const size_t insert_idx = start + fwd + rev;
    neighbors_[insert_idx] = dst;
    left_lower_[insert_idx] = ll;
    left_upper_[insert_idx] = lu;
    right_lower_[insert_idx] = rl;
    right_upper_[insert_idx] = ru;
    deg.rev = static_cast<uint16_t>(rev + 1);
}

/**
 * @brief Recompress a single node's adjacency list to regain slack.
 *
 * @details We rebuild a new forward neighbor list by:
 *  - taking all currently stored edges in the row (fwd + rev),
 *  - re-running DFS dominance compression,
 *  - then keeping Top-K edges by support-weighted score, where K is the original forward
 *    degree (deg.fwd) loaded from disk.
 *
 * After recompression, we rewrite the first `deg.fwd` entries in-place and clear `deg.rev`.
 * The allocated capacity (row_offset_[row+1] - row_offset_[row]) does not change.
 */
void DynamicSegmentGraph::recompress(unsigned label) {
    // recompress() is only called for already-inserted labels (see addReverseEdge()).
    const size_t row = static_cast<size_t>(label_to_row_[label]);

    const size_t start = row_offset_[row];
    const size_t capacity = row_offset_[row + 1] - start;
    auto &deg = node_degrees_[row];
    const size_t fwd = static_cast<size_t>(deg.fwd);
    const size_t rev = static_cast<size_t>(deg.rev);
    const size_t total = fwd + rev;

    // 1. Collect candidates
    std::vector<std::pair<unsigned, DistType>> candidates;
    candidates.reserve(total);
    
    const float* query_vec = data_wrapper->nodes[label];
    
    for (size_t i = 0; i < total; ++i) {
        unsigned nbr = neighbors_[start + i];
        float dist = dist_func_(query_vec, data_wrapper->nodes[nbr], dist_func_param_);
        candidates.push_back({nbr, dist});
    }
    
    // Sort
    std::sort(candidates.begin(), candidates.end(), [](const auto& a, const auto& b) {
        return a.second < b.second;
    });
    
    // 2. DFS
    applyDfsCompression(label, candidates);
    
    // 3. Prune (Strict Top-K by support weight)
    //
    // Important: `M` is the theoretical max branching factor *per query range*.
    // The stored adjacency per node (deg.fwd) can be much larger because DSG
    // stores edges with different (L, R) envelopes. During recompression we
    // should not clamp the stored list to `M`, otherwise we lose coverage.
    //
    // Strategy:
    // - Use the original forward degree as the target budget. In the dynamic setting,
    //   `deg.fwd` is treated as a fixed per-row budget loaded from disk.
    
    auto &scratch = dfs_scratch_;
    const size_t cand_count = scratch.ordered_candidates.size();

    const size_t protect_span = (num_indexed_nodes_ + 49) / 50;

    std::vector<TempEdge> edges;
    edges.reserve(cand_count);
    for (size_t i = 0; i < cand_count; ++i) {
        if (!scratch.is_neighbor[i]) {
            continue;
        }
        edges.push_back(TempEdge{
            scratch.ordered_candidates[i].first,
            scratch.left_lower[i],
            scratch.left_upper[i],
            scratch.right_lower[i],
            scratch.right_upper[i],
        });
    }

    // Just fix as the original fwd.
    // TODO: Maybe we can use a better policy to determine the keep_total_limit.
    
    const std::size_t keep_total_limit = fwd;
    selectEdgesBySupport(edges, label, protect_span, keep_total_limit);
    const std::vector<TempEdge> &final_edges = edges;
    
    // 4. Rewrite
    size_t new_count = final_edges.size();
    for (size_t i = 0; i < new_count; ++i) {
        neighbors_[start + i] = final_edges[i].external_id;
        left_lower_[start + i] = final_edges[i].left_lower;
        left_upper_[start + i] = final_edges[i].left_upper;
        right_lower_[start + i] = final_edges[i].right_lower;
        right_upper_[start + i] = final_edges[i].right_upper;
    }
    
    // Update counts: keep the recompressed edges as forward and clear accumulated reverse edges.
    deg.fwd = static_cast<uint16_t>(new_count);
    deg.rev = 0;
}

void DynamicSegmentGraph::getStats() {
    const std::size_t node_count = node_degrees_.size();
    std::uint64_t sum_fwd = 0;
    std::uint64_t sum_rev = 0;
    for (const auto &deg : node_degrees_) {
        sum_fwd += deg.fwd;
        sum_rev += deg.rev;
    }

    edges_amount = static_cast<std::size_t>(sum_fwd + sum_rev);
    avg_forward_nns = node_count == 0 ? 0.0F : static_cast<float>(sum_fwd) / static_cast<float>(node_count);
    avg_reverse_nns = node_count == 0 ? 0.0F : static_cast<float>(sum_rev) / static_cast<float>(node_count);

    std::cout << "DynamicSegmentGraph Statistics:" << std::endl;
    std::cout << "  Build Time: " << index_time << " seconds" << std::endl;
    std::cout << "  Total Edges: " << edges_amount << std::endl;
    std::cout << "  Avg Forward NNs: " << avg_forward_nns << std::endl;
    std::cout << "  Avg Reverse NNs: " << avg_reverse_nns << std::endl;
}

void DynamicSegmentGraph::initializeTemporaryHnsw(size_t ef_limit) {
    if (space_ == nullptr) {
        throw std::runtime_error("DynamicSegmentGraph::initializeTemporaryHnsw missing space.");
    }
    temp_hnsw_ = std::make_unique<HnswType>(space_, data_wrapper->data_size, M, ef_construction, random_seed);
    // setEF is to set the ef for the search in HNSW
    temp_hnsw_->setEf(ef_limit);
}

void DynamicSegmentGraph::runKnnForLabel(
    unsigned label,
    size_t ef_limit,
    std::vector<std::pair<unsigned, DistType>> &candidates) {
    candidates.clear();

    // This KNN is for a point that is already inside the temporary HNSW.
    // We can skip the usual top-layer navigation and search only on level-0,
    // using the point's own level-0 neighbors as the initial frontier.
    //
    // Important details for DSG build:
    // - We mark `label` as visited so it never appears in results.
    // - We seed `candidate_set` with `label`'s level-0 neighbors (NOT `label` itself).
    // - We still keep "return all candidates ever in top_candidates" semantics by
    //   storing candidates removed due to ef trimming and re-adding them at the end.

    const hnswlib::tableint ep_id = static_cast<hnswlib::tableint>(label);

    const void *query = static_cast<const void *>(data_wrapper->nodes[label]);

    hnswlib::VisitedList *vl = temp_hnsw_->visited_list_pool_->getFreeVisitedList();
    hnswlib::vl_type *visited_array = vl->mass;
    const hnswlib::vl_type visited_array_tag = vl->curV;

    visited_array[ep_id] = visited_array_tag;  // exclude the query node itself

    using InternalId = hnswlib::tableint;
    using InternalCandidate = std::pair<DistType, InternalId>;
    using CompareByFirst = typename HnswType::CompareByFirst;

    // Reuse scratch buffers to avoid per-label allocations.
    auto &top_candidates = knn_scratch_.top_candidates_heap;
    auto &candidate_set = knn_scratch_.candidate_set_heap;
    auto &removed_candidates = knn_scratch_.removed_candidates;
    top_candidates.clear();
    candidate_set.clear();
    removed_candidates.clear();
    if (top_candidates.capacity() < ef_limit) {
        top_candidates.reserve(ef_limit);
    }
    if (candidate_set.capacity() < ef_limit) {
        candidate_set.reserve(ef_limit);
    }
    if (removed_candidates.capacity() < ef_limit) {
        removed_candidates.reserve(ef_limit);
    }
    const CompareByFirst heap_comp{};

    auto heap_push = [&](std::vector<InternalCandidate> &heap,
                         const InternalCandidate &value) {
        heap.push_back(value);
        std::push_heap(heap.begin(), heap.end(), heap_comp);
    };
    auto heap_pop = [&](std::vector<InternalCandidate> &heap) -> InternalCandidate {
        std::pop_heap(heap.begin(), heap.end(), heap_comp);
        InternalCandidate value = heap.back();
        heap.pop_back();
        return value;
    };

    DistType lower_bound = std::numeric_limits<DistType>::max();

    auto distance_to = [&](InternalId internal_id) -> DistType {
        return temp_hnsw_->fstdistfunc_(query,
                                       temp_hnsw_->getDataByInternalId(internal_id),
                                       temp_hnsw_->dist_func_param_);
    };

    // Seed the frontier with entry-point's level-0 neighbors.
    int *seed_data = reinterpret_cast<int *>(temp_hnsw_->get_linklist0(ep_id));
    const std::size_t seed_size =
        static_cast<std::size_t>(temp_hnsw_->getListCount(reinterpret_cast<hnswlib::linklistsizeint *>(seed_data)));

#ifdef USE_SSE
    _mm_prefetch(reinterpret_cast<const char *>(visited_array + *(seed_data + 1)), _MM_HINT_T0);
    _mm_prefetch(reinterpret_cast<const char *>(visited_array + *(seed_data + 1) + 64), _MM_HINT_T0);
    _mm_prefetch(temp_hnsw_->data_level0_memory_ + (*(seed_data + 1)) * temp_hnsw_->size_data_per_element_ +
                     temp_hnsw_->offsetData_,
                 _MM_HINT_T0);
    _mm_prefetch(reinterpret_cast<const char *>(seed_data + 2), _MM_HINT_T0);
#endif

    for (std::size_t j = 1; j <= seed_size; ++j) {
#ifdef USE_SSE
        _mm_prefetch(reinterpret_cast<const char *>(visited_array + *(seed_data + j + 1)), _MM_HINT_T0);
        _mm_prefetch(temp_hnsw_->data_level0_memory_ + (*(seed_data + j + 1)) * temp_hnsw_->size_data_per_element_ +
                         temp_hnsw_->offsetData_,
                     _MM_HINT_T0);
#endif
        const InternalId cand_id = static_cast<InternalId>(*(seed_data + j));
        if (visited_array[cand_id] == visited_array_tag) {
            continue;
        }
        visited_array[cand_id] = visited_array_tag;

        const DistType dist = distance_to(cand_id);
        heap_push(candidate_set, InternalCandidate{-dist, cand_id});
#ifdef USE_SSE
        _mm_prefetch(temp_hnsw_->data_level0_memory_ + candidate_set.front().second * temp_hnsw_->size_data_per_element_ +
                         temp_hnsw_->offsetLevel0_,
                     _MM_HINT_T0);
#endif
        heap_push(top_candidates, InternalCandidate{dist, cand_id});

        if (top_candidates.size() > ef_limit) {
            removed_candidates.emplace_back(heap_pop(top_candidates));
        }
        if (!top_candidates.empty()) {
            lower_bound = top_candidates.front().first;
        }
    }

    // Standard HNSW base-layer best-first traversal (level 0).
    while (!candidate_set.empty()) {
        const auto &current_node_pair = candidate_set.front();
        const DistType candidate_dist = -current_node_pair.first;
        if (candidate_dist > lower_bound) {
            break;
        }
        const auto current_node_pair_value = heap_pop(candidate_set);

        const InternalId current_node_id = current_node_pair_value.second;
        int *data = reinterpret_cast<int *>(temp_hnsw_->get_linklist0(current_node_id));
        const std::size_t size =
            static_cast<std::size_t>(temp_hnsw_->getListCount(reinterpret_cast<hnswlib::linklistsizeint *>(data)));

#ifdef USE_SSE
        _mm_prefetch(reinterpret_cast<const char *>(visited_array + *(data + 1)), _MM_HINT_T0);
        _mm_prefetch(reinterpret_cast<const char *>(visited_array + *(data + 1) + 64), _MM_HINT_T0);
        _mm_prefetch(temp_hnsw_->data_level0_memory_ + (*(data + 1)) * temp_hnsw_->size_data_per_element_ +
                         temp_hnsw_->offsetData_,
                     _MM_HINT_T0);
        _mm_prefetch(reinterpret_cast<const char *>(data + 2), _MM_HINT_T0);
#endif

        for (std::size_t j = 1; j <= size; ++j) {
#ifdef USE_SSE
            _mm_prefetch(reinterpret_cast<const char *>(visited_array + *(data + j + 1)), _MM_HINT_T0);
            _mm_prefetch(temp_hnsw_->data_level0_memory_ + (*(data + j + 1)) * temp_hnsw_->size_data_per_element_ +
                             temp_hnsw_->offsetData_,
                         _MM_HINT_T0);
#endif
            const InternalId cand_id = static_cast<InternalId>(*(data + j));
            if (visited_array[cand_id] == visited_array_tag) {
                continue;
            }
            visited_array[cand_id] = visited_array_tag;

            const DistType dist = distance_to(cand_id);
            if (top_candidates.size() < ef_limit || lower_bound > dist) {
                heap_push(candidate_set, InternalCandidate{-dist, cand_id});
#ifdef USE_SSE
                _mm_prefetch(temp_hnsw_->data_level0_memory_ +
                                 candidate_set.front().second * temp_hnsw_->size_data_per_element_ +
                                 temp_hnsw_->offsetLevel0_,
                             _MM_HINT_T0);
#endif
                heap_push(top_candidates, InternalCandidate{dist, cand_id});

                if (top_candidates.size() > ef_limit) {
                    removed_candidates.emplace_back(heap_pop(top_candidates));
                }
                lower_bound = top_candidates.front().first;
            }
        }
    }

    // Re-add any candidates that were trimmed due to ef_limit (ReturnAll semantics).
    for (const auto &removed : removed_candidates) {
        heap_push(top_candidates, removed);
    }

    temp_hnsw_->visited_list_pool_->releaseVisitedList(vl);

    // `top_candidates` is a max-heap by distance (farthest-first pops).
    // Fill output from back to front so `candidates` becomes nearest-first.
    const std::size_t out_size = top_candidates.size();
    candidates.resize(out_size);
    std::size_t out_pos = out_size;
    while (!top_candidates.empty()) {
        const auto [dist, internal_id] = heap_pop(top_candidates);

        const unsigned neighbor_label =
            static_cast<unsigned>(temp_hnsw_->getExternalLabel(internal_id));
        candidates[--out_pos] = {neighbor_label, dist};
    }
}

/**
 * @brief DFS-based dominance compression for segment edges.
 *
 * @details We traverse candidates in increasing distance order and select up to M neighbors.
 * A candidate u is dominated by an already-selected neighbor v if:
 *   dist(u, v) < dist(center, u) / alpha
 *
 * For every selected neighbor we also maintain an eligibility envelope [L,R] x [ll,lu] x [rl,ru]
 * encoded as (left_lower/upper, right_lower/upper) which is later used by `rangeSearch()` to
 * filter edges for a specific query range.
 *
 * Implementation notes:
 *  - `domination_grid` caches pairwise domination decisions to avoid repeated distance calls.
 *  - Worst-case cache size is O(ef^2) bytes per call (uint8_t per pair).
 */
void DynamicSegmentGraph::applyDfsCompression(
    unsigned center_label,
    std::vector<std::pair<unsigned, DistType>> &candidates) {
    const std::size_t max_neighbors = static_cast<std::size_t>(M);
    const auto &nodes = data_wrapper->nodes;
    const float *const nodes_base = nodes.data();
    const std::size_t nodes_dim = nodes.dim();
    const DistType inv_alpha = static_cast<DistType>(1.0f / alpha);

    auto &ordered = dfs_scratch_.ordered_candidates;
    ordered.assign(candidates.begin(), candidates.end());

    const std::size_t candidate_count = ordered.size();
    auto &prefix = dfs_scratch_.prefix;
    prefix.clear();
    prefix.reserve(max_neighbors);

    dfs_scratch_.is_neighbor.assign(candidate_count, false);
    dfs_scratch_.left_lower.assign(candidate_count, 0);
    dfs_scratch_.left_upper.assign(candidate_count, 0);
    dfs_scratch_.right_lower.assign(candidate_count, 0);
    dfs_scratch_.right_upper.assign(candidate_count, 0);
    
    // Resize and clear the domination grid (0 = unknown)
    const size_t grid_size = candidate_count * candidate_count;
    if (dfs_scratch_.domination_grid.size() < grid_size) {
        dfs_scratch_.domination_grid.resize(grid_size);
    }
    // We only need to clear the relevant part, but given the logic,
    // candidates change every time, so we must reset validity.
    // std::fill is fast enough for typical ef sizes (e.g. 100-500 -> 10KB-250KB).
    // Optimization: only clear if we are reusing a large buffer? 
    // For now, simpler to just clear the needed portion.
    std::fill(dfs_scratch_.domination_grid.begin(), dfs_scratch_.domination_grid.begin() + grid_size, 0);

    const unsigned global_left = 0;
    const unsigned global_right =
        static_cast<unsigned>(data_wrapper->data_size - 1);

    auto dfs = [&](auto &&self, unsigned L, unsigned R, unsigned left_cap,
                   unsigned right_cap) -> void {
        if (prefix.size() >= max_neighbors) {
            return;
        }
        const unsigned start_idx = prefix.empty() ? 0 : prefix.back() + 1;
        for (unsigned idx = start_idx; idx < candidate_count; ++idx) {
            const unsigned candidate_label = ordered[idx].first;
            const DistType candidate_dist = ordered[idx].second;
            const DistType dom_threshold = candidate_dist * inv_alpha;
            if (candidate_label < L || candidate_label > R) {
                continue;
            }

            const float *const cand_vec =
                nodes_base + static_cast<std::size_t>(candidate_label) * nodes_dim;

            bool dominated = false;
            for (unsigned prev_idx : prefix) {
                // Flat 2D index
                const size_t grid_idx = static_cast<size_t>(prev_idx) * candidate_count + idx;
                const uint8_t status = dfs_scratch_.domination_grid[grid_idx];
                
                if (status != 0) {
                    dominated = (status == 1);
                } else {
                    const unsigned prev_label = ordered[prev_idx].first;
                    const float *const prev_vec =
                        nodes_base + static_cast<std::size_t>(prev_label) * nodes_dim;
                    const DistType pair_dist = dist_func_(
                        cand_vec,
                        prev_vec,
                        dist_func_param_);
                    dominated = pair_dist < dom_threshold;
                    dfs_scratch_.domination_grid[grid_idx] = dominated ? 1 : 2;
                }
                
                if (dominated) {
                    break;
                }
            }
            if (dominated) {
                continue;
            }

            unsigned next_left_cap = left_cap;
            unsigned next_right_cap = right_cap;
            if (candidate_label < center_label) {
                next_left_cap = std::min(candidate_label, left_cap);
            } else if (candidate_label > center_label) {
                next_right_cap = std::max(candidate_label, right_cap);
            }

            prefix.push_back(idx);
            if (!dfs_scratch_.is_neighbor[idx]) {
                dfs_scratch_.is_neighbor[idx] = true;
                dfs_scratch_.left_lower[idx] = L;
                dfs_scratch_.left_upper[idx] = next_left_cap;
                dfs_scratch_.right_lower[idx] = next_right_cap;
                dfs_scratch_.right_upper[idx] = R;
            } else {
                dfs_scratch_.left_lower[idx] =
                    std::min(dfs_scratch_.left_lower[idx], L);
                dfs_scratch_.left_upper[idx] =
                    std::max(dfs_scratch_.left_upper[idx], next_left_cap);
                dfs_scratch_.right_lower[idx] =
                    std::min(dfs_scratch_.right_lower[idx], next_right_cap);
                dfs_scratch_.right_upper[idx] =
                    std::max(dfs_scratch_.right_upper[idx], R);
            }

            self(self, L, R, next_left_cap, next_right_cap);

            prefix.pop_back();

            // Shrink [L, R] to avoid exploring label subranges that have already been
            // "covered" by the current candidate relative to the left/right caps.
            if (candidate_label < left_cap) {
                L = candidate_label + 1;
            } else if (candidate_label > right_cap) {
                R = candidate_label - 1;
            } else {
                break;
            }

            if (prefix.size() >= max_neighbors) {
                break;
            }
        }
    };

    dfs(dfs, global_left, global_right, center_label, center_label);
}

} // namespace dsg
