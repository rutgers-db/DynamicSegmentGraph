/**
 * @file ordered_dsg_vs_digra.cc
 * @brief DIGRA-only ordered incremental experiment: 180k build, insert remaining 20k, evaluate QPS on 200k GT.
 *
 * Experiment design:
 * - Total N default = 200,000 points (180k build batch + 20k incremental batch).
 * - Build initial index with the first 180,000 points (90% of total, ascending id).
 * - Insert the remaining 20,000 points (10% of total, ascending id) one-by-one in attribute/id order.
 * - Evaluate query recall and QPS using 200k groundtruth: "...-200k-num1000-k10.arbitrary.cvs".
 *
 * Implementation details:
 * - DIGRA uses RangeHNSW (see digra_adapter.h). Build with initial_N=180k, then addPoint() for the remaining 20k ordered inserts.
 * - This approach keeps incremental inserts under 20% of initial capacity to avoid segmentation faults.
 * - Query pipelines follow existing project conventions and formats.
 * - We report per-ef recall and QPS via the same accumulator format used in other benchmarks.
 *
 * C++ standard: C++17
 *
 * Complexity (high level):
 * - Build (HNSW-like): expected O(N log N) average with graph-degree and ef_construction factors; memory ~ O(N * K).
 * - Ordered inserts: expected O(log N) to O(ef * K) per insertion depending on graph/search parameters.
 * - Query: approximately O(ef * log N) distance evaluations; priority-queue/beam cost hidden in ef.
 */

#include <algorithm>
#include <chrono>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <map>
#include <numeric>
#include <random>
#include <sstream>
#include <string>
#include <sys/stat.h>
#include <tuple>
#include <vector>
#include <cstring>

#include "data_wrapper.h"
#include "utils.h"
#include "digra_adapter.h"

#ifdef __linux__
#include "sys/sysinfo.h"
#include "sys/types.h"
#endif

using std::cout;
using std::endl;
using std::string;
using std::to_string;
using std::vector;

#include <sys/resource.h>
static long getMemoryUsageKB() {
    struct rusage usage;
    getrusage(RUSAGE_SELF, &usage);
    return usage.ru_maxrss; // KB on Linux/macOS
}

class MemoryRecorder {
public:
    explicit MemoryRecorder(const std::string &description)
        : description_(description), before_(getMemoryUsageKB()) {}
    ~MemoryRecorder() {
        auto after = getMemoryUsageKB();
        std::cout << description_ << ": " << (after - before_) << " KB memory used." << std::endl;
    }
private:
    std::string description_;
    long before_;
};

static void ReplaceSubstringInPath(std::string &path, const std::string &old_str, const std::string &new_str) {
    size_t pos = path.find(old_str);
    if (pos != std::string::npos) {
        path.replace(pos, old_str.length(), new_str);
    }
}

static void ReplaceSubstringInPaths(std::vector<std::string> &paths, const std::string &old_str, const std::string &new_str) {
    for (std::string &path : paths) ReplaceSubstringInPath(path, old_str, new_str);
}

static vector<unsigned> makeRange(unsigned start_inclusive, unsigned end_inclusive) {
    vector<unsigned> ids;
    ids.reserve((size_t)end_inclusive - (size_t)start_inclusive + 1);
    for (unsigned i = start_inclusive; i <= end_inclusive; ++i) ids.push_back(i);
    return ids;
}

static float* flatten_subset(const vector<vector<float>> &nodes, const vector<unsigned> &ids, int dim) {
    float *buf = new float[(size_t)ids.size() * (size_t)dim];
    for (size_t i = 0; i < ids.size(); i++) {
        const auto &row = nodes[ids[i]];
        std::memcpy(buf + i * dim, row.data(), sizeof(float) * (size_t)dim);
    }
    return buf;
}

static void log_result_recorder(
    const std::map<int, std::tuple<double, double, double, double>> &result_recorder,
    const std::map<int, std::tuple<float, float>> &comparison_recorder,
    const int amount) {
    for (auto item : result_recorder) {
        const auto &[recall, calDistTime, internal_search_time, fetch_nn_time] = item.second;
        const auto &[comps, hops] = comparison_recorder.at(item.first);
        const auto cur_range_amount = amount / (int)result_recorder.size();
        cout << std::setiosflags(std::ios::fixed) << std::setprecision(4)
             << "range: " << item.first
             << "\t recall: " << recall / cur_range_amount
             << "\t QPS: " << std::setprecision(0)
             << cur_range_amount / internal_search_time << "\t"
             << "Comps: " << comps / cur_range_amount << std::setprecision(4)
             << "\t Hops: " << hops / cur_range_amount << std::setprecision(4) << std::endl;
    }
}

int main(int argc, char **argv) {
    // Defaults
    string dataset = "deep";
    string dataset_path = "";
    string query_path = "";
    unsigned digra_k = 16;             // per dataset below
    unsigned ef_max = 500;              // unused by DIGRA but kept for CLI compatibility
    unsigned ef_construction = 300;     // default for deep; override via CLI
    int query_num = 1000;
    int query_k = 10;

    // Two-batch setting: 183k build then +20k incremental inserts
    const int DEFAULT_INITIAL_N = 183000;
    int total_N = 200000;  // can be overridden by -N
    int initial_N = DEFAULT_INITIAL_N;

    // CLI
    for (int i = 0; i < argc; i++) {
        string arg = argv[i];
        if (arg == "-dataset" && i + 1 < argc) dataset = string(argv[i + 1]);
        if (arg == "-dataset_path" && i + 1 < argc) dataset_path = string(argv[i + 1]);
        if (arg == "-query_path" && i + 1 < argc) query_path = string(argv[i + 1]);
        if (arg == "-k" && i + 1 < argc) digra_k = (unsigned)atoi(argv[i + 1]);
        if (arg == "-ef_max" && i + 1 < argc) ef_max = (unsigned)atoi(argv[i + 1]);
        if (arg == "-ef_construction" && i + 1 < argc) ef_construction = (unsigned)atoi(argv[i + 1]);
        if (arg == "-N" && i + 1 < argc) total_N = atoi(argv[i + 1]);
    }
    if (total_N < initial_N) initial_N = total_N;

    // Per-dataset K for DIGRA
    if (dataset == "deep") {
        digra_k = 16;
    } else if (dataset == "yt8m-video" || dataset == "wiki-image") {
        digra_k = 64;
    }

    string root_path = "/research/projects/zp128/RangeIndexWithRandomInsertion";
    // Groundtruth for 200k evaluation following the existing naming scheme
    string gt_path_200k = root_path + "/groundtruth/ordered_stream/wiki-image_benchmark-groundtruth-deep-200k-num1000-k10.arbitrary.cvs";
    if (dataset != "wiki-image") ReplaceSubstringInPath(gt_path_200k, "wiki-image", dataset);

    DataWrapper data_wrapper(query_num, query_k, dataset, total_N);
    data_wrapper.readData(dataset_path, query_path);

    // ef list similar to ordered_serfVsDsg.cc
    cout << "search ef:" << endl;
    vector<int> searchef_para_range_list;
    const int EF_ST = 32, EF_ED = 128, EF_STRIDE = 16;
    for (int i = 1; i < EF_ST; i += 1) searchef_para_range_list.push_back(i);
    for (int i = EF_ST; i <= EF_ED; i += EF_STRIDE) searchef_para_range_list.push_back(i);
    print_set(searchef_para_range_list);

    cout << "DIGRA K:" << digra_k
         << " ef construction: " << ef_construction
         << " ef_max: " << ef_max << endl;

    const int dim = (int)data_wrapper.data_dim;

    // ===== DIGRA Build on first 180k =====
    vector<unsigned> initial_ids = makeRange(0u, (unsigned)initial_N - 1u);
    float *baseData = flatten_subset(data_wrapper.nodes, initial_ids, dim);
    int *keyList = new int[(size_t)initial_N];
    int *valueList = new int[(size_t)initial_N];
    for (int i = 0; i < initial_N; i++) { keyList[i] = i; valueList[i] = i; }

    long rss_before_kb = getMemoryUsageKB();
    auto t_init_start = std::chrono::high_resolution_clock::now();
    // Use smaller buffer since we're only inserting 20k more (< 20% of initial capacity)
    DigraIndex digra(dim, initial_N, (int)total_N + 70000, baseData, keyList, valueList, (int)digra_k, (int)ef_construction);
    auto t_init_end = std::chrono::high_resolution_clock::now();
    long rss_after_kb = getMemoryUsageKB();
    cout << "DIGRA build " << initial_N << " time(s): "
         << std::setprecision(4) << std::chrono::duration<double>(t_init_end - t_init_start).count() << endl;
    cout << "DIGRA init(" << initial_N << ") RSS delta(MB): "
         << std::setprecision(3) << ((rss_after_kb - rss_before_kb) / 1024.0) << endl;

    // ===== Ordered inserts: remaining 20k to reach 200k =====
    // Strategy: Insert only 10% more elements (20k out of 180k initial) to avoid buffer overflow
    // This keeps incremental insertions well under the 20% safety threshold
    vector<unsigned> incremental_ids;
    if (initial_N < total_N) incremental_ids = makeRange((unsigned)initial_N, (unsigned)total_N - 1u);
    cout << "Incremental inserts: " << incremental_ids.size() << " elements (" 
         << std::setprecision(1) << (100.0 * incremental_ids.size() / initial_N) 
         << "% of initial capacity)" << endl;
    
    if (!incremental_ids.empty()) {
        double total_insert_sec = 0.0;
        for (auto id : incremental_ids) {
            if(id % 1000 == 0) cout << "DIGRA inserting id: " << id << endl;
            auto t0 = std::chrono::high_resolution_clock::now();
            digra.addPoint((int)id, (int)id, (float*)data_wrapper.nodes.at(id).data());
            auto t1 = std::chrono::high_resolution_clock::now();
            total_insert_sec += std::chrono::duration<double>(t1 - t0).count();
        }
        double avg_ms = (total_insert_sec / (double)incremental_ids.size()) * 1000.0;
        cout << "DIGRA avg per-insert time over " << incremental_ids.size() << ": "
             << std::setprecision(3) << avg_ms << " ms" << endl;
    }

    // ===== Evaluate on 200k GT =====
    cout << endl << "==== [DIGRA] Query evaluation on 200k groundtruth (after ordered inserts) =====" << endl;
    data_wrapper.LoadGroundtruth(gt_path_200k);
    for (auto one_se : searchef_para_range_list) {
        std::map<int, std::tuple<double, double, double, double>> result_recorder; // recall, calDist, internal_search, fetch
        std::map<int, std::tuple<float, float>> comparison_recorder; // comps, hops (unused)
        for (int idx = 0; idx < (int)data_wrapper.query_ids.size(); idx++) {
            int one_id = data_wrapper.query_ids.at(idx);
            auto ql = data_wrapper.query_ranges.at(idx).first;
            auto qr = data_wrapper.query_ranges.at(idx).second;
            int query_range = qr - ql + 1;
            auto t1 = std::chrono::high_resolution_clock::now();
            auto pq = digra.queryRange((float*)data_wrapper.querys.at(one_id).data(), ql, qr, data_wrapper.query_k, one_se);
            auto t2 = std::chrono::high_resolution_clock::now();
            double elapsed = std::chrono::duration<double>(t2 - t1).count();
            vector<int> res;
            while (!pq.empty()) { res.push_back((int)pq.top().second); pq.pop(); }
            double prec = countPrecision(data_wrapper.groundtruth.at(idx), res);
            std::get<0>(result_recorder[query_range]) += prec;
            std::get<2>(result_recorder[query_range]) += elapsed; // internal_search_time
            std::get<1>(result_recorder[query_range]) += 0.0;      // cal_dist_time
            std::get<3>(result_recorder[query_range]) += 0.0;      // fetch_nns_time
            std::get<0>(comparison_recorder[query_range]) += 0.0f; // comps
            std::get<1>(comparison_recorder[query_range]) += 0.0f; // hops
        }
        cout << endl << "Search ef: " << one_se << endl << "========================" << endl;
        log_result_recorder(result_recorder, comparison_recorder, (int)data_wrapper.query_ids.size());
        cout << "========================" << endl;
    }

    delete[] baseData;
    delete[] keyList;
    delete[] valueList;
    return 0;
}


