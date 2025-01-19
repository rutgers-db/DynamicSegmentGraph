#include <algorithm>
#include <iostream>
#include <map>
#include <numeric>
#include <random>
#include <sstream>
#include <vector>
#include <tuple>
#include <iomanip>
#include <sys/stat.h>

#include "data_processing.h"
#include "data_wrapper.h"
#include "index_base.h"
#include "logger.h"
#include "reader.h"
#include "compact_graph.h"
#include "segment_graph_2d.h"
#include "utils.h"

#ifdef __linux__
#include "sys/sysinfo.h"
#include "sys/types.h"
#endif

using std::cout;
using std::endl;
using std::string;
using std::to_string;
using std::vector;

#include <sys/resource.h> // Linux/macOS
long getMemoryUsage() {
    struct rusage usage;
    getrusage(RUSAGE_SELF, &usage);
    return usage.ru_maxrss; // Memory usage in kilobytes
}
class MemoryRecorder {
public:
    MemoryRecorder(const std::string &description) :
        description_(description), memoryBefore_(getMemoryUsage()) {
    }

    ~MemoryRecorder() {
        auto memoryAfter = getMemoryUsage();
        std::cout << description_ << ": " << (memoryAfter - memoryBefore_)
#ifdef _WIN32
                  << " bytes memory used." << std::endl;
#else
                  << " KB memory used." << std::endl;
#endif
    }

private:
    std::string description_;
    long memoryBefore_;
};

void log_result_recorder(
    const std::map<int, std::tuple<double, double, double, double>> &result_recorder,
    const std::map<int, std::tuple<float, float>> &comparison_recorder,
    const int amount) {
    for (auto item : result_recorder) {
        const auto &[recall, calDistTime, internal_search_time, fetch_nn_time] = item.second;
        const auto &[comps, hops] = comparison_recorder.at(item.first);
        const auto cur_range_amount = amount / result_recorder.size();
        cout << std::setiosflags(ios::fixed) << std::setprecision(4)
             << "range: " << item.first
             << "\t recall: " << recall / cur_range_amount
             << "\t QPS: " << std::setprecision(0)
             << cur_range_amount / internal_search_time << "\t"
             << "Comps: " << comps / cur_range_amount << std::setprecision(4)
             << "\t Hops: " << hops / cur_range_amount << std::setprecision(4) << std::endl;
    }
}

std::vector<std::vector<unsigned>> generatePermutations(const std::vector<unsigned> &batch_sizes) {
    std::vector<std::vector<unsigned>> permutations;
    unsigned start = 0; // Starting point for the first batch

    for (auto i = 0; i < batch_sizes.size(); i++) {
        auto &size = batch_sizes[i];
        std::vector<unsigned> batch;

        // Generate numbers from 'start' to 'size - 1'
        for (unsigned i = start; i < size; ++i) {
            batch.push_back(i);
        }

        // Store the permutation and update the starting point for the next batch
        permutations.push_back(batch);
        start = size;
    }

    return permutations;
}

// Function to replace a substring in all elements of a vector of strings
void ReplaceSubstringInPaths(std::vector<std::string> &paths, const std::string &old_str, const std::string &new_str) {
    for (std::string &path : paths) {
        size_t pos = path.find(old_str);
        if (pos != std::string::npos) {
            path.replace(pos, old_str.length(), new_str);
        }
    }
}

void tagParameters(vector<string> &gt_paths, unsigned index_k, unsigned ef_max, unsigned ef_construction) {
    string suffix = "_" + to_string(index_k) + "_" + to_string(ef_max) + "_" + to_string(ef_construction);

    for (auto &path : gt_paths) {
        size_t pos = path.find(".bin");
        if (pos != string::npos) {
            path.insert(pos, suffix);
        }
    }
}

bool fileExists(const string &filePath) {
    struct stat buffer;
    return (stat(filePath.c_str(), &buffer) == 0);
}

int main(int argc, char **argv) {
#ifdef USE_SSE
    cout << "Use SSE" << endl;
#endif

    // Parameters
    string dataset = "deep";
    vector<unsigned> batches_size;
    for (unsigned i = 100000; i <= 1000000; i += 100000) {
        batches_size.push_back(i);
    }
    int data_size = 1000000;

    auto insert_batches = generatePermutations(batches_size);
    string dataset_path = "";
    string query_path = "";

    unsigned index_k = 16;
    unsigned ef_max = 500;
    unsigned ef_construction = 100;
    int query_num = 1000;
    int query_k = 10;

    string indexk_str = "";
    string ef_con_str = "";
    string version = "Benchmark";
    // vector<string> gt_paths = {
    //     "/research/projects/zp128/RangeIndexWithRandomInsertion/groundtruth/ordered_stream/wiki-image_benchmark-groundtruth-deep-50k-num1000-k10.arbitrary.cvs",
    //     "/research/projects/zp128/RangeIndexWithRandomInsertion/groundtruth/ordered_stream/wiki-image_benchmark-groundtruth-deep-100k-num1000-k10.arbitrary.cvs",
    //     "/research/projects/zp128/RangeIndexWithRandomInsertion/groundtruth/ordered_stream/wiki-image_benchmark-groundtruth-deep-150k-num1000-k10.arbitrary.cvs",
    //     "/research/projects/zp128/RangeIndexWithRandomInsertion/groundtruth/ordered_stream/wiki-image_benchmark-groundtruth-deep-200k-num1000-k10.arbitrary.cvs"};

    vector<string> gt_paths = {
        "/research/projects/zp128/RangeIndexWithRandomInsertion/groundtruth/ordered_stream/wiki-image_benchmark-groundtruth-deep-100k-num1000-k10.arbitrary.cvs",
        "/research/projects/zp128/RangeIndexWithRandomInsertion/groundtruth/ordered_stream/wiki-image_benchmark-groundtruth-deep-200k-num1000-k10.arbitrary.cvs",
        "/research/projects/zp128/RangeIndexWithRandomInsertion/groundtruth/ordered_stream/wiki-image_benchmark-groundtruth-deep-300k-num1000-k10.arbitrary.cvs",
        "/research/projects/zp128/RangeIndexWithRandomInsertion/groundtruth/ordered_stream/wiki-image_benchmark-groundtruth-deep-400k-num1000-k10.arbitrary.cvs",
        "/research/projects/zp128/RangeIndexWithRandomInsertion/groundtruth/ordered_stream/wiki-image_benchmark-groundtruth-deep-500k-num1000-k10.arbitrary.cvs",
        "/research/projects/zp128/RangeIndexWithRandomInsertion/groundtruth/ordered_stream/wiki-image_benchmark-groundtruth-deep-600k-num1000-k10.arbitrary.cvs",
        "/research/projects/zp128/RangeIndexWithRandomInsertion/groundtruth/ordered_stream/wiki-image_benchmark-groundtruth-deep-700k-num1000-k10.arbitrary.cvs",
        "/research/projects/zp128/RangeIndexWithRandomInsertion/groundtruth/ordered_stream/wiki-image_benchmark-groundtruth-deep-800k-num1000-k10.arbitrary.cvs",
        "/research/projects/zp128/RangeIndexWithRandomInsertion/groundtruth/ordered_stream/wiki-image_benchmark-groundtruth-deep-900k-num1000-k10.arbitrary.cvs",
        "/research/projects/zp128/RangeIndexWithRandomInsertion/groundtruth/ordered_stream/wiki-image_benchmark-groundtruth-deep-1m-num1000-k10.arbitrary.cvs"
    };

    for (int i = 0; i < argc; i++) {
        string arg = argv[i];
        if (arg == "-dataset")
            dataset = string(argv[i + 1]);
        if (arg == "-dataset_path")
            dataset_path = string(argv[i + 1]);
        if (arg == "-query_path")
            query_path = string(argv[i + 1]);
        if (arg == "-k")
            index_k = atoi(argv[i + 1]);
        if (arg == "-ef_max")
            ef_max = atoi(argv[i + 1]);
        if (arg == "-ef_construction")
            ef_construction = atoi(argv[i + 1]);
    }

    string root_path = "/research/projects/zp128/RangeIndexWithRandomInsertion";
    vector<string> index_paths;
    string index_dir = "/index/ordered_stream/dsg/wiki-image/";
    for(auto i = 0 ; i< batches_size.size(); i ++){
        index_paths.emplace_back(root_path + index_dir + to_string(i) + ".bin");
    }

    if (dataset != "wiki-image") {
        ReplaceSubstringInPaths(gt_paths, "wiki-image", dataset);
        ReplaceSubstringInPaths(index_paths, "wiki-image", dataset);
        cout << "Print the first groundtruth path" << gt_paths[0] << endl;
    }

    DataWrapper data_wrapper(query_num, query_k, dataset, data_size);
    data_wrapper.readData(dataset_path, query_path);

    int st = 32;     // starting value
    int ed = 400;    // ending value (inclusive)
    int stride = 16; // stride value

    std::vector<int> searchef_para_range_list;
    // // add small seach ef
    // for (int i = 1; i < st; i += 1) {
    //     searchef_para_range_list.push_back(i);
    // }
    // for (int i = st; i <= ed; i += stride) {
    //     searchef_para_range_list.push_back(i);
    // }
    cout << "search ef:" << endl;
    print_set(searchef_para_range_list);
    cout << "index K:" << index_k << " ef construction: " << ef_construction << " ef_max: " << ef_max << endl;

    data_wrapper.version = version;
    base_hnsw::L2Space ss(data_wrapper.data_dim);
    timeval t1, t2;

    float alpha = 1.0;
    if (dataset == "yt8m-video") {
        alpha = 1.3;
    }
    BaseIndex::IndexParams i_params(index_k, ef_construction,
                                    ef_construction, ef_max, alpha);

    Compact::IndexCompactGraph *index = new Compact::IndexCompactGraph(&ss, &data_wrapper);
    // SeRF::IndexSegmentGraph2D *index = new SeRF::IndexSegmentGraph2D(&ss, &data_wrapper);
    cout << " parameters: ef_construction ( " + to_string(i_params.ef_construction) + " )  index-k( "
         << i_params.K << ")  ef_max (" << i_params.ef_max << ") "
         << endl;
    index->initForScabilityExp(&i_params, &ss);

    for (int i = 0; i < insert_batches.size(); i++) {
        auto &insert_batch = insert_batches[i];
        auto &gt_path = gt_paths[i];
        {
            // A temporary region for record the memory allocation
            MemoryRecorder memoryRecorder("Global Array Allocation");
            index->insert_batch(insert_batch);
            index->save(index_paths[i]);
        }
        index->initLabelSet();
        data_wrapper.LoadGroundtruth(gt_path);
        BaseIndex::SearchInfo search_info(&data_wrapper, &i_params, "SeRF_2D",
                                          "benchmark");
        {
            timeval tt3, tt4;
            BaseIndex::SearchParams s_params;
            s_params.query_K = data_wrapper.query_k;
            for (auto one_searchef : searchef_para_range_list) {
                s_params.search_ef = one_searchef;
                std::map<int, std::tuple<double, double, double, double>> result_recorder; // first->precision, second-> caldist time, third->query_time
                std::map<int, std::tuple<float, float>> comparison_recorder;
                gettimeofday(&tt3, NULL);
                for (int idx = 0; idx < data_wrapper.query_ids.size(); idx++) {
                    int one_id = data_wrapper.query_ids.at(idx);
                    s_params.query_range =
                        data_wrapper.query_ranges.at(idx).second - data_wrapper.query_ranges.at(idx).first + 1;
                    auto res = index->rangeFilteringSearchInRange(
                        &s_params, &search_info, data_wrapper.querys.at(one_id),
                        data_wrapper.query_ranges.at(idx));
                    search_info.precision =
                        countPrecision(data_wrapper.groundtruth.at(idx), res);

                    std::get<0>(result_recorder[s_params.query_range]) += search_info.precision;
                    std::get<1>(result_recorder[s_params.query_range]) += search_info.cal_dist_time;
                    std::get<2>(result_recorder[s_params.query_range]) += search_info.internal_search_time;
                    std::get<3>(result_recorder[s_params.query_range]) += search_info.fetch_nns_time;
                    std::get<0>(comparison_recorder[s_params.query_range]) += search_info.total_comparison;
                    std::get<1>(comparison_recorder[s_params.query_range]) += search_info.path_counter;
                }

                cout << endl
                     << "Search ef: " << one_searchef << endl
                     << "========================" << endl;
                log_result_recorder(result_recorder, comparison_recorder,
                                    data_wrapper.query_ids.size());
                cout << "========================" << endl;
                logTime(tt3, tt4, "total query time");
            }
        }
    }

    return 0;
}