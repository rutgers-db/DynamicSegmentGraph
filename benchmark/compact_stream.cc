/**
 * @file exp_halfbound.cc
 * @author Chaoji Zuo (chaoji.zuo@rutgers.edu)
 * @brief Benchmark Half-Bounded Range Filter Search
 * @date 2023-12-22
 *
 * @copyright Copyright (c) 2023
 */

#include <algorithm>
#include <iostream>
#include <map>
#include <numeric>
#include <random>
#include <sstream>
#include <vector>
#include <tuple>
#include <iomanip>

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
    if (batch_sizes.size() > 1)
        assert(batch_sizes[0] < batch_sizes[1]);
    // Seed the random number generator with current time
    std::default_random_engine rng(std::chrono::system_clock::now().time_since_epoch().count());

    for (unsigned size : batch_sizes) {
        std::vector<unsigned> batch;

        // Generate numbers from 'start' to 'size - 1'
        for (unsigned i = start; i < size; ++i) {
            batch.push_back(i);
        }

        // Shuffle to create a random permutation
        // Only for aribitrary order
        // Attention here!!! TODO
        // std::shuffle(batch.begin(), batch.end(), rng);

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

int main(int argc, char **argv) {
#ifdef USE_SSE
    cout << "Use SSE" << endl;
#endif

    // Parameters
    string dataset = "deep";
    // vector<unsigned> batches_size = {1000, 10000, 100000, 1000000, 10000000};
    vector<unsigned> batches_size = {1150000, 1200000}; //1000000, 1050000, 1100000, 
    // vector<unsigned> batches_size = {1000, 10000};
    // int data_size = 10000000;
    // int data_size = 10000;
    int data_size = 1200000;
    
    auto insert_batches = generatePermutations(batches_size);
    string dataset_path = "";
    string query_path = "";
    string index_path = "";

    unsigned index_k = 16;
    unsigned ef_max = 500;
    unsigned ef_construction = 100;
    int query_num = 1000;
    int query_k = 10;

    string indexk_str = "";
    string ef_con_str = "";
    string version = "Benchmark";
    vector<string> gt_paths = {
        // "/research/projects/zp128/RangeIndexWithRandomInsertion/groundtruth/wiki-image_benchmark-groundtruth-deep-1m-num1000-k10.arbitrary.cvs",
        // "/research/projects/zp128/RangeIndexWithRandomInsertion/groundtruth/wiki-image_benchmark-groundtruth-deep-1050k-num1000-k10.arbitrary.cvs",
        // "/research/projects/zp128/RangeIndexWithRandomInsertion/groundtruth/wiki-image_benchmark-groundtruth-deep-1100k-num1000-k10.arbitrary.cvs",
        "/research/projects/zp128/RangeIndexWithRandomInsertion/groundtruth/wiki-image_benchmark-groundtruth-deep-1150-num1000-k10.arbitrary.cvs",
        "/research/projects/zp128/RangeIndexWithRandomInsertion/groundtruth/wiki-image_benchmark-groundtruth-deep-1200k-num1000-k10.arbitrary.cvs"};

    // vector<string> gt_paths = {
    //     "/research/projects/zp128/RangeIndexWithRandomInsertion/groundtruth/deep_benchmark-groundtruth-deep-1k-num1000-k10.arbitrary.cvs",
    //     "/research/projects/zp128/RangeIndexWithRandomInsertion/groundtruth/deep_benchmark-groundtruth-deep-10k-num1000-k10.arbitrary.cvs",
    //     "/research/projects/zp128/RangeIndexWithRandomInsertion/groundtruth/deep_benchmark-groundtruth-deep-100k-num1000-k10.arbitrary.cvs",
    //     "/research/projects/zp128/RangeIndexWithRandomInsertion/groundtruth/deep_benchmark-groundtruth-deep-1m-num1000-k10.arbitrary.cvs",
    //     "/research/projects/zp128/RangeIndexWithRandomInsertion/groundtruth/deep_benchmark-groundtruth-deep-10m-num1000-k10.arbitrary.cvs"};

    for (int i = 0; i < argc; i++) {
        string arg = argv[i];
        if (arg == "-dataset")
            dataset = string(argv[i + 1]);
        if (arg == "-dataset_path")
            dataset_path = string(argv[i + 1]);
        if (arg == "-query_path")
            query_path = string(argv[i + 1]);
        if (arg == "-index_path")
            index_path = string(argv[i + 1]);
        if (arg == "-k")
            index_k = atoi(argv[i + 1]);
        if (arg == "-ef_max")
            ef_max = atoi(argv[i + 1]);
        if (arg == "-ef_construction")
            ef_construction = atoi(argv[i + 1]);
    }

    if (dataset != "wiki-image") {
        ReplaceSubstringInPaths(gt_paths, "wiki-image", dataset);
        cout << "Print the first groundtruth path" << gt_paths[0] << endl;
    }

    DataWrapper data_wrapper(query_num, query_k, dataset, data_size);
    data_wrapper.readData(dataset_path, query_path);

    int st = 16;     // starting value
    int ed = 400;    // ending value (inclusive)
    int stride = 32; // stride value
    std::vector<int> searchef_para_range_list;
    for (int i = st; i <= ed; i += stride) {
        searchef_para_range_list.push_back(i);
    }
    cout << "search ef:" << endl;
    print_set(searchef_para_range_list);
    cout << "index K:" << index_k << " ef construction: " << ef_construction << " ef_max: " << ef_max << endl;

    data_wrapper.version = version;
    base_hnsw::L2Space ss(data_wrapper.data_dim);
    timeval t1, t2;

    BaseIndex::IndexParams i_params(index_k, ef_construction,
                                    ef_construction, ef_max);

    Compact::IndexCompactGraph *index = new Compact::IndexCompactGraph(&ss, &data_wrapper);
    // SeRF::IndexSegmentGraph2D *index = new SeRF::IndexSegmentGraph2D(&ss, &data_wrapper);
    cout << " parameters: ef_construction ( " + to_string(i_params.ef_construction) + " )  index-k( "
         << i_params.K << ")  ef_max (" << i_params.ef_max << ") "
         << endl;
    index->initForScabilityExp(&i_params, &ss);

    for (int i = 0; i < insert_batches.size(); i++) {
        auto &insert_batch = insert_batches[i];
        auto &gt_path = gt_paths[i];
        index->insert_batch(insert_batch);
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