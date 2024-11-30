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

#include "data_processing.h"
#include "data_wrapper.h"
#include "index_base.h"
#include "logger.h"
#include "reader.h"
#include "segment_graph_2d.h"
#include "compact_graph.h"
#include "utils.h"
#include <iomanip>

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
        //  << "\t Internal Search Time: " << internal_search_time
        //  << "\t Fetch NN Time: " << fetch_nn_time
        //  << "\t CalDist Time: " << calDistTime << std::endl; // 新增一行显示CalDist时间
    }
}

#ifdef SAVESEARCHPATH
std::ofstream Compact::IndexCompactGraph::log_query_path_nns("path_nns_deep_96.bin", std::ios::out | std::ios::binary);
#endif

int main(int argc, char **argv) {
#ifdef USE_SSE
    cout << "Use SSE" << endl;
#endif

    // Parameters
    string dataset = "deep";
    int data_size = 100000;
    string dataset_path = "";
    string method = "";
    string query_path = "";
    string groundtruth_path = "";
    int query_num = 1000;
    int query_k = 10;
    unsigned index_k = 8;
    unsigned ef_max = 500;
    unsigned ef_construction = 100;
    
    string index_path;
    string version = "Benchmark";

    for (int i = 0; i < argc; i++) {
        string arg = argv[i];
        if (arg == "-dataset") dataset = string(argv[i + 1]);
        if (arg == "-N")
            data_size = atoi(argv[i + 1]);
        if (arg == "-dataset_path")
            dataset_path = string(argv[i + 1]);
        if (arg == "-query_path")
            query_path = string(argv[i + 1]);
        if (arg == "-groundtruth_path")
            groundtruth_path = string(argv[i + 1]);
        if (arg == "-method")
            method = string(argv[i + 1]);
        if (arg == "-index_path")
            index_path = string(argv[i + 1]);
        if (arg == "-k")
            index_k = atoi(argv[i + 1]);
        if (arg == "-ef_max")
            ef_max = atoi(argv[i + 1]);
        if (arg == "-ef_construction")
            ef_construction = atoi(argv[i + 1]);
    }

    DataWrapper data_wrapper(query_num, query_k, dataset, data_size);
    data_wrapper.readData(dataset_path, query_path);
    data_wrapper.LoadGroundtruth(groundtruth_path);
    assert(data_wrapper.query_ids.size() == data_wrapper.query_ranges.size());

    int st = 16;     // starting value
    int ed = 400;    // ending value (inclusive)
    int stride = 16; // stride value
    std::vector<int> searchef_para_range_list;
    // searchef_para_range_list.push_back(96);
    // // add small seach ef
    // for (int i = 1; i < st; i += 1) {
    //     searchef_para_range_list.push_back(i);
    // }

    for (int i = st; i <= ed; i += stride) {
        searchef_para_range_list.push_back(i);
    }

    // // further add more large search ef
    // st = 500;     // starting value
    // ed = 1600;    // ending value (inclusive)
    // stride = 100; // stride value
    // for (int i = st; i <= ed; i += stride) {
    //     searchef_para_range_list.push_back(i);
    // }

    cout << "search ef:" << endl;
    print_set(searchef_para_range_list);

    data_wrapper.version = version;

    base_hnsw::L2Space ss(data_wrapper.data_dim);

    timeval t1, t2;

    BaseIndex::IndexParams i_params(index_k, ef_construction,
                                    ef_construction, ef_max);
    BaseIndex* index;
    if(method == "Seg2D"){
        index = new SeRF::IndexSegmentGraph2D(&ss, &data_wrapper);
    }else{
        index = new Compact::IndexCompactGraph(&ss, &data_wrapper);
    }
    BaseIndex::SearchInfo search_info(&data_wrapper, &i_params, "SeRF_2D",
                                      "benchmark");

    gettimeofday(&t1, NULL);
    index->load(index_path);
    gettimeofday(&t2, NULL);
    logTime(t1, t2, "Load Index Time");

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

                // output the range in the log query path
#ifdef SAVESEARCHPATH
                Compact::IndexCompactGraph::log_query_path_nns.write(reinterpret_cast<const char*>(&s_params.query_range), sizeof(s_params.query_range));
#endif
                if(method == "Seg2D"){
                    auto res = index->rangeFilteringSearchOutBound(
                    &s_params, &search_info, data_wrapper.querys.at(one_id),
                    data_wrapper.query_ranges.at(idx));
                search_info.precision =
                    countPrecision(data_wrapper.groundtruth.at(idx), res);
                }else{
                    auto res = index->rangeFilteringSearchInRange(
                    &s_params, &search_info, data_wrapper.querys.at(one_id),
                    data_wrapper.query_ranges.at(idx));
                search_info.precision =
                    countPrecision(data_wrapper.groundtruth.at(idx), res);
                }
                
                std::get<0>(result_recorder[s_params.query_range]) += search_info.precision;
                std::get<1>(result_recorder[s_params.query_range]) += search_info.cal_dist_time;
                std::get<2>(result_recorder[s_params.query_range]) += search_info.internal_search_time;
                std::get<3>(result_recorder[s_params.query_range]) += search_info.fetch_nns_time;
                std::get<0>(comparison_recorder[s_params.query_range]) += search_info.total_comparison;
                std::get<1>(comparison_recorder[s_params.query_range]) += search_info.path_counter;
            }
#ifdef SAVESEARCHPATH
            Compact::IndexCompactGraph::log_query_path_nns.close();
#endif            
            cout << endl
                 << "Search ef: " << one_searchef << endl
                 << "========================" << endl;
            log_result_recorder(result_recorder, comparison_recorder,
                                data_wrapper.query_ids.size());
            cout << "========================" << endl;
            logTime(tt3, tt4, "total query time");
        }
    }

    return 0;
}