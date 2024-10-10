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
    unsigned index_k = 8;
    unsigned ef_max = 500;
    unsigned ef_construction = 100;
    int query_num = 1000;
    int query_k = 10;
    

    string indexk_str = "";
    string ef_con_str = "";
    string version = "Benchmark";
    string index_dir_path;

    for (int i = 0; i < argc; i++) {
        string arg = argv[i];
        if (arg == "-dataset") dataset = string(argv[i + 1]);
        if (arg == "-N")
            data_size = atoi(argv[i + 1]);
        if (arg == "-dataset_path")
            dataset_path = string(argv[i + 1]);
        if (arg == "-index_path")
            index_dir_path = string(argv[i + 1]);
        if (arg == "-method")
            method = string(argv[i + 1]);
        if (arg == "-k")
            index_k = atoi(argv[i + 1]);
        if (arg == "-ef_max")
            ef_max = atoi(argv[i + 1]);
        if (arg == "-ef_construction")
            ef_construction = atoi(argv[i + 1]);

    }

    assert(index_k_list.size() != 0);
    assert(ef_construction_list.size() != 0);

    DataWrapper data_wrapper(query_num, query_k, dataset, data_size);
    data_wrapper.readData(dataset_path, query_path); // query_path is useless when just building index

    cout << "index K:" << index_k<< " ef construction: "<<ef_construction<<" ef_max: "<< ef_max<< endl;

    data_wrapper.version = version;
    base_hnsw::L2Space ss(data_wrapper.data_dim);
    timeval t1, t2;
    BaseIndex* index;

    BaseIndex::IndexParams i_params(index_k, ef_construction,
                                    ef_construction, ef_max);
    i_params.recursion_type = BaseIndex::IndexParams::MAX_POS;
    {
        
        if(method == "Seg2D"){
            index = new SeRF::IndexSegmentGraph2D(&ss, &data_wrapper);
        }else{
            index = new Compact::IndexCompactGraph(&ss, &data_wrapper);
        }

        cout << "method: " << method<<" parameters: ef_construction ( " + to_string(i_params.ef_construction) + " )  index-k( "
                << i_params.K << ")  ef_max (" << i_params.ef_max << ") "
                << endl;
        gettimeofday(&t1, NULL);
        index->buildIndex(&i_params);
        gettimeofday(&t2, NULL);
        logTime(t1, t2, "Build Index Time");
        

        string save_path = index_dir_path + "/" + method+ "_" + std::to_string(index_k) + "_" + std::to_string(ef_max) + "_" + std::to_string(ef_construction) + ".bin";
        index->save(save_path);
    }


    return 0;
}