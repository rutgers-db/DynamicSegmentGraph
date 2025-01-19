/**
 *  KNN First Baseline
 *  with the dataste with 1M splitted into 10 parts with random labels
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
#include <sys/stat.h>

#include "baselines/knn_first_hnsw.h"
#include "data_processing.h"
#include "data_wrapper.h"
#include "index_base.h"
#include "logger.h"
#include "reader.h"
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

#ifdef _WIN32
SIZE_T getMemoryUsage() {
    PROCESS_MEMORY_COUNTERS pmc;
    if (GetProcessMemoryInfo(GetCurrentProcess(), &pmc, sizeof(pmc))) {
        return pmc.WorkingSetSize; // Returns memory usage in bytes
    }
    return 0;
}
#else
long getMemoryUsage() {
    struct rusage usage;
    getrusage(RUSAGE_SELF, &usage);
    return usage.ru_maxrss; // Memory usage in kilobytes
}
#endif

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

// Function to replace a substring in all elements of a vector of strings
void ReplaceSubstringInPath(std::string &path, const std::string &old_str, const std::string &new_str) {
    size_t pos = path.find(old_str);
    if (pos != std::string::npos) {
        path.replace(pos, old_str.length(), new_str);
    }
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

void tagParameters(vector<string> &paths, string file_type, unsigned index_k, unsigned ef_max, unsigned ef_construction) {
    string suffix = "_" + to_string(index_k) + "_" + to_string(ef_max) + "_" + to_string(ef_construction);

    for (auto &path : paths) {
        size_t pos = path.find(file_type);
        if (pos != string::npos) {
            path.insert(pos, suffix);
        }
    }
}

bool fileExists(const string &filePath) {
    struct stat buffer;
    // stat() returns 0 if the file exists
    return (stat(filePath.c_str(), &buffer) == 0);
}

int main(int argc, char **argv) {
#ifdef USE_SSE
    cout << "Use SSE" << endl;
#endif

    // Parameters
    int data_size = 1000000;
    int part_num = 10;
    unsigned index_k = 16;
    unsigned ef_max = 500;
    unsigned ef_construction = 100;
    int query_num = 1000;
    int query_k = 10;

    string dataset = "deep";
    string dataset_path = "";
    string query_path = "";
    string version = "Benchmark";

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
    string gt_dir = "/groundtruth/incremental/wiki-image_1m-num1000-k10/groundtruth_part_";
    string index_dir = "/index/incremental_stream/wiki-image/";
    string log_dir = "/log/incremental_stream/wiki-image/";
    string permutation_path = root_path + "/groundtruth/incremental/wiki-image_1m-num1000-k10/permutation.bin";

    // get all the gt parts file paths
    vector<string> gt_paths;
    vector<string> index_paths;
    vector<string> log_paths;
    for (auto i = 0; i < part_num; i++) {
        gt_paths.emplace_back(root_path + gt_dir + to_string(i) + ".csv");
        index_paths.emplace_back(root_path + index_dir + to_string(i) + ".bin");
        log_paths.emplace_back(root_path + log_dir + to_string(i) + ".log");
    }
    tagParameters(index_paths, ".bin", index_k, ef_max, ef_construction);
    tagParameters(log_paths, ".log", index_k, ef_max, ef_construction);

    if (dataset != "wiki-image") {
        ReplaceSubstringInPaths(gt_paths, "wiki-image", dataset);
        ReplaceSubstringInPaths(index_paths, "wiki-image", dataset);
        ReplaceSubstringInPaths(log_paths, "wiki-image", dataset);
        ReplaceSubstringInPath(permutation_path, "wiki-image", dataset);
        cout << "Print the first groundtruth path" << gt_paths[0] << endl;
    }

    auto insert_batches = ReadAndSplit(permutation_path, part_num);
    DataWrapper data_wrapper(query_num, query_k, dataset, data_size);
    data_wrapper.readData(dataset_path, query_path);

    int st = 16;     // starting value
    int ed = 400;    // ending value (inclusive)
    int stride = 32; // stride value

    std::vector<int> searchef_para_range_list;
    // add small seach ef
    // for (int i = 6; i < st; i += 1) {
    //     searchef_para_range_list.push_back(i);
    // }
    for (int i = st; i <= ed; i += stride) {
        searchef_para_range_list.push_back(i);
    }
    cout << "search ef:" << endl;
    print_set(searchef_para_range_list);
    cout << "index K:" << index_k << " ef construction: " << ef_construction << " ef_max: " << ef_max << endl;

    data_wrapper.version = version;
    timeval t1, t2;

    BaseIndex::IndexParams i_params;
    i_params.ef_construction = ef_construction;
    i_params.K = index_k;

    KnnFirstWrapper index(&data_wrapper);
    auto *ss = new hnswlib_incre::L2Space(data_wrapper.data_dim);
    index.initForBuilding(&i_params, ss);

    cout << " parameters: ef_construction ( " + to_string(i_params.ef_construction) + " )  index-k( "
         << i_params.K << ")  ef_max (" << i_params.ef_max << ") "
         << endl;

    for (int i = 0; i < insert_batches.size(); i++) {
        auto &insert_batch = insert_batches[i];
        auto &gt_path = gt_paths[i];
        {
            // A temporary region for record the memory allocation
            MemoryRecorder memoryRecorder("Global Array Allocation");
            gettimeofday(&t1, NULL);
            index.insertBatch(insert_batch);
            gettimeofday(&t2, NULL);
            logTime(t1, t2, "Build knnfirst HNSW Index Time");
        }
        data_wrapper.LoadGroundtruth(gt_path);
        cout << "HNSW Total # of Neighbors: " << index.index_info->nodes_amount
                << endl;

        cout << "twoside" << endl;
        BaseIndex::SearchInfo search_info(&data_wrapper, &i_params,
                                            "KnnFirstHnsw",
                                            "knnfirst/twoside");
        execute_knn_first_search(index, search_info, data_wrapper,
                                    searchef_para_range_list);
    }

    return 0;
}