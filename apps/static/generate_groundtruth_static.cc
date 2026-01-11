/**
 * @file generate_groundtruth_static.cc
 * @brief Generate groundtruth for the static range-filter workload.
 *
 * This tool generates groundtruth files under `groundtruth/static/` for the
 * standard static evaluation protocol (fixed dataset, fixed query set, fixed
 * query ranges). Dynamic workload groundtruth (if needed) will be added later
 * under apps/dynamic/.
 */

#include <cstdlib>
#include <iostream>
#include <stdexcept>
#include <string>

#include "data_wrapper.h"

using std::cout;
using std::endl;
using std::string;

namespace {

void PrintUsage(const char *prog) {
    cout << "Usage: " << prog
         << " -dataset_path DATA -query_path QUERIES "
            "[-dataset NAME] [-N DATA_SIZE] "
            "[-groundtruth_root DIR] [-query_num NUM] [-query_k TOPK]\n";
    cout << "Defaults: dataset=deep, N=100000, NUM=1000, TOPK=10, "
            "groundtruth_root=./groundtruth/static\n";
}

string FormatSizeSymbol(int data_size) {
    if (data_size % 1'000'000 == 0 && data_size >= 1'000'000) {
        return std::to_string(data_size / 1'000'000) + "m";
    }
    return std::to_string(data_size / 1'000) + "k";
}

} // namespace

int main(int argc, char **argv) {
    string dataset = "deep";
    int data_size = 100000;
    string dataset_path;
    string query_path;
    string groundtruth_root = "./groundtruth/static";
    constexpr int query_num = DataWrapper::kStaticQueryNum;
    constexpr int query_k = DataWrapper::kStaticTopK;

    auto require_value = [&](int &idx, const string &flag) -> string {
        if (idx + 1 >= argc) {
            throw std::invalid_argument(flag + " requires a value.");
        }
        return string(argv[++idx]);
    };

    try {
        for (int i = 1; i < argc; ++i) {
            string arg = argv[i];
            if (arg == "-dataset") {
                dataset = require_value(i, arg);
            } else if (arg == "-N") {
                data_size = std::stoi(require_value(i, arg));
            } else if (arg == "-dataset_path") {
                dataset_path = require_value(i, arg);
            } else if (arg == "-query_path") {
                query_path = require_value(i, arg);
            } else if (arg == "-groundtruth_root") {
                groundtruth_root = require_value(i, arg);
            } else if (arg == "-h" || arg == "--help") {
                PrintUsage(argv[0]);
                return 0;
            } else {
                throw std::invalid_argument("Unknown flag: " + arg);
            }
        }
    } catch (const std::exception &ex) {
        cout << "Argument error: " << ex.what() << endl;
        PrintUsage(argv[0]);
        return 1;
    }

    if (dataset_path.empty() || query_path.empty()) {
        cout << "Both -dataset_path and -query_path are required.\n";
        PrintUsage(argv[0]);
        return 1;
    }

    cout << "Generating groundtruth (static workload) for dataset '" << dataset << "' "
         << "(N=" << data_size << ", queries=" << query_num
         << ", topK=" << query_k << ")\n";
    cout << "Dataset path: " << dataset_path << "\n"
         << "Query path:   " << query_path << "\n"
         << "Output root:  " << groundtruth_root << endl;
    cout << "Size symbol is " << FormatSizeSymbol(data_size) << endl;

    DataWrapper data_wrapper(query_num, query_k, dataset, data_size);
    data_wrapper.readData(dataset_path, query_path);
    data_wrapper.generateRangeFilteringQueriesAndGroundtruthBenchmark(
        groundtruth_root);

    return 0;
}
