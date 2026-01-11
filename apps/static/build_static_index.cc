/**
 * @file build_static_index.cc
 * @brief Build a static DSG index for a fixed dataset (full label set 0..N-1).
 *
 * This CLI is intentionally scoped to the *static workload*:
 * - Build an index for a fixed dataset snapshot.
 * - Save the resulting index to disk (v2 magic-header format).
 *
 * Future dynamic workloads (mixed insert/query, replay traces, partial builds)
 * will live under apps/dynamic/.
 *
 * Author: Zhencan Peng
 */

#include <chrono>
#include <cstring>
#include <iostream>
#include <stdexcept>
#include <string>

#include "base_hnsw/hnswlib.h"
#include "data_wrapper.h"
#include "dsg.h"

using namespace std::chrono;

namespace {
struct BuildConfig {
    std::string dataset = "deep";
    std::string dataset_path;
    std::string query_path;
    std::string index_path = "dsg.index";
    int data_size = 100000;
    int query_num = 1000;
    int query_k = 10;
    unsigned index_k = 16;
    unsigned ef_construction = 100;
    unsigned ef_max = 400;
    unsigned random_seed = 2025;
    float alpha = 1.0F;
};

BuildConfig parseArgs(int argc, char **argv) {
    BuildConfig cfg;
    for (int i = 1; i < argc; ++i) {
        const std::string arg = argv[i];
        auto require_value = [&](const char *flag) -> const char * {
            if (i + 1 >= argc) {
                throw std::invalid_argument(std::string("Missing value for ") + flag);
            }
            return argv[++i];
        };
        if (arg == "-dataset") {
            cfg.dataset = require_value("-dataset");
        } else if (arg == "-dataset_path") {
            cfg.dataset_path = require_value("-dataset_path");
        } else if (arg == "-query_path") {
            cfg.query_path = require_value("-query_path");
        } else if (arg == "-index_path") {
            cfg.index_path = require_value("-index_path");
        } else if (arg == "-N") {
            cfg.data_size = std::stoi(require_value("-N"));
        } else if (arg == "-query_num") {
            cfg.query_num = std::stoi(require_value("-query_num"));
        } else if (arg == "-query_k") {
            cfg.query_k = std::stoi(require_value("-query_k"));
        } else if (arg == "-k") {
            cfg.index_k = static_cast<unsigned>(std::stoul(require_value("-k")));
        } else if (arg == "-ef_construction") {
            cfg.ef_construction = static_cast<unsigned>(std::stoul(require_value("-ef_construction")));
        } else if (arg == "-ef_max") {
            cfg.ef_max = static_cast<unsigned>(std::stoul(require_value("-ef_max")));
        } else if (arg == "-alpha") {
            cfg.alpha = std::stof(require_value("-alpha"));
        } else if (arg == "-seed") {
            cfg.random_seed = static_cast<unsigned>(std::stoul(require_value("-seed")));
        } else if (arg == "-h" || arg == "--help") {
            throw std::invalid_argument("usage");
        }
    }
    if (cfg.dataset_path.empty()) {
        throw std::invalid_argument("dataset_path is required (-dataset_path).");
    }
    return cfg;
}

void printUsage() {
    std::cout << "Usage: build_static_index "
                 "-dataset_path <path> [-dataset name] [-N size] [-k out_degree] "
                 "[-ef_construction val] [-ef_max val] [-alpha val] "
                 "[-query_path path] [-index_path path] [-seed val]\n";
}
} // namespace

int main(int argc, char **argv) {
    try {
        const BuildConfig cfg = parseArgs(argc, argv);

        DataWrapper data_wrapper(cfg.query_num, cfg.query_k, cfg.dataset, cfg.data_size);
        std::string dataset_path = cfg.dataset_path;
        std::string query_path = cfg.query_path;
        data_wrapper.readData(dataset_path, query_path);

        hnswlib::L2Space space(data_wrapper.data_dim);
        dsg::DynamicSegmentGraph index(&space, &data_wrapper);
        index.M = cfg.index_k;
        index.ef_construction = cfg.ef_construction;
        index.ef_max = cfg.ef_max;
        index.alpha = cfg.alpha;
        index.random_seed = cfg.random_seed;

        std::cout << "[DSG][static] dataset=" << cfg.dataset << " N=" << cfg.data_size
                  << " M=" << cfg.index_k
                  << " ef_construction=" << cfg.ef_construction << " ef_max=" << cfg.ef_max
                  << " alpha=" << cfg.alpha << "\n";

        const auto build_start = steady_clock::now();
        // Static workload: build full label set (0..N-1).
        index.build(data_wrapper.labels);
        const auto build_end = steady_clock::now();
        const double seconds = duration<double>(build_end - build_start).count();
        std::cout << "[DSG][static] Build finished in " << seconds << " seconds\n";
        index.getStats();
        index.save(cfg.index_path);
        std::cout << "[DSG][static] Index saved to " << cfg.index_path << "\n";
    } catch (const std::invalid_argument &ex) {
        printUsage();
        std::cerr << "Argument error: " << ex.what() << std::endl;
        return 1;
    } catch (const std::exception &ex) {
        std::cerr << "Build failed: " << ex.what() << std::endl;
        return 1;
    }

    return 0;
}
