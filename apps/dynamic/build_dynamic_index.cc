/**
 * @file build_dynamic_index.cc
 * @brief Build an initial DSG index for dynamic workloads by indexing only a subset of labels.
 *
 * This CLI is the first step toward *dynamic workload* experiments (build + later insert/query):
 * - Load a dataset in the existing unified binary format.
 * - Randomly choose a subset of labels (default: half) using a reproducible RNG seed.
 * - Build the DSG index using only the chosen labels.
 * - Persist the remaining (unchosen) labels to disk so later insertion/query workloads can
 *   replay the exact holdout set.
 *
 * Remaining-labels file format (little-endian):
 * - uint32_t magic   (0x4C424753, ASCII "SGBL")
 * - uint32_t version (1)
 * - uint64_t count
 * - count * uint32_t labels (sorted ascending)
 *
 * Author: Zhencan Peng
 * Date: 2025/11/30
 */

#include <algorithm>
#include <chrono>
#include <cstdint>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <random>
#include <stdexcept>
#include <string>
#include <vector>

#include "base_hnsw/hnswlib.h"
#include "data_wrapper.h"
#include "dsg.h"
#include "reader.h"

using namespace std::chrono;

namespace {
constexpr uint32_t kRemainingLabelsMagic = 0x4C424753U; // "SGBL"
constexpr uint32_t kRemainingLabelsVersion = 1U;
constexpr float kDefaultBuildRatio = 0.5F;

struct BuildConfig {
    std::string dataset = "deep";
    std::string dataset_path;
    std::string query_path; // Optional for building; future workloads may require it.
    std::string index_path = "dsg_dynamic.index";
    std::string remaining_labels_path = "remaining_labels.bin";
    int data_size = 100000;
    int query_num = 1000;
    int query_k = 10;
    unsigned index_k = 16;
    unsigned ef_construction = 100;
    unsigned ef_max = 400;
    unsigned random_seed = 2025;
    float alpha = 1.0F;
    float build_ratio = kDefaultBuildRatio;
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
        } else if (arg == "-remaining_labels_path") {
            cfg.remaining_labels_path = require_value("-remaining_labels_path");
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
        } else if (arg == "-build_ratio") {
            cfg.build_ratio = std::stof(require_value("-build_ratio"));
        } else if (arg == "-h" || arg == "--help") {
            throw std::invalid_argument("usage");
        }
    }

    if (cfg.dataset_path.empty()) {
        throw std::invalid_argument("dataset_path is required (-dataset_path).");
    }
    if (cfg.remaining_labels_path.empty()) {
        throw std::invalid_argument("remaining_labels_path is required (-remaining_labels_path).");
    }
    if (cfg.data_size <= 0) {
        throw std::invalid_argument("N must be > 0.");
    }
    if (!(cfg.build_ratio > 0.0F && cfg.build_ratio < 1.0F)) {
        throw std::invalid_argument("build_ratio must be in (0, 1).");
    }
    return cfg;
}

void printUsage() {
    std::cout << "Usage: build_dynamic_index "
                 "-dataset_path <path> -remaining_labels_path <path> [-index_path path] "
                 "[-dataset name] [-N size] [-k out_degree] [-ef_construction val] "
                 "[-ef_max val] [-alpha val] [-seed val] [-build_ratio float] "
                 "[-query_path path]\n";
}

void writeRemainingLabels(const std::string &file_path,
                          const std::vector<unsigned> &remaining_sorted) {
    std::ofstream out(file_path, std::ios::binary | std::ios::trunc);
    if (!out.is_open()) {
        throw std::runtime_error("Cannot open file for writing remaining labels: " + file_path);
    }

    const uint32_t magic = kRemainingLabelsMagic;
    const uint32_t version = kRemainingLabelsVersion;
    const uint64_t count = static_cast<uint64_t>(remaining_sorted.size());
    out.write(reinterpret_cast<const char *>(&magic), sizeof(uint32_t));
    out.write(reinterpret_cast<const char *>(&version), sizeof(uint32_t));
    out.write(reinterpret_cast<const char *>(&count), sizeof(uint64_t));

    for (unsigned v : remaining_sorted) {
        const uint32_t u = static_cast<uint32_t>(v);
        out.write(reinterpret_cast<const char *>(&u), sizeof(uint32_t));
    }
}
} // namespace

int main(int argc, char **argv) {
    try {
        const BuildConfig cfg = parseArgs(argc, argv);

        DataWrapper data_wrapper(cfg.query_num, cfg.query_k, cfg.dataset, cfg.data_size);
        ReadBinaryVectors(cfg.dataset_path, data_wrapper.nodes, cfg.data_size);
        data_wrapper.data_dim = data_wrapper.nodes.dim();

        hnswlib::L2Space space(data_wrapper.data_dim);
        dsg::DynamicSegmentGraph index(&space, &data_wrapper);
        index.M = cfg.index_k;
        index.ef_construction = cfg.ef_construction;
        index.ef_max = cfg.ef_max;
        index.alpha = cfg.alpha;
        index.random_seed = cfg.random_seed;

        std::vector<unsigned> shuffled = data_wrapper.labels;
        std::mt19937 rng(cfg.random_seed);
        std::shuffle(shuffled.begin(), shuffled.end(), rng);

        const size_t n = shuffled.size();
        const size_t build_n = std::max<size_t>(1, static_cast<size_t>(static_cast<double>(n) * cfg.build_ratio));
        std::vector<unsigned> build_labels(shuffled.begin(), shuffled.begin() + static_cast<std::ptrdiff_t>(build_n));
        std::vector<unsigned> remaining_labels(shuffled.begin() + static_cast<std::ptrdiff_t>(build_n), shuffled.end());
        // DSG build currently requires labels in ascending order.
        std::sort(build_labels.begin(), build_labels.end());
        std::sort(remaining_labels.begin(), remaining_labels.end());

        std::cout << "[DSG][dynamic-build] dataset=" << cfg.dataset << " N=" << cfg.data_size
                  << " build_ratio=" << std::fixed << std::setprecision(3) << cfg.build_ratio
                  << " build_n=" << build_labels.size()
                  << " remaining_n=" << remaining_labels.size()
                  << " M=" << cfg.index_k
                  << " ef_construction=" << cfg.ef_construction
                  << " ef_max=" << cfg.ef_max
                  << " alpha=" << cfg.alpha
                  << " seed=" << cfg.random_seed << "\n";

        const auto build_start = steady_clock::now();
        index.build(build_labels);
        const auto build_end = steady_clock::now();
        const double seconds = duration<double>(build_end - build_start).count();
        std::cout << "[DSG][dynamic-build] Build finished in " << seconds << " seconds\n";

        index.getStats();
        index.save(cfg.index_path);
        std::cout << "[DSG][dynamic-build] Index saved to " << cfg.index_path << "\n";

        writeRemainingLabels(cfg.remaining_labels_path, remaining_labels);
        std::cout << "[DSG][dynamic-build] Remaining labels saved to " << cfg.remaining_labels_path << "\n";
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

