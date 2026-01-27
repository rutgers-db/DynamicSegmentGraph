#include "data_wrapper.h"
#include "reader.h"
#include "utils.h"

#include <algorithm>
#include <cmath>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <random>
#include <sstream>
#include <sys/stat.h>
#include <sys/types.h>

#ifdef _OPENMP
#include <omp.h>
#endif

namespace {
namespace fs = std::filesystem;

enum class GroundtruthFileKind { Ranges, Neighbors };

std::string FormatRangeLabel(double ratio_pct) {
    const int pct = static_cast<int>(ratio_pct * 100 + 0.5);
    std::ostringstream oss;
    oss << "range_" << std::setw(2) << std::setfill('0') << pct << "pct";
    return oss.str();
}

std::string BuildGroundtruthFileName(double ratio,
                                     int topk,
                                     int query_num,
                                     int data_size,
                                     GroundtruthFileKind kind) {
    std::ostringstream oss;
    oss << FormatRangeLabel(ratio) << "_top" << topk << "_q" << query_num;
    oss << "_N" << data_size;
    if (kind == GroundtruthFileKind::Ranges) {
        oss << "_ranges.bin";
    } else {
        oss << "_neighbors.bin";
    }
    return oss.str();
}

struct GroundtruthNeighborsHeader {
    uint32_t query_count;
    uint32_t topk;
};

struct GroundtruthRangesHeader {
    uint32_t query_count;
};

void WriteRangesBinary(const fs::path &file_path,
                       const std::vector<std::pair<int, int>> &bounds) {
    fs::create_directories(file_path.parent_path());
    std::ofstream out(file_path, std::ios::binary | std::ios::trunc);
    if (!out.is_open()) {
        throw std::runtime_error("Cannot open file for writing ranges: " +
                                 file_path.string());
    }
    GroundtruthRangesHeader header{
        static_cast<uint32_t>(bounds.size())};
    out.write(reinterpret_cast<const char *>(&header.query_count),
              sizeof(uint32_t));
    std::vector<int32_t> packed;
    packed.reserve(bounds.size() * 2);
    for (const auto &bound : bounds) {
        packed.emplace_back(static_cast<int32_t>(bound.first));
        packed.emplace_back(static_cast<int32_t>(bound.second));
    }
    out.write(reinterpret_cast<const char *>(packed.data()),
              sizeof(int32_t) * packed.size());
}

void WriteNeighborsBinary(const fs::path &file_path,
                          const FlatVectors<int> &groundtruth,
                          uint32_t topk) {
    fs::create_directories(file_path.parent_path());
    std::ofstream out(file_path, std::ios::binary | std::ios::trunc);
    if (!out.is_open()) {
        throw std::runtime_error("Cannot open file for writing groundtruth: " +
                                 file_path.string());
    }
    GroundtruthNeighborsHeader header{
        static_cast<uint32_t>(groundtruth.size()),
        topk};
    out.write(reinterpret_cast<const char *>(&header.query_count),
              sizeof(uint32_t));
    out.write(reinterpret_cast<const char *>(&header.topk), sizeof(uint32_t));
    for (size_t i = 0; i < groundtruth.size(); ++i) {
        const int32_t *row = groundtruth[i];
        out.write(reinterpret_cast<const char *>(row),
                  sizeof(int32_t) * header.topk);
    }
}

void ReadRangesBinary(const fs::path &file_path,
                      std::vector<std::pair<int, int>> &bounds) {
    std::ifstream in(file_path, std::ios::binary);
    if (!in.is_open()) {
        throw std::runtime_error("Cannot open ranges file: " +
                                 file_path.string());
    }
    GroundtruthRangesHeader header{};
    in.read(reinterpret_cast<char *>(&header.query_count), sizeof(uint32_t));
    bounds.resize(header.query_count);
    for (size_t i = 0; i < bounds.size(); ++i) {
        int32_t l = 0;
        int32_t r = 0;
        in.read(reinterpret_cast<char *>(&l), sizeof(int32_t));
        in.read(reinterpret_cast<char *>(&r), sizeof(int32_t));
        bounds[i] = {l, r};
    }
}

void ReadNeighborsBinary(const fs::path &file_path,
                         FlatVectors<int> &groundtruth) {
    std::ifstream in(file_path, std::ios::binary);
    if (!in.is_open()) {
        throw std::runtime_error("Cannot open groundtruth file: " +
                                 file_path.string());
    }
    GroundtruthNeighborsHeader header{};
    in.read(reinterpret_cast<char *>(&header.query_count), sizeof(uint32_t));
    in.read(reinterpret_cast<char *>(&header.topk), sizeof(uint32_t));
    groundtruth.resize(header.query_count, header.topk);
    for (size_t i = 0; i < groundtruth.size(); ++i) {
        int32_t *row = groundtruth[i];
        in.read(reinterpret_cast<char *>(row),
                sizeof(int32_t) * header.topk);
    }
}
} // namespace

void SynthesizeQuerys(const FlatVectors<float> &nodes,
                      FlatVectors<float> &querys,
                      const int query_num) {
    const int dim = static_cast<int>(nodes.dim());
    std::default_random_engine e;
    std::uniform_int_distribution<int> u(0, static_cast<int>(nodes.size()) - 1);
    querys.resize(query_num, dim);

    for (int n = 0; n < query_num; n++) {
        float *row = querys[n];
        for (int i = 0; i < dim; i++) {
            int select_idx = u(e);
            row[i] = nodes[select_idx][i];
        }
    }
}

void DataWrapper::readData(string &dataset_path, string &query_path) {
    ReadDataWrapper(dataset_path, this->nodes, data_size, query_path,
                    this->querys, query_num);
    cout << "Load vecs from: " << dataset_path << endl;
    cout << "# of vecs: " << nodes.size() << endl;

    if (querys.empty()) {
        cout << "Synthesizing querys..." << endl;
        SynthesizeQuerys(nodes, querys, query_num);
    }

    this->data_dim = this->nodes.dim();
}

void SaveToCSVRow(const string &path, const int idx, const int l_bound, const int r_bound, const int pos_range, const int real_search_key_range, const int K_neighbor, const double &search_time, const vector<int> &gt) {
    std::ofstream file;
    file.open(path, std::ios_base::app);
    if (file) {
        file << idx << "," << l_bound << "," << r_bound << "," << pos_range << ","
             << real_search_key_range << "," << K_neighbor << "," << search_time
             << ",";
        for (auto ele : gt) {
            file << ele << " ";
        }
        file << "\n";
    }
    file.close();
}

void DataWrapper::LoadGroundtruth(const string &gt_root) {
    using std::cout;
    using std::endl;
    fs::path base = gt_root.empty() ? fs::path("./groundtruth/static") : fs::path(gt_root);
    fs::path dataset_dir = base;
    const fs::path dataset_candidate = base / this->dataset;
    if (fs::exists(dataset_candidate) && fs::is_directory(dataset_candidate)) {
        dataset_dir = dataset_candidate;
    } else if (this->dataset == "yt8m-video") {
        const fs::path yt8m_candidate = base / "yt8m";
        if (fs::exists(yt8m_candidate) && fs::is_directory(yt8m_candidate)) {
            dataset_dir = yt8m_candidate;
        }
    }
    if (!fs::exists(dataset_dir) || !fs::is_directory(dataset_dir)) {
        throw std::runtime_error("Groundtruth directory not found: " +
                                 dataset_dir.string());
    }
    cout << "Loading groundtruth from " << dataset_dir << "...";
    for (size_t range_id = 0; range_id < kStaticRangeCount; ++range_id) {
        const auto ratio = kRangeRatios[range_id];
        const auto ranges_file =
            BuildGroundtruthFileName(ratio, this->query_k, this->query_num,
                                     this->data_size,
                                     GroundtruthFileKind::Ranges);
        const auto neighbors_file =
            BuildGroundtruthFileName(ratio, this->query_k, this->query_num,
                                     this->data_size,
                                     GroundtruthFileKind::Neighbors);
        ReadRangesBinary(dataset_dir / ranges_file, static_query_ranges[range_id]);
        ReadNeighborsBinary(dataset_dir / neighbors_file,
                            static_groundtruth[range_id]);
    }
    cout << " done." << endl;
}

void DataWrapper::generateRangeFilteringQueriesAndGroundtruthBenchmark(
    const string &save_root) {
    vector<int> query_range_list;
    query_range_list.reserve(kStaticRangeCount);
    for (const auto ratio : kRangeRatios) {
        int window = static_cast<int>(std::round(this->data_size * ratio));
        window = std::max(1, std::min(window, this->data_size));
        query_range_list.emplace_back(window);
    }

    cout << "Generating Range Filtering Groundtruth..." << endl;
    cout << "Ranges: " << endl;
    print_set(query_range_list);
    vector<double> bf_latency_ave(query_range_list.size(), 0.0);
    const int query_count = static_cast<int>(this->querys.size());
    std::default_random_engine e;
    fs::path base = save_root.empty() ? fs::path("./groundtruth/static")
                                      : fs::path(save_root);
    fs::path dataset_dir = base / this->dataset;
    fs::create_directories(dataset_dir);

    for (size_t range_id = 0; range_id < query_range_list.size(); ++range_id) {
        const int window = query_range_list[range_id];
        auto &range_bounds = static_query_ranges[range_id];
        range_bounds.resize(this->query_num);
        std::uniform_int_distribution<int> u_lbound(
            0, std::max(this->data_size - window, 0));
        for (int i = 0; i < query_count; ++i) {
            int l_bound = u_lbound(e);
            int r_bound = std::min(this->data_size - 1, l_bound + window - 1);
            range_bounds[i] = {l_bound, r_bound};
        }

        auto &range_groundtruth = static_groundtruth[range_id];
        range_groundtruth.resize(this->query_num, this->query_k);

        double range_time_acc = 0.0;
#pragma omp parallel for reduction(+ : range_time_acc) schedule(static)
        for (int i = 0; i < query_count; ++i) {
            const auto bounds = range_bounds[i];
            timeval local_t1, local_t2;
            double greedy_time = 0.0;
            gettimeofday(&local_t1, NULL);
            int *row = range_groundtruth[i];
            greedyNearestInto(this->nodes, this->querys.at(i), bounds.first,
                              bounds.second, this->query_k, row);
            gettimeofday(&local_t2, NULL);
            CountTime(local_t1, local_t2, greedy_time);
            range_time_acc += greedy_time;
        }

        bf_latency_ave[range_id] = range_time_acc / static_cast<double>(query_count);
        const auto ratio = kRangeRatios[range_id];
        const auto ranges_file =
            BuildGroundtruthFileName(ratio, this->query_k, this->query_num,
                                     this->data_size,
                                     GroundtruthFileKind::Ranges);
        const auto neighbors_file =
            BuildGroundtruthFileName(ratio, this->query_k, this->query_num,
                                     this->data_size,
                                     GroundtruthFileKind::Neighbors);
        WriteRangesBinary(dataset_dir / ranges_file, range_bounds);
        WriteNeighborsBinary(dataset_dir / neighbors_file, range_groundtruth,
                             this->query_k);
    }

    cout << "Average brute-force latency per range: " << endl;
    print_set(bf_latency_ave);
    cout << "Groundtruth saved under " << dataset_dir << endl;
}

// TODO: This function is not rewritten yet.
// TODO: Need to rewrite this function to generate incremental insertion groundtruth.
void DataWrapper::generateIncrementalInsertionGroundtruth(
    int num_parts,
    const string &save_dir) {
    // Ensure num_parts is greater than zero
    if (num_parts <= 0) {
        throw std::invalid_argument("num_parts must be greater than zero.");
    }

    // Create the save_dir if it does not exist
    struct stat info;
    if (stat(save_dir.c_str(), &info) != 0) {
        if (mkdir(save_dir.c_str(), 0777) != 0) {
            throw std::runtime_error("Failed to create directory: " + save_dir);
        }
    } else if (!(info.st_mode & S_IFDIR)) {
        throw std::runtime_error(save_dir + " exists but is not a directory.");
    }

    // Generate a random permutation of indices
    vector<int> permutation(this->data_size);
    std::iota(permutation.begin(), permutation.end(), 0);
    std::shuffle(permutation.begin(), permutation.end(), std::default_random_engine{});

    // Save the permutation to a binary file
    WriteVectorToFile(save_dir + "/permutation.bin", permutation);

    // Split the permutation into num_parts parts
    int part_size = this->data_size / num_parts;
    int remainder = this->data_size % num_parts;

    // Temporary storage for inserted nodes and keys
    FlatVectors<float> inserted_nodes;
    inserted_nodes.set_dimension(this->nodes.dim());
    vector<int> inserted_keys;

    vector<int> query_range_list;
    query_range_list.emplace_back(this->data_size * 0.01);
    query_range_list.emplace_back(this->data_size * 0.02);
    query_range_list.emplace_back(this->data_size * 0.04);
    query_range_list.emplace_back(this->data_size * 0.08);
    query_range_list.emplace_back(this->data_size * 0.16);
    query_range_list.emplace_back(this->data_size * 0.32);
    query_range_list.emplace_back(this->data_size * 0.64);
    std::default_random_engine random_engine; // Define the random engine outside
    const size_t query_count = this->querys.size();
    const size_t range_bucket_count = query_range_list.size();

    struct ResultEntry {
        int query_idx;
        int l_bound;
        int r_bound;
        int pos_range;
        int real_search_key_range;
        double search_time;
        vector<int> gt;
    };

    // Generate groundtruth incrementally for each part
    for (int part = 0; part < num_parts; ++part) {
        // Determine start and end indices for the current part
        int start_idx = part * part_size;
        int end_idx = start_idx + part_size;
        if (part == num_parts - 1) {
            end_idx += remainder; // Add the remaining elements to the last part
        }

        // Insert nodes from the current part into the temporary storage
        for (int i = start_idx; i < end_idx; ++i) {
            inserted_nodes.push_back(this->nodes[permutation[i]]);
            inserted_keys.emplace_back(permutation[i]);
        }

        // Generate groundtruth for all queries based on inserted nodes
        vector<vector<int>> part_groundtruth;
        vector<double> avg_greedy_time(range_bucket_count, 0.0);
        vector<ResultEntry> results(range_bucket_count * query_count);
        timeval t1, t2;

        for (size_t range_id = 0; range_id < range_bucket_count; ++range_id) {
            const int range = query_range_list[range_id];
            std::uniform_int_distribution<int> u_lbound(0,
                                                         std::max(this->data_size - range, 0));

// #pragma omp parallel for
            for (size_t i = 0; i < query_count; ++i) {
                int l_bound = u_lbound(random_engine);
                int r_bound = std::min(this->data_size - 1, l_bound + range - 1);

                double greedy_time;
                gettimeofday(&t1, NULL);
                auto gt = scanNearest(inserted_nodes, inserted_keys, this->querys[i], l_bound, r_bound, this->query_k);
                gettimeofday(&t2, NULL);
                CountTime(t1, t2, greedy_time);
                const size_t result_idx = range_id * query_count + i;
                results[result_idx] = ResultEntry{static_cast<int>(i), l_bound, r_bound, range, r_bound - l_bound + 1, greedy_time, gt};
// #pragma omp critical
                {
                    avg_greedy_time[range_id] += greedy_time;
                }
            }
            if (query_count > 0) {
                avg_greedy_time[range_id] /= static_cast<double>(query_count);
            }
        }

        // Save results to a CSV
        string range_file = save_dir + "/groundtruth_part_" + std::to_string(part) + ".csv";
        std::ofstream res_file(range_file);
        for (const auto &entry : results) {
            SaveToCSVRow(range_file, entry.query_idx, entry.l_bound, entry.r_bound, entry.pos_range, entry.real_search_key_range, this->query_k, entry.search_time, entry.gt);
        }
        res_file.close();

        // Save average greedy time to a file
        string avg_time_file = save_dir + "/avg_greedy_time_part_" + std::to_string(part) + ".csv";
        std::ofstream time_file(avg_time_file);
        for (size_t range_id = 0; range_id < avg_greedy_time.size(); ++range_id) {
            double qps = 1.0 / avg_greedy_time[range_id]; // Calculate QPS
            time_file << query_range_list[range_id] << "," << avg_greedy_time[range_id] << "," << qps << "\n";
        }
        time_file.close();

        cout << "Saved groundtruth for part " << part << " to " << range_file << endl;
        cout << "Saved average greedy time for part " << part << " to " << avg_time_file << endl;
    }

    cout << "Incremental groundtruth generation completed. Permutation saved to " << save_dir << "/permutation.txt" << endl;
}
