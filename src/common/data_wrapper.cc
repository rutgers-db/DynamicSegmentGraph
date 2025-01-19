#include "data_wrapper.h"
#include "reader.h"
#include "utils.h"
#include <sys/stat.h>
#include <sys/types.h>
// #include <omp.h>

void SynthesizeQuerys(const vector<vector<float>> &nodes,
                      vector<vector<float>> &querys,
                      const int query_num) {
    int dim = nodes.front().size();
    std::default_random_engine e;
    std::uniform_int_distribution<int> u(0, nodes.size() - 1);
    querys.clear();
    querys.resize(query_num);

    for (unsigned n = 0; n < query_num; n++) {
        for (unsigned i = 0; i < dim; i++) {
            int select_idx = u(e);
            querys[n].emplace_back(nodes[select_idx][i]);
        }
    }
}

void DataWrapper::readData(string &dataset_path, string &query_path) {
    ReadDataWrapper(dataset, dataset_path, this->nodes, data_size, query_path,
                    this->querys, query_num, this->nodes_keys);
    cout << "Load vecs from: " << dataset_path << endl;
    cout << "# of vecs: " << nodes.size() << endl;

    // already sort data in sorted_data
    // if (dataset != "wiki-image" && dataset != "yt8m") {
    //   nodes_keys.resize(nodes.size());
    //   iota(nodes_keys.begin(), nodes_keys.end(), 0);
    // }

    if (querys.empty()) {
        cout << "Synthesizing querys..." << endl;
        SynthesizeQuerys(nodes, querys, query_num);
    }

    this->real_keys = false;
    vector<size_t> index_permutation; // already sort data ahead

    // if (dataset == "wiki-image" || dataset == "yt8m") {
    //   cout << "first search_key before sorting: " << nodes_keys.front() <<
    //   endl; cout << "sorting dataset: " << dataset << endl; index_permutation =
    //   sort_permutation(nodes_keys); apply_permutation_in_place(nodes,
    //   index_permutation); apply_permutation_in_place(nodes_keys,
    //   index_permutation); cout << "Dimension: " << nodes.front().size() <<
    //   endl; cout << "first search_key: " << nodes_keys.front() << endl;
    //   this->real_keys = true;
    // }
    this->data_dim = this->nodes.front().size();
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

void DataWrapper::LoadGroundtruth(const string &gt_path) {
    this->groundtruth.clear();
    this->query_ranges.clear();
    this->query_ids.clear();
    cout << "Loading Groundtruth from" << gt_path << "...";
    ReadGroundtruthQuery(this->groundtruth, this->query_ranges, this->query_ids,
                         gt_path);
    cout << "    Done!" << endl;
}

void DataWrapper::generateRangeFilteringQueriesAndGroundtruthBenchmark(
    bool is_save_to_file,
    const string save_path) {
    timeval t1, t2;

    vector<int> query_range_list;
    query_range_list.emplace_back(this->data_size * 0.01);
    query_range_list.emplace_back(this->data_size * 0.02);
    query_range_list.emplace_back(this->data_size * 0.04);
    query_range_list.emplace_back(this->data_size * 0.08);
    query_range_list.emplace_back(this->data_size * 0.16);
    query_range_list.emplace_back(this->data_size * 0.32);
    query_range_list.emplace_back(this->data_size * 0.64);
    // query_range_list.emplace_back(this->data_size);

    cout << "Generating Range Filtering Groundtruth...";
    cout << endl
         << "Ranges: " << endl;
    print_set(query_range_list);
    vector<double> bf_latency_aveInEachRange(query_range_list.size(), 0);
    std::default_random_engine e;

    for (int range_id = 0; range_id < query_range_list.size(); range_id++) {
        auto &range = query_range_list[range_id];
        std::uniform_int_distribution<int> u_lbound(0,
                                                    std::max(this->data_size - range - 1, 0));
        for (int i = 0; i < this->querys.size(); i++) {
            int l_bound = u_lbound(e);
            int r_bound = std::min(this->data_size - 1, l_bound + range - 1);
            int search_key_range = r_bound - l_bound + 1;
            // if (this->real_keys)
            //   search_key_range =
            //       this->nodes_keys.at(r_bound) - this->nodes_keys.at(l_bound);
            query_ranges.emplace_back(std::make_pair(l_bound, r_bound));
            double greedy_time;
            gettimeofday(&t1, NULL);
            auto gt = greedyNearest(this->nodes, this->querys.at(i), l_bound, r_bound,
                                    this->query_k);
            gettimeofday(&t2, NULL);
            CountTime(t1, t2, greedy_time);
            bf_latency_aveInEachRange[range_id] += greedy_time;

            groundtruth.emplace_back(gt);
            query_ids.emplace_back(i);
            if (is_save_to_file) {
                SaveToCSVRow(save_path, i, l_bound, r_bound, range, search_key_range,
                             this->query_k, greedy_time, gt);
            }
        }
        bf_latency_aveInEachRange[range_id] /= this->querys.size();
    }
    cout << "Here is bruteforce average time cost when generating groundtruth for each range:" << endl;
    print_set(bf_latency_aveInEachRange);
    if (is_save_to_file) {
        cout << "Save GroundTruth to path: " << save_path << endl;
    }
}

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
    vector<vector<float>> inserted_nodes;
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
        vector<double> avg_greedy_time(query_range_list.size(), 0.0);
        vector<ResultEntry> results(query_range_list.size() * this->querys.size());
        timeval t1, t2;

        for (int range_id = 0; range_id < query_range_list.size(); ++range_id) {
            auto &range = query_range_list[range_id];
            std::uniform_int_distribution<int> u_lbound(0,
                                                         std::max(this->data_size - range, 0));

// #pragma omp parallel for
            for (int i = 0; i < this->querys.size(); ++i) {
                int l_bound = u_lbound(random_engine);
                int r_bound = std::min(this->data_size - 1, l_bound + range - 1);

                double greedy_time;
                gettimeofday(&t1, NULL);
                auto gt = scanNearest(inserted_nodes, inserted_keys, this->querys[i], l_bound, r_bound, this->query_k);
                gettimeofday(&t2, NULL);
                CountTime(t1, t2, greedy_time);
                auto result_idx = range_id * this->querys.size() + i;
                results[result_idx] = ResultEntry{i, l_bound, r_bound, range, r_bound - l_bound + 1, greedy_time, gt};
// #pragma omp critical
                {
                    avg_greedy_time[range_id] += greedy_time;
                }
            }
            avg_greedy_time[range_id] /= this->querys.size();
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
