/**
 * @file base_index.h
 *
 * This header isolates the abstract BaseIndex interface so other components
 * can evolve independently from legacy range-search implementations.
 */

#pragma once

// Created by Zhencan Peng on 2025-12-01: refactored BaseIndex definition.

#include <cstddef>
#include <string>
#include <vector>

#include "data_wrapper.h"

static const unsigned default_M = 16;
static const unsigned default_ef_construction = 400;

/**
 * @brief Abstract base class for all range-filter-aware indexes.
 */
class BaseIndex {
public:
    // theoretical max out degree
    // in dsg, it is the theoretical max out degree of the DSG given a range
    // So in dsg, per-node's edge count is actually much larger than M

    unsigned M = default_M;
    // ef construction
    unsigned ef_construction = default_ef_construction;
    // random seed
    unsigned random_seed = 2025;
    // ef max
    unsigned ef_max = 1000;
    // alpha used in Vamana
    float alpha = 1.0F;

    // index record
    double index_time = 0.0;
    std::size_t window_count = 0;
    std::size_t edges_amount = 0;
    float avg_forward_nns = 0.0F;
    float avg_reverse_nns = 0.0F;


    explicit BaseIndex(const DataWrapper *data) : data_wrapper(data) {}
    virtual ~BaseIndex() = default;

    /**
     * @brief Build an index over a selected subset of labels.
     * @param labels External labels in [0, data_size) to include in the index.
     */
    virtual void build(const vector<unsigned> &labels) = 0;
    virtual void save(const std::string &file_path) = 0;
    virtual void load(const std::string &file_path) = 0;

    const DataWrapper *data_wrapper = nullptr;
    // returned nns
    vector<unsigned> returned_nns;
    // range search

    // search parameters
    unsigned query_topK = 10;
    unsigned search_ef = 100;

    /**
     * Set the _search_ef_ parameter for controlling search breadth.
     */
    void setSearchEf(unsigned ef) {
        this->search_ef = ef;
    }

    void setQueryTopK(unsigned topK) {
        this->query_topK = topK;
        returned_nns.resize(topK);
    }

    virtual void rangeSearch(const float *query, const std::pair<int, int> query_bound) = 0;
};

