#include <algorithm>
#include <boost/functional/hash.hpp>
#include <ctime>
#include <iostream>
#include <limits>
#include <numeric>
#include <queue>
#include <vector>
#include <fstream>

#include "baselines/hnswalg.h"
#include "baselines/hnswlib.h"
#include "infrastructure/io/data_loader.h"
#include "infrastructure/io/index_serializer.h"
#include "infrastructure/utils/utils.h"
#include "infrastructure/utils/distance.h"
#include "core/algorithms/segment_graph_2d.h"

using namespace base_hnsw;
using std::pair;

namespace SeRF {

// ========== OneSegmentNeighbors 实现 ==========
OneSegmentNeighbors::OneSegmentNeighbors() : batch(0) {}

OneSegmentNeighbors::OneSegmentNeighbors(unsigned num) : batch(num) {}

OneSegmentNeighbors::OneSegmentNeighbors(unsigned num, int start, int end) 
    : batch(num), start(start), end(end) {}

const unsigned OneSegmentNeighbors::size() {
    return nns_id.size();
}

// ========== SegmentGraph2DHNSW 实现 ==========

template <typename dist_t>
SegmentGraph2DHNSW<dist_t>::SegmentGraph2DHNSW(
    const BaseIndex::IndexParams &index_params,
    base_hnsw::SpaceInterface<float> *s,
    size_t max_elements,
    size_t M,
    size_t ef_construction,
    size_t random_seed) 
    : base_hnsw::HierarchicalNSW<float>(s, max_elements, M, index_params.ef_construction, random_seed) {
    params = &index_params;
    ef_max_ = index_params.ef_max;
}

template <typename dist_t>
std::priority_queue<std::pair<dist_t, base_hnsw::tableint>,
                    std::vector<std::pair<dist_t, base_hnsw::tableint>>,
                    base_hnsw::HierarchicalNSW<float>::CompareByFirst>
SegmentGraph2DHNSW<dist_t>::searchBaseLayerLevel0(base_hnsw::tableint ep_id, const void *data_point, int layer) {
    hnswlib_incre::VisitedList *vl = visited_list_pool_->getFreeVisitedList();
    hnswlib_incre::vl_type *visited_array = vl->mass;
    hnswlib_incre::vl_type visited_array_tag = vl->curV;

    std::priority_queue<std::pair<dist_t, base_hnsw::tableint>,
                        std::vector<std::pair<dist_t, base_hnsw::tableint>>,
                        base_hnsw::HierarchicalNSW<float>::CompareByFirst>
        top_candidates;
    std::priority_queue<std::pair<dist_t, base_hnsw::tableint>,
                        std::vector<std::pair<dist_t, base_hnsw::tableint>>,
                        base_hnsw::HierarchicalNSW<float>::CompareByFirst>
        candidateSet;

    std::vector<pair<dist_t, base_hnsw::tableint>> deleted_list;
    size_t ef_construction = ef_max_;

    dist_t lowerBound;
    if (!isMarkedDeleted(ep_id)) {
        dist_t dist = fstdistfunc_(data_point, getDataByInternalId(ep_id), dist_func_param_);
        top_candidates.emplace(dist, ep_id);
        lowerBound = dist;
        candidateSet.emplace(-dist, ep_id);
    } else {
        lowerBound = std::numeric_limits<dist_t>::max();
        candidateSet.emplace(-lowerBound, ep_id);
    }
    visited_array[ep_id] = visited_array_tag;

    while (!candidateSet.empty()) {
        std::pair<dist_t, base_hnsw::tableint> curr_el_pair = candidateSet.top();
        if ((-curr_el_pair.first) > lowerBound) {
            break;
        }
        candidateSet.pop();

        base_hnsw::tableint curNodeNum = curr_el_pair.second;
        std::unique_lock<std::mutex> lock(link_list_locks_[curNodeNum]);

        int *data;
        if (layer == 0) {
            data = (int *)get_linklist0(curNodeNum);
        } else {
            data = (int *)get_linklist(curNodeNum, layer);
        }
        size_t size = getListCount((base_hnsw::linklistsizeint *)data);
        base_hnsw::tableint *datal = (base_hnsw::tableint *)(data + 1);

#ifdef USE_SSE
        _mm_prefetch((char *)(visited_array + *(data + 1)), _MM_HINT_T0);
        _mm_prefetch((char *)(visited_array + *(data + 1) + 64), _MM_HINT_T0);
        _mm_prefetch(getDataByInternalId(*datal), _MM_HINT_T0);
        _mm_prefetch(getDataByInternalId(*(datal + 1)), _MM_HINT_T0);
#endif

        for (size_t j = 0; j < size; j++) {
            base_hnsw::tableint candidate_id = *(datal + j);
#ifdef USE_SSE
            _mm_prefetch((char *)(visited_array + *(datal + j + 1)), _MM_HINT_T0);
            _mm_prefetch(getDataByInternalId(*(datal + j + 1)), _MM_HINT_T0);
#endif
            if (visited_array[candidate_id] == visited_array_tag)
                continue;
            visited_array[candidate_id] = visited_array_tag;

            char *currObj1 = (getDataByInternalId(candidate_id));
            dist_t dist1 = fstdistfunc_(data_point, currObj1, dist_func_param_);

            if (top_candidates.size() < ef_construction || lowerBound > dist1) {
                candidateSet.emplace(-dist1, candidate_id);
#ifdef USE_SSE
                _mm_prefetch(getDataByInternalId(candidateSet.top().second), _MM_HINT_T0);
#endif

                if (!isMarkedDeleted(candidate_id))
                    top_candidates.emplace(dist1, candidate_id);

                if (top_candidates.size() > ef_construction) {
                    deleted_list.emplace_back(top_candidates.top());
                    top_candidates.pop();
                }

                if (!top_candidates.empty())
                    lowerBound = top_candidates.top().first;
            }
        }
    }

    visited_list_pool_->releaseVisitedList(vl);

    for (auto deleted_candidate : deleted_list) {
        top_candidates.emplace(deleted_candidate);
    }

    return top_candidates;
}

template <typename dist_t>
base_hnsw::tableint SegmentGraph2DHNSW<dist_t>::mutuallyConnectNewElementLevel0(
    const void *data_point,
    base_hnsw::tableint cur_c,
    std::priority_queue<std::pair<dist_t, base_hnsw::tableint>,
                        std::vector<std::pair<dist_t, base_hnsw::tableint>>,
                        base_hnsw::HierarchicalNSW<float>::CompareByFirst> &top_candidates,
    int level,
    bool isUpdate) {
    
    size_t Mcurmax = maxM0_;
    int external_id = getExternalLabel(cur_c);
    base_hnsw::tableint next_closest_entry_point = 0;

    std::vector<base_hnsw::tableint> selectedNeighbors;
    selectedNeighbors.reserve(M_);

    {
        std::priority_queue<std::pair<dist_t, base_hnsw::tableint>> queue_closest;
        while (!top_candidates.empty()) {
            queue_closest.emplace(-top_candidates.top().first, top_candidates.top().second);
            top_candidates.pop();
        }

        int external_left_most = -1;
        int last_batch_left_most = -1;
        unsigned iter_counter = 0;
        unsigned batch_counter = 0;
        std::vector<std::pair<dist_t, base_hnsw::tableint>> return_list;
        std::vector<int> return_external_list;

        vector<pair<int, pair<dist_t, base_hnsw::tableint>>> buffer_candidates;

        while (!queue_closest.empty()) {
            if (return_list.size() >= Mcurmax || iter_counter >= ef_basic_construction_) {
                if (batch_counter == 0) {
                    next_closest_entry_point = return_list.front().second;
                    for (std::pair<dist_t, base_hnsw::tableint> curent_pair : return_list) {
                        selectedNeighbors.push_back(curent_pair.second);
                    }
                }

                for (pair<int, pair<dist_t, base_hnsw::tableint>> curent_buffer : buffer_candidates) {
                    if (curent_buffer.first > external_left_most) {
                        queue_closest.emplace(curent_buffer.second);
                    }
                }

                OneSegmentNeighbors one_segment(batch_counter, last_batch_left_most + 1, external_left_most);
                one_segment.nns_id.swap(return_external_list);
                segment_graph->at(external_id).forward_nns.emplace_back(one_segment);

                return_list.clear();
                return_external_list.clear();
                iter_counter = 0;
                batch_counter++;
                last_batch_left_most = external_left_most;
            }

            std::pair<dist_t, base_hnsw::tableint> curent_pair = queue_closest.top();
            dist_t dist_to_query = -curent_pair.first;
            queue_closest.pop();

            int curent_external_id = getExternalLabel(curent_pair.second);
            if (curent_external_id < last_batch_left_most) {
                continue;
            }
            iter_counter++;
            bool good = true;

            for (std::pair<dist_t, base_hnsw::tableint> second_pair : return_list) {
                dist_t curdist = fstdistfunc_(getDataByInternalId(second_pair.second),
                                              getDataByInternalId(curent_pair.second),
                                              dist_func_param_);

                if (curdist < dist_to_query) {
                    good = false;
                    break;
                }
            }
            if (good) {
                return_list.push_back(curent_pair);
                return_external_list.push_back(curent_external_id);
                if (curent_external_id > external_left_most) {
                    external_left_most = curent_external_id;
                }
            } else {
                if (curent_external_id > external_left_most) {
                    buffer_candidates.emplace_back(make_pair(curent_external_id, curent_pair));
                }
            }
        }

        if (batch_counter == 0) {
            next_closest_entry_point = return_list.front().second;
            for (std::pair<dist_t, base_hnsw::tableint> curent_pair : return_list) {
                selectedNeighbors.push_back(curent_pair.second);
            }
        }

        if (!return_list.empty()) {
            OneSegmentNeighbors one_segment(batch_counter, last_batch_left_most + 1, external_id);
            one_segment.nns_id.swap(return_external_list);
            segment_graph->at(external_id).forward_nns.emplace_back(one_segment);
        }
    }

    {
        base_hnsw::linklistsizeint *ll_cur;
        ll_cur = get_linklist0(cur_c);

        if (*ll_cur && !isUpdate) {
            throw std::runtime_error("The newly inserted element should have blank link list");
        }
        setListCount(ll_cur, selectedNeighbors.size());
        base_hnsw::tableint *data = (base_hnsw::tableint *)(ll_cur + 1);
        for (size_t idx = 0; idx < selectedNeighbors.size(); idx++) {
            if (data[idx] && !isUpdate)
                throw std::runtime_error("Possible memory corruption");
            if (level > element_levels_[selectedNeighbors[idx]])
                throw std::runtime_error("Trying to make a link on a non-existent level");

            data[idx] = selectedNeighbors[idx];
        }
    }

    for (size_t idx = 0; idx < selectedNeighbors.size(); idx++) {
        std::unique_lock<std::mutex> lock(link_list_locks_[selectedNeighbors[idx]]);

        base_hnsw::linklistsizeint *ll_other;
        ll_other = get_linklist0(selectedNeighbors[idx]);

        size_t sz_link_list_other = getListCount(ll_other);

        if (sz_link_list_other > Mcurmax)
            throw std::runtime_error("Bad value of sz_link_list_other");
        if (selectedNeighbors[idx] == cur_c)
            throw std::runtime_error("Trying to connect an element to itself");
        if (level > element_levels_[selectedNeighbors[idx]])
            throw std::runtime_error("Trying to make a link on a non-existent level");

        base_hnsw::tableint *data = (base_hnsw::tableint *)(ll_other + 1);

        bool is_cur_c_present = false;
        if (isUpdate) {
            for (size_t j = 0; j < sz_link_list_other; j++) {
                if (data[j] == cur_c) {
                    is_cur_c_present = true;
                    break;
                }
            }
        }

        if (!is_cur_c_present) {
            if (sz_link_list_other < Mcurmax) {
                data[sz_link_list_other] = cur_c;
                setListCount(ll_other, sz_link_list_other + 1);
            } else {
                dist_t d_max = fstdistfunc_(getDataByInternalId(cur_c),
                                            getDataByInternalId(selectedNeighbors[idx]), 
                                            dist_func_param_);
                std::priority_queue<std::pair<dist_t, base_hnsw::tableint>,
                                    std::vector<std::pair<dist_t, base_hnsw::tableint>>,
                                    base_hnsw::HierarchicalNSW<float>::CompareByFirst>
                    candidates;
                candidates.emplace(d_max, cur_c);

                for (size_t j = 0; j < sz_link_list_other; j++) {
                    candidates.emplace(fstdistfunc_(getDataByInternalId(data[j]),
                                                    getDataByInternalId(selectedNeighbors[idx]),
                                                    dist_func_param_),
                                       data[j]);
                }

                getNeighborsByHeuristic2(candidates, Mcurmax);

                int indx = 0;
                while (candidates.size() > 0) {
                    data[indx] = candidates.top().second;
                    candidates.pop();
                    indx++;
                }
                setListCount(ll_other, indx);
            }
        }
    }

    return next_closest_entry_point;
}

// 显式模板实例化
template class SegmentGraph2DHNSW<float>;

// ========== IndexSegmentGraph2D 实现 ==========

IndexSegmentGraph2D::IndexSegmentGraph2D(base_hnsw::SpaceInterface<float> *s,
                                         const DataWrapper *data)
    : BaseIndex(data) {
    fstdistfunc_ = s->get_dist_func();
    dist_func_param_ = s->get_dist_func_param();
    index_info = new IndexInfo();
    index_info->index_version_type = "IndexSegmentGraph2D";
}

void IndexSegmentGraph2D::processReverseNeighbors() {
    for (size_t i = 0; i < data_wrapper->data_size; ++i) {
        for (auto batch : this->directed_indexed_arr.at(i).forward_nns) {
            for (auto nn : batch.nns_id) {
                this->directed_indexed_arr.at(nn).reverse_nns.insert(
                    this->directed_indexed_arr.at(nn).reverse_nns.begin(), i);
            }
        }
    }
}

void IndexSegmentGraph2D::processReverseNeighbors(vector<unsigned> &nodes_ids) {
    for (auto &i : nodes_ids) {
        for (auto batch : this->directed_indexed_arr.at(i).forward_nns) {
            for (auto nn : batch.nns_id) {
                this->directed_indexed_arr.at(nn).reverse_nns.insert(
                    this->directed_indexed_arr.at(nn).reverse_nns.begin(), i);
            }
        }
    }
}

void IndexSegmentGraph2D::printOnebatch() {
    cout << "Print one batch" << endl;
    for (auto nns : directed_indexed_arr[data_wrapper->data_size / 2].forward_nns) {
        cout << "Forward batch: " << nns.batch << "(" << nns.start << "," << nns.end << ")" << endl;
        print_set(nns.nns_id);
        cout << endl;
    }
    cout << endl;

    cout << "Reverse batch: " << endl;
    print_set(directed_indexed_arr[data_wrapper->data_size / 2].reverse_nns);
    cout << endl << endl;
}

void IndexSegmentGraph2D::countNeighbrs() {
    double batch_counter = 0;
    double max_batch_counter = 0;
    size_t max_reverse_nn = 0;

    if (!directed_indexed_arr.empty()) {
        for (unsigned j = 0; j < directed_indexed_arr.size(); j++) {
            int temp_size = 0;
            for (const auto &nns : directed_indexed_arr[j].forward_nns) {
                temp_size += nns.nns_id.size();
            }
            batch_counter += directed_indexed_arr[j].forward_nns.size();
            index_info->nodes_amount += temp_size;
        }
    }

    index_info->avg_forward_nns = index_info->nodes_amount / static_cast<float>(data_wrapper->data_size);

    if (isLog) {
        cout << "Max. forward batch nn #: " << max_batch_counter << endl;
        cout << "Avg. forward nn #: " << index_info->nodes_amount / static_cast<float>(data_wrapper->data_size) << endl;
        cout << "Avg. forward batch #: " << batch_counter / static_cast<float>(data_wrapper->data_size) << endl;
        batch_counter = 0;

        int reverse_node_amount = 0;
        for (unsigned j = 0; j < directed_indexed_arr.size(); j++) {
            reverse_node_amount += directed_indexed_arr[j].reverse_nns.size();
            batch_counter += 1;
            max_reverse_nn = std::max(max_reverse_nn, directed_indexed_arr[j].reverse_nns.size());
        }

        index_info->nodes_amount += reverse_node_amount;
        index_info->avg_reverse_nns = reverse_node_amount / static_cast<float>(data_wrapper->data_size);

        cout << "Max. reverse nn #: " << max_reverse_nn << endl;
        cout << "Avg. reverse nn #: " << reverse_node_amount / static_cast<float>(data_wrapper->data_size) << endl;
        cout << "Avg. reverse batch #: " << batch_counter / static_cast<float>(data_wrapper->data_size) << endl;
        cout << "Avg. delta nn #: " << index_info->nodes_amount / static_cast<float>(data_wrapper->data_size) << endl;
    }
}

void IndexSegmentGraph2D::buildIndex(const IndexParams *index_params) {
    cout << "Building Index using " << index_info->index_version_type << endl;
    timeval tt1, tt2;
    visited_list_pool_ = new hnswlib_incre::VisitedListPool(1, data_wrapper->data_size);

    index_params_ = index_params;
    index_k = index_params->K;

    base_hnsw::L2Space space(data_wrapper->data_dim);
    hnsw = new SegmentGraph2DHNSW<float>(*index_params, &space, 2 * data_wrapper->data_size,
                                         index_params->K, index_params->ef_construction,
                                         index_params->random_seed);

    directed_indexed_arr.clear();
    directed_indexed_arr.resize(data_wrapper->data_size);
    hnsw->segment_graph = &directed_indexed_arr;
    gettimeofday(&tt1, NULL);

    for (size_t i = 0; i < data_wrapper->data_size; ++i) {
        hnsw->addPoint(data_wrapper->nodes.at(i).data(), i);
    }
    gettimeofday(&tt2, NULL);
    index_info->index_time = CountTime(tt1, tt2);

    processReverseNeighbors();
    countNeighbrs();

    if (index_params->print_one_batch) {
        printOnebatch();
    }
}

void IndexSegmentGraph2D::initForScabilityExp(const IndexParams *index_params, base_hnsw::L2Space *space) {
    if (visited_list_pool_ == nullptr)
        visited_list_pool_ = new hnswlib_incre::VisitedListPool(1, data_wrapper->data_size);
    index_params_ = index_params;
    
    hnsw = new SegmentGraph2DHNSW<float>(*index_params, space, 2 * data_wrapper->data_size,
                                         index_params->K, index_params->ef_construction,
                                         index_params->random_seed);

    directed_indexed_arr.resize(data_wrapper->data_size);
    hnsw->segment_graph = &directed_indexed_arr;
}

void IndexSegmentGraph2D::insert_batch(vector<unsigned> &nodes_ids) {
    timeval tt1, tt2;
    gettimeofday(&tt1, NULL);
    for (auto i : nodes_ids) {
        hnsw->addPoint(data_wrapper->nodes.at(i).data(), i);
    }

    gettimeofday(&tt2, NULL);
    index_info->index_time = CountTime(tt1, tt2);
    cout << "Insert a " << nodes_ids.size() << " batch need " << index_info->index_time << endl;

    processReverseNeighbors(nodes_ids);
    countNeighbrs();
}

SearchResult IndexSegmentGraph2D::searchKnn(
    const SearchParams *search_params,
    SearchInfo *search_info,
    const vector<float> &query) {
    // 这是一个占位实现，实际可能需要实现具体的KNN搜索逻辑
    SearchResult result;
    return result;
}

vector<OneSegmentNeighbors>::const_iterator IndexSegmentGraph2D::decompressForwardPath(
    const vector<OneSegmentNeighbors> &forward_nns,
    const int lbound) {
    auto forward_batch_it = forward_nns.begin();
    while (forward_batch_it != forward_nns.end()) {
        if (lbound < forward_batch_it->end) {
            break;
        }
        forward_batch_it++;
    }
    return forward_batch_it;
}

vector<OneSegmentNeighbors>::const_iterator IndexSegmentGraph2D::decompressReversePath(
    const vector<OneSegmentNeighbors> &reverse_nns,
    const int rbound) {
    auto reverse_batch_it = reverse_nns.begin();
    while (reverse_batch_it != reverse_nns.end()) {
        if (rbound > reverse_batch_it->start) {
            break;
        }
        reverse_batch_it++;
    }
    return reverse_batch_it;
}

vector<int> IndexSegmentGraph2D::rangeFilteringSearchInRange(
    const SearchParams *search_params,
    SearchInfo *search_info,
    const vector<float> &query,
    const std::pair<int, int> query_bound) {
    
    timeval tt1, tt2, tt3, tt4;

    hnswlib_incre::VisitedList *vl = visited_list_pool_->getFreeVisitedList();
    hnswlib_incre::vl_type *visited_array = vl->mass;
    hnswlib_incre::vl_type visited_array_tag = vl->curV;
    float lower_bound = std::numeric_limits<float>::max();
    std::priority_queue<pair<float, int>> top_candidates;
    std::priority_queue<pair<float, int>> candidate_set;

    const int data_size = data_wrapper->data_size;
    const int two_batch_threshold = data_size * search_params->control_batch_threshold;
    search_info->path_counter = 0;
    search_info->total_comparison = 0;
    search_info->internal_search_time = 0;
    search_info->cal_dist_time = 0;
    search_info->fetch_nns_time = 0;
    num_search_comparison = 0;

    vector<int> enter_list;
    {
        int lbound = query_bound.first;
        int interval = (query_bound.second - lbound) / 3;
        for (size_t i = 0; i < 3; i++) {
            int point = lbound + interval * i;
            float dist = EuclideanDistance(data_wrapper->nodes[point], query);
            candidate_set.push(make_pair(-dist, point));
            enter_list.emplace_back(point);
            visited_array[point] = visited_array_tag;
        }
    }
    gettimeofday(&tt3, NULL);

    size_t hop_counter = 0;
    while (!candidate_set.empty()) {
        std::pair<float, int> current_node_pair = candidate_set.top();
        int current_node_id = current_node_pair.second;

        if (-current_node_pair.first > lower_bound) {
            break;
        }

        hop_counter++;
        candidate_set.pop();

        vector<int> current_neighbors;
        vector<const vector<int> *> neighbor_iterators;

        gettimeofday(&tt1, NULL);
        {
            auto forward_it = decompressForwardPath(
                directed_indexed_arr[current_node_id].forward_nns,
                query_bound.first);
            if (forward_it != directed_indexed_arr[current_node_id].forward_nns.end()) {
                neighbor_iterators.emplace_back(&forward_it->nns_id);
                if (current_node_id - query_bound.first < two_batch_threshold) {
                    forward_it++;
                    if (forward_it != directed_indexed_arr[current_node_id].forward_nns.end()) {
                        neighbor_iterators.emplace_back(&forward_it->nns_id);
                    }
                }
            }

            neighbor_iterators.emplace_back(&directed_indexed_arr.at(current_node_id).reverse_nns);
        }
        gettimeofday(&tt2, NULL);
        AccumulateTime(tt1, tt2, search_info->fetch_nns_time);

        gettimeofday(&tt1, NULL);
        for (auto batch_it : neighbor_iterators) {
            for (auto candidate_id : *batch_it) {
                if (candidate_id < query_bound.first || candidate_id > query_bound.second)
                    continue;
                if (!(visited_array[candidate_id] == visited_array_tag)) {
                    visited_array[candidate_id] = visited_array_tag;

                    float dist = fstdistfunc_(query.data(),
                                              data_wrapper->nodes[candidate_id].data(),
                                              dist_func_param_);

                    num_search_comparison++;
                    if (top_candidates.size() < search_params->search_ef || lower_bound > dist) {
                        candidate_set.push(make_pair(-dist, candidate_id));
                        top_candidates.push(make_pair(dist, candidate_id));
                        if (top_candidates.size() > search_params->search_ef) {
                            top_candidates.pop();
                        }
                        if (!top_candidates.empty()) {
                            lower_bound = top_candidates.top().first;
                        }
                    }
                }
            }
        }
        gettimeofday(&tt2, NULL);
        AccumulateTime(tt1, tt2, search_info->cal_dist_time);
    }

    vector<int> res;
    while (top_candidates.size() > search_params->query_K) {
        top_candidates.pop();
    }

    while (!top_candidates.empty()) {
        res.emplace_back(top_candidates.top().second);
        top_candidates.pop();
    }
    search_info->total_comparison += num_search_comparison;

    visited_list_pool_->releaseVisitedList(vl);
    gettimeofday(&tt4, NULL);
    CountTime(tt3, tt4, search_info->internal_search_time);
    return res;
}

vector<int> IndexSegmentGraph2D::rangeFilteringSearchOutBound(
    const SearchParams *search_params,
    SearchInfo *search_info,
    const vector<float> &query,
    const std::pair<int, int> query_bound) {
    
    timeval tt1, tt2, tt3, tt4;

    hnswlib_incre::VisitedList *vl = visited_list_pool_->getFreeVisitedList();
    hnswlib_incre::vl_type *visited_array = vl->mass;
    hnswlib_incre::vl_type visited_array_tag = vl->curV;
    float lower_bound = std::numeric_limits<float>::max();
    std::priority_queue<pair<float, int>> top_candidates;
    std::priority_queue<pair<float, int>> candidate_set;

    search_info->path_counter = 0;
    search_info->total_comparison = 0;
    search_info->internal_search_time = 0;
    search_info->cal_dist_time = 0;
    search_info->fetch_nns_time = 0;
    num_search_comparison = 0;
    
    vector<int> enter_list;
    {
        int lbound = query_bound.first;
        int interval = (query_bound.second - lbound) / 3;
        for (size_t i = 0; i < 3; i++) {
            int point = lbound + interval * i;
            float dist = EuclideanDistance(data_wrapper->nodes[point], query);
            candidate_set.push(make_pair(-dist, point));
            enter_list.emplace_back(point);
            visited_array[point] = visited_array_tag;
        }
    }
    gettimeofday(&tt3, NULL);

    size_t hop_counter = 0;

    while (!candidate_set.empty()) {
        std::pair<float, int> current_node_pair = candidate_set.top();
        int current_node_id = current_node_pair.second;

        if (-current_node_pair.first > lower_bound) {
            break;
        }

        hop_counter++;
        candidate_set.pop();
        
        vector<int> current_neighbors;
        vector<const vector<int> *> neighbor_iterators;

        gettimeofday(&tt1, NULL);
        {
            auto forward_it = decompressForwardPath(
                directed_indexed_arr[current_node_id].forward_nns,
                query_bound.first);
            if (forward_it != directed_indexed_arr[current_node_id].forward_nns.end()) {
                neighbor_iterators.emplace_back(&forward_it->nns_id);
            }
            neighbor_iterators.emplace_back(&directed_indexed_arr.at(current_node_id).reverse_nns);
        }

        gettimeofday(&tt2, NULL);
        AccumulateTime(tt1, tt2, search_info->fetch_nns_time);
        gettimeofday(&tt1, NULL);

        for (auto batch_it : neighbor_iterators) {
            unsigned visited_nn_num = 0;
            for (auto candidate_id : *batch_it) {
                if (candidate_id < query_bound.first || candidate_id > query_bound.second)
                    continue;
                visited_nn_num++;

                if (visited_nn_num > 2 * index_k)
                    break;

                if (!(visited_array[candidate_id] == visited_array_tag)) {
                    visited_array[candidate_id] = visited_array_tag;

                    float dist = fstdistfunc_(query.data(),
                                              data_wrapper->nodes[candidate_id].data(),
                                              dist_func_param_);

                    num_search_comparison++;
                    if (top_candidates.size() < search_params->search_ef || lower_bound > dist) {
                        candidate_set.emplace(-dist, candidate_id);
                        top_candidates.emplace(dist, candidate_id);
                        if (top_candidates.size() > search_params->search_ef) {
                            top_candidates.pop();
                        }
                        if (!top_candidates.empty()) {
                            lower_bound = top_candidates.top().first;
                        }
                    }
                }
            }
        }
        gettimeofday(&tt2, NULL);
        AccumulateTime(tt1, tt2, search_info->cal_dist_time);
    }

    vector<int> res;
    while (top_candidates.size() > search_params->query_K) {
        top_candidates.pop();
    }

    while (!top_candidates.empty()) {
        res.emplace_back(top_candidates.top().second);
        top_candidates.pop();
    }
    search_info->path_counter += hop_counter;
    search_info->total_comparison += num_search_comparison;

    visited_list_pool_->releaseVisitedList(vl);

    gettimeofday(&tt4, NULL);
    CountTime(tt3, tt4, search_info->internal_search_time);

    return res;
}

void IndexSegmentGraph2D::save(const string &file_path) {
    std::ofstream output(file_path, std::ios::binary);
    unsigned counter = 0;
    base_hnsw::writeBinaryPOD(output, index_k);
    for (auto &segment : directed_indexed_arr) {
        base_hnsw::writeBinaryPOD(output, (int)segment.forward_nns.size());
        base_hnsw::writeBinaryPOD(output, (int)segment.reverse_nns.size());

        counter += 2;
        for (auto &nn : segment.forward_nns) {
            base_hnsw::writeBinaryPOD(output, nn.batch);
            base_hnsw::writeBinaryPOD(output, nn.start);
            base_hnsw::writeBinaryPOD(output, nn.end);
            base_hnsw::writeBinaryPOD(output, (int)nn.nns_id.size());
            for (auto &nn_id : nn.nns_id) {
                base_hnsw::writeBinaryPOD(output, nn_id);
                counter += 1;
            }
            counter += 4;
        }
        for (auto &nn_id : segment.reverse_nns) {
            base_hnsw::writeBinaryPOD(output, nn_id);
            counter += 1;
        }
    }
    cout << "Total write " << counter << " (int) to file " << file_path << endl;
}

void IndexSegmentGraph2D::load(const string &file_path) {
    std::ifstream input(file_path, std::ios::binary);
    if (!input.is_open()) throw std::runtime_error("Cannot open file");
    directed_indexed_arr.clear();
    directed_indexed_arr.resize(data_wrapper->data_size);
    base_hnsw::readBinaryPOD(input, index_k);
    cout << "Index K is " << index_k << endl;
    visited_list_pool_ = new hnswlib_incre::VisitedListPool(1, data_wrapper->data_size);
    
    int forward_num;
    int reverse_num;
    int batch_num;
    int start_pos;
    int end_pos;
    int nn_size;
    int one_nn;
    
    for (size_t i = 0; i < data_wrapper->data_size; i++) {
        base_hnsw::readBinaryPOD(input, forward_num);
        base_hnsw::readBinaryPOD(input, reverse_num);

        vector<OneSegmentNeighbors> neighbors;
        for (size_t j = 0; j < forward_num; j++) {
            base_hnsw::readBinaryPOD(input, batch_num);
            base_hnsw::readBinaryPOD(input, start_pos);
            base_hnsw::readBinaryPOD(input, end_pos);
            base_hnsw::readBinaryPOD(input, nn_size);
            vector<int> forward_nns;
            for (size_t k = 0; k < nn_size; k++) {
                base_hnsw::readBinaryPOD(input, one_nn);
                forward_nns.emplace_back(one_nn);
            }
            OneSegmentNeighbors one_forward_segment(batch_num, start_pos, end_pos);
            one_forward_segment.nns_id.swap(forward_nns);
            neighbors.emplace_back(one_forward_segment);
        }

        vector<int> reverse_nns;
        for (size_t j = 0; j < reverse_num; j++) {
            base_hnsw::readBinaryPOD(input, one_nn);
            reverse_nns.emplace_back(one_nn);
        }
        directed_indexed_arr[i].forward_nns.swap(neighbors);
        directed_indexed_arr[i].reverse_nns.swap(reverse_nns);
    }
    printOnebatch();
    countNeighbrs();
    cout << "Total # of neighbors: " << index_info->nodes_amount << endl;
}

IndexSegmentGraph2D::~IndexSegmentGraph2D() {
    delete hnsw;
    delete index_info;
    directed_indexed_arr.clear();
    delete visited_list_pool_;
}

} // namespace SeRF
