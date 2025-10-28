#include "infrastructure/utils/utils.h"
#include <cmath>
#include <iomanip>
#include <unordered_set>
#include <limits>

float EuclideanDistance(const vector<float> &lhs, const vector<float> &rhs,
                        const int &startDim, int lensDim) {
    float acc = 0.0f;
    const int endDim = startDim + lensDim;
    for (int i = startDim; i < endDim; ++i) {
        float d = lhs[i] - rhs[i];
        acc += d * d;
    }
    return std::sqrt(acc);
}

float EuclideanDistance(const vector<float> &lhs, const vector<float> &rhs) {
    float acc = 0.0f;
    const size_t n = lhs.size();
    for (size_t i = 0; i < n; ++i) {
        float d = lhs[i] - rhs[i];
        acc += d * d;
    }
    return std::sqrt(acc);
}

float EuclideanDistanceSquare(const vector<float> &lhs,
                              const vector<float> &rhs) {
    float acc = 0.0f;
    const size_t n = lhs.size();
    for (size_t i = 0; i < n; ++i) {
        float d = lhs[i] - rhs[i];
        acc += d * d;
    }
    return acc;
}

// 注意：头文件声明为 (t2, t1, val_time)，项目调用基本都是 (start, end)
// 为获得正时间增量，这里按 (end - start) 实际计算，即 val_time += (t1 - t2)
void AccumulateTime(timeval &t2, timeval &t1, double &val_time) {
    val_time += (t1.tv_sec - t2.tv_sec) +
                (t1.tv_usec - t2.tv_usec) / 1000000.0;
}

void CountTime(timeval &t1, timeval &t2, double &val_time) {
    val_time += (t2.tv_sec - t1.tv_sec) +
                (t2.tv_usec - t1.tv_usec) / 1000000.0;
}

double CountTime(timeval &t1, timeval &t2) {
    return (t2.tv_sec - t1.tv_sec) +
           (t2.tv_usec - t1.tv_usec) / 1000000.0;
}

void logTime(timeval &begin, timeval &end, const string &log) {
    double sec = CountTime(begin, end);
    cout << log << ": " << std::fixed << std::setprecision(6) << sec << "s" << endl;
}

double countPrecision(const vector<int> &truth, const vector<int> &pred) {
    if (truth.empty()) return 0.0;
    std::unordered_set<int> s(truth.begin(), truth.end());
    size_t hit = 0;
    for (int v : pred) if (s.count(v)) ++hit;
    return static_cast<double>(hit) / static_cast<double>(truth.size());
}

double countApproximationRatio(const vector<vector<float>> &raw_data,
                               const vector<int> &truth,
                               const vector<int> &pred,
                               const vector<float> &query) {
    auto avg_dist = [&](const vector<int> &ids) -> double {
        if (ids.empty()) return 0.0;
        double sum = 0.0;
        for (int id : ids) {
            sum += EuclideanDistance(raw_data[id], query);
        }
        return sum / static_cast<double>(ids.size());
    };
    double denom = avg_dist(truth);
    double numer = avg_dist(pred);
    if (denom == 0.0) return numer == 0.0 ? 1.0 : std::numeric_limits<double>::infinity();
    return numer / denom;
}

vector<int> greedyNearest(const vector<vector<float>> &dpts,
                          const vector<float> query,
                          const int k_smallest) {
    std::priority_queue<std::pair<float,int>> topk;
    const int N = static_cast<int>(dpts.size());
    const int K = std::max(0, k_smallest);
    for (int i = 0; i < N; ++i) {
        float dist = EuclideanDistance(dpts[i], query);
        if ((int)topk.size() < K) topk.emplace(dist, i);
        else if (dist < topk.top().first) { topk.pop(); topk.emplace(dist, i); }
    }
    vector<int> res;
    res.reserve(topk.size());
    while (!topk.empty()) { res.emplace_back(topk.top().second); topk.pop(); }
    std::reverse(res.begin(), res.end());
    return res;
}

vector<int> greedyNearest(const vector<vector<float>> &dpts,
                          const vector<float> query,
                          const int l_bound,
                          const int r_bound,
                          const int k_smallest) {
    std::priority_queue<std::pair<float,int>> topk;
    const int L = std::max(0, l_bound);
    const int R = std::min((int)dpts.size() - 1, r_bound);
    const int K = std::max(0, k_smallest);
    if (L <= R) {
        for (int i = L; i <= R; ++i) {
            float dist = EuclideanDistance(dpts[i], query);
            if ((int)topk.size() < K) topk.emplace(dist, i);
            else if (dist < topk.top().first) { topk.pop(); topk.emplace(dist, i); }
        }
    }
    vector<int> res;
    res.reserve(topk.size());
    while (!topk.empty()) { res.emplace_back(topk.top().second); topk.pop(); }
    std::reverse(res.begin(), res.end());
    return res;
}

// 非关键流程占位以满足链接（如未被调用仅为空实现）
void rangeGreedy(const vector<vector<float>> &nodes, const int k_smallest,
                 const int l_bound, const int r_bound) {
    (void)nodes; (void)k_smallest; (void)l_bound; (void)r_bound;
}

void greedyNearest(const int query_pos, const vector<vector<float>> &dpts,
                   const int k_smallest, const int l_bound, const int r_bound) {
    (void)query_pos; (void)dpts; (void)k_smallest; (void)l_bound; (void)r_bound;
}