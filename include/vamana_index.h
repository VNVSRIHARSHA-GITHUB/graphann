#pragma once

#include <cstdint>
#include <vector>
#include <mutex>
#include <string>

// Result of a single query search.
struct SearchResult {
    std::vector<uint32_t> ids;      // nearest neighbor IDs (sorted by distance)
    uint32_t dist_cmps;             // number of distance computations
    double latency_us;              // search latency in microseconds
};

// Vamana graph-based approximate nearest neighbor index.
class VamanaIndex {
public:
    VamanaIndex() = default;
    ~VamanaIndex();

    // ---- Build ----
    void build(const std::string& data_path, uint32_t R, uint32_t L,
               float alpha, float gamma);

    // ---- Search ----
    SearchResult search(const float* query, uint32_t K, uint32_t L) const;

    // ---- Persistence ----
    void save(const std::string& path) const;
    void load(const std::string& index_path, const std::string& data_path);

    uint32_t get_npts() const { return npts_; }
    uint32_t get_dim()  const { return dim_; }

private:
    // A candidate = (distance, node_id). Ordered by distance.
    using Candidate = std::pair<float, uint32_t>;

    // ---- Core algorithms ----
    std::pair<std::vector<Candidate>, uint32_t>
    greedy_search(const float* query, uint32_t L) const;

    void robust_prune(uint32_t node, std::vector<Candidate>& candidates,
                      float alpha, uint32_t R);

    std::vector<Candidate> search_from(uint32_t seed,
                                       const float* query,
                                       uint32_t L) const;

    void run_pass(const std::vector<uint32_t>& perm,
                  float alpha, uint32_t R, uint32_t L,
                  uint32_t gamma_R);

    // ---- Improvement 6: Local-Density Adaptive Pruning ----
    float compute_rho(uint32_t point_id, uint32_t k);
    std::vector<float> densities_;   // ρ(p) for each point
    float max_rho_ = 0.0f;           // global max density

    // ---- Data ----
    float*   data_      = nullptr;
    uint32_t npts_      = 0;
    uint32_t dim_       = 0;
    bool     owns_data_ = false;

    // ---- Graph ----
    std::vector<std::vector<uint32_t>> graph_;
    uint32_t start_node_ = 0;

    // ---- Concurrency ----
    mutable std::vector<std::mutex> locks_;

    // ---- Helpers ----
    const float* get_vector(uint32_t id) const {
        return data_ + (size_t)id * dim_;
    }
};