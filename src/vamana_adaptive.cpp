#include "vamana_index.h"
#include "distance.h"
#include "io_utils.h"
#include "timer.h"

#include <algorithm>
#include <fstream>
#include <iostream>
#include <numeric>
#include <random>
#include <set>
#include <stdexcept>
#include <cstdlib>

// ============================================================================
// Destructor
// ============================================================================

VamanaIndex::~VamanaIndex() {
    if (owns_data_ && data_) {
        std::free(data_);
        data_ = nullptr;
    }
}

// ============================================================================
// Greedy Search (FIXED: vector<bool> → vector<char>)
// ============================================================================

std::pair<std::vector<VamanaIndex::Candidate>, uint32_t>
VamanaIndex::greedy_search(const float* query, uint32_t L) const {
    std::set<Candidate> candidate_set;
    std::vector<char> visited(npts_, 0);           // ← CRITICAL FIX
    uint32_t dist_cmps = 0;

    float start_dist = compute_l2sq(query, get_vector(start_node_), dim_);
    dist_cmps++;
    candidate_set.insert({start_dist, start_node_});
    visited[start_node_] = 1;

    std::set<uint32_t> expanded;
    while (true) {
        uint32_t best_node = UINT32_MAX;
        for (const auto& [dist, id] : candidate_set) {
            if (expanded.find(id) == expanded.end()) {
                best_node = id;
                break;
            }
        }
        if (best_node == UINT32_MAX) break;

        expanded.insert(best_node);

        std::vector<uint32_t> neighbors;
        {
            std::lock_guard<std::mutex> lock(locks_[best_node]);
            neighbors = graph_[best_node];
        }

        for (uint32_t nbr : neighbors) {
            if (visited[nbr]) continue;
            visited[nbr] = 1;

            float d = compute_l2sq(query, get_vector(nbr), dim_);
            dist_cmps++;

            if (candidate_set.size() < L) {
                candidate_set.insert({d, nbr});
            } else {
                auto worst = std::prev(candidate_set.end());
                if (d < worst->first) {
                    candidate_set.erase(worst);
                    candidate_set.insert({d, nbr});
                }
            }
        }
    }

    std::vector<Candidate> results(candidate_set.begin(), candidate_set.end());
    return {results, dist_cmps};
}

// ============================================================================
// Robust Prune — Adaptive Alpha
// ============================================================================

void VamanaIndex::robust_prune(uint32_t node, std::vector<Candidate>& candidates,
                               float alpha_base, uint32_t R) {
    candidates.erase(
        std::remove_if(candidates.begin(), candidates.end(),
                       [node](const Candidate& c){ return c.second == node; }),
        candidates.end());

    std::sort(candidates.begin(), candidates.end());

    float beta = alpha_base - 1.0f;
    float local_alpha = 1.0f + beta * (densities_[node] / (max_rho_ + 1e-9f));

    std::vector<uint32_t> new_nbrs;
    new_nbrs.reserve(R);

    for (const auto& [dn, cid] : candidates) {
        if (new_nbrs.size() >= R) break;

        bool keep = true;
        for (uint32_t sel : new_nbrs) {
            float dc = compute_l2sq(get_vector(cid), get_vector(sel), dim_);
            if (dn > local_alpha * dc) {
                keep = false;
                break;
            }
        }
        if (keep) new_nbrs.push_back(cid);
    }

    graph_[node] = std::move(new_nbrs);
}

// ============================================================================
// compute_rho
// ============================================================================

float VamanaIndex::compute_rho(uint32_t point_id, uint32_t k) {
    auto [candidates, _] = greedy_search(get_vector(point_id), k);
    if (candidates.empty()) return 1.0f;

    float sum_dist = 0.0f;
    for (const auto& c : candidates) sum_dist += std::sqrt(c.first);
    return sum_dist / candidates.size();
}

// ============================================================================
// Build — Local-Density Adaptive Pruning
// ============================================================================

void VamanaIndex::build(const std::string& data_path, uint32_t R, uint32_t L,
                        float alpha, float gamma) {
    std::cout << "[Adaptive Density Pruning] Loading data..." << std::endl;
    FloatMatrix mat = load_fbin(data_path);
    npts_ = mat.npts;
    dim_  = mat.dims;
    data_ = mat.data.release();
    owns_data_ = true;

    std::cout << "  Points: " << npts_ << ", Dims: " << dim_ << std::endl;
    if (L < R) L = R;

    graph_.assign(npts_, std::vector<uint32_t>());
    locks_ = std::vector<std::mutex>(npts_);
    densities_.assign(npts_, 0.0f);
    max_rho_ = 0.0f;

    std::vector<uint32_t> perm(npts_);
    std::iota(perm.begin(), perm.end(), 0);
    std::mt19937 rng(42);
    std::shuffle(perm.begin(), perm.end(), rng);

    start_node_ = perm[0];

    // Step 1: Estimate local densities
    std::cout << "Estimating local densities..." << std::endl;
#pragma omp parallel for reduction(max:max_rho_)
    for (uint32_t i = 0; i < npts_; i++) {
        float rho = compute_rho(i, 5);
        densities_[i] = rho;
        if (rho > max_rho_) max_rho_ = rho;
    }
    if (max_rho_ == 0.0f) max_rho_ = 1.0f;

    // Step 2: Main build
    uint32_t gamma_R = static_cast<uint32_t>(gamma * R);
    Timer build_timer;

#pragma omp parallel for schedule(dynamic, 64)
    for (size_t idx = 0; idx < npts_; idx++) {
        uint32_t point = perm[idx];

        auto [candidates, _] = greedy_search(get_vector(point), L);
        robust_prune(point, candidates, alpha, R);

        for (uint32_t nbr : graph_[point]) {
            std::lock_guard<std::mutex> lock(locks_[nbr]);
            graph_[nbr].push_back(point);

            if (graph_[nbr].size() > gamma_R) {
                std::vector<Candidate> nbr_cands;
                for (uint32_t nn : graph_[nbr]) {
                    nbr_cands.push_back({compute_l2sq(get_vector(nbr), get_vector(nn), dim_), nn});
                }
                robust_prune(nbr, nbr_cands, alpha, R);
            }
        }

        if (idx % 50000 == 0) {
#pragma omp critical
            std::cout << "\r  Inserted " << idx << " / " << npts_ << " points" << std::flush;
        }
    }

    std::cout << "\nBuild complete in " << build_timer.elapsed_seconds() << " seconds." << std::endl;

    size_t total = 0;
    for (uint32_t i = 0; i < npts_; i++) total += graph_[i].size();
    std::cout << "Average degree: " << (double)total / npts_ << std::endl;
}

// ============================================================================
// Search, Save, Load (unchanged)
// ============================================================================

SearchResult VamanaIndex::search(const float* query, uint32_t K, uint32_t L) const {
    if (L < K) L = K;
    Timer t;
    auto [candidates, dist_cmps] = greedy_search(query, L);
    double latency = t.elapsed_us();

    SearchResult result;
    result.dist_cmps = dist_cmps;
    result.latency_us = latency;
    result.ids.reserve(K);
    for (uint32_t i = 0; i < K && i < candidates.size(); i++) {
        result.ids.push_back(candidates[i].second);
    }
    return result;
}

void VamanaIndex::save(const std::string& path) const {
    std::ofstream out(path, std::ios::binary);
    if (!out.is_open()) throw std::runtime_error("Cannot open: " + path);
    out.write(reinterpret_cast<const char*>(&npts_), 4);
    out.write(reinterpret_cast<const char*>(&dim_), 4);
    out.write(reinterpret_cast<const char*>(&start_node_), 4);
    for (uint32_t i = 0; i < npts_; i++) {
        uint32_t deg = graph_[i].size();
        out.write(reinterpret_cast<const char*>(&deg), 4);
        if (deg > 0)
            out.write(reinterpret_cast<const char*>(graph_[i].data()),
                      deg * sizeof(uint32_t));
    }
    std::cout << "Index saved to " << path << std::endl;
}

void VamanaIndex::load(const std::string& index_path,
                       const std::string& data_path) {
    FloatMatrix mat = load_fbin(data_path);
    npts_ = mat.npts; dim_ = mat.dims;
    data_ = mat.data.release(); owns_data_ = true;

    std::ifstream in(index_path, std::ios::binary);
    if (!in.is_open()) throw std::runtime_error("Cannot open: " + index_path);
    uint32_t fn, fd;
    in.read(reinterpret_cast<char*>(&fn), 4);
    in.read(reinterpret_cast<char*>(&fd), 4);
    in.read(reinterpret_cast<char*>(&start_node_), 4);
    if (fn != npts_ || fd != dim_) throw std::runtime_error("Mismatch");

    graph_.resize(npts_);
    locks_ = std::vector<std::mutex>(npts_);
    for (uint32_t i = 0; i < npts_; i++) {
        uint32_t deg;
        in.read(reinterpret_cast<char*>(&deg), 4);
        graph_[i].resize(deg);
        if (deg > 0)
            in.read(reinterpret_cast<char*>(graph_[i].data()),
                    deg * sizeof(uint32_t));
    }
    std::cout << "Index loaded: " << npts_ << " pts" << std::endl;
}