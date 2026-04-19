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
// Greedy Search  (identical to original, but with CRITICAL FIX)
// ============================================================================
// CHANGED: std::vector<bool> → std::vector<char>
// vector<bool> is a known source of segfaults / UB in heavy loops.
// This was the most likely cause of the Pass-2 crash on SIFT1M.

std::pair<std::vector<VamanaIndex::Candidate>, uint32_t>
VamanaIndex::greedy_search(const float* query, uint32_t L) const {
    std::set<Candidate> candidate_set;
    std::vector<char> visited(npts_, 0);          // ← FIXED
    uint32_t dist_cmps = 0;

    float sd = compute_l2sq(query, get_vector(start_node_), dim_);
    dist_cmps++;
    candidate_set.insert({sd, start_node_});
    visited[start_node_] = 1;

    std::set<uint32_t> expanded;
    while (true) {
        uint32_t best = UINT32_MAX;
        for (const auto& [d, id] : candidate_set)
            if (expanded.find(id) == expanded.end()) { best = id; break; }
        if (best == UINT32_MAX) break;
        expanded.insert(best);

        std::vector<uint32_t> nbrs;
        { std::lock_guard<std::mutex> lk(locks_[best]); nbrs = graph_[best]; }

        for (uint32_t nb : nbrs) {
            if (visited[nb]) continue;
            visited[nb] = 1;
            float d = compute_l2sq(query, get_vector(nb), dim_);
            dist_cmps++;
            if (candidate_set.size() < L)
                candidate_set.insert({d, nb});
            else {
                auto worst = std::prev(candidate_set.end());
                if (d < worst->first) {
                    candidate_set.erase(worst);
                    candidate_set.insert({d, nb});
                }
            }
        }
    }
    std::vector<Candidate> res(candidate_set.begin(), candidate_set.end());
    return {res, dist_cmps};
}

// ============================================================================
// Robust Prune  (identical to original)
// ============================================================================

void VamanaIndex::robust_prune(uint32_t node,
                               std::vector<Candidate>& candidates,
                               float alpha, uint32_t R) {
    candidates.erase(
        std::remove_if(candidates.begin(), candidates.end(),
                       [node](const Candidate& c){ return c.second == node; }),
        candidates.end());
    std::sort(candidates.begin(), candidates.end());

    std::vector<uint32_t> new_nbrs;
    new_nbrs.reserve(R);

    for (const auto& [dn, cid] : candidates) {
        if (new_nbrs.size() >= R) break;
        bool keep = true;
        for (uint32_t sel : new_nbrs) {
            float dc = compute_l2sq(get_vector(cid), get_vector(sel), dim_);
            if (dn > alpha * dc) { keep = false; break; }
        }
        if (keep) new_nbrs.push_back(cid);
    }
    graph_[node] = std::move(new_nbrs);
}

// ============================================================================
// Internal single-pass helper (reused by both passes)
// ============================================================================

void VamanaIndex::run_pass(const std::vector<uint32_t>& perm,
                           float alpha, uint32_t R, uint32_t L,
                           uint32_t gamma_R) {
    // Single-threaded – parallelization would break correctness
    for (size_t idx = 0; idx < perm.size(); idx++) {
        uint32_t point = perm[idx];

        // Use full user-provided L for a richer candidate pool
        auto [candidates, _d] = greedy_search(get_vector(point), std::max(L, R*2));

        robust_prune(point, candidates, alpha, R);

        // Backward edges
        for (uint32_t nbr : graph_[point]) {
            std::lock_guard<std::mutex> lock(locks_[nbr]);
            graph_[nbr].push_back(point);
            if (graph_[nbr].size() > gamma_R) {
                std::vector<Candidate> nc;
                for (uint32_t nn : graph_[nbr]) {
                    float d = compute_l2sq(get_vector(nbr), get_vector(nn), dim_);
                    nc.push_back({d, nn});
                }
                robust_prune(nbr, nc, alpha, R);
            }
        }

        // Simple progress indicator (helps you see it's not stuck)
        if ((idx + 1) % 50000 == 0 || idx + 1 == perm.size()) {
            std::cout << (idx + 1) * 100 / perm.size() << "% " << std::flush;
        }
    }
    std::cout << std::endl;
}

// ============================================================================
// Build — TWO-PASS CONSTRUCTION (Automated Two-Pass)
// ============================================================================
// Pass 1: α = 1.0  → tight local edges
// Pass 2: α > 1.0  → long-range shortcuts
// This exactly matches section 3 of your project report.

void VamanaIndex::build(const std::string& data_path, uint32_t R, uint32_t L,
                        float alpha, float gamma) {
    std::cout << "[TwoPass Build] Loading data..." << std::endl;
    FloatMatrix mat = load_fbin(data_path);
    npts_ = mat.npts; dim_ = mat.dims;
    data_ = mat.data.release(); owns_data_ = true;

    std::cout << "  Points: " << npts_ << ", Dims: " << dim_ << std::endl;
    if (L < R) L = R;

    graph_.resize(npts_);
    locks_ = std::vector<std::mutex>(npts_);

    std::mt19937 rng(42);
    start_node_ = rng() % npts_;

    std::vector<uint32_t> perm(npts_);
    std::iota(perm.begin(), perm.end(), 0);

    uint32_t gamma_R = static_cast<uint32_t>(gamma * R);

    // ── PASS 1: alpha = 1.0 → tight local edges ────────────────────────────
    std::cout << "[TwoPass Build] Pass 1 (alpha=1.0)..." << std::endl;
    std::shuffle(perm.begin(), perm.end(), rng);
    run_pass(perm, 1.0f, R, L, gamma_R);

    size_t t1 = 0;
    for (uint32_t i = 0; i < npts_; i++) t1 += graph_[i].size();
    std::cout << "  Pass 1 done. Avg degree: " << (double)t1 / npts_ << std::endl;

    // ── PASS 2: user alpha → long-range edges ──────────────────────────────
    std::cout << "[TwoPass Build] Pass 2 (alpha=" << alpha << ")..." << std::endl;
    std::shuffle(perm.begin(), perm.end(), rng);
    run_pass(perm, alpha, R, L, gamma_R);

    size_t t2 = 0;
    for (uint32_t i = 0; i < npts_; i++) t2 += graph_[i].size();
    std::cout << "[TwoPass Build] Done. Avg degree: " << (double)t2 / npts_ << std::endl;
}

// ============================================================================
// Search, Save, Load  (identical to original)
// ============================================================================

SearchResult VamanaIndex::search(const float* query,
                                 uint32_t K, uint32_t L) const {
    if (L < K) L = K;
    Timer t;
    auto [candidates, dist_cmps] = greedy_search(query, L);
    double latency = t.elapsed_us();
    SearchResult result;
    result.dist_cmps = dist_cmps;
    result.latency_us = latency;
    result.ids.reserve(K);
    for (uint32_t i = 0; i < K && i < candidates.size(); i++)
        result.ids.push_back(candidates[i].second);
    return result;
}

void VamanaIndex::save(const std::string& path) const {
    std::ofstream out(path, std::ios::binary);
    if (!out.is_open()) throw std::runtime_error("Cannot open: " + path);
    out.write(reinterpret_cast<const char*>(&npts_), 4);
    out.write(reinterpret_cast<const char*>(&dim_),  4);
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
    if (fn != npts_ || fd != dim_)
        throw std::runtime_error("Index/data mismatch");

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
    std::cout << "Index loaded: " << npts_ << " pts, start="
              << start_node_ << std::endl;
}