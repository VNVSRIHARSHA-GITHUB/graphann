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
// Greedy Search (Fixed: vector<bool> → vector<char>)
// ============================================================================

std::pair<std::vector<VamanaIndex::Candidate>, uint32_t>
VamanaIndex::greedy_search(const float* query, uint32_t L) const {
    std::set<Candidate> cset;
    std::vector<char> visited(npts_, 0);
    uint32_t dist_cmps = 0;

    float sd = compute_l2sq(query, get_vector(start_node_), dim_);
    dist_cmps++;
    cset.insert({sd, start_node_});
    visited[start_node_] = 1;

    std::set<uint32_t> expanded;
    while (true) {
        uint32_t best = UINT32_MAX;
        for (const auto& [d, id] : cset)
            if (!expanded.count(id)) { best = id; break; }
        if (best == UINT32_MAX) break;
        expanded.insert(best);

        std::vector<uint32_t> nbrs;
        { std::lock_guard<std::mutex> lk(locks_[best]); nbrs = graph_[best]; }

        for (uint32_t nb : nbrs) {
            if (visited[nb]) continue;
            visited[nb] = 1;
            float d = compute_l2sq(query, get_vector(nb), dim_);
            dist_cmps++;
            if (cset.size() < L)
                cset.insert({d, nb});
            else {
                auto worst = std::prev(cset.end());
                if (d < worst->first) { cset.erase(worst); cset.insert({d, nb}); }
            }
        }
    }
    return {std::vector<Candidate>(cset.begin(), cset.end()), dist_cmps};
}

// ============================================================================
// Robust Prune
// ============================================================================

void VamanaIndex::robust_prune(uint32_t node,
                               std::vector<Candidate>& candidates,
                               float alpha, uint32_t R) {
    candidates.erase(
        std::remove_if(candidates.begin(), candidates.end(),
                       [node](const Candidate& c){ return c.second == node; }),
        candidates.end());
    std::sort(candidates.begin(), candidates.end());

    std::vector<uint32_t> nn;
    nn.reserve(R);
    for (const auto& [dn, cid] : candidates) {
        if (nn.size() >= R) break;
        bool keep = true;
        for (uint32_t sel : nn) {
            float dc = compute_l2sq(get_vector(cid), get_vector(sel), dim_);
            if (dn > alpha * dc) { keep = false; break; }
        }
        if (keep) nn.push_back(cid);
    }
    graph_[node] = std::move(nn);
}

// ============================================================================
// search_from (kept for compatibility)
// ============================================================================

std::vector<VamanaIndex::Candidate>
VamanaIndex::search_from(uint32_t seed, const float* query, uint32_t L) const {
    std::set<Candidate> cset;
    std::vector<char> vis(npts_, 0);
    cset.insert({compute_l2sq(query, get_vector(seed), dim_), seed});
    vis[seed] = 1;

    std::set<uint32_t> exp;
    while (true) {
        uint32_t best = UINT32_MAX;
        for (const auto& [d, id] : cset)
            if (!exp.count(id)) { best = id; break; }
        if (best == UINT32_MAX) break;
        exp.insert(best);

        std::vector<uint32_t> nbrs;
        { std::lock_guard<std::mutex> lk(locks_[best]); nbrs = graph_[best]; }

        for (uint32_t nb : nbrs) {
            if (vis[nb]) continue;
            vis[nb] = 1;
            float d = compute_l2sq(query, get_vector(nb), dim_);
            if (cset.size() < L)
                cset.insert({d, nb});
            else {
                auto worst = std::prev(cset.end());
                if (d < worst->first) { cset.erase(worst); cset.insert({d, nb}); }
            }
        }
    }
    return std::vector<Candidate>(cset.begin(), cset.end());
}

// ============================================================================
// Dummy run_pass
// ============================================================================

void VamanaIndex::run_pass(const std::vector<uint32_t>& perm,
                           float alpha, uint32_t R, uint32_t L,
                           uint32_t gamma_R) {
    (void)perm; (void)alpha; (void)R; (void)L; (void)gamma_R;
}

// ============================================================================
// Build — SELECTIVE BACKWARD EDGES (Corrected & Safe)
// ============================================================================

void VamanaIndex::build(const std::string& data_path, uint32_t R, uint32_t L,
                        float alpha, float gamma) {
    std::cout << "[Selective Backward Edges] Loading data..." << std::endl;
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
    std::shuffle(perm.begin(), perm.end(), rng);

    uint32_t gamma_R = static_cast<uint32_t>(gamma * R);

    std::cout << "[Selective Build] R=" << R << " L=" << L
              << " alpha=" << alpha << std::endl;

    // Single-threaded (safe)
    for (size_t idx = 0; idx < npts_; idx++) {
        uint32_t point = perm[idx];

        auto [candidates, _d] = greedy_search(get_vector(point), L);
        robust_prune(point, candidates, alpha, R);

        // ── SELECTIVE backward edges ───────────────────────────────
        for (uint32_t nbr : graph_[point]) {
            std::lock_guard<std::mutex> lock(locks_[nbr]);

            float d_nbr_point = compute_l2sq(get_vector(nbr), get_vector(point), dim_);

            bool passes = true;
            for (uint32_t existing : graph_[nbr]) {
                float d_existing_point = compute_l2sq(get_vector(existing),
                                                      get_vector(point), dim_);
                // Correct condition: d(nbr, point) <= α * d(existing, point)
                if (d_nbr_point > alpha * d_existing_point) {
                    passes = false;
                    break;
                }
            }

            if (passes) {
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
        }

        if ((idx + 1) % 50000 == 0 || idx + 1 == npts_)
            std::cout << "\r  Progress: " << (idx + 1) * 100 / npts_ << "%" << std::flush;
    }
    std::cout << std::endl;

    size_t total = 0;
    for (uint32_t i = 0; i < npts_; i++) total += graph_[i].size();
    std::cout << "[Selective Build] Done. Avg degree: "
              << (double)total / npts_ << std::endl;
}

// ============================================================================
// Search, Save, Load (unchanged)
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