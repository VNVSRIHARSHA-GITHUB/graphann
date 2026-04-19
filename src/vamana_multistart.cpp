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
// Greedy Search  (identical to original)
// ============================================================================

std::pair<std::vector<VamanaIndex::Candidate>, uint32_t>
VamanaIndex::greedy_search(const float* query, uint32_t L) const {
    std::set<Candidate> candidate_set;
    std::vector<bool> visited(npts_, false);
    uint32_t dist_cmps = 0;

    float sd = compute_l2sq(query, get_vector(start_node_), dim_);
    dist_cmps++;
    candidate_set.insert({sd, start_node_});
    visited[start_node_] = true;

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
            visited[nb] = true;
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
// Helper: run greedy search from a specific seed node (not start_node_)
// ============================================================================

std::vector<VamanaIndex::Candidate>
VamanaIndex::search_from(uint32_t seed, const float* query, uint32_t L) const {
    std::set<Candidate> cset;
    std::vector<bool> vis(npts_, false);

    float sd = compute_l2sq(query, get_vector(seed), dim_);
    cset.insert({sd, seed});
    vis[seed] = true;

    std::set<uint32_t> exp;
    while (true) {
        uint32_t best = UINT32_MAX;
        for (const auto& [d, id] : cset)
            if (exp.find(id) == exp.end()) { best = id; break; }
        if (best == UINT32_MAX) break;
        exp.insert(best);

        std::vector<uint32_t> nbrs;
        { std::lock_guard<std::mutex> lk(locks_[best]); nbrs = graph_[best]; }

        for (uint32_t nb : nbrs) {
            if (vis[nb]) continue;
            vis[nb] = true;
            float d = compute_l2sq(query, get_vector(nb), dim_);
            if (cset.size() < L)
                cset.insert({d, nb});
            else {
                auto worst = std::prev(cset.end());
                if (d < worst->first) {
                    cset.erase(worst);
                    cset.insert({d, nb});
                }
            }
        }
    }
    return std::vector<Candidate>(cset.begin(), cset.end());
}

// ============================================================================
// Build — WITH MULTIPLE ENTRY POINTS
// ============================================================================
// Change from baseline:
//   Instead of one greedy search from start_node_, we run M searches
//   from M diverse start nodes and union their visited sets before pruning.
//   This prevents disconnected components in clustered datasets.
// ============================================================================

void VamanaIndex::build(const std::string& data_path, uint32_t R, uint32_t L,
                        float alpha, float gamma) {
    std::cout << "[MultiStart Build] Loading data..." << std::endl;
    FloatMatrix mat = load_fbin(data_path);
    npts_ = mat.npts; dim_ = mat.dims;
    data_ = mat.data.release(); owns_data_ = true;

    std::cout << "  Points: " << npts_ << ", Dims: " << dim_ << std::endl;
    if (L < R) L = R;

    graph_.resize(npts_);
    locks_ = std::vector<std::mutex>(npts_);

    std::mt19937 rng(42);
    start_node_ = rng() % npts_;

    // ── pick M diverse start nodes evenly spaced across dataset ──────────
    const uint32_t M = 3;
    std::vector<uint32_t> start_nodes(M);
    uint32_t step = npts_ / M;
    for (uint32_t i = 0; i < M; i++)
        start_nodes[i] = i * step;

    std::cout << "  Using " << M << " start nodes: ";
    for (auto s : start_nodes) std::cout << s << " ";
    std::cout << std::endl;

    std::vector<uint32_t> perm(npts_);
    std::iota(perm.begin(), perm.end(), 0);
    std::shuffle(perm.begin(), perm.end(), rng);

    uint32_t gamma_R = static_cast<uint32_t>(gamma * R);

    std::cout << "[MultiStart Build] Building..." << std::endl;

    #pragma omp parallel for schedule(dynamic, 64)
    for (size_t idx = 0; idx < npts_; idx++) {
        uint32_t point = perm[idx];

        // ── search from each start node, union the candidates ─────────────
        std::vector<Candidate> merged;
        for (uint32_t s : start_nodes) {
            auto cands = search_from(s, get_vector(point), L);
            for (auto& c : cands) merged.push_back(c);
        }

        // deduplicate by id, keep smallest distance per id
        std::sort(merged.begin(), merged.end(),
                  [](const Candidate& a, const Candidate& b){
                      return a.second < b.second ||
                             (a.second == b.second && a.first < b.first);
                  });
        merged.erase(std::unique(merged.begin(), merged.end(),
                                 [](const Candidate& a, const Candidate& b){
                                     return a.second == b.second;
                                 }), merged.end());

        // sort by distance for prune
        std::sort(merged.begin(), merged.end());

        // prune over unioned candidate pool
        robust_prune(point, merged, alpha, R);

        // backward edges
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
    }

    size_t total = 0;
    for (uint32_t i = 0; i < npts_; i++) total += graph_[i].size();
    std::cout << "[MultiStart Build] Done. Avg degree: "
              << (double)total / npts_ << std::endl;
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