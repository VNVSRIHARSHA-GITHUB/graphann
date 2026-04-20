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

    float start_dist = compute_l2sq(query, get_vector(start_node_), dim_);
    dist_cmps++;
    candidate_set.insert({start_dist, start_node_});
    visited[start_node_] = true;

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
            visited[nbr] = true;

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
// Robust Prune  (identical to original)
// ============================================================================

void VamanaIndex::robust_prune(uint32_t node,
                               std::vector<Candidate>& candidates,
                               float alpha, uint32_t R) {
    candidates.erase(
        std::remove_if(candidates.begin(), candidates.end(),
                       [node](const Candidate& c) { return c.second == node; }),
        candidates.end());

    std::sort(candidates.begin(), candidates.end());

    std::vector<uint32_t> new_neighbors;
    new_neighbors.reserve(R);

    for (const auto& [dist_to_node, cand_id] : candidates) {
        if (new_neighbors.size() >= R) break;

        bool keep = true;
        for (uint32_t selected : new_neighbors) {
            float dist_cand_to_selected =
                compute_l2sq(get_vector(cand_id), get_vector(selected), dim_);
            if (dist_to_node > alpha * dist_cand_to_selected) {
                keep = false;
                break;
            }
        }
        if (keep) new_neighbors.push_back(cand_id);
    }

    graph_[node] = std::move(new_neighbors);
}

// ============================================================================
// Build — WITH VAMANA REFINE
// ============================================================================
// Change from baseline:
//   After initial prune gives tentative neighbors, we do a second local
//   greedy search starting FROM point itself. We merge Vinit + Vlocal
//   and re-prune to get final neighbors.
// ============================================================================

void VamanaIndex::build(const std::string& data_path, uint32_t R, uint32_t L,
                        float alpha, float gamma) {
    std::cout << "[Refine Build] Loading data..." << std::endl;
    FloatMatrix mat = load_fbin(data_path);
    npts_ = mat.npts;
    dim_  = mat.dims;
    data_ = mat.data.release();
    owns_data_ = true;

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

    std::cout << "[Refine Build] Building (R=" << R << " L=" << L
              << " alpha=" << alpha << ")..." << std::endl;

    #pragma omp parallel for schedule(dynamic, 64)
    for (size_t idx = 0; idx < npts_; idx++) {
        uint32_t point = perm[idx];

        // ── STEP 1: initial search from medoid → Vinit ──────────────────
        auto [init_cands, _d1] = greedy_search(get_vector(point), L);

        // ── STEP 2: initial prune → tentative neighbors ──────────────────
        robust_prune(point, init_cands, alpha, R);

        // ── STEP 3: local search starting FROM point itself → Vlocal ─────
        // Temporarily redirect start_node_ per-thread using a local copy.
        // We cannot safely write start_node_ (shared), so we call the
        // internal search with point as seed manually.
        std::vector<Candidate> local_cands;
        {
            // seed the local search with point's current neighbors
            std::set<Candidate> cset;
            std::vector<bool> vis(npts_, false);
            vis[point] = true;

            // add point's new neighbors as starting seeds
            for (uint32_t nbr : graph_[point]) {
                float d = compute_l2sq(get_vector(point), get_vector(nbr), dim_);
                cset.insert({d, nbr});
                vis[nbr] = true;
            }

            std::set<uint32_t> exp;
            while (true) {
                uint32_t best = UINT32_MAX;
                for (const auto& [d, id] : cset) {
                    if (exp.find(id) == exp.end()) { best = id; break; }
                }
                if (best == UINT32_MAX) break;
                exp.insert(best);

                std::vector<uint32_t> nbrs;
                {
                    std::lock_guard<std::mutex> lk(locks_[best]);
                    nbrs = graph_[best];
                }
                for (uint32_t nb : nbrs) {
                    if (vis[nb]) continue;
                    vis[nb] = true;
                    float d = compute_l2sq(get_vector(point), get_vector(nb), dim_);
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
            local_cands.assign(cset.begin(), cset.end());
        }

        // ── STEP 4: merge Vinit neighbors + Vlocal, re-prune ─────────────
        std::vector<Candidate> merged;
        // add current neighbors of point (from step 2)
        for (uint32_t nbr : graph_[point]) {
            float d = compute_l2sq(get_vector(point), get_vector(nbr), dim_);
            merged.push_back({d, nbr});
        }
        // add local candidates
        for (auto& c : local_cands) merged.push_back(c);

        // deduplicate by id
        std::sort(merged.begin(), merged.end(),
                  [](const Candidate& a, const Candidate& b){
                      return a.second < b.second;
                  });
        merged.erase(std::unique(merged.begin(), merged.end(),
                                 [](const Candidate& a, const Candidate& b){
                                     return a.second == b.second;
                                 }), merged.end());

        // final prune over merged pool
        robust_prune(point, merged, alpha, R);

        // ── STEP 5: backward edges (same as baseline) ────────────────────
        for (uint32_t nbr : graph_[point]) {
            std::lock_guard<std::mutex> lock(locks_[nbr]);
            graph_[nbr].push_back(point);
            if (graph_[nbr].size() > gamma_R) {
                std::vector<Candidate> nbr_cands;
                for (uint32_t nn : graph_[nbr]) {
                    float d = compute_l2sq(get_vector(nbr), get_vector(nn), dim_);
                    nbr_cands.push_back({d, nn});
                }
                robust_prune(nbr, nbr_cands, alpha, R);
            }
        }
    }

    size_t total = 0;
    for (uint32_t i = 0; i < npts_; i++) total += graph_[i].size();
    std::cout << "[Refine Build] Done. Avg degree: "
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
    result.dist_cmps  = dist_cmps;
    result.latency_us = latency;
    result.ids.reserve(K);
    for (uint32_t i = 0; i < K && i < candidates.size(); i++)
        result.ids.push_back(candidates[i].second);
    return result;
}

void VamanaIndex::save(const std::string& path) const {
    std::ofstream out(path, std::ios::binary);
    if (!out.is_open())
        throw std::runtime_error("Cannot open: " + path);
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
    if (!in.is_open())
        throw std::runtime_error("Cannot open: " + index_path);

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