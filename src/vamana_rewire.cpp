#include "vamana_index.h"
#include "distance.h"
#include "io_utils.h"
#include "timer.h"

#include <algorithm>
#include <atomic>
#include <fstream>
#include <iostream>
#include <numeric>
#include <random>
#include <set>
#include <stdexcept>
#include <thread>
#include <cstdlib>

VamanaIndex::~VamanaIndex() {
    if (owns_data_ && data_) { std::free(data_); data_ = nullptr; }
}

std::pair<std::vector<VamanaIndex::Candidate>, uint32_t>
VamanaIndex::greedy_search(const float* query, uint32_t L) const {
    std::set<Candidate> cset;
    std::vector<bool> visited(npts_, false);
    uint32_t dist_cmps = 0;
    float sd = compute_l2sq(query, get_vector(start_node_), dim_);
    dist_cmps++;
    cset.insert({sd, start_node_});
    visited[start_node_] = true;
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
            visited[nb] = true;
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
    return { std::vector<Candidate>(cset.begin(), cset.end()), dist_cmps };
}

void VamanaIndex::robust_prune(uint32_t node,
                               std::vector<Candidate>& candidates,
                               float alpha, uint32_t R) {
    candidates.erase(
        std::remove_if(candidates.begin(), candidates.end(),
                       [node](const Candidate& c){ return c.second == node; }),
        candidates.end());
    std::sort(candidates.begin(), candidates.end());
    std::vector<uint32_t> nn; nn.reserve(R);
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

std::vector<VamanaIndex::Candidate>
VamanaIndex::search_from(uint32_t seed, const float* query, uint32_t L) const {
    std::set<Candidate> cset;
    std::vector<bool> vis(npts_, false);
    cset.insert({ compute_l2sq(query, get_vector(seed), dim_), seed });
    vis[seed] = true;
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
            vis[nb] = true;
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

void VamanaIndex::run_pass(const std::vector<uint32_t>& perm,
                           float alpha, uint32_t R, uint32_t L,
                           uint32_t gamma_R) {
    (void)perm; (void)alpha; (void)R; (void)L; (void)gamma_R;
}

// ============================================================================
// Build — POST-BUILD SELF-CONSISTENCY REWIRING
// Step 1: run standard baseline build (single pass)
// Step 2: count how many times each node was rejected during pruning
//         (approximated here by nodes with below-average degree —
//          these were likely rejected most and have suboptimal edges)
// Step 3: for top 15% of such nodes, re-run greedy search starting FROM
//         the node itself, then re-prune to update its edges on the
//         now-complete graph
// ============================================================================
void VamanaIndex::build(const std::string& data_path, uint32_t R, uint32_t L,
                        float alpha, float gamma) {
    std::cout << "[Rewire Build] Loading data..." << std::endl;
    FloatMatrix mat = load_fbin(data_path);
    npts_ = mat.npts; dim_ = mat.dims;
    data_ = mat.data.release(); owns_data_ = true;
    if (L < R) L = R;

    graph_.resize(npts_);
    locks_ = std::vector<std::mutex>(npts_);

    std::mt19937 rng(42);
    start_node_ = rng() % npts_;

    std::vector<uint32_t> perm(npts_);
    std::iota(perm.begin(), perm.end(), 0);
    std::shuffle(perm.begin(), perm.end(), rng);

    uint32_t gamma_R = static_cast<uint32_t>(gamma * R);

    // ── PHASE 1: standard baseline build ─────────────────────────────────
    std::cout << "[Rewire Build] Phase 1: standard build..." << std::endl;

    uint32_t num_threads = std::thread::hardware_concurrency();
    if (num_threads == 0) num_threads = 4;
    std::atomic<size_t> counter(0);
    std::mutex prog_mtx;

    auto worker_phase1 = [&](size_t start, size_t end) {
        for (size_t idx = start; idx < end; idx++) {
            uint32_t point = perm[idx];
            auto [candidates, _d] = greedy_search(get_vector(point), L);
            robust_prune(point, candidates, alpha, R);

            for (uint32_t nbr : graph_[point]) {
                std::lock_guard<std::mutex> lock(locks_[nbr]);
                graph_[nbr].push_back(point);
                if (graph_[nbr].size() > gamma_R) {
                    std::vector<Candidate> nc;
                    for (uint32_t nn : graph_[nbr]) {
                        float d = compute_l2sq(get_vector(nbr),
                                               get_vector(nn), dim_);
                        nc.push_back({d, nn});
                    }
                    robust_prune(nbr, nc, alpha, R);
                }
            }
            size_t done = ++counter;
            if (done % 10000 == 0) {
                std::lock_guard<std::mutex> lk(prog_mtx);
                std::cout << "\r  Phase1: " << done << " / "
                          << npts_ << std::flush;
            }
        }
    };

    {
        size_t chunk = npts_ / num_threads;
        std::vector<std::thread> threads;
        for (uint32_t t = 0; t < num_threads; t++) {
            size_t s = t * chunk;
            size_t e = (t == num_threads - 1) ? npts_ : s + chunk;
            threads.emplace_back(worker_phase1, s, e);
        }
        for (auto& t : threads) t.join();
    }

    std::cout << "\n[Rewire Build] Phase 1 done." << std::endl;

    // ── PHASE 2: identify top 15% high-churn nodes ────────────────────────
    // Nodes with the lowest degree after build were rejected most often
    // during pruning — their edges are the least reliable.
    std::vector<uint32_t> degrees(npts_);
    for (uint32_t i = 0; i < npts_; i++)
        degrees[i] = graph_[i].size();

    // sort node IDs by degree ascending
    std::vector<uint32_t> order(npts_);
    std::iota(order.begin(), order.end(), 0);
    std::sort(order.begin(), order.end(),
              [&](uint32_t a, uint32_t b){ return degrees[a] < degrees[b]; });

    uint32_t rewire_count = npts_ * 15 / 100;   // top 15%
    std::vector<uint32_t> rewire_nodes(
        order.begin(), order.begin() + rewire_count);

    std::cout << "[Rewire Build] Phase 2: rewiring "
              << rewire_count << " nodes..." << std::endl;

    // ── PHASE 2 build: re-run search FROM each high-churn node ───────────
    counter = 0;
    auto worker_phase2 = [&](size_t start, size_t end) {
        for (size_t idx = start; idx < end; idx++) {
            uint32_t node = rewire_nodes[idx];

            // search starting from the node itself
            auto local_cands = search_from(node, get_vector(node), L);

            // re-prune on the now-complete graph
            robust_prune(node, local_cands, alpha, R);

            // update backward edges for new neighbors
            for (uint32_t nbr : graph_[node]) {
                std::lock_guard<std::mutex> lock(locks_[nbr]);
                // only add if not already present
                bool found = false;
                for (uint32_t ex : graph_[nbr])
                    if (ex == node) { found = true; break; }
                if (!found) {
                    graph_[nbr].push_back(node);
                    if (graph_[nbr].size() > gamma_R) {
                        std::vector<Candidate> nc;
                        for (uint32_t nn : graph_[nbr]) {
                            float d = compute_l2sq(get_vector(nbr),
                                                   get_vector(nn), dim_);
                            nc.push_back({d, nn});
                        }
                        robust_prune(nbr, nc, alpha, R);
                    }
                }
            }

            size_t done = ++counter;
            if (done % 5000 == 0) {
                std::lock_guard<std::mutex> lk(prog_mtx);
                std::cout << "\r  Phase2: " << done << " / "
                          << rewire_count << std::flush;
            }
        }
    };

    {
        size_t chunk = rewire_count / num_threads;
        if (chunk == 0) chunk = 1;
        std::vector<std::thread> threads;
        size_t pos = 0;
        for (uint32_t t = 0; t < num_threads && pos < rewire_count; t++) {
            size_t s = pos;
            size_t e = (t == num_threads - 1) ? rewire_count
                                               : std::min(pos + chunk,
                                                         (size_t)rewire_count);
            threads.emplace_back(worker_phase2, s, e);
            pos = e;
        }
        for (auto& t : threads) t.join();
    }

    size_t total = 0;
    for (uint32_t i = 0; i < npts_; i++) total += graph_[i].size();
    std::cout << "\n[Rewire Build] Done. Avg degree: "
              << (double)total / npts_ << std::endl;
}

SearchResult VamanaIndex::search(const float* query,
                                 uint32_t K, uint32_t L) const {
    if (L < K) L = K;
    Timer t;
    auto [candidates, dist_cmps] = greedy_search(query, L);
    double latency = t.elapsed_us();
    SearchResult result;
    result.dist_cmps = dist_cmps; result.latency_us = latency;
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