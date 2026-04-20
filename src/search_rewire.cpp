#include "vamana_index.h"
#include "io_utils.h"
#include "timer.h"

#include <algorithm>
#include <iostream>
#include <iomanip>
#include <numeric>
#include <string>
#include <sstream>
#include <vector>
#include <cstdlib>

// -------------------- Helper --------------------

static void print_usage(const char* prog) {
    std::cerr << "Usage: " << prog
              << " --index <index_path>"
              << " --data <fbin_path>"
              << " --queries <query_fbin_path>"
              << " --gt <ground_truth_ibin_path>"
              << " --K <num_neighbors>"
              << " --L <comma_separated_L_values>"
              << std::endl;
}

static std::vector<uint32_t> parse_L_values(const std::string& s) {
    std::vector<uint32_t> values;
    std::istringstream stream(s);
    std::string token;
    while (std::getline(stream, token, ',')) {
        values.push_back(std::atoi(token.c_str()));
    }
    std::sort(values.begin(), values.end());
    return values;
}

// -------------------- Recall@K --------------------

static double compute_recall(const std::vector<uint32_t>& result,
                             const uint32_t* gt, uint32_t K) {
    uint32_t found = 0;
    for (uint32_t i = 0; i < K && i < result.size(); i++) {
        for (uint32_t j = 0; j < K; j++) {
            if (result[i] == gt[j]) {
                found++;
                break;
            }
        }
    }
    return (double)found / K;
}

// -------------------- Recall@1 --------------------

static double compute_recall1(const std::vector<uint32_t>& result,
                              const uint32_t* gt) {
    if (result.empty()) return 0.0;
    return (result[0] == gt[0]) ? 1.0 : 0.0;
}

// -------------------- MAIN --------------------

int main(int argc, char** argv) {
    std::string index_path, data_path, query_path, gt_path, L_str;
    uint32_t K = 10;

    for (int i = 1; i < argc; i++) {
        std::string arg = argv[i];
        if (arg == "--index" && i + 1 < argc) index_path = argv[++i];
        else if (arg == "--data" && i + 1 < argc) data_path = argv[++i];
        else if (arg == "--queries" && i + 1 < argc) query_path = argv[++i];
        else if (arg == "--gt" && i + 1 < argc) gt_path = argv[++i];
        else if (arg == "--K" && i + 1 < argc) K = std::atoi(argv[++i]);
        else if (arg == "--L" && i + 1 < argc) L_str = argv[++i];
        else if (arg == "--help") {
            print_usage(argv[0]);
            return 0;
        }
    }

    if (index_path.empty() || data_path.empty() ||
        query_path.empty() || gt_path.empty() || L_str.empty()) {
        print_usage(argv[0]);
        return 1;
    }

    std::vector<uint32_t> L_values = parse_L_values(L_str);

    // -------------------- Load --------------------

    std::cout << "Loading index..." << std::endl;
    VamanaIndex index;
    index.load(index_path, data_path);

    std::cout << "Loading queries..." << std::endl;
    FloatMatrix queries = load_fbin(query_path);

    std::cout << "Loading ground truth..." << std::endl;
    IntMatrix gt = load_ibin(gt_path);

    uint32_t nq = queries.npts;

    // -------------------- Header --------------------

    std::cout << "\n=== Search Results (K=" << K << ") ===\n";
    std::cout << std::setw(8)  << "L"
              << std::setw(14) << "Recall@1"
              << std::setw(14) << "Recall@" + std::to_string(K)
              << std::setw(16) << "Avg Dist Cmps"
              << std::setw(18) << "Avg Latency (us)"
              << std::setw(18) << "P99 Latency (us)"
              << std::endl;

    std::cout << std::string(90, '-') << std::endl;

    // -------------------- Loop over L --------------------

    for (uint32_t L : L_values) {
        std::vector<double> r1(nq);
        std::vector<double> rK(nq);
        std::vector<uint32_t> cmps(nq);
        std::vector<double> lat(nq);

        #pragma omp parallel for schedule(dynamic, 16)
        for (uint32_t q = 0; q < nq; q++) {
            SearchResult res = index.search(queries.row(q), K, L);

            r1[q] = compute_recall1(res.ids, gt.row(q));
            rK[q] = compute_recall(res.ids, gt.row(q), K);
            cmps[q] = res.dist_cmps;
            lat[q] = res.latency_us;
        }

        double avg_r1 = std::accumulate(r1.begin(), r1.end(), 0.0) / nq;
        double avg_rK = std::accumulate(rK.begin(), rK.end(), 0.0) / nq;
        double avg_cmps = (double)std::accumulate(cmps.begin(), cmps.end(), 0ULL) / nq;
        double avg_lat = std::accumulate(lat.begin(), lat.end(), 0.0) / nq;

        std::sort(lat.begin(), lat.end());
        double p99 = lat[(size_t)(0.99 * nq)];

        std::cout << std::setw(8) << L
                  << std::setw(14) << std::fixed << std::setprecision(4) << avg_r1
                  << std::setw(14) << std::fixed << std::setprecision(4) << avg_rK
                  << std::setw(16) << std::fixed << std::setprecision(1) << avg_cmps
                  << std::setw(18) << std::fixed << std::setprecision(1) << avg_lat
                  << std::setw(18) << std::fixed << std::setprecision(1) << p99
                  << std::endl;
    }

    std::cout << "\nDone.\n";
    return 0;
}