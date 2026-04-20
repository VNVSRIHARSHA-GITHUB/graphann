#include "vamana_index.h"
#include "timer.h"

#include <iostream>
#include <string>
#include <cstdlib>

static void print_usage(const char* prog) {
    std::cerr << "Usage: " << prog
              << " --data <fbin_path>"
              << " --output <index_path>"
              << " [--R <32>] [--L <75>]"
              << " [--alpha <1.2>] [--gamma <1.5>]"
              << std::endl;
}

int main(int argc, char** argv) {
    std::string data_path, output_path;
    uint32_t R = 32;
    uint32_t L = 75;
    float alpha = 1.2f;
    float gamma = 1.5f;

    for (int i = 1; i < argc; i++) {
        std::string arg = argv[i];
        if      (arg == "--data"   && i+1 < argc) data_path   = argv[++i];
        else if (arg == "--output" && i+1 < argc) output_path = argv[++i];
        else if (arg == "--R"      && i+1 < argc) R     = std::atoi(argv[++i]);
        else if (arg == "--L"      && i+1 < argc) L     = std::atoi(argv[++i]);
        else if (arg == "--alpha"  && i+1 < argc) alpha = std::atof(argv[++i]);
        else if (arg == "--gamma"  && i+1 < argc) gamma = std::atof(argv[++i]);
        else if (arg == "--help" || arg == "-h") {
            print_usage(argv[0]);
            return 0;
        }
    }

    if (data_path.empty() || output_path.empty()) {
        print_usage(argv[0]);
        return 1;
    }

    std::cout << "=== Adaptive Density Pruning Builder ===" << std::endl;
    std::cout << "R="     << R 
              << "  L="   << L 
              << "  alpha=" << alpha 
              << "  gamma=" << gamma << std::endl;

    VamanaIndex index;
    Timer total_timer;

    index.build(data_path, R, L, alpha, gamma);

    double total_time = total_timer.elapsed_seconds();
    std::cout << "\nTotal build time: " << total_time << " seconds" << std::endl;

    index.save(output_path);
    std::cout << "Done." << std::endl;
    return 0;
}