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
    uint32_t R = 32, L = 75;
    float alpha = 1.2f, gamma = 1.5f;

    for (int i = 1; i < argc; i++) {
        std::string arg = argv[i];
        if      (arg == "--data"   && i+1 < argc) data_path   = argv[++i];
        else if (arg == "--output" && i+1 < argc) output_path = argv[++i];
        else if (arg == "--R"      && i+1 < argc) R     = std::atoi(argv[++i]);
        else if (arg == "--L"      && i+1 < argc) L     = std::atoi(argv[++i]);
        else if (arg == "--alpha"  && i+1 < argc) alpha = std::atof(argv[++i]);
        else if (arg == "--gamma"  && i+1 < argc) gamma = std::atof(argv[++i]);
        else if (arg == "--help") { print_usage(argv[0]); return 0; }
    }

    if (data_path.empty() || output_path.empty()) {
        print_usage(argv[0]); return 1;
    }

    std::cout << "=== MultiStart Index Builder ===" << std::endl;
    std::cout << "R=" << R << " L=" << L
              << " alpha=" << alpha << " gamma=" << gamma << std::endl;

    VamanaIndex index;
    Timer t;
    index.build(data_path, R, L, alpha, gamma);
    std::cout << "Total build time: " << t.elapsed_seconds() << "s" << std::endl;
    index.save(output_path);
    return 0;
}