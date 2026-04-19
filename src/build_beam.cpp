#include "vamana_index.h"
#include "timer.h"
#include <iostream>
#include <string>
#include <cstdlib>

int main(int argc, char** argv) {
    std::string data_path, output_path;
    uint32_t R = 32, L = 75;
    float alpha = 1.2f, gamma = 1.5f;

    for (int i = 1; i < argc; i++) {
        std::string a = argv[i];
        if      (a == "--data"   && i+1 < argc) data_path   = argv[++i];
        else if (a == "--output" && i+1 < argc) output_path = argv[++i];
        else if (a == "--R"      && i+1 < argc) R     = std::atoi(argv[++i]);
        else if (a == "--L"      && i+1 < argc) L     = std::atoi(argv[++i]);
        else if (a == "--alpha"  && i+1 < argc) alpha = std::atof(argv[++i]);
        else if (a == "--gamma"  && i+1 < argc) gamma = std::atof(argv[++i]);
    }
    if (data_path.empty() || output_path.empty()) {
        std::cerr << "Usage: build_beam --data <f> --output <f> "
                     "[--R 32] [--L 75] [--alpha 1.2] [--gamma 1.5]\n";
        return 1;
    }
    std::cout << "=== Beam Build ===" << std::endl;
    VamanaIndex index;
    Timer t;
    index.build(data_path, R, L, alpha, gamma);
    std::cout << "Build time: " << t.elapsed_seconds() << "s" << std::endl;
    index.save(output_path);
    return 0;
}