#pragma once

#include <chrono>

// Simple stopwatch timer for benchmarking.
class Timer {
  public:
    Timer() { reset(); }

    void reset() { start_ = std::chrono::high_resolution_clock::now(); }

    double elapsed_seconds() const {
        auto now = std::chrono::high_resolution_clock::now();
        return std::chrono::duration<double>(now - start_).count();
    }

    double elapsed_ms() const { return elapsed_seconds() * 1000.0; }

    double elapsed_us() const { return elapsed_seconds() * 1e6; }

  private:
    std::chrono::high_resolution_clock::time_point start_;
};
