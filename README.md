# GraphANN — Educational Vamana Index

A clean, modular C++ implementation of the **Vamana graph-based approximate nearest neighbor (ANN) index** for students to learn, experiment with, and extend.

Implements the core algorithm from [*DiskANN: Fast Accurate Billion-point Nearest Neighbor Search on a Single Node*](https://proceedings.neurips.cc/paper/2019/hash/09853c7fb1d3f8ee67a61b6bf4a7f8e6-Abstract.html) (NeurIPS 2019).

---

## Algorithm Overview

### Build Phase
For each point (in a random order, parallelized with OpenMP):

1. **Greedy Search**: Search the current graph for the point itself, producing a candidate list of size `L`
2. **Robust Prune (α-RNG)**: Prune candidates to at most `R` diverse neighbors using the alpha-RNG rule — a candidate `c` is kept only if `dist(node, c) ≤ α · dist(c, n)` for all already-selected neighbors `n`
3. **Add Edges**: Set forward edges; add backward edges to each neighbor
4. **Degree Check**: If any neighbor's degree exceeds `γR`, prune its neighborhood

Per-node mutexes ensure correctness during parallel construction.

### Search Phase
Greedy beam search starting from a fixed start node, maintaining a candidate set bounded at size `L`. Returns the top-`K` closest points found.

### Parameters
| Parameter | Typical Range | Description |
|-----------|--------------|-------------|
| `R` | 32–64 | Max out-degree (graph connectivity) |
| `L` (build) | 75–200 | Search list size during construction (≥ R) |
| `α` (alpha) | 1.0–1.5 | RNG pruning threshold (> 1 keeps long-range edges) |
| `γ` (gamma) | 1.2–1.5 | Degree multiplier triggering neighbor pruning |
| `L` (search) | 10–200 | Search list size at query time (≥ K) |
| `K` | 1–100 | Number of nearest neighbors to return |

---

## Project Structure

```
graphann/
├── CMakeLists.txt              # Build config (C++17, OpenMP, -O3 -march=native)
├── README.md
├── include/
│   ├── distance.h              # Squared L2 distance function
│   ├── io_utils.h              # fbin/ibin file loaders
│   ├── timer.h                 # Simple chrono-based timer
│   └── vamana_index.h          # VamanaIndex class declaration
├── scripts/
│   ├── convert_vecs.py         # Convert fvecs/ivecs → fbin/ibin
│   └── run_sift1m.sh           # One-command SIFT1M download, build & search
└── src/
    ├── distance.cpp            # Distance implementation
    ├── io_utils.cpp            # File I/O implementation
    ├── vamana_index.cpp        # Core: greedy_search, robust_prune, build, search
    ├── build_index.cpp         # CLI: build index from data
    └── search_index.cpp        # CLI: search + recall/latency evaluation
```

### Key files to study
- **`src/vamana_index.cpp`** — the core algorithm: `greedy_search()`, `robust_prune()`, `build()`
- **`include/vamana_index.h`** — data structures (adjacency list graph, per-node locks)

---

## Build

Requirements: C++17 compiler with OpenMP support (GCC ≥ 7, Clang ≥ 10).

```bash
mkdir build && cd build
cmake .. -DCMAKE_BUILD_TYPE=Release
make -j
```

This produces two executables: `build_index` and `search_index`.

---

## Quick Start — SIFT1M end-to-end

A single script downloads the [SIFT1M](http://corpus-texmex.irisa.fr/) dataset, converts it to binary format, builds a Vamana index, and runs search with recall evaluation:

```bash
./scripts/run_sift1m.sh
```

This will:
1. Build the project (cmake + make)
2. Download SIFT1M (1M base vectors, 10K queries, ground truth) into `tmp/sift/`
3. Convert `.fvecs`/`.ivecs` files to `.fbin`/`.ibin` format in `tmp/`
4. Build a Vamana index with default parameters (R=32, L=75, α=1.2, γ=1.5)
5. Run search at multiple `L` values and report recall@10, latency, and distance computations

Requires: `curl`, `python3` with `numpy`, and a C++17 compiler with OpenMP.

---

## Usage

### File Formats

**fbin** (float binary): Used for dataset and query vectors.
```
[4 bytes: uint32 npts] [4 bytes: uint32 dims] [npts * dims * 4 bytes: float32 row-major vectors]
```

**ibin** (int binary): Used for ground truth neighbor IDs.
```
[4 bytes: uint32 npts] [4 bytes: uint32 dims] [npts * dims * 4 bytes: uint32 row-major IDs]
```

Standard ANN benchmark datasets (SIFT, GIST, GloVe, etc.) are available in this format from [ANN Benchmarks](http://ann-benchmarks.com/) and [big-ann-benchmarks](https://big-ann-benchmarks.com/).

### Build an Index

```bash
./build_index \
  --data /path/to/base.fbin \
  --output /path/to/index.bin \
  --R 32 --L 75 --alpha 1.2 --gamma 1.5
```

### Search and Evaluate

```bash
./search_index \
  --index /path/to/index.bin \
  --data /path/to/base.fbin \
  --queries /path/to/query.fbin \
  --gt /path/to/gt.ibin \
  --K 10 \
  --L 10,20,30,50,75,100,150,200
```

Output:
```
=== Search Results (K=10) ===
       L     Recall@10   Avg Dist Cmps  Avg Latency (us)  P99 Latency (us)
--------------------------------------------------------------------------
      10         0.5432           320.5             125.3             412.7
      20         0.7891           580.2             198.4             623.1
      50         0.9234          1205.8             385.6            1102.3
     100         0.9812          2280.4             702.1            2015.8
     200         0.9965          4350.2            1305.7            3812.4
```

---

## Performance Notes

- **Parallelism**: OpenMP `parallel for schedule(dynamic)` for both build (point insertion) and search (queries)
- **Memory layout**: Contiguous row-major float arrays, 64-byte aligned for SIMD
- **Vectorization**: `-O3 -march=native` auto-vectorizes the L2 distance loop — no manual intrinsics needed
- **Lock granularity**: Per-node `std::mutex` — threads only contend when updating the *same* node's adjacency list
- **No external dependencies** beyond OpenMP

---

## Some sample things to try, and start the experimenting with!

0. **Code understanding**: Use AI tools to understand the logic of algorithm and how it is a hueristic approximation of what we discussed in class

1. **Beam width experiments**: Try different `L` values during build and measure recall vs build time. What's the sweet spot?

2. **Medoid start node**: Replace the random start node with the *medoid* — the point closest to the centroid of the dataset. How does this affect search recall?

3. **Change the edges in index build**: Run the build twice — second pass starts from the graph produced by the first. How does recall change?

4. **Change the search algorithm**: Plot the histogram of node degrees. Is it uniform? What happens with different `α` values?

5. **Concurrent search optimization**: Replace `std::vector<bool> visited` in `greedy_search()` with a pre-allocated scratch buffer to avoid per-query allocation.

---

## References

- Subramanya et al., *DiskANN: Fast Accurate Billion-point Nearest Neighbor Search on a Single Node*, NeurIPS 2019

