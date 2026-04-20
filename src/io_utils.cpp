#include "io_utils.h"

#include <fstream>
#include <stdexcept>
#include <cstdlib>

// Allocates 64-byte-aligned memory (SIMD friendly).
static void* aligned_alloc_wrapper(size_t size) {
    // Round up to multiple of 64 for aligned_alloc requirement
    size_t aligned_size = (size + 63) & ~(size_t)63;
    void* ptr = std::aligned_alloc(64, aligned_size);
    if (!ptr)
        throw std::runtime_error("Failed to allocate " + std::to_string(size) + " bytes");
    return ptr;
}

FloatMatrix load_fbin(const std::string& path) {
    std::ifstream in(path, std::ios::binary);
    if (!in.is_open())
        throw std::runtime_error("Cannot open file: " + path);

    uint32_t npts, dims;
    in.read(reinterpret_cast<char*>(&npts), 4);
    in.read(reinterpret_cast<char*>(&dims), 4);

    if (!in.good())
        throw std::runtime_error("Failed to read header from: " + path);

    size_t data_size = (size_t)npts * dims * sizeof(float);

    // Verify file has enough data
    auto cur = in.tellg();
    in.seekg(0, std::ios::end);
    auto file_size = in.tellg();
    if ((size_t)(file_size - cur) < data_size)
        throw std::runtime_error("File too small: expected " +
            std::to_string(data_size) + " data bytes, file has " +
            std::to_string(file_size - cur) + " after header");
    in.seekg(cur);

    FloatMatrix mat;
    mat.npts = npts;
    mat.dims = dims;
    mat.data = std::unique_ptr<float[], void(*)(void*)>(
        static_cast<float*>(aligned_alloc_wrapper(data_size)), std::free);

    in.read(reinterpret_cast<char*>(mat.data.get()), data_size);
    if (!in.good())
        throw std::runtime_error("Failed to read data from: " + path);

    return mat;
}

IntMatrix load_ibin(const std::string& path) {
    std::ifstream in(path, std::ios::binary);
    if (!in.is_open())
        throw std::runtime_error("Cannot open file: " + path);

    uint32_t npts, dims;
    in.read(reinterpret_cast<char*>(&npts), 4);
    in.read(reinterpret_cast<char*>(&dims), 4);

    if (!in.good())
        throw std::runtime_error("Failed to read header from: " + path);

    size_t data_size = (size_t)npts * dims * sizeof(uint32_t);

    auto cur = in.tellg();
    in.seekg(0, std::ios::end);
    auto file_size = in.tellg();
    if ((size_t)(file_size - cur) < data_size)
        throw std::runtime_error("File too small: expected " +
            std::to_string(data_size) + " data bytes, file has " +
            std::to_string(file_size - cur) + " after header");
    in.seekg(cur);

    IntMatrix mat;
    mat.npts = npts;
    mat.dims = dims;
    mat.data = std::unique_ptr<uint32_t[], void(*)(void*)>(
        static_cast<uint32_t*>(aligned_alloc_wrapper(data_size)), std::free);

    in.read(reinterpret_cast<char*>(mat.data.get()), data_size);
    if (!in.good())
        throw std::runtime_error("Failed to read data from: " + path);

    return mat;
}
