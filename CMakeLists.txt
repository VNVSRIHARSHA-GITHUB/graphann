cmake_minimum_required(VERSION 3.14)
project(graphann LANGUAGES CXX)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_FLAGS_RELEASE "-O3 -march=native -DNDEBUG")
if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release)
endif()
find_package(OpenMP REQUIRED)

# ── original library (DO NOT CHANGE) ──────────────────────────────────────
add_library(graphann_lib STATIC
    src/distance.cpp
    src/io_utils.cpp
    src/vamana_index.cpp
)
target_include_directories(graphann_lib PUBLIC include)
target_link_libraries(graphann_lib PUBLIC OpenMP::OpenMP_CXX)

# original executables (DO NOT CHANGE)
add_executable(build_index src/build_index.cpp)
target_link_libraries(build_index PRIVATE graphann_lib)

add_executable(search_index src/search_index.cpp)
target_link_libraries(search_index PRIVATE graphann_lib)

# ── NEW: Improvement 1 — Vamana Refine ────────────────────────────────────
add_library(graphann_refine_lib STATIC
    src/distance.cpp
    src/io_utils.cpp
    src/vamana_refine.cpp
)
target_include_directories(graphann_refine_lib PUBLIC include)
target_link_libraries(graphann_refine_lib PUBLIC OpenMP::OpenMP_CXX)

add_executable(build_refine src/build_refine.cpp)
target_link_libraries(build_refine PRIVATE graphann_refine_lib)

# ── NEW: Improvement 2 — Multiple Entry Points ────────────────────────────
add_library(graphann_multistart_lib STATIC
    src/distance.cpp
    src/io_utils.cpp
    src/vamana_multistart.cpp
)
target_include_directories(graphann_multistart_lib PUBLIC include)
target_link_libraries(graphann_multistart_lib PUBLIC OpenMP::OpenMP_CXX)

add_executable(build_multistart src/build_multistart.cpp)
target_link_libraries(build_multistart PRIVATE graphann_multistart_lib)

# ── NEW: Improvement 3 — Two-Pass Construction ────────────────────────────
add_library(graphann_twopass_lib STATIC
    src/distance.cpp
    src/io_utils.cpp
    src/vamana_twopass.cpp
)
target_include_directories(graphann_twopass_lib PUBLIC include)
target_link_libraries(graphann_twopass_lib PUBLIC OpenMP::OpenMP_CXX)

add_executable(build_twopass src/build_twopass.cpp)
target_link_libraries(build_twopass PRIVATE graphann_twopass_lib)