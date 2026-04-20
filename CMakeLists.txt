cmake_minimum_required(VERSION 3.14)
project(graphann LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_FLAGS_RELEASE "-O3 -march=native -DNDEBUG")

if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release)
endif()

find_package(OpenMP REQUIRED)

# =============================================================================
# 1. Original Baseline
# =============================================================================
add_library(graphann_lib STATIC
    src/distance.cpp
    src/io_utils.cpp
    src/vamana_index.cpp
)
target_include_directories(graphann_lib PUBLIC include)
target_link_libraries(graphann_lib PUBLIC OpenMP::OpenMP_CXX)

add_executable(build_index src/build_index.cpp)
add_executable(search_index src/search_index.cpp)

# =============================================================================
# 2. Vamana Refine
# =============================================================================
add_library(graphann_refine_lib STATIC
    src/distance.cpp src/io_utils.cpp src/vamana_refine.cpp
)
target_include_directories(graphann_refine_lib PUBLIC include)
target_link_libraries(graphann_refine_lib PUBLIC OpenMP::OpenMP_CXX)

add_executable(build_refine src/build_refine.cpp)
add_executable(search_refine src/search_refine.cpp)

# =============================================================================
# 3. Multi-Start
# =============================================================================
add_library(graphann_multistart_lib STATIC
    src/distance.cpp src/io_utils.cpp src/vamana_multistart.cpp
)
target_include_directories(graphann_multistart_lib PUBLIC include)
target_link_libraries(graphann_multistart_lib PUBLIC OpenMP::OpenMP_CXX)

add_executable(build_multistart src/build_multistart.cpp)
add_executable(search_multistart src/search_multistart.cpp)

# =============================================================================
# 4. Two-Pass
# =============================================================================
add_library(graphann_twopass_lib STATIC
    src/distance.cpp src/io_utils.cpp src/vamana_twopass.cpp
)
target_include_directories(graphann_twopass_lib PUBLIC include)
target_link_libraries(graphann_twopass_lib PUBLIC OpenMP::OpenMP_CXX)

add_executable(build_twopass src/build_twopass.cpp)
add_executable(search_twopass src/search_twopass.cpp)

# =============================================================================
# 5. Beam Width
# =============================================================================
add_library(graphann_beam_lib STATIC
    src/distance.cpp src/io_utils.cpp src/vamana_beam.cpp
)
target_include_directories(graphann_beam_lib PUBLIC include)
target_link_libraries(graphann_beam_lib PUBLIC OpenMP::OpenMP_CXX)

add_executable(build_beam src/build_beam.cpp)
add_executable(search_beam src/search_beam.cpp)

# =============================================================================
# 6. Selective Backward Edges (you just added)
# =============================================================================
add_library(graphann_selective_lib STATIC
    src/distance.cpp src/io_utils.cpp src/vamana_selective.cpp
)
target_include_directories(graphann_selective_lib PUBLIC include)
target_link_libraries(graphann_selective_lib PUBLIC OpenMP::OpenMP_CXX)

add_executable(build_selective src/build_selective.cpp)
add_executable(search_selective src/search_selective.cpp)

# =============================================================================
# 7. Post-Build Rewiring
# =============================================================================
add_library(graphann_rewire_lib STATIC
    src/distance.cpp src/io_utils.cpp src/vamana_rewire.cpp
)
target_include_directories(graphann_rewire_lib PUBLIC include)
target_link_libraries(graphann_rewire_lib PUBLIC OpenMP::OpenMP_CXX)

add_executable(build_rewire src/build_rewire.cpp)
add_executable(search_rewire src/search_rewire.cpp)

# =============================================================================
# Link all executables correctly
# =============================================================================
target_link_libraries(build_index      PRIVATE graphann_lib)
target_link_libraries(search_index     PRIVATE graphann_lib)

target_link_libraries(build_refine     PRIVATE graphann_refine_lib)
target_link_libraries(search_refine    PRIVATE graphann_refine_lib)

target_link_libraries(build_multistart PRIVATE graphann_multistart_lib)
target_link_libraries(search_multistart PRIVATE graphann_multistart_lib)

target_link_libraries(build_twopass    PRIVATE graphann_twopass_lib)
target_link_libraries(search_twopass   PRIVATE graphann_twopass_lib)

target_link_libraries(build_beam       PRIVATE graphann_beam_lib)
target_link_libraries(search_beam      PRIVATE graphann_beam_lib)

target_link_libraries(build_selective  PRIVATE graphann_selective_lib)
target_link_libraries(search_selective PRIVATE graphann_selective_lib)

target_link_libraries(build_rewire     PRIVATE graphann_rewire_lib)
target_link_libraries(search_rewire    PRIVATE graphann_rewire_lib)

# Give all executables the include directory
foreach(tgt 
    build_index search_index
    build_refine search_refine
    build_multistart search_multistart
    build_twopass search_twopass
    build_beam search_beam
    build_selective search_selective
    build_rewire search_rewire)
    target_include_directories(${tgt} PRIVATE include)
endforeach()