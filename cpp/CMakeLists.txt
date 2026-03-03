#=============================================================================
# cmake-format: off
# SPDX-FileCopyrightText: Copyright (c) 2018-2026, NVIDIA CORPORATION.
# SPDX-License-Identifier: Apache-2.0
# cmake-format: on
#=============================================================================

# Modifications Copyright (c) 2025, AA-I Technologies Ltd.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
# NOTICE: This file has been modified by AA-I Technologies Ltd. from the original.
cmake_minimum_required(VERSION 3.30.4 FATAL_ERROR)

include(../cmake/rapids_config.cmake)
include(rapids-cmake)
include(rapids-cpm)
include(rapids-cuda)
include(rapids-export)
include(rapids-find)
include(CMakeDependentOption)

rapids_cuda_init_architectures(CUGRAPH)

project(CUGRAPH VERSION "${RAPIDS_VERSION}" LANGUAGES C CXX CUDA)

if(CMAKE_CXX_COMPILER_ID STREQUAL "GNU" AND
   CMAKE_CXX_COMPILER_VERSION VERSION_LESS 9.3)
    message(FATAL_ERROR "GCC compiler must be at least 9.3")
endif()

# Write the version header
rapids_cmake_write_version_file(include/cugraph/version_config.hpp)
rapids_cmake_write_version_file(include/cugraph_c/version_config.hpp)

################################################################################
# - build type -----------------------------------------------------------------

# Set a default build type if none was specified
rapids_cmake_build_type(Release)

################################################################################
# - User Options  --------------------------------------------------------------

option(BUILD_SHARED_LIBS "Build cuGraph shared libraries" ON)
option(BUILD_CUGRAPH_MG_TESTS "Build cuGraph multigpu algorithm tests" OFF)
option(CMAKE_CUDA_LINEINFO "Enable the -lineinfo option for nvcc (useful for cuda-memcheck / profiler" OFF)
option(BUILD_TESTS "Configure CMake to build tests" ON)
option(USE_RAFT_STATIC "Build raft as a static library" OFF)
option(CUGRAPH_COMPILE_RAFT_LIB "Compile the raft library instead of using it header-only" ON)
option(CUGRAPH_COMPILE_CUVS "Always compile cuVS from source" OFF)
cmake_dependent_option(CUGRAPH_USE_CUVS_STATIC
  "Build and statically link the cuVS library" ON
  "CUGRAPH_COMPILE_CUVS" OFF)

# AAI target GPU selection (required for AAI kernel specializations)
set(TARGET_GPU "" CACHE STRING "Target GPU for AAI kernels (A100, L4, A10G)")
set_property(CACHE TARGET_GPU PROPERTY STRINGS "A100" "L4" "A10G")

message(VERBOSE "CUGRAPH: CUGRAPH_COMPILE_CUVS=${CUGRAPH_COMPILE_CUVS}")
message(VERBOSE "CUGRAPH: CUGRAPH_USE_CUVS_STATIC=${CUGRAPH_USE_CUVS_STATIC}")

# Validate and configure TARGET_GPU
if(NOT TARGET_GPU)
    message(FATAL_ERROR "TARGET_GPU must be set. Use -DTARGET_GPU=<A100|L4|A10G>")
endif()

if(TARGET_GPU STREQUAL "A100")
    set(AAI_TARGET_GPU_VALUE "AAI_GPU_A100")
    set(AAI_TARGET_GPU_SUFFIX "a100")
elseif(TARGET_GPU STREQUAL "L4")
    set(AAI_TARGET_GPU_VALUE "AAI_GPU_L4")
    set(AAI_TARGET_GPU_SUFFIX "l4")
elseif(TARGET_GPU STREQUAL "A10G")
    set(AAI_TARGET_GPU_VALUE "AAI_GPU_A10G")
    set(AAI_TARGET_GPU_SUFFIX "a10g")
else()
    message(FATAL_ERROR "Invalid TARGET_GPU '${TARGET_GPU}'. Must be A100, L4, or A10G.")
endif()

message(STATUS "CUGRAPH: TARGET_GPU=${TARGET_GPU} (AAI kernels optimized for ${TARGET_GPU})")

################################################################################
# - compiler options -----------------------------------------------------------

# For now, disable CMake's automatic module scanning for C++ files. There is an sccache bug in the
# version RAPIDS uses in CI that causes it to handle the resulting -M* flags incorrectly with
# gcc>=14. We can remove this once we upgrade to a newer sccache version.
set(CMAKE_CXX_SCAN_FOR_MODULES OFF)

# CUDA runtime
rapids_cuda_init_runtime(USE_STATIC ON)

rapids_find_package(CUDAToolkit REQUIRED
    BUILD_EXPORT_SET    cugraph-exports
    INSTALL_EXPORT_SET  cugraph-exports
    )

set(CUGRAPH_C_FLAGS "")
set(CUGRAPH_CXX_FLAGS "")
set(CUGRAPH_CUDA_FLAGS "")

if(CMAKE_COMPILER_IS_GNUCXX)
    list(APPEND CUGRAPH_CXX_FLAGS -Werror -Wno-error=deprecated-declarations -Wno-deprecated-declarations -DRAFT_HIDE_DEPRECATION_WARNINGS)
endif(CMAKE_COMPILER_IS_GNUCXX)


message("-- Building for GPU_ARCHS = ${CMAKE_CUDA_ARCHITECTURES}")

list(APPEND CUGRAPH_CUDA_FLAGS --expt-extended-lambda --expt-relaxed-constexpr)
list(APPEND CUGRAPH_CUDA_FLAGS -Werror=cross-execution-space-call -Wno-deprecated-declarations -DRAFT_HIDE_DEPRECATION_WARNINGS -Xptxas=--disable-warnings)
list(APPEND CUGRAPH_CUDA_FLAGS -Xcompiler=-Wall,-Wno-error=sign-compare,-Wno-error=unused-but-set-variable)
include(${rapids-cmake-dir}/cuda/enable_fatbin_compression.cmake)
rapids_cuda_enable_fatbin_compression(VARIABLE CUGRAPH_CUDA_FLAGS TUNE_FOR rapids)

# Option to enable line info in CUDA device compilation to allow introspection when profiling /
# memchecking
if (CMAKE_CUDA_LINEINFO)
    list(APPEND CUGRAPH_CUDA_FLAGS -lineinfo)
endif()

# Debug options
if(CMAKE_BUILD_TYPE MATCHES Debug)
    message(STATUS "Building with debugging flags")
    list(APPEND CUGRAPH_CUDA_FLAGS -G -Xcompiler=-rdynamic)
endif()

###################################################################################################
# - find CPM based dependencies  ------------------------------------------------------------------

rapids_cpm_init()

###
# Linking to the `raft::raft` target implicitly links cugraph targets to the
# following public header-only  raft dependencies:
# * CCCL
# * RMM
# * GTest/GMock
#
# The CMakeLists.txt for each of these projects are properly configured
# to generate a build and install export-set, so reimplementing finding or
# fetching those targets in cuGraph is redundant (at best), and potentially
# error-prone if something about those targets change and our implementation
# lags behind.
###

# Need CCCL, then rmm, then cuCollections, then RAFT.
# This ensures that libraries can be overridden for testing.
include(cmake/thirdparty/get_cccl.cmake)
include(${rapids-cmake-dir}/cpm/rmm.cmake)
rapids_cpm_rmm(BUILD_EXPORT_SET cugraph-exports)
include(${rapids-cmake-dir}/cpm/cuco.cmake)
rapids_cpm_cuco(BUILD_EXPORT_SET cugraph-exports)
include(cmake/thirdparty/get_raft.cmake)

# Always include cuVS for spectral clustering enhancements
include(cmake/thirdparty/get_cuvs.cmake)

if (BUILD_CUGRAPH_MTMG_TESTS)
    if(NOT TARGET ucx::ucp)
        find_package(ucx REQUIRED)
    endif()

    if(NOT TARGET ucxx::ucxx)
        find_package(ucxx REQUIRED)
    endif()
endif()

if(BUILD_TESTS)
  include(${rapids-cmake-dir}/cpm/gtest.cmake)
  rapids_cpm_gtest(BUILD_STATIC)
endif()

################################################################################
# - libcugraph library target --------------------------------------------------

# NOTE: The most expensive compilations are listed first
#       since ninja will run them in parallel in this order,
#       which should give us a better parallel schedule.

################################################################################
# - AAI sources (GPU-specific implementations + integration layer) -------------
# AAI implementation files are organized by GPU target in src/aai/impl/<gpu>/
# Integration (glue) files are in src/aai/integration/

# Per-algorithm AAI routing control.
# AAI_ROUTED_ALGORITHMS: semicolon-separated list of algorithm IDs to route to AAI.
# Empty string (default) means ALL algorithms are routed. "NONE" disables all routing.
set(AAI_ROUTED_ALGORITHMS "" CACHE STRING
    "Algorithms to route to AAI (empty=all, NONE=disable). Example: BFS;SSSP;PAGERANK")

set(_AAI_ALL_ALGORITHMS
    BFS SSSP K_HOP_NBRS PAGERANK HITS
    BETWEENNESS_CENTRALITY EIGENVECTOR_CENTRALITY KATZ_CENTRALITY
    LOUVAIN LEIDEN ECG TRIANGLE_COUNT K_TRUSS EGONET
    SPECTRAL_MODULARITY_MAXIMIZATION
    ANALYZE_CLUSTERING_MODULARITY ANALYZE_CLUSTERING_EDGE_CUT ANALYZE_CLUSTERING_RATIO_CUT
    WCC SCC CORE_NUMBER K_CORE COSINE JACCARD OVERLAP SORENSEN MST
)

if("${AAI_ROUTED_ALGORITHMS}" STREQUAL "NONE")
    set(_aai_route_list "")
elseif("${AAI_ROUTED_ALGORITHMS}" STREQUAL "")
    set(_aai_route_list ${_AAI_ALL_ALGORITHMS})
else()
    set(_aai_route_list ${AAI_ROUTED_ALGORITHMS})
endif()

foreach(_algo ${_aai_route_list})
    list(FIND _AAI_ALL_ALGORITHMS "${_algo}" _idx)
    if(_idx EQUAL -1)
        message(FATAL_ERROR "Unknown AAI algorithm: ${_algo}. Valid: ${_AAI_ALL_ALGORITHMS}")
    endif()
endforeach()

# =========================================================================
# Mapping from algorithm routing names to impl file glob prefixes.
# Each algorithm maps to one or more filename prefixes (relative to the
# impl/<gpu>/<category>/ directory).  Only impl files matching a routed
# algorithm are compiled, avoiding duplicate-symbol linker errors from
# independently-generated solutions.
# =========================================================================
set(_AAI_IMPL_DIR "src/aai/impl/${AAI_TARGET_GPU_SUFFIX}")

set(_AAI_IMPL_GLOBS_BFS
    "${_AAI_IMPL_DIR}/traversal/bfs*")
set(_AAI_IMPL_GLOBS_SSSP
    "${_AAI_IMPL_DIR}/traversal/sssp*")
set(_AAI_IMPL_GLOBS_K_HOP_NBRS
    "${_AAI_IMPL_DIR}/traversal/k_hop_nbrs*")
set(_AAI_IMPL_GLOBS_PAGERANK
    "${_AAI_IMPL_DIR}/link_analysis/pagerank*")
set(_AAI_IMPL_GLOBS_HITS
    "${_AAI_IMPL_DIR}/link_analysis/hits*")
set(_AAI_IMPL_GLOBS_BETWEENNESS_CENTRALITY
    "${_AAI_IMPL_DIR}/centrality/betweenness_centrality*"
    "${_AAI_IMPL_DIR}/centrality/edge_betweenness_centrality*")
set(_AAI_IMPL_GLOBS_EIGENVECTOR_CENTRALITY
    "${_AAI_IMPL_DIR}/centrality/eigenvector_centrality*")
set(_AAI_IMPL_GLOBS_KATZ_CENTRALITY
    "${_AAI_IMPL_DIR}/centrality/katz_centrality*")
set(_AAI_IMPL_GLOBS_LOUVAIN
    "${_AAI_IMPL_DIR}/community/louvain*")
set(_AAI_IMPL_GLOBS_LEIDEN
    "${_AAI_IMPL_DIR}/community/leiden*")
set(_AAI_IMPL_GLOBS_ECG
    "${_AAI_IMPL_DIR}/community/ecg*")
set(_AAI_IMPL_GLOBS_TRIANGLE_COUNT
    "${_AAI_IMPL_DIR}/community/triangle_count*")
set(_AAI_IMPL_GLOBS_K_TRUSS
    "${_AAI_IMPL_DIR}/community/k_truss*")
set(_AAI_IMPL_GLOBS_EGONET
    "${_AAI_IMPL_DIR}/community/extract_ego*")
set(_AAI_IMPL_GLOBS_SPECTRAL_MODULARITY_MAXIMIZATION
    "${_AAI_IMPL_DIR}/community/spectral_modularity_maximization*")
set(_AAI_IMPL_GLOBS_ANALYZE_CLUSTERING_MODULARITY
    "${_AAI_IMPL_DIR}/community/analyze_clustering_modularity*")
set(_AAI_IMPL_GLOBS_ANALYZE_CLUSTERING_EDGE_CUT
    "${_AAI_IMPL_DIR}/community/analyze_clustering_edge_cut*")
set(_AAI_IMPL_GLOBS_ANALYZE_CLUSTERING_RATIO_CUT
    "${_AAI_IMPL_DIR}/community/analyze_clustering_ratio_cut*")
set(_AAI_IMPL_GLOBS_WCC
    "${_AAI_IMPL_DIR}/components/wcc*"
    "${_AAI_IMPL_DIR}/components/weakly_connected_components*")
set(_AAI_IMPL_GLOBS_SCC
    "${_AAI_IMPL_DIR}/components/scc*"
    "${_AAI_IMPL_DIR}/components/strongly_connected_components*")
set(_AAI_IMPL_GLOBS_CORE_NUMBER
    "${_AAI_IMPL_DIR}/cores/core_number*")
set(_AAI_IMPL_GLOBS_K_CORE
    "${_AAI_IMPL_DIR}/cores/k_core*")
set(_AAI_IMPL_GLOBS_COSINE
    "${_AAI_IMPL_DIR}/link_prediction/cosine*")
set(_AAI_IMPL_GLOBS_JACCARD
    "${_AAI_IMPL_DIR}/link_prediction/jaccard*")
set(_AAI_IMPL_GLOBS_OVERLAP
    "${_AAI_IMPL_DIR}/link_prediction/overlap*")
set(_AAI_IMPL_GLOBS_SORENSEN
    "${_AAI_IMPL_DIR}/link_prediction/sorensen*")
set(_AAI_IMPL_GLOBS_MST
    "${_AAI_IMPL_DIR}/tree/mst*")

# Collect impl sources only for routed algorithms
set(AAI_IMPL_SOURCES "")
foreach(_algo ${_aai_route_list})
    if(DEFINED _AAI_IMPL_GLOBS_${_algo})
        foreach(_pattern ${_AAI_IMPL_GLOBS_${_algo}})
            file(GLOB _matched "${_pattern}")
            foreach(_f ${_matched})
                # Only include .cu files, not .cu.flags
                if(_f MATCHES "\\.cu$")
                    list(APPEND AAI_IMPL_SOURCES "${_f}")
                endif()
            endforeach()
        endforeach()
    else()
        message(WARNING "AAI: No impl glob mapping for algorithm ${_algo}")
    endif()
endforeach()
list(REMOVE_DUPLICATES AAI_IMPL_SOURCES)
message(STATUS "AAI: ${_aai_route_list}")
message(STATUS "AAI: compiling ${CMAKE_CURRENT_LIST_DIR} impl sources: ${AAI_IMPL_SOURCES}")
list(LENGTH AAI_IMPL_SOURCES _n_impl)
message(STATUS "AAI: ${_n_impl} impl source files for routed algorithms")

# Glob AAI integration (glue) sources - always compiled
file(GLOB_RECURSE AAI_INTEGRATION_SOURCES
    "src/aai/integration/*.cu"
)

# Combine into AAI_SOURCES
set(AAI_SOURCES
    ${AAI_IMPL_SOURCES}
    ${AAI_INTEGRATION_SOURCES}
)

# Split AAI impl sources into regular and RDC-requiring lists.
# A file needs RDC if its .cu.flags sidecar contains "--rdc=true".
# RDC files are compiled into a separate static library (cugraph_aai_rdc)
# because CUDA_SEPARABLE_COMPILATION must be set at the target level.
set(AAI_RDC_SOURCES "")
set(_AAI_NON_RDC_IMPL_SOURCES "")
foreach(_cu_file ${AAI_IMPL_SOURCES})
    set(_needs_rdc FALSE)
    set(_extra_flags "")
    if(EXISTS "${_cu_file}.flags")
        file(STRINGS "${_cu_file}.flags" _extra_flags)
        if("--rdc=true" IN_LIST _extra_flags)
            set(_needs_rdc TRUE)
            list(REMOVE_ITEM _extra_flags "--rdc=true")
        endif()
    endif()
    if(_needs_rdc)
        list(APPEND AAI_RDC_SOURCES "${_cu_file}")
        if(_extra_flags)
            set_source_files_properties("${_cu_file}"
                PROPERTIES COMPILE_OPTIONS "$<$<COMPILE_LANGUAGE:CUDA>:${_extra_flags}>"
            )
        endif()
        message(STATUS "AAI: RDC file: ${_cu_file}")
    else()
        list(APPEND _AAI_NON_RDC_IMPL_SOURCES "${_cu_file}")
        if(_extra_flags)
            set_source_files_properties("${_cu_file}"
                PROPERTIES COMPILE_OPTIONS "$<$<COMPILE_LANGUAGE:CUDA>:${_extra_flags}>"
            )
            message(STATUS "AAI: extra flags for ${_cu_file}: ${_extra_flags}")
        endif()
    endif()
endforeach()

# Rebuild AAI_SOURCES without RDC files (those go into cugraph_aai_rdc)
set(AAI_SOURCES
    ${_AAI_NON_RDC_IMPL_SOURCES}
    ${AAI_INTEGRATION_SOURCES}
)

# RDC sub-library for AAI files that use cooperative kernels.
if(AAI_RDC_SOURCES)
    list(LENGTH AAI_RDC_SOURCES _n_rdc)
    message(STATUS "AAI: ${_n_rdc} RDC source files -> cugraph_aai_rdc")

    add_library(cugraph_aai_rdc STATIC ${AAI_RDC_SOURCES})
    set_target_properties(cugraph_aai_rdc PROPERTIES
        CUDA_SEPARABLE_COMPILATION  ON
        CUDA_RESOLVE_DEVICE_SYMBOLS ON
        CUDA_STANDARD               20
        CUDA_STANDARD_REQUIRED      ON
        POSITION_INDEPENDENT_CODE   ON
    )
    target_compile_options(cugraph_aai_rdc
        PRIVATE "$<$<COMPILE_LANGUAGE:CUDA>:${CUGRAPH_CUDA_FLAGS}>"
    )
    target_compile_definitions(cugraph_aai_rdc
        PRIVATE AAI_TARGET_GPU=${AAI_TARGET_GPU_VALUE}
                CUDA_API_PER_THREAD_DEFAULT_STREAM
    )
    target_include_directories(cugraph_aai_rdc PRIVATE
        "${CMAKE_CURRENT_SOURCE_DIR}/../thirdparty"
        "${CMAKE_CURRENT_SOURCE_DIR}/src"
        "${CMAKE_CURRENT_SOURCE_DIR}/include"
    )
    target_link_libraries(cugraph_aai_rdc
        PRIVATE
            rmm::rmm
            raft::raft
    )
endif()

set(CUGRAPH_SOURCES
    src/sampling/temporal_sampling_sg_v32_e32.cu
    src/sampling/temporal_sampling_sg_v64_e64.cu
    src/sampling/temporal_sampling_mg_v32_e32.cu
    src/sampling/temporal_sampling_mg_v64_e64.cu
    src/sampling/detail/update_temporal_edge_mask_sg_v32_e32.cu
    src/sampling/detail/update_temporal_edge_mask_sg_v64_e64.cu
    src/sampling/detail/update_temporal_edge_mask_mg_v32_e32.cu
    src/sampling/detail/update_temporal_edge_mask_mg_v64_e64.cu
    src/utilities/shuffle_vertices_mg_v32.cu
    src/utilities/shuffle_vertices_mg_v64.cu
    src/utilities/shuffle_properties_mg.cu
    src/utilities/validation_checks.cu
    src/detail/permute_range_v32.cu
    src/detail/permute_range_v64.cu
    src/utilities/shuffle_vertex_pairs_mg_v32_e32.cu
    src/utilities/shuffle_vertex_pairs_mg_v64_e64.cu
    src/utilities/shuffle_local_edge_srcs_dsts_mg_v32.cu
    src/utilities/shuffle_local_edge_srcs_dsts_mg_v64.cu
    src/detail/collect_local_vertex_values_sg_v32_e32.cu
    src/detail/collect_local_vertex_values_sg_v64_e64.cu
    src/detail/collect_local_vertex_values_mg_v32_e32.cu
    src/detail/collect_local_vertex_values_mg_v64_e64.cu
    src/detail/groupby_and_count_mg_v32_e32.cu
    src/detail/groupby_and_count_mg_v64_e64.cu
    src/detail/collect_comm_wrapper_mg_v32_e32.cu
    src/detail/collect_comm_wrapper_mg_v64_e64.cu
    src/sampling/detail/conversion_utilities.cu
    src/sampling/random_walks_mg_v64_e64.cu
    src/sampling/random_walks_mg_v32_e32.cu
    src/community/detail/common_methods_mg_v64_e64.cu
    src/community/detail/common_methods_mg_v32_e32.cu
    src/community/detail/common_methods_sg_v64_e64.cu
    src/community/detail/common_methods_sg_v32_e32.cu
    src/community/detail/refine_sg_v64_e64.cu
    src/community/detail/refine_sg_v32_e32.cu
    src/community/detail/refine_mg_v64_e64.cu
    src/community/detail/refine_mg_v32_e32.cu
    src/community/edge_triangle_count_sg_v64_e64.cu
    src/community/edge_triangle_count_sg_v32_e32.cu
    src/community/edge_triangle_count_mg_v64_e64.cu
    src/community/edge_triangle_count_mg_v32_e32.cu
    src/community/detail/maximal_independent_moves_sg_v64_e64.cu
    src/community/detail/maximal_independent_moves_sg_v32_e32.cu
    src/community/detail/maximal_independent_moves_mg_v64_e64.cu
    src/community/detail/maximal_independent_moves_mg_v32_e32.cu
    src/detail/utility_wrappers_32.cu
    src/detail/utility_wrappers_64.cu
    src/structure/graph_view_mg_v64_e64.cu
    src/structure/graph_view_mg_v32_e32.cu
    src/structure/remove_self_loops_sg_v32_e32.cu
    src/structure/remove_self_loops_sg_v64_e64.cu
    src/structure/remove_multi_edges_sg_v32_e32.cu
    src/structure/remove_multi_edges_sg_v64_e64.cu
    src/utilities/path_retrieval_sg_v32_e32.cu
    src/utilities/path_retrieval_sg_v64_e64.cu
    src/structure/legacy/graph.cu
    src/linear_assignment/legacy/hungarian.cu
    src/link_prediction/jaccard_sg_v64_e64.cu
    src/link_prediction/sorensen_sg_v64_e64.cu
    src/link_prediction/overlap_sg_v64_e64.cu
    src/link_prediction/cosine_sg_v64_e64.cu
    src/link_prediction/jaccard_mg_v64_e64.cu
    src/link_prediction/jaccard_mg_v32_e32.cu
    src/link_prediction/sorensen_mg_v64_e64.cu
    src/link_prediction/sorensen_mg_v32_e32.cu
    src/link_prediction/overlap_mg_v64_e64.cu
    src/link_prediction/overlap_mg_v32_e32.cu
    src/link_prediction/cosine_mg_v64_e64.cu
    src/link_prediction/cosine_mg_v32_e32.cu
    src/layout/legacy/force_atlas2.cu
    src/converters/legacy/COOtoCSR.cu
    src/community/louvain_sg_v64_e64.cu
    src/community/louvain_mg_v64_e64.cu
    src/community/louvain_mg_v32_e32.cu
    src/community/leiden_sg_v64_e64.cu
    src/community/leiden_mg_v64_e64.cu
    src/community/leiden_mg_v32_e32.cu
    src/community/ecg_sg_v64_e64.cu
    src/community/ecg_mg_v64_e64.cu
    src/community/ecg_mg_v32_e32.cu
    src/community/egonet_sg_v64_e64.cu
    src/community/egonet_mg_v64_e64.cu
    src/community/egonet_mg_v32_e32.cu
    src/community/k_truss_sg_v64_e64.cu
    src/community/k_truss_mg_v64_e64.cu
    src/community/k_truss_mg_v32_e32.cu
    src/lookup/lookup_src_dst_mg_v32_e32.cu
    src/lookup/lookup_src_dst_mg_v64_e64.cu
    src/lookup/lookup_src_dst_sg_v32_e32.cu
    src/lookup/lookup_src_dst_sg_v64_e64.cu
    src/sampling/random_walks_sg_v64_e64.cu
    src/sampling/random_walks_sg_v32_e32.cu
    src/sampling/detail/prepare_next_frontier_sg_v64_e64.cu
    src/sampling/detail/prepare_next_frontier_sg_v32_e32.cu
    src/sampling/detail/prepare_next_frontier_mg_v64_e64.cu
    src/sampling/detail/prepare_next_frontier_mg_v32_e32.cu
    src/sampling/detail/remove_visited_vertices_from_frontier_sg_v32_e32.cu
    src/sampling/detail/remove_visited_vertices_from_frontier_sg_v64_e64.cu
    src/sampling/detail/check_edge_bias_values_sg_v64_e64.cu
    src/sampling/detail/check_edge_bias_values_sg_v32_e32.cu
    src/sampling/detail/check_edge_bias_values_mg_v64_e64.cu
    src/sampling/detail/check_edge_bias_values_mg_v32_e32.cu
    src/sampling/detail/temporal_partition_vertices_v64.cu
    src/sampling/detail/temporal_partition_vertices_v32.cu
    src/sampling/detail/gather_one_hop_mg_v32_e32.cu
    src/sampling/detail/gather_one_hop_mg_v64_e64.cu
    src/sampling/detail/gather_one_hop_sg_v32_e32.cu
    src/sampling/detail/gather_one_hop_sg_v64_e64.cu
    src/sampling/detail/gather_sampled_properties.cu
    src/sampling/detail/sample_edges_sg_v64_e64.cu
    src/sampling/detail/sample_edges_sg_v32_e32.cu
    src/sampling/detail/sample_edges_mg_v64_e64.cu
    src/sampling/detail/sample_edges_mg_v32_e32.cu
    src/sampling/detail/temporal_sample_edges_sg_v64_e64.cu
    src/sampling/detail/temporal_sample_edges_sg_v32_e32.cu
    src/sampling/detail/temporal_sample_edges_mg_v64_e64.cu
    src/sampling/detail/temporal_sample_edges_mg_v32_e32.cu
    src/sampling/detail/shuffle_and_organize_output.cu
    src/sampling/neighbor_sampling_mg_v32_e32.cu
    src/sampling/neighbor_sampling_mg_v64_e64.cu
    src/sampling/neighbor_sampling_sg_v32_e32.cu
    src/sampling/neighbor_sampling_sg_v64_e64.cu
    src/sampling/negative_sampling_sg_v32_e32.cu
    src/sampling/negative_sampling_sg_v64_e64.cu
    src/sampling/negative_sampling_mg_v32_e32.cu
    src/sampling/negative_sampling_mg_v64_e64.cu
    src/sampling/sampling_post_processing_sg_v64_e64.cu
    src/sampling/sampling_post_processing_sg_v32_e32.cu
    src/cores/core_number_sg_v64_e64.cu
    src/cores/core_number_mg_v64_e64.cu
    src/cores/core_number_mg_v32_e32.cu
    src/cores/k_core_sg_v64_e64.cu
    src/cores/k_core_mg_v64_e64.cu
    src/cores/k_core_mg_v32_e32.cu
    src/generators/generate_rmat_edgelist_sg_v32_e32.cu
    src/generators/generate_rmat_edgelist_sg_v64_e64.cu
    src/generators/generate_bipartite_rmat_edgelist_sg_v32_e32.cu
    src/generators/generate_bipartite_rmat_edgelist_sg_v64_e64.cu
    src/generators/generator_tools_sg_v32_e32.cu
    src/generators/generator_tools_sg_v64_e64.cu
    src/generators/simple_generators_sg_v32_e32.cu
    src/generators/simple_generators_sg_v64_e64.cu
    src/generators/erdos_renyi_generator_sg_v32_e32.cu
    src/generators/erdos_renyi_generator_sg_v64_e64.cu
    src/structure/graph_sg_v64_e64.cu
    src/structure/graph_sg_v32_e32.cu
    src/structure/graph_mg_v64_e64.cu
    src/structure/graph_mg_v32_e32.cu
    src/structure/graph_view_sg_v64_e64.cu
    src/structure/graph_view_sg_v32_e32.cu
    src/structure/decompress_to_edgelist_sg_v64_e64.cu
    src/structure/decompress_to_edgelist_sg_v32_e32.cu
    src/structure/decompress_to_edgelist_mg_v64_e64.cu
    src/structure/decompress_to_edgelist_mg_v32_e32.cu
    src/structure/symmetrize_graph_sg_v64_e64.cu
    src/structure/symmetrize_graph_sg_v32_e32.cu
    src/structure/symmetrize_graph_mg_v64_e64.cu
    src/structure/symmetrize_graph_mg_v32_e32.cu
    src/structure/transpose_graph_sg_v64_e64.cu
    src/structure/transpose_graph_sg_v32_e32.cu
    src/structure/transpose_graph_mg_v64_e64.cu
    src/structure/transpose_graph_mg_v32_e32.cu
    src/structure/transpose_graph_storage_sg_v64_e64.cu
    src/structure/transpose_graph_storage_sg_v32_e32.cu
    src/structure/transpose_graph_storage_mg_v64_e64.cu
    src/structure/transpose_graph_storage_mg_v32_e32.cu
    src/structure/coarsen_graph_sg_v64_e64.cu
    src/structure/coarsen_graph_sg_v32_e32.cu
    src/structure/coarsen_graph_mg_v64_e64.cu
    src/structure/coarsen_graph_mg_v32_e32.cu
    src/structure/graph_weight_utils_mg_v64_e64.cu
    src/structure/graph_weight_utils_mg_v32_e32.cu
    src/structure/graph_weight_utils_sg_v64_e64.cu
    src/structure/graph_weight_utils_sg_v32_e32.cu
    src/structure/renumber_edgelist_sg_v64_e64.cu
    src/structure/renumber_edgelist_sg_v32_e32.cu
    src/structure/renumber_edgelist_mg_v64_e64.cu
    src/structure/renumber_edgelist_mg_v32_e32.cu
    src/structure/renumber_utils_sg_v64_e64.cu
    src/structure/renumber_utils_sg_v32_e32.cu
    src/structure/renumber_utils_mg_v64_e64.cu
    src/structure/renumber_utils_mg_v32_e32.cu
    src/structure/relabel_sg_v64_e64.cu
    src/structure/relabel_sg_v32_e32.cu
    src/structure/relabel_mg_v64_e64.cu
    src/structure/relabel_mg_v32_e32.cu
    src/structure/induced_subgraph_sg_v64_e64.cu
    src/structure/induced_subgraph_sg_v32_e32.cu
    src/structure/induced_subgraph_mg_v64_e64.cu
    src/structure/induced_subgraph_mg_v32_e32.cu
    src/structure/select_random_vertices_sg_v64_e64.cu
    src/structure/select_random_vertices_sg_v32_e32.cu
    src/structure/select_random_vertices_mg_v64_e64.cu
    src/structure/select_random_vertices_mg_v32_e32.cu
    src/traversal/extract_bfs_paths_sg_v64_e64.cu
    src/traversal/extract_bfs_paths_sg_v32_e32.cu
    src/traversal/extract_bfs_paths_mg_v64_e64.cu
    src/traversal/extract_bfs_paths_mg_v32_e32.cu
    src/traversal/bfs_sg_v64_e64.cu
    src/traversal/bfs_mg_v64_e64.cu
    src/traversal/bfs_mg_v32_e32.cu
    src/traversal/sssp_sg_v64_e64.cu
    src/traversal/od_shortest_distances_sg_v64_e64.cu
    src/traversal/od_shortest_distances_sg_v32_e32.cu
    src/traversal/sssp_mg_v64_e64.cu
    src/traversal/sssp_mg_v32_e32.cu
    src/link_analysis/hits_sg_v64_e64.cu
    src/link_analysis/hits_mg_v64_e64.cu
    src/link_analysis/hits_mg_v32_e32.cu
    src/link_analysis/pagerank_sg_v64_e64.cu
    src/link_analysis/pagerank_mg_v64_e64.cu
    src/link_analysis/pagerank_mg_v32_e32.cu
    src/centrality/katz_centrality_sg_v64_e64.cu
    src/centrality/katz_centrality_mg_v64_e64.cu
    src/centrality/katz_centrality_mg_v32_e32.cu
    src/centrality/eigenvector_centrality_sg_v64_e64.cu
    src/centrality/eigenvector_centrality_mg_v64_e64.cu
    src/centrality/eigenvector_centrality_mg_v32_e32.cu
    src/centrality/betweenness_centrality_sg_v64_e64.cu
    src/centrality/betweenness_centrality_mg_v64_e64.cu
    src/centrality/betweenness_centrality_mg_v32_e32.cu
    src/centrality/edge_betweenness_centrality_sg_v32_e32.cu
    src/components/weakly_connected_components_sg_v64_e64.cu
    src/components/weakly_connected_components_mg_v64_e64.cu
    src/components/weakly_connected_components_mg_v32_e32.cu
    src/components/mis_sg_v64_e64.cu
    src/components/mis_sg_v32_e32.cu
    src/components/mis_mg_v64_e64.cu
    src/components/mis_mg_v32_e32.cu
    src/components/vertex_coloring_sg_v64_e64.cu
    src/components/vertex_coloring_sg_v32_e32.cu
    src/components/vertex_coloring_mg_v64_e64.cu
    src/components/vertex_coloring_mg_v32_e32.cu
    src/structure/create_graph_from_edgelist_sg_v64_e64.cu
    src/structure/create_graph_from_edgelist_sg_v32_e32.cu
    src/structure/create_graph_from_edgelist_mg_v64_e64.cu
    src/structure/create_graph_from_edgelist_mg_v32_e32.cu
    src/structure/symmetrize_edgelist_sg_v64_e64.cu
    src/structure/symmetrize_edgelist_sg_v32_e32.cu
    src/structure/symmetrize_edgelist_mg_v64_e64.cu
    src/structure/symmetrize_edgelist_mg_v32_e32.cu
    src/community/triangle_count_sg_v64_e64.cu
    src/community/triangle_count_mg_v64_e64.cu
    src/community/triangle_count_mg_v32_e32.cu
    src/community/approx_weighted_matching_sg_v64_e64.cu
    src/community/approx_weighted_matching_sg_v32_e32.cu
    src/community/approx_weighted_matching_mg_v64_e64.cu
    src/community/approx_weighted_matching_mg_v32_e32.cu
    src/traversal/k_hop_nbrs_sg_v64_e64.cu
    src/traversal/k_hop_nbrs_mg_v64_e64.cu
    src/traversal/k_hop_nbrs_mg_v32_e32.cu
    src/mtmg/vertex_result_sg_v32_e32.cu
    src/mtmg/vertex_result_sg_v64_e64.cu
    src/mtmg/vertex_result_mg_v32_e32.cu
    src/mtmg/vertex_result_mg_v64_e64.cu
    src/mtmg/vertex_pairs_result_sg_v32_e32.cu
    src/mtmg/vertex_pairs_result_sg_v64_e64.cu
    src/mtmg/vertex_pairs_result_mg_v32_e32.cu
    src/mtmg/vertex_pairs_result_mg_v64_e64.cu
    # AAI GPU-specific kernels (selected by TARGET_GPU)
    ${AAI_SOURCES}
)

add_library(cugraph ${CUGRAPH_SOURCES})

set_target_properties(cugraph
    PROPERTIES BUILD_RPATH                         "\$ORIGIN"
               INSTALL_RPATH                       "\$ORIGIN"
               # set target compile options
               CXX_STANDARD                        20
               CXX_STANDARD_REQUIRED               ON
               CUDA_STANDARD                       20
               CUDA_STANDARD_REQUIRED              ON
               POSITION_INDEPENDENT_CODE           ON
               INTERFACE_POSITION_INDEPENDENT_CODE ON
)

target_compile_options(cugraph
            PRIVATE "$<$<COMPILE_LANGUAGE:CXX>:${CUGRAPH_CXX_FLAGS}>"
                    "$<$<COMPILE_LANGUAGE:CUDA>:${CUGRAPH_CUDA_FLAGS}>"
)

# Per-thread default stream option see https://docs.nvidia.com/cuda/cuda-runtime-api/stream-sync-behavior.html
# The per-thread default stream does not synchronize with other streams
target_compile_definitions(cugraph PUBLIC CUDA_API_PER_THREAD_DEFAULT_STREAM)

# AAI target GPU definition
target_compile_definitions(cugraph PRIVATE AAI_TARGET_GPU=${AAI_TARGET_GPU_VALUE})

# Apply per-algorithm AAI routing compile definitions per-file to avoid cache invalidation
# Map algorithm names to their integration file paths
set(_algo_file_map
    "BFS:traversal/bfs.cu"
    "SSSP:traversal/sssp.cu"
    "K_HOP_NBRS:traversal/k_hop_nbrs.cu"
    "PAGERANK:link_analysis/pagerank.cu"
    "HITS:link_analysis/hits.cu"
    "BETWEENNESS_CENTRALITY:centrality/betweenness_centrality.cu"
    "EIGENVECTOR_CENTRALITY:centrality/eigenvector_centrality.cu"
    "KATZ_CENTRALITY:centrality/katz_centrality.cu"
    "LOUVAIN:community/louvain.cu"
    "LEIDEN:community/leiden.cu"
    "ECG:community/ecg.cu"
    "TRIANGLE_COUNT:community/triangle_count.cu"
    "K_TRUSS:community/k_truss.cu"
    "EGONET:community/egonet.cu"
    "SPECTRAL_MODULARITY_MAXIMIZATION:community/spectral.cu"
    "ANALYZE_CLUSTERING_MODULARITY:community/spectral.cu"
    "ANALYZE_CLUSTERING_EDGE_CUT:community/spectral.cu"
    "ANALYZE_CLUSTERING_RATIO_CUT:community/spectral.cu"
    "WCC:components/wcc.cu"
    "SCC:components/scc_legacy.cu"
    "CORE_NUMBER:cores/core_number.cu"
    "K_CORE:cores/k_core.cu"
    "COSINE:link_prediction/cosine.cu"
    "JACCARD:link_prediction/jaccard.cu"
    "OVERLAP:link_prediction/overlap.cu"
    "SORENSEN:link_prediction/sorensen.cu"
    "MST:tree/mst.cu"
)

foreach(_algo ${_aai_route_list})
    # Find the integration file for this algorithm
    set(_found FALSE)
    foreach(_mapping ${_algo_file_map})
        string(REGEX MATCH "^([^:]+):(.+)$" _ ${_mapping})
        set(_map_algo ${CMAKE_MATCH_1})
        set(_map_file ${CMAKE_MATCH_2})
        if(_algo STREQUAL _map_algo)
            set(_integration_file "${CMAKE_CURRENT_SOURCE_DIR}/src/aai/integration/${_map_file}")
            if(EXISTS ${_integration_file})
                set_property(SOURCE ${_integration_file} APPEND PROPERTY COMPILE_DEFINITIONS "AAI_ROUTE_${_algo}")
                set(_found TRUE)
            endif()
            break()
        endif()
    endforeach()
    if(NOT _found)
        message(FATAL_ERROR "No integration file mapping found for algorithm: ${_algo}. "
                            "Please add mapping to _algo_file_map in CMakeLists.txt")
    endif()
endforeach()
message(STATUS "AAI routed algorithms: ${_aai_route_list}")

file(WRITE "${CUGRAPH_BINARY_DIR}/fatbin.ld"
[=[
SECTIONS
{
  .nvFatBinSegment : { *(.nvFatBinSegment) }
  .nv_fatbin : { *(.nv_fatbin) }
}
]=])
target_link_options(cugraph PRIVATE "${CUGRAPH_BINARY_DIR}/fatbin.ld")

add_library(cugraph::cugraph ALIAS cugraph)

################################################################################
# - include paths --------------------------------------------------------------

target_include_directories(cugraph
    PRIVATE
        "${CMAKE_CURRENT_SOURCE_DIR}/../thirdparty"
        "${CMAKE_CURRENT_SOURCE_DIR}/src"
    PUBLIC
        "$<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>"
        "$<INSTALL_INTERFACE:include>"
)

set(COMPILED_RAFT_LIB )
if(CUGRAPH_COMPILE_RAFT_LIB)
  set(COMPILED_RAFT_LIB raft::compiled)
  if(USE_RAFT_STATIC)
    set(COMPILED_RAFT_LIB raft::compiled_static)
  endif()
endif()

set(CUVS_LIB cuvs::cuvs)
if(CUGRAPH_USE_CUVS_STATIC)
  set(CUVS_LIB cuvs::cuvs_static)
endif()

################################################################################
# - link libraries -------------------------------------------------------------
target_link_libraries(cugraph
    PUBLIC
        rmm::rmm
        raft::raft
        $<BUILD_LOCAL_INTERFACE:CUDA::toolkit>
    PRIVATE
        ${COMPILED_RAFT_LIB}
        ${CUVS_LIB}
        cuco::cuco
        $<TARGET_NAME_IF_EXISTS:cugraph_aai_rdc>
    )

################################################################################
# - C-API library --------------------------------------------------------------

add_library(cugraph_c
        src/c_api/resource_handle.cpp
        src/c_api/array.cpp
        src/c_api/degrees.cu
        src/c_api/degrees_result.cpp
        src/c_api/error.cpp
        src/c_api/graph_sg.cpp
        src/c_api/graph_mg.cpp
        src/c_api/graph_functions.cpp
        src/c_api/pagerank.cpp
        src/c_api/katz.cpp
        src/c_api/centrality_result.cpp
        src/c_api/eigenvector_centrality.cpp
        src/c_api/betweenness_centrality.cpp
        src/c_api/core_number.cpp
        src/c_api/k_truss.cpp
        src/c_api/core_result.cpp
        src/c_api/extract_ego.cpp
        src/c_api/ecg.cpp
        src/c_api/k_core.cpp
        src/c_api/hierarchical_clustering_result.cpp
        src/c_api/induced_subgraph.cpp
        src/c_api/capi_helper.cu
        src/c_api/legacy_spectral.cpp
        src/c_api/graph_helper_sg.cu
        src/c_api/graph_helper_mg.cu
        src/c_api/graph_generators.cpp
        src/c_api/induced_subgraph_result.cpp
        src/c_api/hits.cpp
        src/c_api/bfs.cpp
        src/c_api/sssp.cpp
        src/c_api/extract_paths.cpp
        src/c_api/random_walks.cpp
        src/c_api/random.cpp
        src/c_api/similarity.cpp
        src/c_api/leiden.cpp
        src/c_api/lookup_src_dst.cpp
        src/c_api/louvain.cpp
        src/c_api/triangle_count.cpp
        src/c_api/neighbor_sampling.cpp
        src/c_api/sampling_result.cpp
        src/c_api/temporal_neighbor_sampling.cpp
        src/c_api/negative_sampling.cpp
        src/c_api/labeling_result.cpp
        src/c_api/weakly_connected_components.cpp
        src/c_api/strongly_connected_components.cpp
        src/c_api/allgather.cpp
        src/c_api/decompress_to_edgelist.cpp
        src/c_api/edgelist.cpp
        src/c_api/renumber_arbitrary_edgelist.cu
        src/c_api/legacy_fa2.cpp
        src/c_api/legacy_mst.cpp
        src/c_api/extract_vertex_list.cpp
        )
add_library(cugraph::cugraph_c ALIAS cugraph_c)

# Currently presuming we aren't calling any CUDA kernels in cugraph_c

set_target_properties(cugraph_c
    PROPERTIES BUILD_RPATH                         "\$ORIGIN"
               INSTALL_RPATH                       "\$ORIGIN"
               # set target compile options
               CXX_STANDARD                        20
               CXX_STANDARD_REQUIRED               ON
               CUDA_STANDARD                       20
               CUDA_STANDARD_REQUIRED              ON
               POSITION_INDEPENDENT_CODE           ON
               INTERFACE_POSITION_INDEPENDENT_CODE ON
)

target_compile_options(cugraph_c
             PRIVATE "$<$<COMPILE_LANGUAGE:CXX>:${CUGRAPH_CXX_FLAGS}>"
                     "$<$<COMPILE_LANGUAGE:CUDA>:${CUGRAPH_CUDA_FLAGS}>"
)

# Per-thread default stream option see https://docs.nvidia.com/cuda/cuda-runtime-api/stream-sync-behavior.html
# The per-thread default stream does not synchronize with other streams
target_compile_definitions(cugraph_c PUBLIC CUDA_API_PER_THREAD_DEFAULT_STREAM)

target_link_options(cugraph_c PRIVATE "${CUGRAPH_BINARY_DIR}/fatbin.ld")

################################################################################
# - C-API include paths --------------------------------------------------------
target_include_directories(cugraph_c
    PRIVATE
        "${CMAKE_CURRENT_SOURCE_DIR}/src"
    PUBLIC
        "$<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>"
        "$<INSTALL_INTERFACE:include>"
)

################################################################################
# - C-API link libraries -------------------------------------------------------
target_link_libraries(cugraph_c PRIVATE cugraph::cugraph)

################################################################################
# - generate tests -------------------------------------------------------------

if(BUILD_TESTS)
  include(CTest)
  add_subdirectory(tests)
endif()

################################################################################
# - install targets ------------------------------------------------------------
rapids_cmake_install_lib_dir( lib_dir )
include(CPack)

install(TARGETS cugraph
        DESTINATION ${lib_dir}
        EXPORT cugraph-exports)

install(DIRECTORY include/cugraph/
        DESTINATION include/cugraph)

install(FILES ${CMAKE_CURRENT_BINARY_DIR}/include/cugraph/version_config.hpp
        DESTINATION include/cugraph)

install(TARGETS cugraph_c
        DESTINATION ${lib_dir}
        EXPORT cugraph-exports)

install(DIRECTORY include/cugraph_c/
        DESTINATION include/cugraph_c)

install(FILES ${CMAKE_CURRENT_BINARY_DIR}/include/cugraph_c/version_config.hpp
        DESTINATION include/cugraph_c)

################################################################################
# - install export -------------------------------------------------------------

set(doc_string
[=[
Provide targets for cuGraph.

cuGraph library is a collection of GPU accelerated graph algorithms that process data found in
[GPU DataFrames](https://github.com/rapidsai/cudf).

]=])

rapids_export(INSTALL cugraph
    EXPORT_SET cugraph-exports
    GLOBAL_TARGETS cugraph cugraph_c
    NAMESPACE cugraph::
    DOCUMENTATION doc_string
    )

################################################################################
# - build export ---------------------------------------------------------------
rapids_export(BUILD cugraph
    EXPORT_SET cugraph-exports
    GLOBAL_TARGETS cugraph cugraph_c
    NAMESPACE cugraph::
    DOCUMENTATION doc_string
    )

################################################################################
# - make documentation ---------------------------------------------------------
# requires doxygen and graphviz to be installed
# from build directory, run make docs_cugraph

# doc targets for cugraph
find_package(Doxygen 1.8.11)
if(Doxygen_FOUND)
    add_custom_command(OUTPUT CUGRAPH_DOXYGEN
                       WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/doxygen
                       COMMAND ${CMAKE_COMMAND} -E env "RAPIDS_VERSION_MAJOR_MINOR=${RAPIDS_VERSION_MAJOR_MINOR}" doxygen Doxyfile
                       VERBATIM)

    add_custom_target(docs_cugraph DEPENDS CUGRAPH_DOXYGEN)
endif()
