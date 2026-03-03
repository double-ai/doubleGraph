/*
 * Copyright (c) 2025, AA-I Technologies Ltd.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */
// Include original implementation for the obsolete void signatures
#include "link_analysis/pagerank_impl.cuh"

namespace cugraph {

// =============================================================================
// Obsolete void signatures - keep as template instantiations (original impl)
// =============================================================================

template void pagerank(raft::handle_t const& handle,
                       graph_view_t<int32_t, int32_t, true, false> const& graph_view,
                       std::optional<edge_property_view_t<int32_t, float const*>> edge_weight_view,
                       std::optional<float const*> precomputed_vertex_out_weight_sums,
                       std::optional<int32_t const*> personalization_vertices,
                       std::optional<float const*> personalization_values,
                       std::optional<int32_t> personalization_vector_size,
                       float* pageranks,
                       float alpha,
                       float epsilon,
                       size_t max_iterations,
                       bool has_initial_guess,
                       bool do_expensive_check);

template void pagerank(raft::handle_t const& handle,
                       graph_view_t<int32_t, int32_t, true, false> const& graph_view,
                       std::optional<edge_property_view_t<int32_t, double const*>> edge_weight_view,
                       std::optional<double const*> precomputed_vertex_out_weight_sums,
                       std::optional<int32_t const*> personalization_vertices,
                       std::optional<double const*> personalization_values,
                       std::optional<int32_t> personalization_vector_size,
                       double* pageranks,
                       double alpha,
                       double epsilon,
                       size_t max_iterations,
                       bool has_initial_guess,
                       bool do_expensive_check);

}  // namespace cugraph

#ifdef AAI_ROUTE_PAGERANK

#include <cugraph/aai/algorithms.hpp>
#include <cugraph/utilities/error.hpp>

#include <cuda_runtime.h>

namespace cugraph {

// =============================================================================
// Newer centrality_algorithm_metadata_t signatures - specialize to route to AAI
// =============================================================================

// Float specialization (4 routing paths)
template <>
std::tuple<rmm::device_uvector<float>, centrality_algorithm_metadata_t>
pagerank<int32_t, int32_t, float, float, false>(
    raft::handle_t const& handle,
    graph_view_t<int32_t, int32_t, true, false> const& graph_view,
    std::optional<edge_property_view_t<int32_t, float const*>> edge_weight_view,
    std::optional<raft::device_span<float const>> precomputed_vertex_out_weight_sums,
    std::optional<std::tuple<raft::device_span<int32_t const>, raft::device_span<float const>>>
        personalization,
    std::optional<raft::device_span<float const>> initial_pageranks,
    float alpha,
    float epsilon,
    size_t max_iterations,
    bool do_expensive_check)
{
  auto compact_graph = aai::graph32_t::from_graph_view(graph_view);

  // Preconditions (match original pagerank_impl.cuh)
  CUGRAPH_EXPECTS(compact_graph.is_csc, "PageRank requires CSC format (is_csc=true)");
  CUGRAPH_EXPECTS((alpha >= 0.0f) && (alpha <= 1.0f),
                  "Invalid input argument: alpha should be in [0.0, 1.0].");
  CUGRAPH_EXPECTS(epsilon >= 0.0f,
                  "Invalid input argument: epsilon should be non-negative.");
  CUGRAPH_EXPECTS(!personalization.has_value() ||
                      (std::get<0>(*personalization).size() == std::get<1>(*personalization).size()),
                  "Invalid input argument: personalization vertices and values size mismatch.");
  CUGRAPH_EXPECTS(!personalization.has_value() || (std::get<0>(*personalization).size() > 0),
                  "Invalid input argument: personalization vector size should not be 0.");

  // Allocate output
  rmm::device_uvector<float> local_pageranks(
      graph_view.local_vertex_partition_range_size(), handle.get_stream());

  const float* initial_ptr = initial_pageranks ? initial_pageranks->data() : nullptr;
  const float* precomputed_sums_ptr = precomputed_vertex_out_weight_sums ? precomputed_vertex_out_weight_sums->data() : nullptr;

  // =========================================================================
  // ROUTE TO AAI IMPLEMENTATION
  // =========================================================================

  bool has_weights = edge_weight_view.has_value();
  bool has_personalization = personalization.has_value();
  bool has_mask = (compact_graph.edge_mask != nullptr);

  // Sync stream before calling AAI (AAI uses default stream)
  handle.sync_stream();

  aai::PageRankResult result;

  bool has_segments = compact_graph.segment_offsets.has_value();

  // 8-way dispatch: mask x personalization x (weights x segments)
  if (has_mask) {
    if (has_personalization) {
      const int32_t* pers_verts = std::get<0>(*personalization).data();
      const float* pers_vals = std::get<1>(*personalization).data();
      std::size_t pers_size = std::get<0>(*personalization).size();

      if (has_weights) {
        const float* weights = edge_weight_view->value_firsts()[0];
        if (has_segments) {
          result = aai::personalized_pagerank_seg_mask(
              compact_graph, weights, pers_verts, pers_vals, pers_size,
              local_pageranks.data(), precomputed_sums_ptr, alpha, epsilon, max_iterations, initial_ptr);
        } else {
          result = aai::personalized_pagerank_mask(
              compact_graph, weights, pers_verts, pers_vals, pers_size,
              local_pageranks.data(), precomputed_sums_ptr, alpha, epsilon, max_iterations, initial_ptr);
        }
      } else {
        if (has_segments) {
          result = aai::personalized_pagerank_seg_mask(
              compact_graph, pers_verts, pers_vals, pers_size,
              local_pageranks.data(), precomputed_sums_ptr, alpha, epsilon, max_iterations, initial_ptr);
        } else {
          result = aai::personalized_pagerank_mask(
              compact_graph, pers_verts, pers_vals, pers_size,
              local_pageranks.data(), precomputed_sums_ptr, alpha, epsilon, max_iterations, initial_ptr);
        }
      }
    } else {
      if (has_weights) {
        const float* weights = edge_weight_view->value_firsts()[0];
        if (has_segments) {
          result = aai::pagerank_seg_mask(
              compact_graph, weights,
              local_pageranks.data(), precomputed_sums_ptr, alpha, epsilon, max_iterations, initial_ptr);
        } else {
          result = aai::pagerank_mask(
              compact_graph, weights,
              local_pageranks.data(), precomputed_sums_ptr, alpha, epsilon, max_iterations, initial_ptr);
        }
      } else {
        if (has_segments) {
          result = aai::pagerank_seg_mask(
              compact_graph,
              local_pageranks.data(), precomputed_sums_ptr, alpha, epsilon, max_iterations, initial_ptr);
        } else {
          result = aai::pagerank_mask(
              compact_graph,
              local_pageranks.data(), precomputed_sums_ptr, alpha, epsilon, max_iterations, initial_ptr);
        }
      }
    }
  } else {
    if (has_personalization) {
      const int32_t* pers_verts = std::get<0>(*personalization).data();
      const float* pers_vals = std::get<1>(*personalization).data();
      std::size_t pers_size = std::get<0>(*personalization).size();

      if (has_weights) {
        const float* weights = edge_weight_view->value_firsts()[0];
        if (has_segments) {
          result = aai::personalized_pagerank_seg(
              compact_graph, weights, pers_verts, pers_vals, pers_size,
              local_pageranks.data(), precomputed_sums_ptr, alpha, epsilon, max_iterations, initial_ptr);
        } else {
          result = aai::personalized_pagerank(
              compact_graph, weights, pers_verts, pers_vals, pers_size,
              local_pageranks.data(), precomputed_sums_ptr, alpha, epsilon, max_iterations, initial_ptr);
        }
      } else {
        if (has_segments) {
          result = aai::personalized_pagerank_seg(
              compact_graph, pers_verts, pers_vals, pers_size,
              local_pageranks.data(), precomputed_sums_ptr, alpha, epsilon, max_iterations, initial_ptr);
        } else {
          result = aai::personalized_pagerank(
              compact_graph, pers_verts, pers_vals, pers_size,
              local_pageranks.data(), precomputed_sums_ptr, alpha, epsilon, max_iterations, initial_ptr);
        }
      }
    } else {
      if (has_weights) {
        const float* weights = edge_weight_view->value_firsts()[0];
        if (has_segments) {
          result = aai::pagerank_seg(
              compact_graph, weights,
              local_pageranks.data(), precomputed_sums_ptr, alpha, epsilon, max_iterations, initial_ptr);
        } else {
          result = aai::pagerank(
              compact_graph, weights,
              local_pageranks.data(), precomputed_sums_ptr, alpha, epsilon, max_iterations, initial_ptr);
        }
      } else {
        if (has_segments) {
          result = aai::pagerank_seg(
              compact_graph,
              local_pageranks.data(), precomputed_sums_ptr, alpha, epsilon, max_iterations, initial_ptr);
        } else {
          result = aai::pagerank(
              compact_graph,
              local_pageranks.data(), precomputed_sums_ptr, alpha, epsilon, max_iterations, initial_ptr);
        }
      }
    }
  }

  cudaDeviceSynchronize();
  {
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
      CUGRAPH_FAIL((std::string("AAI CUDA error after cudaDeviceSynchronize: ") + cudaGetErrorString(err)).c_str());
    }
  }

  return std::make_tuple(std::move(local_pageranks),
                         centrality_algorithm_metadata_t{result.iterations, result.converged});
}

// Double specialization (weighted only)
// Note: cuGraph's unweighted algorithms default weight_t to float32, so there is no
// unweighted double PageRank variant. Double precision requires explicit edge weights.
template <>
std::tuple<rmm::device_uvector<double>, centrality_algorithm_metadata_t>
pagerank<int32_t, int32_t, double, double, false>(
    raft::handle_t const& handle,
    graph_view_t<int32_t, int32_t, true, false> const& graph_view,
    std::optional<edge_property_view_t<int32_t, double const*>> edge_weight_view,
    std::optional<raft::device_span<double const>> precomputed_vertex_out_weight_sums,
    std::optional<std::tuple<raft::device_span<int32_t const>, raft::device_span<double const>>>
        personalization,
    std::optional<raft::device_span<double const>> initial_pageranks,
    double alpha,
    double epsilon,
    size_t max_iterations,
    bool do_expensive_check)
{
  auto compact_graph = aai::graph32_t::from_graph_view(graph_view);

  // Preconditions (match original pagerank_impl.cuh)
  CUGRAPH_EXPECTS(compact_graph.is_csc, "PageRank requires CSC format (is_csc=true)");
  CUGRAPH_EXPECTS((alpha >= 0.0) && (alpha <= 1.0),
                  "Invalid input argument: alpha should be in [0.0, 1.0].");
  CUGRAPH_EXPECTS(epsilon >= 0.0,
                  "Invalid input argument: epsilon should be non-negative.");
  CUGRAPH_EXPECTS(!personalization.has_value() ||
                      (std::get<0>(*personalization).size() == std::get<1>(*personalization).size()),
                  "Invalid input argument: personalization vertices and values size mismatch.");
  CUGRAPH_EXPECTS(!personalization.has_value() || (std::get<0>(*personalization).size() > 0),
                  "Invalid input argument: personalization vector size should not be 0.");

  // Allocate output
  rmm::device_uvector<double> local_pageranks(
      graph_view.local_vertex_partition_range_size(), handle.get_stream());

  const double* initial_ptr = initial_pageranks ? initial_pageranks->data() : nullptr;
  const double* precomputed_sums_ptr = precomputed_vertex_out_weight_sums ? precomputed_vertex_out_weight_sums->data() : nullptr;

  // =========================================================================
  // ROUTE TO AAI IMPLEMENTATION
  // =========================================================================

  CUGRAPH_EXPECTS(edge_weight_view.has_value(),
                  "Invalid input argument: double-precision PageRank requires edge weights.");

  bool has_mask = (compact_graph.edge_mask != nullptr);

  // Sync stream before calling AAI (AAI uses default stream)
  handle.sync_stream();

  aai::PageRankResult result;
  const double* weights = edge_weight_view->value_firsts()[0];
  bool has_segments = compact_graph.segment_offsets.has_value();

  // 4-way dispatch: mask x (personalization x segments)
  if (has_mask) {
    if (personalization.has_value()) {
      const int32_t* pers_verts = std::get<0>(*personalization).data();
      const double* pers_vals = std::get<1>(*personalization).data();
      std::size_t pers_size = std::get<0>(*personalization).size();

      if (has_segments) {
        result = aai::personalized_pagerank_seg_mask(
            compact_graph, weights, pers_verts, pers_vals, pers_size,
            local_pageranks.data(), precomputed_sums_ptr, alpha, epsilon, max_iterations, initial_ptr);
      } else {
        result = aai::personalized_pagerank_mask(
            compact_graph, weights, pers_verts, pers_vals, pers_size,
            local_pageranks.data(), precomputed_sums_ptr, alpha, epsilon, max_iterations, initial_ptr);
      }
    } else {
      if (has_segments) {
        result = aai::pagerank_seg_mask(
            compact_graph, weights,
            local_pageranks.data(), precomputed_sums_ptr, alpha, epsilon, max_iterations, initial_ptr);
      } else {
        result = aai::pagerank_mask(
            compact_graph, weights,
            local_pageranks.data(), precomputed_sums_ptr, alpha, epsilon, max_iterations, initial_ptr);
      }
    }
  } else {
    if (personalization.has_value()) {
      const int32_t* pers_verts = std::get<0>(*personalization).data();
      const double* pers_vals = std::get<1>(*personalization).data();
      std::size_t pers_size = std::get<0>(*personalization).size();

      if (has_segments) {
        result = aai::personalized_pagerank_seg(
            compact_graph, weights, pers_verts, pers_vals, pers_size,
            local_pageranks.data(), precomputed_sums_ptr, alpha, epsilon, max_iterations, initial_ptr);
      } else {
        result = aai::personalized_pagerank(
            compact_graph, weights, pers_verts, pers_vals, pers_size,
            local_pageranks.data(), precomputed_sums_ptr, alpha, epsilon, max_iterations, initial_ptr);
      }
    } else {
      if (has_segments) {
        result = aai::pagerank_seg(
            compact_graph, weights,
            local_pageranks.data(), precomputed_sums_ptr, alpha, epsilon, max_iterations, initial_ptr);
      } else {
        result = aai::pagerank(
            compact_graph, weights,
            local_pageranks.data(), precomputed_sums_ptr, alpha, epsilon, max_iterations, initial_ptr);
      }
    }
  }

  cudaDeviceSynchronize();
  {
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
      CUGRAPH_FAIL((std::string("AAI CUDA error after cudaDeviceSynchronize: ") + cudaGetErrorString(err)).c_str());
    }
  }

  return std::make_tuple(std::move(local_pageranks),
                         centrality_algorithm_metadata_t{result.iterations, result.converged});
}

}  // namespace cugraph

#else

namespace cugraph {
template std::tuple<rmm::device_uvector<float>, centrality_algorithm_metadata_t> pagerank<int32_t, int32_t, float, float, false>(raft::handle_t const&, graph_view_t<int32_t, int32_t, true, false> const&, std::optional<edge_property_view_t<int32_t, float const*>>, std::optional<raft::device_span<float const>>, std::optional<std::tuple<raft::device_span<int32_t const>, raft::device_span<float const>>>, std::optional<raft::device_span<float const>>, float, float, size_t, bool);
template std::tuple<rmm::device_uvector<double>, centrality_algorithm_metadata_t> pagerank<int32_t, int32_t, double, double, false>(raft::handle_t const&, graph_view_t<int32_t, int32_t, true, false> const&, std::optional<edge_property_view_t<int32_t, double const*>>, std::optional<raft::device_span<double const>>, std::optional<std::tuple<raft::device_span<int32_t const>, raft::device_span<double const>>>, std::optional<raft::device_span<double const>>, double, double, size_t, bool);
}  // namespace cugraph

#endif
