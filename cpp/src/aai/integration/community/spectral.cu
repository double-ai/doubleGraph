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
#include <cugraph/algorithms.hpp>
#include <cugraph/detail/utility_wrappers.hpp>
#include <cugraph/legacy/graph.hpp>
#include <cugraph/utilities/error.hpp>

#include <raft/core/copy.hpp>
#include <raft/random/rng_state.hpp>
#include <raft/sparse/convert/coo.cuh>
#include <raft/sparse/convert/csr.cuh>
#include <raft/spectral/modularity_maximization.cuh>
#include <raft/spectral/partition.cuh>

#include <rmm/device_vector.hpp>
#include <rmm/exec_policy.hpp>

#include <thrust/transform.h>

#include <cuvs/cluster/spectral.hpp>
#include <cuvs/preprocessing/spectral_embedding.hpp>

#include <ctime>

namespace cugraph {

namespace ext_raft {

namespace detail {

template <typename vertex_t, typename edge_t, typename weight_t>
void balancedCutClustering_impl(raft::handle_t const& handle,
                                raft::random::RngState& rng_state,
                                legacy::GraphCSRView<vertex_t, edge_t, weight_t> const& graph,
                                vertex_t n_clusters,
                                vertex_t n_eig_vects,
                                weight_t evs_tolerance,
                                int evs_max_iter,
                                weight_t kmean_tolerance,
                                int kmean_max_iter,
                                vertex_t* clustering,
                                weight_t* eig_vals,
                                weight_t* eig_vects)
{
  RAFT_EXPECTS(graph.edge_data != nullptr, "API error, graph must have weights");
  RAFT_EXPECTS(evs_tolerance >= weight_t{0.0},
               "API error, evs_tolerance must be between 0.0 and 1.0");
  RAFT_EXPECTS(evs_tolerance < weight_t{1.0},
               "API error, evs_tolerance must be between 0.0 and 1.0");
  RAFT_EXPECTS(kmean_tolerance >= weight_t{0.0},
               "API error, kmean_tolerance must be between 0.0 and 1.0");
  RAFT_EXPECTS(kmean_tolerance < weight_t{1.0},
               "API error, kmean_tolerance must be between 0.0 and 1.0");
  RAFT_EXPECTS(n_clusters > 1, "API error, must specify more than 1 cluster");
  RAFT_EXPECTS(n_clusters < graph.number_of_vertices,
               "API error, number of clusters must be smaller than number of vertices");
  RAFT_EXPECTS(n_eig_vects <= n_clusters,
               "API error, cannot specify more eigenvectors than clusters");
  RAFT_EXPECTS(clustering != nullptr, "API error, must specify valid clustering");
  RAFT_EXPECTS(eig_vals != nullptr, "API error, must specify valid eigenvalues");
  RAFT_EXPECTS(eig_vects != nullptr, "API error, must specify valid eigenvectors");

  // Convert CSR to COO using raft::sparse::convert::csr_to_coo
  rmm::device_uvector<vertex_t> src_indices(graph.number_of_edges, handle.get_stream());
  rmm::device_uvector<vertex_t> dst_indices(graph.number_of_edges, handle.get_stream());

  // Copy destination indices (already in COO format)
  raft::copy(dst_indices.data(), graph.indices, graph.number_of_edges, handle.get_stream());

  // Convert CSR row offsets to COO source indices
  raft::sparse::convert::csr_to_coo<vertex_t>(graph.offsets,
                                              static_cast<vertex_t>(graph.number_of_vertices),
                                              src_indices.data(),
                                              static_cast<edge_t>(graph.number_of_edges),
                                              handle.get_stream());

  // Create coordinate structure view from converted COO data
  auto coord_view = raft::make_device_coordinate_structure_view<vertex_t, vertex_t, vertex_t>(
    src_indices.data(),
    dst_indices.data(),
    graph.number_of_vertices,
    graph.number_of_vertices,
    graph.number_of_edges);

  // Create COO matrix view using coordinate structure view and CSR edge data
  auto coo_matrix = raft::make_device_coo_matrix_view<weight_t>(graph.edge_data, coord_view);

  cuvs::cluster::spectral::params params;

  params.rng_state    = rng_state;
  params.n_clusters   = n_clusters;
  params.n_components = n_eig_vects;
  params.n_init       = 10;  // Multiple initializations for better results
  params.n_neighbors =
    std::min(static_cast<int>(graph.number_of_vertices) - 1, 15);  // Adaptive neighbor count
  params.tolerance = evs_tolerance;  // Eigensolver convergence tolerance

  cuvs::cluster::spectral::fit_predict(
    handle,
    params,
    coo_matrix,
    raft::make_device_vector_view<vertex_t, vertex_t>(clustering, graph.number_of_vertices));
}

template <typename vertex_t, typename edge_t, typename weight_t>
void spectralModularityMaximization_impl(
  raft::handle_t const& handle,
  raft::random::RngState& rng_state,
  legacy::GraphCSRView<vertex_t, edge_t, weight_t> const& graph,
  vertex_t n_clusters,
  vertex_t n_eig_vects,
  weight_t evs_tolerance,
  int evs_max_iter,
  weight_t kmean_tolerance,
  int kmean_max_iter,
  vertex_t* clustering,
  weight_t* eig_vals,
  weight_t* eig_vects)
{
  RAFT_EXPECTS(graph.edge_data != nullptr, "API error, graph must have weights");
  RAFT_EXPECTS(evs_tolerance >= weight_t{0.0},
               "API error, evs_tolerance must be between 0.0 and 1.0");
  RAFT_EXPECTS(evs_tolerance < weight_t{1.0},
               "API error, evs_tolerance must be between 0.0 and 1.0");
  RAFT_EXPECTS(kmean_tolerance >= weight_t{0.0},
               "API error, kmean_tolerance must be between 0.0 and 1.0");
  RAFT_EXPECTS(kmean_tolerance < weight_t{1.0},
               "API error, kmean_tolerance must be between 0.0 and 1.0");
  RAFT_EXPECTS(n_clusters > 1, "API error, must specify more than 1 cluster");
  RAFT_EXPECTS(n_clusters < graph.number_of_vertices,
               "API error, number of clusters must be smaller than number of vertices");
  RAFT_EXPECTS(n_eig_vects <= n_clusters,
               "API error, cannot specify more eigenvectors than clusters");
  RAFT_EXPECTS(clustering != nullptr, "API error, must specify valid clustering");
  RAFT_EXPECTS(eig_vals != nullptr, "API error, must specify valid eigenvalues");
  RAFT_EXPECTS(eig_vects != nullptr, "API error, must specify valid eigenvectors");

  // Convert CSR to COO using raft::sparse::convert::csr_to_coo
  rmm::device_uvector<vertex_t> src_indices(graph.number_of_edges, handle.get_stream());
  rmm::device_uvector<vertex_t> dst_indices(graph.number_of_edges, handle.get_stream());

  // Copy destination indices (already in COO format)
  raft::copy(dst_indices.data(), graph.indices, graph.number_of_edges, handle.get_stream());

  // Convert CSR row offsets to COO source indices
  raft::sparse::convert::csr_to_coo<vertex_t>(graph.offsets,
                                              static_cast<vertex_t>(graph.number_of_vertices),
                                              src_indices.data(),
                                              static_cast<edge_t>(graph.number_of_edges),
                                              handle.get_stream());

  // Create coordinate structure view from converted COO data
  auto coord_view = raft::make_device_coordinate_structure_view<vertex_t, vertex_t, vertex_t>(
    src_indices.data(),
    dst_indices.data(),
    graph.number_of_vertices,
    graph.number_of_vertices,
    graph.number_of_edges);

  // Create COO matrix view using coordinate structure view and CSR edge data
  auto coo_matrix = raft::make_device_coo_matrix_view<weight_t>(graph.edge_data, coord_view);

  cuvs::cluster::spectral::params params;

  params.rng_state    = rng_state;
  params.n_clusters   = n_clusters;
  params.n_components = n_eig_vects;
  params.n_init       = 10;  // Multiple initializations for better results
  params.n_neighbors =
    std::min(static_cast<int>(graph.number_of_vertices) - 1, 15);  // Adaptive neighbor count
  params.tolerance = evs_tolerance;  // Eigensolver convergence tolerance

  cuvs::cluster::spectral::fit_predict(
    handle,
    params,
    coo_matrix,
    raft::make_device_vector_view<vertex_t, vertex_t>(clustering, graph.number_of_vertices));
}

template <typename vertex_t, typename edge_t, typename weight_t>
void analyzeModularityClustering_impl(raft::handle_t const& handle,
                                      legacy::GraphCSRView<vertex_t, edge_t, weight_t> const& graph,
                                      int n_clusters,
                                      vertex_t const* clustering,
                                      weight_t* modularity)
{
  using index_type = vertex_t;
  using value_type = weight_t;
  using nnz_type   = edge_t;

  raft::spectral::matrix::sparse_matrix_t<index_type, value_type, nnz_type> const r_csr_m{handle,
                                                                                          graph};

  weight_t mod;
  raft::spectral::analyzeModularity(handle, r_csr_m, n_clusters, clustering, mod);
  *modularity = mod;
}

template <typename vertex_t, typename edge_t, typename weight_t>
void analyzeBalancedCut_impl(raft::handle_t const& handle,
                             legacy::GraphCSRView<vertex_t, edge_t, weight_t> const& graph,
                             vertex_t n_clusters,
                             vertex_t const* clustering,
                             weight_t* edgeCut,
                             weight_t* ratioCut)
{
  RAFT_EXPECTS(n_clusters <= graph.number_of_vertices,
               "API error: number of clusters must be <= number of vertices");
  RAFT_EXPECTS(n_clusters > 0, "API error: number of clusters must be > 0)");

  weight_t edge_cut;
  weight_t cost{0};

  using index_type = vertex_t;
  using value_type = weight_t;
  using nnz_type   = edge_t;

  raft::spectral::matrix::sparse_matrix_t<index_type, value_type, nnz_type> const r_csr_m{handle,
                                                                                          graph};

  raft::spectral::analyzePartition(handle, r_csr_m, n_clusters, clustering, edge_cut, cost);

  *edgeCut  = edge_cut;
  *ratioCut = cost;
}

}  // namespace detail

template <typename VT, typename ET, typename WT>
void balancedCutClustering(raft::handle_t const& handle,
                           raft::random::RngState& rng_state,
                           legacy::GraphCSRView<VT, ET, WT> const& graph,
                           VT num_clusters,
                           VT num_eigen_vects,
                           WT evs_tolerance,
                           int evs_max_iter,
                           WT kmean_tolerance,
                           int kmean_max_iter,
                           VT* clustering)
{
  rmm::device_uvector<WT> eig_vals(num_eigen_vects, handle.get_stream());
  rmm::device_uvector<WT> eig_vects(num_eigen_vects * graph.number_of_vertices,
                                    handle.get_stream());

  detail::balancedCutClustering_impl(handle,
                                     rng_state,
                                     graph,
                                     num_clusters,
                                     num_eigen_vects,
                                     evs_tolerance,
                                     evs_max_iter,
                                     kmean_tolerance,
                                     kmean_max_iter,
                                     clustering,
                                     eig_vals.data(),
                                     eig_vects.data());
}

template <typename VT, typename ET, typename WT>
void spectralModularityMaximization(raft::handle_t const& handle,
                                    raft::random::RngState& rng_state,
                                    legacy::GraphCSRView<VT, ET, WT> const& graph,
                                    VT n_clusters,
                                    VT n_eigen_vects,
                                    WT evs_tolerance,
                                    int evs_max_iter,
                                    WT kmean_tolerance,
                                    int kmean_max_iter,
                                    VT* clustering)
{
  rmm::device_uvector<WT> eig_vals(n_eigen_vects, handle.get_stream());
  rmm::device_uvector<WT> eig_vects(n_eigen_vects * graph.number_of_vertices, handle.get_stream());

  detail::spectralModularityMaximization_impl(handle,
                                              rng_state,
                                              graph,
                                              n_clusters,
                                              n_eigen_vects,
                                              evs_tolerance,
                                              evs_max_iter,
                                              kmean_tolerance,
                                              kmean_max_iter,
                                              clustering,
                                              eig_vals.data(),
                                              eig_vects.data());
}

// 6-param versions are the primary implementations. segment_offsets is ignored
// in the generic (non-AAI) path; AAI explicit specializations below use it to
// dispatch to _seg kernel variants.
template <typename VT, typename ET, typename WT>
void analyzeClustering_modularity(raft::handle_t const& handle,
                                  legacy::GraphCSRView<VT, ET, WT> const& graph,
                                  int n_clusters,
                                  VT const* clustering,
                                  WT* score,
                                  std::vector<VT> const* /*segment_offsets*/)
{
  detail::analyzeModularityClustering_impl(handle, graph, n_clusters, clustering, score);
}

template <typename VT, typename ET, typename WT>
void analyzeClustering_edge_cut(raft::handle_t const& handle,
                                legacy::GraphCSRView<VT, ET, WT> const& graph,
                                int n_clusters,
                                VT const* clustering,
                                WT* score,
                                std::vector<VT> const* /*segment_offsets*/)
{
  WT dummy{0.0};
  detail::analyzeBalancedCut_impl(handle, graph, n_clusters, clustering, score, &dummy);
}

template <typename VT, typename ET, typename WT>
void analyzeClustering_ratio_cut(raft::handle_t const& handle,
                                 legacy::GraphCSRView<VT, ET, WT> const& graph,
                                 int n_clusters,
                                 VT const* clustering,
                                 WT* score,
                                 std::vector<VT> const* /*segment_offsets*/)
{
  WT dummy{0.0};
  detail::analyzeBalancedCut_impl(handle, graph, n_clusters, clustering, &dummy, score);
}

// 5-param shims: delegate to the 6-param versions with segment_offsets=nullptr
template <typename VT, typename ET, typename WT>
void analyzeClustering_modularity(raft::handle_t const& handle,
                                  legacy::GraphCSRView<VT, ET, WT> const& graph,
                                  int n_clusters,
                                  VT const* clustering,
                                  WT* score)
{
  analyzeClustering_modularity(handle, graph, n_clusters, clustering, score,
                               static_cast<std::vector<VT> const*>(nullptr));
}

template <typename VT, typename ET, typename WT>
void analyzeClustering_edge_cut(raft::handle_t const& handle,
                                legacy::GraphCSRView<VT, ET, WT> const& graph,
                                int n_clusters,
                                VT const* clustering,
                                WT* score)
{
  analyzeClustering_edge_cut(handle, graph, n_clusters, clustering, score,
                             static_cast<std::vector<VT> const*>(nullptr));
}

template <typename VT, typename ET, typename WT>
void analyzeClustering_ratio_cut(raft::handle_t const& handle,
                                 legacy::GraphCSRView<VT, ET, WT> const& graph,
                                 int n_clusters,
                                 VT const* clustering,
                                 WT* score)
{
  analyzeClustering_ratio_cut(handle, graph, n_clusters, clustering, score,
                              static_cast<std::vector<VT> const*>(nullptr));
}

}  // namespace ext_raft
}  // namespace cugraph

// AAI includes (outside cugraph namespace to avoid conflicts)
#if defined(AAI_ROUTE_SPECTRAL_MODULARITY_MAXIMIZATION) || defined(AAI_ROUTE_ANALYZE_CLUSTERING_MODULARITY) || defined(AAI_ROUTE_ANALYZE_CLUSTERING_EDGE_CUT) || defined(AAI_ROUTE_ANALYZE_CLUSTERING_RATIO_CUT)
#include <cugraph/aai/algorithms.hpp>
#include <cuda_runtime.h>
#endif

// balancedCutClustering routes to AAI's spectral_modularity_maximization because cuGraph's
// balanced cut clustering is a direct pass-through to spectral modularity maximization -- both
// use identical CUVS spectral clustering (cuvs::cluster::spectral::fit_predict) with the same
// parameters. There is no separate AAI implementation for balanced cut.
#ifdef AAI_ROUTE_SPECTRAL_MODULARITY_MAXIMIZATION

namespace cugraph {
namespace ext_raft {

// =============================================================================
// balancedCutClustering - Routed to AAI spectral_modularity_maximization
// =============================================================================

// Float specialization - routes to AAI spectral_modularity_maximization
template <>
void balancedCutClustering<int, int, float>(raft::handle_t const& handle,
                                            raft::random::RngState& rng_state,
                                            legacy::GraphCSRView<int, int, float> const& graph,
                                            int num_clusters,
                                            int num_eigen_vects,
                                            float evs_tolerance,
                                            int evs_max_iter,
                                            float kmean_tolerance,
                                            int kmean_max_iter,
                                            int* clustering)
{
  // Preconditions (from detail::balancedCutClustering_impl, lines 48-64)
  CUGRAPH_EXPECTS(graph.edge_data != nullptr, "API error, graph must have weights");
  CUGRAPH_EXPECTS(evs_tolerance >= 0.0f,
                  "API error, evs_tolerance must be between 0.0 and 1.0");
  CUGRAPH_EXPECTS(evs_tolerance < 1.0f,
                  "API error, evs_tolerance must be between 0.0 and 1.0");
  CUGRAPH_EXPECTS(kmean_tolerance >= 0.0f,
                  "API error, kmean_tolerance must be between 0.0 and 1.0");
  CUGRAPH_EXPECTS(kmean_tolerance < 1.0f,
                  "API error, kmean_tolerance must be between 0.0 and 1.0");
  CUGRAPH_EXPECTS(num_clusters > 1, "API error, must specify more than 1 cluster");
  CUGRAPH_EXPECTS(num_clusters < graph.number_of_vertices,
                  "API error, number of clusters must be smaller than number of vertices");
  CUGRAPH_EXPECTS(num_eigen_vects <= num_clusters,
                  "API error, cannot specify more eigenvectors than clusters");
  CUGRAPH_EXPECTS(clustering != nullptr, "API error, must specify valid clustering");

  aai::graph32_t compact_graph{
      .offsets            = graph.offsets,
      .indices            = graph.indices,
      .number_of_vertices = static_cast<int32_t>(graph.number_of_vertices),
      .number_of_edges    = static_cast<int32_t>(graph.number_of_edges),
      .is_symmetric       = true,
      .is_multigraph      = false,
      .is_csc             = false,
      .segment_offsets    = std::nullopt,
  };

  handle.sync_stream();

  aai::spectral_modularity_maximization(compact_graph,
                               graph.edge_data,
                               num_clusters,
                               num_eigen_vects,
                               evs_tolerance,
                               evs_max_iter,
                               kmean_tolerance,
                               kmean_max_iter,
                               clustering);

  cudaDeviceSynchronize();
  {
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
      CUGRAPH_FAIL((std::string("AAI CUDA error after cudaDeviceSynchronize: ") + cudaGetErrorString(err)).c_str());
    }
  }
}

// Double specialization - routes to AAI spectral_modularity_maximization
template <>
void balancedCutClustering<int, int, double>(raft::handle_t const& handle,
                                             raft::random::RngState& rng_state,
                                             legacy::GraphCSRView<int, int, double> const& graph,
                                             int num_clusters,
                                             int num_eigen_vects,
                                             double evs_tolerance,
                                             int evs_max_iter,
                                             double kmean_tolerance,
                                             int kmean_max_iter,
                                             int* clustering)
{
  CUGRAPH_EXPECTS(graph.edge_data != nullptr, "API error, graph must have weights");
  CUGRAPH_EXPECTS(evs_tolerance >= 0.0,
                  "API error, evs_tolerance must be between 0.0 and 1.0");
  CUGRAPH_EXPECTS(evs_tolerance < 1.0,
                  "API error, evs_tolerance must be between 0.0 and 1.0");
  CUGRAPH_EXPECTS(kmean_tolerance >= 0.0,
                  "API error, kmean_tolerance must be between 0.0 and 1.0");
  CUGRAPH_EXPECTS(kmean_tolerance < 1.0,
                  "API error, kmean_tolerance must be between 0.0 and 1.0");
  CUGRAPH_EXPECTS(num_clusters > 1, "API error, must specify more than 1 cluster");
  CUGRAPH_EXPECTS(num_clusters < graph.number_of_vertices,
                  "API error, number of clusters must be smaller than number of vertices");
  CUGRAPH_EXPECTS(num_eigen_vects <= num_clusters,
                  "API error, cannot specify more eigenvectors than clusters");
  CUGRAPH_EXPECTS(clustering != nullptr, "API error, must specify valid clustering");

  aai::graph32_t compact_graph{
      .offsets            = graph.offsets,
      .indices            = graph.indices,
      .number_of_vertices = static_cast<int32_t>(graph.number_of_vertices),
      .number_of_edges    = static_cast<int32_t>(graph.number_of_edges),
      .is_symmetric       = true,
      .is_multigraph      = false,
      .is_csc             = false,
      .segment_offsets    = std::nullopt,
  };

  handle.sync_stream();

  aai::spectral_modularity_maximization(compact_graph,
                               graph.edge_data,
                               num_clusters,
                               num_eigen_vects,
                               evs_tolerance,
                               evs_max_iter,
                               kmean_tolerance,
                               kmean_max_iter,
                               clustering);

  cudaDeviceSynchronize();
  {
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
      CUGRAPH_FAIL((std::string("AAI CUDA error after cudaDeviceSynchronize: ") + cudaGetErrorString(err)).c_str());
    }
  }
}

}  // namespace ext_raft
}  // namespace cugraph

#else  // !AAI_ROUTE_SPECTRAL_MODULARITY_MAXIMIZATION (balanced cut fallback)

namespace cugraph {
namespace ext_raft {
template void balancedCutClustering<int, int, float>(raft::handle_t const&, raft::random::RngState&, legacy::GraphCSRView<int, int, float> const&, int, int, float, int, float, int, int*);
template void balancedCutClustering<int, int, double>(raft::handle_t const&, raft::random::RngState&, legacy::GraphCSRView<int, int, double> const&, int, int, double, int, double, int, int*);
}  // namespace ext_raft
}  // namespace cugraph

#endif  // AAI_ROUTE_SPECTRAL_MODULARITY_MAXIMIZATION (balanced cut)

#ifdef AAI_ROUTE_SPECTRAL_MODULARITY_MAXIMIZATION

namespace cugraph {
namespace ext_raft {

// =============================================================================
// spectralModularityMaximization - Specialized to route to AAI
// =============================================================================

// Float specialization - routes to AAI
template <>
void spectralModularityMaximization<int, int, float>(raft::handle_t const& handle,
                                                     raft::random::RngState& rng_state,
                                                     legacy::GraphCSRView<int, int, float> const& graph,
                                                     int n_clusters,
                                                     int n_eigen_vects,
                                                     float evs_tolerance,
                                                     int evs_max_iter,
                                                     float kmean_tolerance,
                                                     int kmean_max_iter,
                                                     int* clustering)
{
  CUGRAPH_EXPECTS(graph.edge_data != nullptr, "API error, graph must have weights");
  CUGRAPH_EXPECTS(evs_tolerance >= 0.0f,
                  "API error, evs_tolerance must be between 0.0 and 1.0");
  CUGRAPH_EXPECTS(evs_tolerance < 1.0f,
                  "API error, evs_tolerance must be between 0.0 and 1.0");
  CUGRAPH_EXPECTS(kmean_tolerance >= 0.0f,
                  "API error, kmean_tolerance must be between 0.0 and 1.0");
  CUGRAPH_EXPECTS(kmean_tolerance < 1.0f,
                  "API error, kmean_tolerance must be between 0.0 and 1.0");
  CUGRAPH_EXPECTS(n_clusters > 1, "API error, must specify more than 1 cluster");
  CUGRAPH_EXPECTS(n_clusters < graph.number_of_vertices,
                  "API error, number of clusters must be smaller than number of vertices");
  CUGRAPH_EXPECTS(n_eigen_vects <= n_clusters,
                  "API error, cannot specify more eigenvectors than clusters");
  CUGRAPH_EXPECTS(clustering != nullptr, "API error, must specify valid clustering");

  aai::graph32_t compact_graph{
      .offsets            = graph.offsets,
      .indices            = graph.indices,
      .number_of_vertices = static_cast<int32_t>(graph.number_of_vertices),
      .number_of_edges    = static_cast<int32_t>(graph.number_of_edges),
      .is_symmetric       = true,
      .is_multigraph      = false,
      .is_csc             = false,
      .segment_offsets    = std::nullopt,
  };

  handle.sync_stream();

  aai::spectral_modularity_maximization(compact_graph,
                                        graph.edge_data,
                                        n_clusters,
                                        n_eigen_vects,
                                        evs_tolerance,
                                        evs_max_iter,
                                        kmean_tolerance,
                                        kmean_max_iter,
                                        clustering);

  cudaDeviceSynchronize();
  {
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
      CUGRAPH_FAIL((std::string("AAI CUDA error after cudaDeviceSynchronize: ") + cudaGetErrorString(err)).c_str());
    }
  }
}

// Double specialization - routes to AAI
template <>
void spectralModularityMaximization<int, int, double>(raft::handle_t const& handle,
                                                      raft::random::RngState& rng_state,
                                                      legacy::GraphCSRView<int, int, double> const& graph,
                                                      int n_clusters,
                                                      int n_eigen_vects,
                                                      double evs_tolerance,
                                                      int evs_max_iter,
                                                      double kmean_tolerance,
                                                      int kmean_max_iter,
                                                      int* clustering)
{
  CUGRAPH_EXPECTS(graph.edge_data != nullptr, "API error, graph must have weights");
  CUGRAPH_EXPECTS(evs_tolerance >= 0.0,
                  "API error, evs_tolerance must be between 0.0 and 1.0");
  CUGRAPH_EXPECTS(evs_tolerance < 1.0,
                  "API error, evs_tolerance must be between 0.0 and 1.0");
  CUGRAPH_EXPECTS(kmean_tolerance >= 0.0,
                  "API error, kmean_tolerance must be between 0.0 and 1.0");
  CUGRAPH_EXPECTS(kmean_tolerance < 1.0,
                  "API error, kmean_tolerance must be between 0.0 and 1.0");
  CUGRAPH_EXPECTS(n_clusters > 1, "API error, must specify more than 1 cluster");
  CUGRAPH_EXPECTS(n_clusters < graph.number_of_vertices,
                  "API error, number of clusters must be smaller than number of vertices");
  CUGRAPH_EXPECTS(n_eigen_vects <= n_clusters,
                  "API error, cannot specify more eigenvectors than clusters");
  CUGRAPH_EXPECTS(clustering != nullptr, "API error, must specify valid clustering");

  aai::graph32_t compact_graph{
      .offsets            = graph.offsets,
      .indices            = graph.indices,
      .number_of_vertices = static_cast<int32_t>(graph.number_of_vertices),
      .number_of_edges    = static_cast<int32_t>(graph.number_of_edges),
      .is_symmetric       = true,
      .is_multigraph      = false,
      .is_csc             = false,
      .segment_offsets    = std::nullopt,
  };

  handle.sync_stream();

  aai::spectral_modularity_maximization(compact_graph,
                                        graph.edge_data,
                                        n_clusters,
                                        n_eigen_vects,
                                        evs_tolerance,
                                        evs_max_iter,
                                        kmean_tolerance,
                                        kmean_max_iter,
                                        clustering);

  cudaDeviceSynchronize();
  {
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
      CUGRAPH_FAIL((std::string("AAI CUDA error after cudaDeviceSynchronize: ") + cudaGetErrorString(err)).c_str());
    }
  }
}

}  // namespace ext_raft
}  // namespace cugraph

#else  // !AAI_ROUTE_SPECTRAL_MODULARITY_MAXIMIZATION

namespace cugraph {
namespace ext_raft {
template void spectralModularityMaximization<int, int, float>(raft::handle_t const&, raft::random::RngState&, legacy::GraphCSRView<int, int, float> const&, int, int, float, int, float, int, int*);
template void spectralModularityMaximization<int, int, double>(raft::handle_t const&, raft::random::RngState&, legacy::GraphCSRView<int, int, double> const&, int, int, double, int, double, int, int*);
}  // namespace ext_raft
}  // namespace cugraph

#endif  // AAI_ROUTE_SPECTRAL_MODULARITY_MAXIMIZATION

#ifdef AAI_ROUTE_ANALYZE_CLUSTERING_MODULARITY

namespace cugraph {
namespace ext_raft {

// =============================================================================
// analyzeClustering_modularity - Specialized to route to AAI
// =============================================================================

// Float specialization - routes to AAI
template <>
void analyzeClustering_modularity<int, int, float>(
  raft::handle_t const& handle,
  legacy::GraphCSRView<int, int, float> const& graph,
  int n_clusters,
  int const* clustering,
  float* score,
  std::vector<int> const* segment_offsets)
{
  CUGRAPH_EXPECTS(graph.edge_data != nullptr, "API error, graph must have weights");
  CUGRAPH_EXPECTS(clustering != nullptr, "API error, clustering is null");
  CUGRAPH_EXPECTS(score != nullptr, "API error, score is null");

  aai::graph32_t compact_graph{
      .offsets            = graph.offsets,
      .indices            = graph.indices,
      .number_of_vertices = static_cast<int32_t>(graph.number_of_vertices),
      .number_of_edges    = static_cast<int32_t>(graph.number_of_edges),
      .is_symmetric       = true,
      .is_multigraph      = false,
      .is_csc             = false,
      .segment_offsets    = segment_offsets
          ? std::make_optional(std::vector<int32_t>(segment_offsets->begin(), segment_offsets->end()))
          : std::nullopt,
  };

  handle.sync_stream();

  double result;
  if (compact_graph.segment_offsets.has_value()) {
    result = aai::analyze_clustering_modularity_seg(
        compact_graph, graph.edge_data, static_cast<std::size_t>(n_clusters), clustering);
  } else {
    result = aai::analyze_clustering_modularity(
        compact_graph, graph.edge_data, static_cast<std::size_t>(n_clusters), clustering);
  }

  cudaDeviceSynchronize();
  {
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
      CUGRAPH_FAIL((std::string("AAI CUDA error after cudaDeviceSynchronize: ") + cudaGetErrorString(err)).c_str());
    }
  }

  *score = static_cast<float>(result);
}

// Double specialization - routes to AAI
template <>
void analyzeClustering_modularity<int, int, double>(
  raft::handle_t const& handle,
  legacy::GraphCSRView<int, int, double> const& graph,
  int n_clusters,
  int const* clustering,
  double* score,
  std::vector<int> const* segment_offsets)
{
  CUGRAPH_EXPECTS(graph.edge_data != nullptr, "API error, graph must have weights");
  CUGRAPH_EXPECTS(clustering != nullptr, "API error, clustering is null");
  CUGRAPH_EXPECTS(score != nullptr, "API error, score is null");

  aai::graph32_t compact_graph{
      .offsets            = graph.offsets,
      .indices            = graph.indices,
      .number_of_vertices = static_cast<int32_t>(graph.number_of_vertices),
      .number_of_edges    = static_cast<int32_t>(graph.number_of_edges),
      .is_symmetric       = true,
      .is_multigraph      = false,
      .is_csc             = false,
      .segment_offsets    = segment_offsets
          ? std::make_optional(std::vector<int32_t>(segment_offsets->begin(), segment_offsets->end()))
          : std::nullopt,
  };

  handle.sync_stream();

  double result;
  if (compact_graph.segment_offsets.has_value()) {
    result = aai::analyze_clustering_modularity_seg(
        compact_graph, graph.edge_data, static_cast<std::size_t>(n_clusters), clustering);
  } else {
    result = aai::analyze_clustering_modularity(
        compact_graph, graph.edge_data, static_cast<std::size_t>(n_clusters), clustering);
  }

  cudaDeviceSynchronize();
  {
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
      CUGRAPH_FAIL((std::string("AAI CUDA error after cudaDeviceSynchronize: ") + cudaGetErrorString(err)).c_str());
    }
  }

  *score = result;
}

// Explicit instantiations for the 5-param shims (defined above, delegate to 6-param)
template void analyzeClustering_modularity<int, int, float>(raft::handle_t const&, legacy::GraphCSRView<int, int, float> const&, int, int const*, float*);
template void analyzeClustering_modularity<int, int, double>(raft::handle_t const&, legacy::GraphCSRView<int, int, double> const&, int, int const*, double*);

}  // namespace ext_raft
}  // namespace cugraph

#else  // !AAI_ROUTE_ANALYZE_CLUSTERING_MODULARITY

namespace cugraph {
namespace ext_raft {
template void analyzeClustering_modularity<int, int, float>(raft::handle_t const&, legacy::GraphCSRView<int, int, float> const&, int, int const*, float*, std::vector<int> const*);
template void analyzeClustering_modularity<int, int, double>(raft::handle_t const&, legacy::GraphCSRView<int, int, double> const&, int, int const*, double*, std::vector<int> const*);
template void analyzeClustering_modularity<int, int, float>(raft::handle_t const&, legacy::GraphCSRView<int, int, float> const&, int, int const*, float*);
template void analyzeClustering_modularity<int, int, double>(raft::handle_t const&, legacy::GraphCSRView<int, int, double> const&, int, int const*, double*);
}  // namespace ext_raft
}  // namespace cugraph

#endif  // AAI_ROUTE_ANALYZE_CLUSTERING_MODULARITY

#ifdef AAI_ROUTE_ANALYZE_CLUSTERING_EDGE_CUT

namespace cugraph {
namespace ext_raft {

// =============================================================================
// analyzeClustering_edge_cut - Specialized to route to AAI
// =============================================================================

// Float specialization - routes to AAI
template <>
void analyzeClustering_edge_cut<int, int, float>(
  raft::handle_t const& handle,
  legacy::GraphCSRView<int, int, float> const& graph,
  int n_clusters,
  int const* clustering,
  float* score,
  std::vector<int> const* segment_offsets)
{
  CUGRAPH_EXPECTS(graph.edge_data != nullptr, "API error, graph must have weights");
  CUGRAPH_EXPECTS(n_clusters <= graph.number_of_vertices,
                  "API error: number of clusters must be <= number of vertices");
  CUGRAPH_EXPECTS(n_clusters > 0, "API error: number of clusters must be > 0");
  CUGRAPH_EXPECTS(clustering != nullptr, "API error, clustering is null");
  CUGRAPH_EXPECTS(score != nullptr, "API error, score is null");

  aai::graph32_t compact_graph{
      .offsets            = graph.offsets,
      .indices            = graph.indices,
      .number_of_vertices = static_cast<int32_t>(graph.number_of_vertices),
      .number_of_edges    = static_cast<int32_t>(graph.number_of_edges),
      .is_symmetric       = true,
      .is_multigraph      = false,
      .is_csc             = false,
      .segment_offsets    = segment_offsets
          ? std::make_optional(std::vector<int32_t>(segment_offsets->begin(), segment_offsets->end()))
          : std::nullopt,
  };

  handle.sync_stream();

  double result;
  if (compact_graph.segment_offsets.has_value()) {
    result = aai::analyze_clustering_edge_cut_seg(
        compact_graph, graph.edge_data, static_cast<std::size_t>(n_clusters), clustering);
  } else {
    result = aai::analyze_clustering_edge_cut(
        compact_graph, graph.edge_data, static_cast<std::size_t>(n_clusters), clustering);
  }

  cudaDeviceSynchronize();
  {
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
      CUGRAPH_FAIL((std::string("AAI CUDA error after cudaDeviceSynchronize: ") + cudaGetErrorString(err)).c_str());
    }
  }

  *score = static_cast<float>(result);
}

template <>
void analyzeClustering_edge_cut<int, int, double>(
  raft::handle_t const& handle,
  legacy::GraphCSRView<int, int, double> const& graph,
  int n_clusters,
  int const* clustering,
  double* score,
  std::vector<int> const* segment_offsets)
{
  CUGRAPH_EXPECTS(graph.edge_data != nullptr, "API error, graph must have weights");
  CUGRAPH_EXPECTS(n_clusters <= graph.number_of_vertices,
                  "API error: number of clusters must be <= number of vertices");
  CUGRAPH_EXPECTS(n_clusters > 0, "API error: number of clusters must be > 0");
  CUGRAPH_EXPECTS(clustering != nullptr, "API error, clustering is null");
  CUGRAPH_EXPECTS(score != nullptr, "API error, score is null");

  aai::graph32_t compact_graph{
      .offsets            = graph.offsets,
      .indices            = graph.indices,
      .number_of_vertices = static_cast<int32_t>(graph.number_of_vertices),
      .number_of_edges    = static_cast<int32_t>(graph.number_of_edges),
      .is_symmetric       = true,
      .is_multigraph      = false,
      .is_csc             = false,
      .segment_offsets    = segment_offsets
          ? std::make_optional(std::vector<int32_t>(segment_offsets->begin(), segment_offsets->end()))
          : std::nullopt,
  };

  handle.sync_stream();

  double result;
  if (compact_graph.segment_offsets.has_value()) {
    result = aai::analyze_clustering_edge_cut_seg(
        compact_graph, graph.edge_data, static_cast<std::size_t>(n_clusters), clustering);
  } else {
    result = aai::analyze_clustering_edge_cut(
        compact_graph, graph.edge_data, static_cast<std::size_t>(n_clusters), clustering);
  }

  cudaDeviceSynchronize();
  {
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
      CUGRAPH_FAIL((std::string("AAI CUDA error after cudaDeviceSynchronize: ") + cudaGetErrorString(err)).c_str());
    }
  }

  *score = result;
}

// Explicit instantiations for the 5-param shims (defined above, delegate to 6-param)
template void analyzeClustering_edge_cut<int, int, float>(raft::handle_t const&, legacy::GraphCSRView<int, int, float> const&, int, int const*, float*);
template void analyzeClustering_edge_cut<int, int, double>(raft::handle_t const&, legacy::GraphCSRView<int, int, double> const&, int, int const*, double*);

}  // namespace ext_raft
}  // namespace cugraph

#else  // !AAI_ROUTE_ANALYZE_CLUSTERING_EDGE_CUT

namespace cugraph {
namespace ext_raft {
template void analyzeClustering_edge_cut<int, int, float>(raft::handle_t const&, legacy::GraphCSRView<int, int, float> const&, int, int const*, float*, std::vector<int> const*);
template void analyzeClustering_edge_cut<int, int, double>(raft::handle_t const&, legacy::GraphCSRView<int, int, double> const&, int, int const*, double*, std::vector<int> const*);
template void analyzeClustering_edge_cut<int, int, float>(raft::handle_t const&, legacy::GraphCSRView<int, int, float> const&, int, int const*, float*);
template void analyzeClustering_edge_cut<int, int, double>(raft::handle_t const&, legacy::GraphCSRView<int, int, double> const&, int, int const*, double*);
}  // namespace ext_raft
}  // namespace cugraph

#endif  // AAI_ROUTE_ANALYZE_CLUSTERING_EDGE_CUT

#ifdef AAI_ROUTE_ANALYZE_CLUSTERING_RATIO_CUT

namespace cugraph {
namespace ext_raft {

// =============================================================================
// analyzeClustering_ratio_cut - Specialized to route to AAI
// =============================================================================

// Float specialization - routes to AAI
template <>
void analyzeClustering_ratio_cut<int, int, float>(
  raft::handle_t const& handle,
  legacy::GraphCSRView<int, int, float> const& graph,
  int n_clusters,
  int const* clustering,
  float* score,
  std::vector<int> const* segment_offsets)
{
  CUGRAPH_EXPECTS(graph.edge_data != nullptr, "API error, graph must have weights");
  CUGRAPH_EXPECTS(n_clusters <= graph.number_of_vertices,
                  "API error: number of clusters must be <= number of vertices");
  CUGRAPH_EXPECTS(n_clusters > 0, "API error: number of clusters must be > 0");
  CUGRAPH_EXPECTS(clustering != nullptr, "API error, clustering is null");
  CUGRAPH_EXPECTS(score != nullptr, "API error, score is null");

  aai::graph32_t compact_graph{
      .offsets            = graph.offsets,
      .indices            = graph.indices,
      .number_of_vertices = static_cast<int32_t>(graph.number_of_vertices),
      .number_of_edges    = static_cast<int32_t>(graph.number_of_edges),
      .is_symmetric       = true,
      .is_multigraph      = false,
      .is_csc             = false,
      .segment_offsets    = segment_offsets
          ? std::make_optional(std::vector<int32_t>(segment_offsets->begin(), segment_offsets->end()))
          : std::nullopt,
  };

  handle.sync_stream();

  double result;
  if (compact_graph.segment_offsets.has_value()) {
    result = aai::analyze_clustering_ratio_cut_seg(
        compact_graph, graph.edge_data, static_cast<std::size_t>(n_clusters), clustering);
  } else {
    result = aai::analyze_clustering_ratio_cut(
        compact_graph, graph.edge_data, static_cast<std::size_t>(n_clusters), clustering);
  }

  cudaDeviceSynchronize();
  {
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
      CUGRAPH_FAIL((std::string("AAI CUDA error after cudaDeviceSynchronize: ") + cudaGetErrorString(err)).c_str());
    }
  }

  *score = static_cast<float>(result);
}

template <>
void analyzeClustering_ratio_cut<int, int, double>(
  raft::handle_t const& handle,
  legacy::GraphCSRView<int, int, double> const& graph,
  int n_clusters,
  int const* clustering,
  double* score,
  std::vector<int> const* segment_offsets)
{
  CUGRAPH_EXPECTS(graph.edge_data != nullptr, "API error, graph must have weights");
  CUGRAPH_EXPECTS(n_clusters <= graph.number_of_vertices,
                  "API error: number of clusters must be <= number of vertices");
  CUGRAPH_EXPECTS(n_clusters > 0, "API error: number of clusters must be > 0");
  CUGRAPH_EXPECTS(clustering != nullptr, "API error, clustering is null");
  CUGRAPH_EXPECTS(score != nullptr, "API error, score is null");

  aai::graph32_t compact_graph{
      .offsets            = graph.offsets,
      .indices            = graph.indices,
      .number_of_vertices = static_cast<int32_t>(graph.number_of_vertices),
      .number_of_edges    = static_cast<int32_t>(graph.number_of_edges),
      .is_symmetric       = true,
      .is_multigraph      = false,
      .is_csc             = false,
      .segment_offsets    = segment_offsets
          ? std::make_optional(std::vector<int32_t>(segment_offsets->begin(), segment_offsets->end()))
          : std::nullopt,
  };

  handle.sync_stream();

  double result;
  if (compact_graph.segment_offsets.has_value()) {
    result = aai::analyze_clustering_ratio_cut_seg(
        compact_graph, graph.edge_data, static_cast<std::size_t>(n_clusters), clustering);
  } else {
    result = aai::analyze_clustering_ratio_cut(
        compact_graph, graph.edge_data, static_cast<std::size_t>(n_clusters), clustering);
  }

  cudaDeviceSynchronize();
  {
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
      CUGRAPH_FAIL((std::string("AAI CUDA error after cudaDeviceSynchronize: ") + cudaGetErrorString(err)).c_str());
    }
  }

  *score = result;
}

// Explicit instantiations for the 5-param shims (defined above, delegate to 6-param)
template void analyzeClustering_ratio_cut<int, int, float>(raft::handle_t const&, legacy::GraphCSRView<int, int, float> const&, int, int const*, float*);
template void analyzeClustering_ratio_cut<int, int, double>(raft::handle_t const&, legacy::GraphCSRView<int, int, double> const&, int, int const*, double*);

}  // namespace ext_raft
}  // namespace cugraph

#else  // !AAI_ROUTE_ANALYZE_CLUSTERING_RATIO_CUT

namespace cugraph {
namespace ext_raft {
template void analyzeClustering_ratio_cut<int, int, float>(raft::handle_t const&, legacy::GraphCSRView<int, int, float> const&, int, int const*, float*, std::vector<int> const*);
template void analyzeClustering_ratio_cut<int, int, double>(raft::handle_t const&, legacy::GraphCSRView<int, int, double> const&, int, int const*, double*, std::vector<int> const*);
template void analyzeClustering_ratio_cut<int, int, float>(raft::handle_t const&, legacy::GraphCSRView<int, int, float> const&, int, int const*, float*);
template void analyzeClustering_ratio_cut<int, int, double>(raft::handle_t const&, legacy::GraphCSRView<int, int, double> const&, int, int const*, double*);
}  // namespace ext_raft
}  // namespace cugraph

#endif  // AAI_ROUTE_ANALYZE_CLUSTERING_RATIO_CUT
