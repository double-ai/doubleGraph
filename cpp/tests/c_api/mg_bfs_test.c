/*
 * SPDX-FileCopyrightText: Copyright (c) 2022, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 * Modifications Copyright (c) 2025, AA-I Technologies Ltd.
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
 *
 * NOTICE: This file has been modified by AA-I Technologies Ltd. from the original.
 */

#include "mg_test_utils.h" /* RUN_TEST */

#include <cugraph_c/algorithms.h>
#include <cugraph_c/graph.h>

#include <math.h>
#include <stdint.h>

typedef int32_t vertex_t;
typedef int32_t edge_t;
typedef float weight_t;

int edge_exists(vertex_t* h_src, vertex_t* h_dst, size_t num_edges, vertex_t u, vertex_t v)
{
  for (size_t i = 0; i < num_edges; ++i) {
    if (h_src[i] == u && h_dst[i] == v) return 1;
  }
  return 0;
}

int generic_bfs_test(const cugraph_resource_handle_t* p_handle,
                     vertex_t* h_src,
                     vertex_t* h_dst,
                     weight_t* h_wgt,
                     vertex_t* h_seeds,
                     vertex_t const* expected_distances,
                     size_t num_vertices,
                     size_t num_edges,
                     size_t num_seeds,
                     size_t depth_limit,
                     bool_t store_transposed)
{
  int test_ret_value = 0;

  cugraph_error_code_t ret_code = CUGRAPH_SUCCESS;
  cugraph_error_t* ret_error;

  cugraph_graph_t* p_graph                               = NULL;
  cugraph_paths_result_t* paths_result                   = NULL;
  cugraph_type_erased_device_array_t* p_sources          = NULL;
  cugraph_type_erased_device_array_view_t* p_source_view = NULL;

  if (cugraph_resource_handle_get_rank(p_handle) != 0) num_seeds = 0;

  ret_code =
    cugraph_type_erased_device_array_create(p_handle, num_seeds, INT32, &p_sources, &ret_error);
  TEST_ASSERT(test_ret_value, ret_code == CUGRAPH_SUCCESS, "p_sources create failed.");

  p_source_view = cugraph_type_erased_device_array_view(p_sources);

  ret_code = cugraph_type_erased_device_array_view_copy_from_host(
    p_handle, p_source_view, (byte_t*)h_seeds, &ret_error);
  TEST_ASSERT(test_ret_value, ret_code == CUGRAPH_SUCCESS, "src copy_from_host failed.");

  ret_code = create_mg_test_graph(
    p_handle, h_src, h_dst, h_wgt, num_edges, store_transposed, FALSE, &p_graph, &ret_error);

  TEST_ASSERT(test_ret_value, ret_code == CUGRAPH_SUCCESS, "create_mg_test_graph failed.");

  ret_code = cugraph_bfs(
    p_handle, p_graph, p_source_view, FALSE, 10000000, TRUE, TRUE, &paths_result, &ret_error);

  TEST_ASSERT(test_ret_value, ret_code == CUGRAPH_SUCCESS, cugraph_error_message(ret_error));
  TEST_ASSERT(test_ret_value, ret_code == CUGRAPH_SUCCESS, "cugraph_bfs failed.");

  cugraph_type_erased_device_array_view_t* vertices;
  cugraph_type_erased_device_array_view_t* distances;
  cugraph_type_erased_device_array_view_t* predecessors;

  vertices     = cugraph_paths_result_get_vertices(paths_result);
  predecessors = cugraph_paths_result_get_predecessors(paths_result);
  distances    = cugraph_paths_result_get_distances(paths_result);

  vertex_t h_vertices[num_vertices];
  vertex_t h_predecessors[num_vertices];
  vertex_t h_distances[num_vertices];

  ret_code = cugraph_type_erased_device_array_view_copy_to_host(
    p_handle, (byte_t*)h_vertices, vertices, &ret_error);
  TEST_ASSERT(test_ret_value, ret_code == CUGRAPH_SUCCESS, "copy_to_host failed.");

  ret_code = cugraph_type_erased_device_array_view_copy_to_host(
    p_handle, (byte_t*)h_distances, distances, &ret_error);
  TEST_ASSERT(test_ret_value, ret_code == CUGRAPH_SUCCESS, "copy_to_host failed.");

  ret_code = cugraph_type_erased_device_array_view_copy_to_host(
    p_handle, (byte_t*)h_predecessors, predecessors, &ret_error);
  TEST_ASSERT(test_ret_value, ret_code == CUGRAPH_SUCCESS, "copy_to_host failed.");

  size_t num_local_vertices = cugraph_type_erased_device_array_view_size(vertices);

  // Build a vertex-to-distance lookup for predecessor validation.
  // Initialize to -1 so we can detect vertices not owned by this GPU.
  vertex_t dist_by_vertex[num_vertices];
  for (int i = 0; i < num_vertices; ++i) {
    dist_by_vertex[i] = -1;
  }
  for (int i = 0; i < num_local_vertices; ++i) {
    dist_by_vertex[h_vertices[i]] = h_distances[i];
  }

  // Validate that BFS results represent a valid shortest-path tree rather than
  // checking exact predecessors, since BFS predecessor tie-breaking is non-deterministic.
  for (int i = 0; (i < num_local_vertices) && (test_ret_value == 0); ++i) {
    TEST_ASSERT(test_ret_value,
                expected_distances[h_vertices[i]] == h_distances[i],
                "bfs distances don't match");

    if (h_distances[i] == 0 || h_distances[i] == INT32_MAX) {
      TEST_ASSERT(test_ret_value,
                  h_predecessors[i] == -1,
                  "bfs predecessor should be -1 for source or unreachable vertex");
    } else {
      TEST_ASSERT(test_ret_value,
                  h_predecessors[i] != -1,
                  "bfs predecessor should not be -1 for reachable vertex");

      TEST_ASSERT(test_ret_value,
                  edge_exists(h_src, h_dst, num_edges, h_predecessors[i], h_vertices[i]),
                  "bfs predecessor edge doesn't exist in graph");

      // Only check predecessor distance if the predecessor is owned by this GPU.
      if (dist_by_vertex[h_predecessors[i]] != -1) {
        TEST_ASSERT(test_ret_value,
                    dist_by_vertex[h_predecessors[i]] == h_distances[i] - 1,
                    "bfs predecessor not on shortest path");
      }
    }
  }

  cugraph_paths_result_free(paths_result);
  cugraph_graph_free(p_graph);
  cugraph_error_free(ret_error);

  return test_ret_value;
}

int test_bfs(const cugraph_resource_handle_t* p_handle)
{
  size_t num_edges    = 8;
  size_t num_vertices = 6;

  vertex_t src[]                = {0, 1, 1, 2, 2, 2, 3, 4};
  vertex_t dst[]                = {1, 3, 4, 0, 1, 3, 5, 5};
  weight_t wgt[]                = {0.1f, 2.1f, 1.1f, 5.1f, 3.1f, 4.1f, 7.2f, 3.2f};
  vertex_t seeds[]              = {0};
  vertex_t expected_distances[] = {0, 1, 2147483647, 2, 2, 3};

  // Bfs wants store_transposed = FALSE
  return generic_bfs_test(p_handle,
                          src,
                          dst,
                          wgt,
                          seeds,
                          expected_distances,
                          num_vertices,
                          num_edges,
                          1,
                          10,
                          FALSE);
}

int main(int argc, char** argv)
{
  void* raft_handle                 = create_mg_raft_handle(argc, argv);
  cugraph_resource_handle_t* handle = cugraph_create_resource_handle(raft_handle);

  int result = 0;
  result |= RUN_MG_TEST(test_bfs, handle);

  cugraph_free_resource_handle(handle);
  free_mg_raft_handle(raft_handle);

  return result;
}
