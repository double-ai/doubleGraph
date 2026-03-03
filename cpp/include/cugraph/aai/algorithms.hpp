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
 *
 * AAI Algorithms - Aggregating Header
 *
 * This header includes all AAI algorithm declarations for backward compatibility.
 * For new code, prefer including the specific category headers from <cugraph/aai/api/>.
 */
#pragma once

// Types and result structs
#include <cugraph/aai/types.hpp>

// Graph data structure
#include <cugraph/aai/compact_graph.hpp>

// Cache pool for GPU resource management
#include <cugraph/aai/cache_pool.hpp>

// Algorithm categories
#include <cugraph/aai/api/traversal.hpp>
#include <cugraph/aai/api/link_analysis.hpp>
#include <cugraph/aai/api/centrality.hpp>
#include <cugraph/aai/api/community.hpp>
#include <cugraph/aai/api/link_prediction.hpp>
#include <cugraph/aai/api/components.hpp>
#include <cugraph/aai/api/cores.hpp>
#include <cugraph/aai/api/tree.hpp>
