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
 * Target GPU configuration for AAI kernel specializations.
 *
 * This header is configured by CMake based on the TARGET_GPU option.
 * Supported targets: A100, L4, A10G
 */
#pragma once

// =============================================================================
// Target GPU identifiers
// =============================================================================

#define AAI_GPU_A100 1
#define AAI_GPU_L4   2
#define AAI_GPU_A10G 3

// =============================================================================
// Active target (set by CMake via -DAAI_TARGET_GPU=AAI_GPU_<name>)
// =============================================================================

#ifndef AAI_TARGET_GPU
#error "AAI_TARGET_GPU must be defined. Set TARGET_GPU in CMake (A100, L4, or A10G)."
#endif

// Validate target
#if AAI_TARGET_GPU != AAI_GPU_A100 && \
    AAI_TARGET_GPU != AAI_GPU_L4 && \
    AAI_TARGET_GPU != AAI_GPU_A10G
#error "Invalid AAI_TARGET_GPU. Must be AAI_GPU_A100, AAI_GPU_L4, or AAI_GPU_A10G."
#endif

// =============================================================================
// Convenience macros for conditional compilation
// =============================================================================

#define AAI_IS_A100 (AAI_TARGET_GPU == AAI_GPU_A100)
#define AAI_IS_L4   (AAI_TARGET_GPU == AAI_GPU_L4)
#define AAI_IS_A10G (AAI_TARGET_GPU == AAI_GPU_A10G)
