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
 * AAI Cache Pool - Thread-local LRU pool for GPU resource caches
 *
 * Algorithm implementations allocate GPU resources (device memory, streams, etc.)
 * in per-function Cache structs. Rather than keeping all caches alive for the
 * lifetime of the thread, the CachePool enforces a capacity limit and evicts the
 * least-recently-used caches when the limit is reached. Eviction destroys the
 * cache object, freeing its GPU resources.
 *
 * Usage in algorithm files:
 *
 *   struct Cache : Cacheable {
 *       int32_t* buf = nullptr;
 *       Cache()  { cudaMalloc(&buf, ...); }
 *       ~Cache() override { if (buf) cudaFree(buf); }
 *   };
 *
 *   void my_algorithm(...) {
 *       static int tag;
 *       auto& cache = cache_pool().acquire<Cache>(&tag);
 *       // use cache.buf ...
 *   }
 */
#pragma once

#include <cstddef>
#include <list>
#include <memory>
#include <unordered_map>

namespace aai {

// Base class for all algorithm caches. Provides the virtual destructor
// needed for type-erased ownership by the pool.
struct Cacheable {
  virtual ~Cacheable() = default;
};

// Thread-local LRU pool that manages algorithm cache lifetimes.
//
// Each cache is identified by a stable pointer tag (typically the address of a
// function-local static variable). On acquire(), existing entries are promoted
// to most-recently-used. New entries may trigger eviction of the
// least-recently-used entry, whose destructor frees GPU resources.
class CachePool {
 public:
  static constexpr size_t DEFAULT_CAPACITY = 8;

  explicit CachePool(size_t capacity = DEFAULT_CAPACITY) : capacity_(capacity) {}

  CachePool(const CachePool&)            = delete;
  CachePool& operator=(const CachePool&) = delete;

  // Acquire a cache of type T, identified by tag.
  // Creates a new T() if not present (evicting LRU if at capacity).
  // Promotes the entry to most-recently-used on every call.
  template <typename T>
  T& acquire(const void* tag)
  {
    auto it = map_.find(tag);
    if (it != map_.end()) {
      // Promote to MRU
      order_.splice(order_.begin(), order_, it->second.order_it);
      return static_cast<T&>(*it->second.ptr);
    }
    // Evict LRU entries until under capacity
    while (map_.size() >= capacity_) {
      auto lru_tag = order_.back();
      order_.pop_back();
      map_.erase(lru_tag);  // unique_ptr dtor frees GPU resources
    }
    // Create new entry at MRU position
    auto ptr = std::make_unique<T>();
    T& ref   = *ptr;
    order_.push_front(tag);
    map_[tag] = Entry{std::move(ptr), order_.begin()};
    return ref;
  }

  size_t size() const { return map_.size(); }
  size_t capacity() const { return capacity_; }

  void set_capacity(size_t cap)
  {
    capacity_ = cap;
    while (map_.size() > capacity_) {
      auto lru_tag = order_.back();
      order_.pop_back();
      map_.erase(lru_tag);
    }
  }

 private:
  struct Entry {
    std::unique_ptr<Cacheable> ptr;
    std::list<const void*>::iterator order_it;
  };

  size_t capacity_;
  std::list<const void*> order_;                        // front = MRU, back = LRU
  std::unordered_map<const void*, Entry> map_;           // tag -> entry
};

// Access the thread-local cache pool instance.
inline CachePool& cache_pool()
{
  thread_local static CachePool pool;
  return pool;
}

}  // namespace aai
