#pragma once

#include <iostream>
#include <unordered_map>

#include "debug.h"

class RFCacheStats {
 public:
  void print(unsigned long long gpu_tot_sim_cycle,
             unsigned long long gpu_sim_cycle, unsigned tot_num_ocs) const {
    double oc_cycles = (gpu_sim_cycle + gpu_tot_sim_cycle) * tot_num_ocs;
    double avg_fill_buffer_size = m_tot_fill_buffer_size / oc_cycles;
    // aggregate stats
    // for (auto oc : m_n_read_hits) {
    //     tot_read_hits += oc.second;
    // }

    // for (auto oc : m_n_read_misses) {
    //     tot_read_misses += oc.second;
    // }

    // for (auto shader : m_n_writes) {
    //     tot_writes += shader.second;
    // }

    // print stats
    std::cout << "rfcache_read_hits = " << m_tot_read_hits << std::endl;
    std::cout << "rfcache_read_misses = " << m_tot_read_misses << std::endl;
    std::cout << "rfcache_writes = " << m_tot_writes << std::endl;
    std::cout << "rfcache_max_fill_buffer_size = " << m_max_fill_buffer_length << std::endl;
    std::cout << "rfcache_avg_fill_buffer_size = " << avg_fill_buffer_size << std::endl;
    // clear per kernel stats
    // m_n_writes.clear();
    // m_n_read_misses.clear();
    // m_n_read_hits.clear();
  }
  void inc_read_hits(unsigned oc_id) {
    DDDDPRINTF("Inc Hit " << oc_id);
    m_tot_read_hits++;
    // m_n_read_hits[oc_id]++;
  }
  void inc_read_misses(unsigned oc_id) {
    DDDDPRINTF("Inc Miss " << oc_id)
    m_tot_read_misses++;
    // m_n_read_misses[oc_id]++;
  }
  void inc_writes(unsigned shader_id) {
    DDDDPRINTF("Inc Write " << shader_id)
    m_tot_writes++;
    // m_n_writes[shader_id]++;
  }
  void reg_fill_buffer_size(size_t sz) {
    if (sz > m_max_fill_buffer_length) m_max_fill_buffer_length = sz;
    m_tot_fill_buffer_size += sz;
  }

 private:
  //   mutable std::unordered_map<unsigned, unsigned long long> m_n_read_hits;
  //   // read hits per oc mutable std::unordered_map<unsigned, unsigned long
  //   long> m_n_read_misses; // read misses per oc mutable
  //   std::unordered_map<unsigned, unsigned long long> m_n_writes; // writes
  //   per each sm
  unsigned long long m_tot_read_hits = 0;
  unsigned long long m_tot_read_misses = 0;
  unsigned long long m_tot_writes = 0;
  size_t m_max_fill_buffer_length = 0;
  unsigned long long m_tot_fill_buffer_size = 0;
};