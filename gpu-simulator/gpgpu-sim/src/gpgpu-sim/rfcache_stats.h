#pragma once

#include <cassert>
#include <iostream>
#include <unordered_map>

#include "debug.h"

class RFCacheStats {
 public:
  void print(unsigned long long gpu_tot_sim_cycle,
             unsigned long long gpu_sim_cycle, unsigned tot_num_ocs) const {
    assert((m_tot_oc_alloc_r_stalls + m_tot_oc_alloc_nr_stalls) ==
           m_tot_oc_alloc_stalls);
    assert((m_bdt_bst_stalls + m_nw_nv_stalls + m_ow_nv_stalls) ==
           m_tot_oc_alloc_r_stalls);
    auto tot_oc_allocated =
        m_tot_oc_alloc -
        m_tot_oc_alloc_stalls;  // number of times we successfully allocated ocs
                                // without stalling
    assert((m_oc_alloc_ow + m_oc_alloc_nw_wp + m_oc_alloc_nw_wop) ==
           tot_oc_allocated);
    double oc_cycles = (gpu_sim_cycle + gpu_tot_sim_cycle) * tot_num_ocs;
    double avg_fill_buffer_size = m_tot_fill_buffer_size / oc_cycles;

    auto ow_stalls_waiting_for_dispatch =
        m_ow_nv_stalls -
        m_ow_nv_stalls_waiting_for_ops;  // number of stalls in ow allocation
                                         // because it is waiting for dispatch

    assert(m_oc_alloc_nw_wp == 0);
    assert(m_oc_alloc_nw_wop ==
           (m_nw_adt_first_time_oc_allocation + m_nw_adt_done_exit_or_null +
            m_nw_bdt_ast_done_exit_or_null + m_nw_adt_not_valid_inst +
            m_nw_bdt_ast_not_valid_inst + m_nw_adt_valid_on_barrier +
            m_nw_bdt_ast_valid_on_barrier + m_nw_adt_valid_on_scoreboard +
            m_nw_bdt_ast_valid_on_scoreboard + m_nw_adt_can_progress +
            m_nw_bdt_ast_can_progress));
    assert(m_bdt_bst_stalls ==
           (m_nw_bdt_bst_done_exit_or_null + m_nw_bdt_bst_not_valid_inst +
            m_nw_bdt_bst_valid_on_barrier + m_nw_bdt_bst_valid_on_scoreboard +
            m_nw_bdt_bst_can_progress));
    // print stats
    std::cout << "rfcache_read_hits = " << m_tot_read_hits << std::endl;
    std::cout << "rfcache_read_misses = " << m_tot_read_misses << std::endl;
    std::cout << "rfcache_writes = " << m_tot_writes << std::endl;
    std::cout << "rfcache_max_fill_buffer_size = " << m_max_fill_buffer_length
              << std::endl;
    std::cout << "rfcache_avg_fill_buffer_size = " << avg_fill_buffer_size
              << std::endl;
    std::cout << "rfcache_tot_steps = " << m_tot_steps << std::endl;
    std::cout << "rfcache_tot_oc_alloc = " << m_tot_oc_alloc << std::endl;
    std::cout << "rfcache_tot_oc_alloc_stalls = " << m_tot_oc_alloc_stalls
              << std::endl;
    std::cout << "rfcache_tot_oc_alloc_nr_stalls = " << m_tot_oc_alloc_nr_stalls
              << std::endl;
    std::cout << "rfcache_tot_oc_alloc_r_stalls = " << m_tot_oc_alloc_r_stalls
              << std::endl;
    std::cout << "rfcache_tot_oc_alloc_ow_nv_stalls = " << m_ow_nv_stalls
              << std::endl;
    std::cout << "rfcache_tot_oc_alloc_nw_nv_stalls = " << m_nw_nv_stalls
              << std::endl;
    std::cout << "rfcache_tot_oc_alloc_bdt_bst_stalls = " << m_bdt_bst_stalls
              << std::endl;
    std::cout << "rfcache_tot_oc_alloc_ow = " << m_oc_alloc_ow << std::endl;
    std::cout << "rfcache_tot_oc_alloc_nw_wp = " << m_oc_alloc_nw_wp
              << std::endl;
    std::cout << "rfcache_tot_oc_alloc_nw_wop = " << m_oc_alloc_nw_wop
              << std::endl;
    std::cout << "rfcache_tot_oc_alloc_ow_nv_stalls_waiting_for_ops = "
              << m_ow_nv_stalls_waiting_for_ops << std::endl;
    std::cout << "rfcache_tot_oc_alloc_ow_nv_stalls_waiting_for_dispatch = "
              << ow_stalls_waiting_for_dispatch << std::endl;
    
    
    std::cout << "rfcache_nw_adt_first_time_oc_allocation = "
              << m_nw_adt_first_time_oc_allocation << std::endl;
    std::cout << "rfcache_nw_adt_done_exit_or_null = "
              << m_nw_adt_done_exit_or_null << std::endl;
    std::cout << "rfcache_nw_adt_not_valid_inst = "
              << m_nw_adt_not_valid_inst << std::endl;
    std::cout << "rfcache_nw_adt_valid_on_barrier = "
              << m_nw_adt_valid_on_barrier << std::endl;
    std::cout << "rfcache_nw_adt_valid_on_scoreboard = "
              << m_nw_adt_valid_on_scoreboard << std::endl;
    std::cout << "rfcache_nw_adt_can_progress = "
              << m_nw_adt_can_progress << std::endl;
    std::cout << "rfcache_nw_bdt_ast_done_exit_or_null = "
              << m_nw_bdt_ast_done_exit_or_null << std::endl;
    std::cout << "rfcache_nw_bdt_ast_not_valid_inst = "
              << m_nw_bdt_ast_not_valid_inst << std::endl;
    std::cout << "rfcache_nw_bdt_ast_valid_on_barrier = "
              << m_nw_bdt_ast_valid_on_barrier << std::endl;
    std::cout << "rfcache_nw_bdt_ast_valid_on_scoreboard = "
              << m_nw_bdt_ast_valid_on_scoreboard << std::endl;
    std::cout << "rfcache_nw_bdt_ast_can_progress = "
              << m_nw_bdt_ast_can_progress << std::endl;
    std::cout << "rfcache_nw_bdt_bst_done_exit_or_null = "
              << m_nw_bdt_bst_done_exit_or_null << std::endl;
    std::cout << "rfcache_nw_bdt_bst_not_valid_inst = "
              << m_nw_bdt_bst_not_valid_inst << std::endl;
    std::cout << "rfcache_nw_bdt_bst_valid_on_barrier = "
              << m_nw_bdt_bst_valid_on_barrier << std::endl;
    std::cout << "rfcache_nw_bdt_bst_valid_on_scoreboard = "
              << m_nw_bdt_bst_valid_on_scoreboard << std::endl;
    std::cout << "rfcache_nw_bdt_bst_can_progress = "
              << m_nw_bdt_bst_can_progress << std::endl;
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

  void inc_steps() { m_tot_steps++; }

  void inc_oc_alloc() { m_tot_oc_alloc++; }

  void inc_oc_alloc_stalls() { m_tot_oc_alloc_stalls++; }

  void inc_oc_alloc_nr_stall() { m_tot_oc_alloc_nr_stalls++; }

  void inc_oc_alloc_r_stalls() { m_tot_oc_alloc_r_stalls++; }

  void inc_ow_nv_stalls() {
    // this type of stall happens when we have new instructions to a warp
    // already in ocs but this oc is not free
    m_ow_nv_stalls++;
  }

  void inc_nw_nv_stalls() {
    // this stall type happens because we have instructions to new warps but no
    // oc id free
    m_nw_nv_stalls++;
  }

  void inc_bdt_bst_stalls() {
    // this stall happens because we are bellow both thresholds
    m_bdt_bst_stalls++;
  }

  void inc_alloc_ow() {
    // count number of times we allocate an OC for an old warp
    m_oc_alloc_ow++;
  }

  void inc_alloc_nw_wp() {
    // number of times we allocate new warp with pending instruction left for
    // the old warp
    m_oc_alloc_nw_wp++;
  }

  void inc_alloc_nw_wop() {
    // number of times we allocate new warp without pending instruction left in
    // the latch for old warp
    m_oc_alloc_nw_wop++;
  }
  void inc_ow_nv_stalls_waiting_for_ops() {
    // count number of ow stalls because waiting for ops from MRF
    m_ow_nv_stalls_waiting_for_ops++;
  }
  void inc_nw_first_time_oc_allocation() {
    m_nw_adt_first_time_oc_allocation++;
  }

  void inc_nw_adt_done_exit_or_null() { m_nw_adt_done_exit_or_null++; }

  void inc_nw_bdt_ast_done_exit_or_null() { m_nw_bdt_ast_done_exit_or_null++; }
  void inc_nw_bdt_bst_done_exit_or_null() { m_nw_bdt_bst_done_exit_or_null++; }
  void inc_nw_adt_not_valid_inst() { m_nw_adt_not_valid_inst++; }
  void inc_nw_bdt_ast_not_valid_inst() { m_nw_bdt_ast_not_valid_inst++; }
  void inc_nw_bdt_bst_not_valid_inst() { m_nw_bdt_bst_not_valid_inst++; }
  void inc_nw_adt_valid_on_barrier() { m_nw_adt_valid_on_barrier++; }
  void inc_nw_bdt_ast_valid_on_barrier() { m_nw_bdt_ast_valid_on_barrier++; }
  void inc_nw_bdt_bst_valid_on_barrier() { m_nw_bdt_bst_valid_on_barrier++; }
  void inc_nw_adt_valid_on_scoreboard() { m_nw_adt_valid_on_scoreboard++; }
  void inc_nw_bdt_ast_valid_on_scoreboard() {
    m_nw_bdt_ast_valid_on_scoreboard++;
  }
  void inc_nw_bdt_bst_valid_on_scoreboard() {
    m_nw_bdt_bst_valid_on_scoreboard++;
  }
  void inc_nw_adt_can_progress() { m_nw_adt_can_progress++; }
  void inc_nw_bdt_ast_can_progress() { m_nw_bdt_ast_can_progress++; }
  void inc_nw_bdt_bst_can_progress() { m_nw_bdt_bst_can_progress++; }

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
  unsigned long long m_tot_steps = 0;
  unsigned long long m_tot_oc_alloc = 0;
  unsigned long long m_tot_oc_alloc_stalls = 0;
  unsigned long long m_tot_oc_alloc_nr_stalls =
      0;  // there is no ready inst in the latch between issue and oc allocation
  unsigned long long m_tot_oc_alloc_r_stalls =
      0;  // there were ready insts but oc allocation stalled
  unsigned long long m_ow_nv_stalls =
      0;  // stalls because of instructions to warps in collector units that are
          // not free
  unsigned long long m_nw_nv_stalls =
      0;  // stalls because of instructions to warps that are not in collector
          // units yet there is no free oc for allocation
  unsigned long long m_bdt_bst_stalls =
      0;  // stalls because of candidate is below distance threshold and stall
          // threshold has not reached
  unsigned long long m_oc_alloc_ow = 0;  // number of times we allocate old warp
  unsigned long long m_oc_alloc_nw_wp =
      0;  // number of times we allocate for new warp but the old warp has
          // pending instructions in the latch
  unsigned long long m_oc_alloc_nw_wop =
      0;  // number of times we allocate for new warp but the old warp has no
          // pending inst in the latch
  unsigned long long m_ow_nv_stalls_waiting_for_ops =
      0;  // old warp stall because waitng for operands
  unsigned long long m_nw_adt_first_time_oc_allocation =
      0;  // the first time we allocate the oc for a new warp
  unsigned long long m_nw_adt_done_exit_or_null =
      0;  // the new warp replaces a warp which already done and has the
          // distance above threshold
  unsigned long long m_nw_bdt_ast_done_exit_or_null =
      0;  // the new warp replaces a warp which already done and has the reuse
          // distance below threshold and stalls above threshold
  unsigned long long m_nw_bdt_bst_done_exit_or_null =
      0;  // the new warp replaces a warp which already done and has the reuse
          // distance below threshold and stalls below threshold
  unsigned long long m_nw_adt_not_valid_inst =
      0;  // the new warp replaces a warp which has no valid insts in I-Buffer
          // and has the distance above threshold
  unsigned long long m_nw_bdt_ast_not_valid_inst =
      0;  // the new warp replaces a warp which has no valid insts in I-Buffer
          // and has the reuse distance below threshold and stalls above
          // threshold
  unsigned long long m_nw_bdt_bst_not_valid_inst =
      0;  // the new warp replaces a warp which has no valid insts in I-Buffer
          // and has the reuse distance below threshold and stalls below
          // threshold
  unsigned long long m_nw_adt_valid_on_barrier =
      0;  // the new warp replaces a warp which has valid insts but waiting on
          // barrier and has the distance above threshold
  unsigned long long m_nw_bdt_ast_valid_on_barrier =
      0;  // the new warp replaces a warp which has valid insts but waiting on
          // barrier has the reuse distance below threshold and stalls above
          // threshold
  unsigned long long m_nw_bdt_bst_valid_on_barrier =
      0;  // the new warp replaces a warp which has valid insts but waiting on
          // barrier has the reuse distance below threshold and stalls below
          // threshold
  unsigned long long m_nw_adt_valid_on_scoreboard =
      0;  // the new warp replaces a warp which has valid insts but waiting on
          // barrier and has the distance above threshold
  unsigned long long m_nw_bdt_ast_valid_on_scoreboard =
      0;  // the new warp replaces a warp which has valid insts but waiting on
          // barrier has the reuse distance below threshold and stalls above
          // threshold
  unsigned long long m_nw_bdt_bst_valid_on_scoreboard =
      0;  // the new warp replaces a warp which has valid insts but waiting on
          // barrier has the reuse distance below threshold and stalls below
          // threshold
  unsigned long long m_nw_adt_can_progress =
      0;  // the new warp replaces a warp which has valid insts but waiting on
          // barrier and has the distance above threshold
  unsigned long long m_nw_bdt_ast_can_progress =
      0;  // the new warp replaces a warp which has valid insts but waiting on
          // barrier has the reuse distance below threshold and stalls above
          // threshold
  unsigned long long m_nw_bdt_bst_can_progress =
      0;  // the new warp replaces a warp which has valid insts but waiting on
          // barrier has the reuse distance below threshold and stalls below
          // threshold
};