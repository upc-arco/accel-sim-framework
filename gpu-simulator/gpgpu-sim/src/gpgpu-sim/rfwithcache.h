#pragma once

#include "rfcache.h"
#include "rfcache_config.h"
#include "shader.h"
class RFWithCache : public opndcoll_rfu_t {
 public:
  RFWithCache(const shader_core_config *config);
  void add_cu_set(unsigned cu_set, unsigned num_cu, unsigned num_dispatch);
  virtual void init(unsigned num_banks, shader_core_ctx *shader);
  virtual void allocate_cu(unsigned port_num);
  virtual void dispatch_ready_cu();
 private:
  const RFCacheConfig &m_rfcache_config;
  class modified_collector_unit_t : public collector_unit_t {
   public:
    modified_collector_unit_t() = delete;
    modified_collector_unit_t(std::size_t sz);
    bool operator==(const collector_unit_t &r) {
      return get_id() == r.get_id();
    }

   private:
    RFCache m_rfcache;
  };
  class OCAllocator {
   public:
    OCAllocator(std::size_t num_ocs, std::size_t num_warps_per_shader);
    void add_oc(modified_collector_unit_t &);
    std::pair<bool, RFWithCache::modified_collector_unit_t &> allocate(unsigned);
    void dispatch(unsigned);
    void dump();
   private:
    size_t m_n_available;
    size_t m_n_ocs;
    size_t m_n_warps_per_shader;
    struct per_oc_info {
      per_oc_info(modified_collector_unit_t &oc, bool available)
          : m_oc(oc), m_availble(available) {}
      modified_collector_unit_t &m_oc;
      bool m_availble;
    };
    std::unordered_map<unsigned, per_oc_info> m_info_table;
    ReplacementPolicy<unsigned> m_lru_policy;
  };
  OCAllocator m_oc_allocator;
};
