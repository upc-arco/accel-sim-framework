#pragma once

#include <bitset>
#include <cassert>
#include <iostream>
#include <unordered_map>
#include <utility>

#include "shader.h"

#ifdef DDEBUG
#define DPRINTF(X) X
#else
#define DPRINTF(X)
#endif

    const unsigned MAX_OC_COUNT = 100;
const unsigned MAX_SLOTS_PER_OC = 6;  // at most 3 OPs, 2 slots each

using tag_t = std::pair<unsigned, unsigned>;  // warp_id and register_id maps to
                                              // unique physical register_id
using missed_oc_slots_t =
    std::bitset<MAX_SLOTS_PER_OC *
                MAX_OC_COUNT>;  // miss status holding bits should be at most
                                // equal to the number of slots in OCs

enum CACHE_ACCESS_STAT_t { Hit, Miss, Reservation, Invalid_Access };
class RFCache {
 public:
  void init(shader_core_ctx *shader) { m_shader = shader; }
  virtual CACHE_ACCESS_STAT_t access(const tag_t &) = 0;
  virtual CACHE_ACCESS_STAT_t read_access(const tag_t &, unsigned oc_slot) = 0;
  virtual CACHE_ACCESS_STAT_t write_access(const tag_t &) = 0;

  virtual missed_oc_slots_t service_read_miss(const tag_t &) = 0;
  virtual void allocate_for_write(const tag_t &) = 0;

 protected:
  shader_core_ctx *m_shader;
};

// A hash function used to hash a pair of any kind
struct hash_pair {
  template <class T1, class T2>
  size_t operator()(const std::pair<T1, T2> &p) const {
    auto hash1 = std::hash<T1>{}(p.first);
    auto hash2 = std::hash<T2>{}(p.second);
    return hash1 ^ hash2;
  }
};
class IdealRFCache : public RFCache {
 public:
  virtual CACHE_ACCESS_STAT_t access(const tag_t &) {}
  virtual CACHE_ACCESS_STAT_t read_access(const tag_t &, unsigned oc_slot);
  virtual CACHE_ACCESS_STAT_t write_access(const tag_t &);
  virtual missed_oc_slots_t service_read_miss(const tag_t &);
  virtual void allocate_for_write(const tag_t &);
 private:
  std::unordered_map<tag_t, missed_oc_slots_t, hash_pair>
      m_cache_table;  // we don't need data for RFCache
};
class RFWithCache : public opndcoll_rfu_t {
 public:
  RFWithCache() : m_cache(new IdealRFCache()) {}
  virtual ~RFWithCache();
  virtual void init(unsigned num_banks, shader_core_ctx *shader);
  virtual void add_cu_set(unsigned cu_set, unsigned num_cu,
                          unsigned num_dispatch);
  virtual void allocate_reads();
  virtual bool writeback(warp_inst_t &warp);
 private:
  shader_core_ctx *m_shader;  // shader core
  RFCache *m_cache;
  class modified_collector_unit_t : public opndcoll_rfu_t::collector_unit_t {
   public:
    modified_collector_unit_t(RFCache *cache) : m_cache(cache) {}
    virtual bool allocate(register_set *pipeline_reg, register_set *output_reg);
    virtual void collect_operand(unsigned op) {
      assert(m_not_ready.test(op));
      DPRINTF(std::cout << "sid: " << m_shader->get_sid() << " op_id: " << op
                        << " cid: " << get_id()
                        << " V: " << m_src_op[op].valid() << " Res: "
                        << m_src_op[op].valid() << " not_r" << std::endl;)
      m_not_ready.reset(op);
      m_src_op[op].unreserve();
    }

   private:
    RFCache *m_cache;
  };
  class modified_arbiter_t : public opndcoll_rfu_t::arbiter_t {
   public:
    modified_arbiter_t(shader_core_ctx *shader) : m_shader{shader} {}

   private:
    virtual void add_read_requests(collector_unit_t *cu);

    shader_core_ctx *m_shader;
  };
};