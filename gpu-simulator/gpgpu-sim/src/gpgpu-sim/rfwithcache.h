#pragma once

#include "rfcache_config.h"
#include "rfcache_rep_policy.h"
#include "shader.h"

using tag_t = std::pair<unsigned, unsigned>;  // <wid, regid>

class RFWithCache : public opndcoll_rfu_t {
 public:
  RFWithCache(const shader_core_config *config);
  void add_cu_set(unsigned cu_set, unsigned num_cu, unsigned num_dispatch);
  void init(unsigned num_banks, shader_core_ctx *shader) override;
  virtual void allocate_cu(unsigned port_num);
  virtual void dispatch_ready_cu();

 private:
  const RFCacheConfig &m_rfcache_config;
  class modified_arbiter_t : public arbiter_t {
   public:
    virtual void add_read_requests(collector_unit_t *cu);
  };
  class modified_collector_unit_t : public collector_unit_t {
   public:
    modified_collector_unit_t() = delete;
    modified_collector_unit_t(std::size_t sz);
    bool operator==(const collector_unit_t &r) {
      return get_id() == r.get_id();
    }
    bool can_be_allocated(const warp_inst_t &inst) const;
    bool allocate(register_set *, register_set *) override;
    bool is_not_ready(unsigned op_id) { return m_not_ready.test(op_id); }
    enum access_t {
      Hit,
      Miss,
    };
    void collect_operand(unsigned op) override {
      DDDPRINTF("Collect Operand <" << m_warp_id << ", "
                                    << m_src_op[op].get_reg() << ">")
      m_not_ready.reset(op);
      auto wid = m_src_op[op].get_wid();
      assert(m_warp_id == wid);
      auto regid = m_src_op[op].get_reg();
      tag_t tag(wid, regid);
      m_rfcache.unlock(tag);
    }

   private:
    mutable unsigned counter = 0;
    std::vector<unsigned> get_src_ops(const warp_inst_t &inst) const;
    class RFCache {
     public:
      RFCache(std::size_t sz);
      access_t read_access(unsigned regid, unsigned wid);
      void flush() {
        /*remove everything in cache and fill buffer*/
        assert(m_n_locked == 0);
        m_n_available = m_size;
        m_rpolicy.reset();
        m_cache_table.clear();
        m_fill_buffer.flush();
      }
      void cycle() { /* drain fill buffer */
      }
      bool can_allocate(const std::vector<unsigned> &ops, unsigned wid) const;
      void lock(const tag_t &tag);
      void unlock(const tag_t &tag);
      void dump();

     private:
      std::size_t m_size;
      std::size_t m_n_available;
      std::size_t m_n_locked;
      bool check_size() const;
      struct pair_hash {
        template <class T1, class T2>
        std::size_t operator()(const std::pair<T1, T2> &p) const {
          auto h1 = std::hash<T1>{}(p.first);
          auto h2 = std::hash<T2>{}(p.second);
          return h1 ^ h2;
        }
      };
      ReplacementPolicy<tag_t, pair_hash> m_rpolicy;
      class CacheBlock {
       public:
        void lock() { m_lock = true; }
        void unlock() { m_lock = false; }
        bool is_locked() { return m_lock; }
        void dump(const tag_t &tag) {
          std::stringstream ss;
          ss << (m_lock ? 'L' : 'U');
          DDDPRINTF("<" << tag.first << ", " << tag.second
                        << ">: " << ss.str());
        }
        bool m_lock = false;
        // bool m_lock;
        // RFWithCache::op_t m_data;
        // size_t m_op_idx;
      };
      using cache_table_t = std::unordered_map<tag_t, CacheBlock, pair_hash>;
      cache_table_t m_cache_table;

      class FillBuffer {
       public:
        bool find(const tag_t &tag) const;
        bool has_pending_writes() const;
        void flush() { m_buffer.clear(); }

       private:
        std::list<tag_t> m_buffer;
      };
      FillBuffer m_fill_buffer;
    };
    RFCache m_rfcache;
  };
  class OCAllocator {
   public:
    OCAllocator(std::size_t num_ocs, std::size_t num_warps_per_shader);
    void add_oc(modified_collector_unit_t &);
    std::pair<bool, RFWithCache::modified_collector_unit_t &> allocate(
        const warp_inst_t &inst);
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
