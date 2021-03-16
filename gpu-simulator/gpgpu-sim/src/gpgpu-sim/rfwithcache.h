#pragma once

#include "rfcache_config.h"
#include "rfcache_rep_policy.h"
#include "shader.h"

using tag_t = std::pair<unsigned, unsigned>;  // <wid, regid>

class RFWithCache : public opndcoll_rfu_t {
 public:
  RFWithCache(const shader_core_config *config, RFCacheStats &rfcache_stats, const shader_core_ctx *shader);
  void add_cu_set(unsigned cu_set, unsigned num_cu, unsigned num_dispatch);
  void init(unsigned num_banks, shader_core_ctx *shader) override;
  virtual void allocate_cu(unsigned port_num);
  virtual void dispatch_ready_cu();
  virtual void step() override;
  virtual bool writeback(warp_inst_t &warp);
  void cache_cycle();
  void process_writes();
  void process_fill_buffer();
 private:
  const shader_core_ctx *m_shdr;
  RFCacheStats &m_rfcache_stats;
  const RFCacheConfig &m_rfcache_config;
  
  class WriteReqs { 
    public:
      void push(unsigned oc_id, unsigned wid, unsigned regid);
      bool has_req(unsigned oc_id) const;
      void flush(unsigned oc_id);
      std::pair<unsigned, unsigned> pop(unsigned oc_id);
      size_t size() const { return m_size; }
    private:
      size_t m_size;
      std::unordered_map<unsigned, std::list<std::pair<unsigned, unsigned>>> m_write_reqs; // <ocid, list<wid, reg_id> last cycle write reqs
  };
  WriteReqs m_write_reqs;
  class modified_arbiter_t : public arbiter_t {
   public:
    virtual void add_read_requests(collector_unit_t *cu);
  };
  class modified_collector_unit_t : public collector_unit_t {
   public:
    modified_collector_unit_t() = delete;
    modified_collector_unit_t(std::size_t sz, unsigned sid, unsigned num_oc_per_core, unsigned oc_id, RFCacheStats &rfcache_stats);
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
    void collect_operand(unsigned op) override;
    void process_write(unsigned regid);
    void process_fill_buffer();
   private:
    unsigned m_sid;
    std::vector<unsigned> get_src_ops(const warp_inst_t &inst) const;
    class RFCache {
     public:
      RFCache(std::size_t sz, RFCacheStats &rfcache_stats, unsigned global_oc_id);
      access_t read_access(unsigned regid, unsigned wid);
      void flush();
      void write_access(unsigned wid, unsigned regid);
      bool can_allocate(const std::vector<unsigned> &ops, unsigned wid) const;
      void lock(const tag_t &tag);
      void unlock(const tag_t &tag);
      void dump();
      void process_fill_buffer();
     private:
      unsigned m_global_oc_id;
      RFCacheStats &m_rfcache_stats;
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
        void flush() { m_buffer.clear(); m_redundant_write_tracker.clear();}
        void dump();
        bool push_back(const tag_t &tag);
        bool pop_front(tag_t &tag);
        size_t size() const { return m_buffer.size(); }
       private:
        std::unordered_map<tag_t, unsigned, pair_hash> m_redundant_write_tracker;
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
