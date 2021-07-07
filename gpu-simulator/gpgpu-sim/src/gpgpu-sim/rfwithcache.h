#pragma once

#include "rfcache_config.h"
#include "rfcache_rep_policy.h"
#include "shader.h"

using tag_t = std::pair<unsigned, unsigned>;  // <wid, regid>

class RFWithCache : public opndcoll_rfu_t {
 public:
  RFWithCache(const shader_core_config *config, RFCacheStats &rfcache_stats,
              shader_core_ctx *shader, std::vector<shd_warp_t *> *warp,
              Scoreboard **scoreboard);
  void add_cu_set(unsigned cu_set, unsigned num_cu, unsigned num_dispatch);
  void init(unsigned num_banks, unsigned num_scheds,
            std::vector<scheduler_unit *> *schedulers,
            shader_core_ctx *shader) override;
  virtual void allocate_cu(unsigned port_num);
  virtual void dispatch_ready_cu();
  virtual void step() override;
  virtual bool writeback(warp_inst_t &warp);
  void cache_cycle();
  void process_writes();
  void process_fill_buffer();
  typedef std::unordered_map<std::string, register_set *>
      out_port_vec_t;  // execution unit type name
  void add_port(port_vector_t &input, out_port_vec_t &output,
                uint_vector_t cu_sets) {
    // m_num_ports++;
    // m_num_collectors += num_collector_units;
    // m_input.resize(m_num_ports);
    // m_output.resize(m_num_ports);
    // m_num_collector_units.resize(m_num_ports);
    // m_input[m_num_ports-1]=input_port;
    // m_output[m_num_ports-1]=output_port;
    // m_num_collector_units[m_num_ports-1]=num_collector_units;
    m_in_ports.push_back(in_port_t(input, output, cu_sets));
  }
 private:
  class in_port_t {
   public:
    in_port_t(port_vector_t &input, out_port_vec_t &output,
              uint_vector_t cu_sets)
        : m_in(input), m_out(output), m_cu_sets(cu_sets) {
      //assert(input.size() == output.size());
      assert(not m_cu_sets.empty());
    }
    // private:
    port_vector_t m_in;
    out_port_vec_t m_out;
    uint_vector_t m_cu_sets;
  };
  std::vector<shd_warp_t *> *m_warp;
  Scoreboard *m_scoreboard;
  void signal_schedulers(unsigned last_wid, unsigned new_wid);
  void sort_next_input_port(unsigned port_num, in_port_t &inp);
  bool priority_func(const warp_inst_t &lhs, const warp_inst_t &rhs,
                     std::list<unsigned> warps_in_ocs) const;
  std::list<std::pair<warp_inst_t **, register_set *>>
      m_prioritized_input_port;  // keep next cycle prioritized list of
                                 // instructions in the input port and the
                                 // target output port
  std::unordered_map<unsigned, unsigned>
      m_pending_insts_in_latch;  // number of pending insts each warp has in the
                                 // latch
  void dump_in_latch(unsigned port_num) const;
  std::vector<scheduler_unit *> *m_schedulers;
  unsigned m_n_schedulers;
  shader_core_ctx *m_shdr;
  RFCacheStats &m_rfcache_stats;
  const RFCacheConfig &m_rfcache_config;

  class WriteReqs {
   public:
    class WriteReq {
     public:
      WriteReq(unsigned wid, unsigned regid, int pending_reuse,
               int reuse_distance)
          : m_wid{wid},
            m_regid{regid},
            m_pending_reuse{pending_reuse},
            m_reuse_distance{reuse_distance} {
        assert(pending_reuse >= 0);
        assert(reuse_distance > -2);
      }
      unsigned wid() const { return m_wid; }
      unsigned regid() const { return m_regid; }
      int pending_reuses() const { return m_pending_reuse; }
      int reuse_distance() const { return m_reuse_distance; }

     private:
      unsigned m_wid;
      unsigned m_regid;
      int m_pending_reuse;
      int m_reuse_distance;
    };

    void push(unsigned oc_id, unsigned wid, unsigned regid, int pending_reuse,
              int reuse_distance);
    bool has_req(unsigned oc_id) const;
    void flush(unsigned oc_id);
    WriteReq pop(unsigned oc_id);
    size_t size() const { return m_size; }

   private:
    size_t m_size;
    std::unordered_map<unsigned, std::list<WriteReq>>
        m_write_reqs;  // <ocid, list<wid, reg_id> last cycle write reqs

    using out_port_vec = std::unordered_map<exec_unit_type_t, register_set *>;
  };
  WriteReqs m_write_reqs;
  class modified_arbiter_t : public arbiter_t {
   public:
    virtual void add_read_requests(collector_unit_t *cu);
  };
  class modified_collector_unit_t : public collector_unit_t {
   public:
    modified_collector_unit_t() = delete;
    modified_collector_unit_t(std::size_t sz, unsigned sid,
                              unsigned num_oc_per_core, unsigned oc_id,
                              RFCacheStats &rfcache_stats);
    bool operator==(const collector_unit_t &r) {
      return get_id() == r.get_id();
    }
    bool can_be_allocated(const warp_inst_t &inst) const;
    bool allocate(warp_inst_t **, register_set *);
    bool is_not_ready(unsigned op_id) { return m_not_ready.test(op_id); }
    enum access_t {
      Hit,
      Miss,
    };
    void collect_operand(unsigned op) override;
    bool waiting_for_opernads() { return m_not_ready.any(); }
    void process_write(unsigned regid, int pending_reuse, int reuse_distance);
    void process_fill_buffer();
    unsigned min_reuse_dist() const { return m_rfcache.min_reuse_dist(); }
    unsigned n_lives() const { return m_rfcache.n_lives(); }

   private:
    unsigned m_sid;
    std::vector<unsigned> get_src_ops(const warp_inst_t &inst) const;
    class RFCache {
     public:
      RFCache(std::size_t sz, RFCacheStats &rfcache_stats,
              unsigned global_oc_id);
      access_t read_access(unsigned regid, int pending_reuses,
                           int reuse_distance, unsigned wid);
      void flush();
      void write_access(unsigned wid, unsigned regid, int pending_reuse,
                        int reuse_distance);
      bool can_allocate(const std::vector<unsigned> &ops, unsigned wid) const;
      void lock(const tag_t &tag);
      void unlock(const tag_t &tag);
      bool replacement(const tag_t &tag, tag_t &replaced_tag);
      void replace(const tag_t &tag) {
        assert(m_cache_table.find(tag) != m_cache_table.end());
        if (m_cache_table[tag].m_pending_reuses > 0) {
          assert(m_n_lives > 0);
          assert(m_cache_table[tag].m_reuse_distance >=
                 0);  // live value should have a non-negative reuse distance
          assert(
              m_min_reuse_distance >=
              0);  // while there is a live value min distance could not be neg
          m_n_lives--;

          // update min reuse distance for live values
          if (m_cache_table[tag].m_reuse_distance == m_min_reuse_distance) {
            // in case we replace min reuse distance element we should recompute
            // min reuse distance
            m_min_reuse_distance = comp_min_rd(tag);
          }
          assert((m_n_lives == 0 && m_min_reuse_distance == -1) ||
                 (m_n_lives > 0 && m_min_reuse_distance != -1));
        } else if (m_cache_table[tag].m_pending_reuses == 0) {
          assert(m_n_deads > 0);
          assert(m_cache_table[tag].m_reuse_distance ==
                 -1);  // dead values have reuse distance of -1
          assert((m_n_lives == 0 && m_min_reuse_distance == -1) ||
                 (m_n_lives > 0 && m_min_reuse_distance != -1));

          m_n_deads--;

          // we don't compute reuse distance for dead vals
        }
        m_cache_table[tag].m_pending_reuses =
            -1;  // when we use replacement to multiple writes to the same reg
                 // pending reuse will be owerwritten by allocate
        m_cache_table[tag].m_reuse_distance =
            -2;  // unallocated entries have reuse distance equal -2
      }
      void allocate(const tag_t &tag, int pending_reuses, int reuse_distance) {
        assert(
            (pending_reuses > 0 && reuse_distance >= 0) ||
            (pending_reuses == 0 &&
             reuse_distance == -1));  // new allocation cannot have negative
                                      // pending reuse reuse distance = -1 when
                                      // no pending otherwise it is positive
        assert((m_cache_table[tag].m_pending_reuses == 0 &&
                m_cache_table[tag].m_reuse_distance == -1) ||
               (m_cache_table[tag].m_pending_reuses == -1 &&
                m_cache_table[tag].m_reuse_distance ==
                    -2));  // wheather new entry or there is no reuse to that

        m_cache_table[tag].m_pending_reuses = pending_reuses;
        m_cache_table[tag].m_reuse_distance = reuse_distance;
        if (pending_reuses > 0) {
          // it is live
          m_n_lives++;
        } else {
          // it is dead
          m_n_deads++;
        }
        if (reuse_distance > -1) {
          // this is a live value and has a reuse distence
          // we only compute min reuse distance based on live values
          if (reuse_distance < m_min_reuse_distance) {
            // new reuse distance changes the minimum
            m_min_reuse_distance = reuse_distance;  // update min reuse distance
          }
        }
      }
      void dump();
      void process_fill_buffer();
      unsigned min_reuse_dist() const { return m_min_reuse_distance; }
      unsigned n_lives() const { return m_n_lives; }

     private:
      unsigned m_global_oc_id;
      RFCacheStats &m_rfcache_stats;
      std::size_t m_size;
      std::size_t m_n_available;
      std::size_t m_n_locked;
      std::size_t m_n_lives;
      std::size_t m_n_deads;
      unsigned m_min_reuse_distance;
      bool check() const;
      unsigned comp_min_rd(const tag_t &tag = tag_t(-1, -1))
          const;  // in normal cases nothing will be excluded but in replacement
                  // we have to not to consider the replaced entry
      struct pair_hash {
        template <class T1, class T2>
        std::size_t operator()(const std::pair<T1, T2> &p) const {
          auto h1 = std::hash<T1>{}(p.first);
          auto h2 = std::hash<T2>{}(p.second);
          return h1 ^ h2;
        }
      };
      //      ReplacementPolicy<tag_t, pair_hash> m_rpolicy;
      class CacheBlock {
       public:
        CacheBlock()
            : m_pending_reuses{-1}, m_reuse_distance{-2}, m_lock{false} {}
        void lock() { m_lock = true; }
        void unlock() { m_lock = false; }
        bool is_locked() { return m_lock; }
        void dump(const tag_t &tag) {
          std::stringstream ss;
          ss << (m_lock ? 'L' : 'U');
          ss << " PR: " << m_pending_reuses;
          ss << " RD: " << m_reuse_distance;
          DDDPRINTF("<" << tag.first << ", " << tag.second
                        << ">: " << ss.str());
        }
        bool m_lock;
        int m_pending_reuses;
        int m_reuse_distance;
      };
      using cache_table_t = std::unordered_map<tag_t, CacheBlock, pair_hash>;
      cache_table_t m_cache_table;

      class FillBuffer {
       public:
        struct Entry {
          tag_t m_tag;
          int m_pending_reuse;
          int m_reuse_distance;
        };

        bool find(const tag_t &tag) const;
        bool has_pending_writes() const;
        void flush() {
          m_buffer.clear();
          m_redundant_write_tracker.clear();
        }
        void dump();
        bool push_back(const Entry &entry);
        bool pop_front(Entry &entry);
        size_t size() const { return m_buffer.size(); }

       private:
        std::unordered_map<tag_t, unsigned, pair_hash>
            m_redundant_write_tracker;

        std::list<Entry> m_buffer;
      };
      FillBuffer m_fill_buffer;
    };
    RFCache m_rfcache;
  };
  class OCAllocator {
   public:
    OCAllocator(std::size_t num_ocs, std::size_t num_warps_per_shader,
                RFCacheStats &rfcache_stats, std::vector<shd_warp_t *> *warp,
                shader_core_ctx *shader, Scoreboard **scoreboard);
    void add_oc(modified_collector_unit_t &);
    std::pair<bool, RFWithCache::modified_collector_unit_t &> allocate(
        const warp_inst_t &inst);
    std::pair<bool, RFWithCache::modified_collector_unit_t &>
    allocate_distance_liveness(const warp_inst_t &inst);
    void dispatch(unsigned);

    shd_warp_t &warp(int i) { return *((*m_warp)[i]); }
    bool warp_done_exit_or_null(int i) const {
      return ((*m_warp)[i] == NULL) || (*m_warp)[i]->done_exit();
    }
    void dump();
    std::list<unsigned> get_warps_in_ocs() const {
      std::list<unsigned> result;
      for (auto iter = m_info_table.begin(); iter != m_info_table.end();
           iter++) {
        result.push_back(iter->first);
      }
      return result;
    }

   private:
    enum NewWarpStallAllocType {
      NW_ADT,      // new warp above distance threshold - allocated
      NW_BDT_AST,  // new warp below distance threshold but above stall
                   // threshold - allocated
      NW_BDT_BST   // new warp below distance and stall threshold - stalled
    };
    enum BeforSchedulingStatus {
      DoneExitorNull,
      NoValidInst,
      ValidonBarrier,
      ValidonScoreboard,
      CanProgress
    };
    void analyze_before_scheduling_status(
        unsigned owid,
        NewWarpStallAllocType
            allocStallType);  // analyze I-Buffer, scoreboard and SIMT status to
                              // understand befor scheduling status
    void update_before_scheduling_status(NewWarpStallAllocType allocStallType,
                                         BeforSchedulingStatus bfSchStatus);
    std::vector<shd_warp_t *> *m_warp;
    Scoreboard **m_scoreboard;
    shader_core_ctx *m_shader;
    RFCacheStats &m_rfcache_stats;
    unsigned
        m_stalls_in_a_row;  // number of stalls in a row we can cause after that
                            // we have to replace the less promissing oc
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
    bool extract_avail_ocs(
        std::vector<std::unordered_map<unsigned, per_oc_info>::iterator> &,
        unsigned thrld);
    void sort_ocs(
        std::vector<std::unordered_map<unsigned, per_oc_info>::iterator> &);
    ReplacementPolicy<unsigned> m_lru_policy;
  };
  OCAllocator m_oc_allocator;

  std::vector<in_port_t> m_in_ports;
};
