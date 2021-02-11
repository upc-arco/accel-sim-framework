#pragma once

#include <bitset>
#include <cassert>
#include <iostream>
#include <unordered_map>
#include <utility>

#include "shader.h"

#define DPRINTF(X)

#define DDEBUG
#ifdef DDEBUG
#define DDPRINTF(X) X
#else
#define DPRINTF(X)
#endif

const unsigned MAX_OC_COUNT = 1;
const unsigned MAX_SLOTS_PER_OC = 6;  // at most 3 OPs, 2 slots each

using tag_t = std::pair<unsigned, unsigned>;  // warp_id and register_id maps to
                                              // unique physical register_id
using missed_oc_slots_t =
    std::bitset<MAX_SLOTS_PER_OC *
                MAX_OC_COUNT>;  // miss status holding bits should be at most
                                // equal to the number of slots in OCs
using replacement_order_list_t = std::list<std::pair<tag_t, bool>>;
using replacement_order_list_iter = replacement_order_list_t::iterator;
class first_cache_block_t {
 public:
  bool is_locked() { return m_locked; }
  void lock() { m_locked = true; }
  void unlock() {
    assert(m_valid);
    m_locked = false;
  }
  void allocate_for_write() {
    m_dirty = true;
    m_valid = true;
    m_missed_oc_slots.reset();
    unlock();
  }

  void allocate_for_read(unsigned oc_slot) {
    m_dirty = false;
    m_valid = true;
    reserve(oc_slot);
    unlock();
  }

  void allocate_and_lock_for_write() {
    allocate_for_write();
    lock();
  }

  void allocate_and_lock_for_read(unsigned oc_slot) {
    assert(m_locked == false);
    reset_missed_oc_slots_mask();
    allocate_for_read(oc_slot);
    lock();
  }

  void reserve(unsigned oc_slot) { 
    DDPRINTF(std::cout << "missed_oc_slot set " << oc_slot << std::endl;)
    m_missed_oc_slots.set(oc_slot); 
  }
  missed_oc_slots_t get_missed_oc_slots_mask() const {
    return m_missed_oc_slots;
  }
  void reset_missed_oc_slots_mask() { m_missed_oc_slots.reset(); }
  void fill_block(const opndcoll_rfu_t::op_t &op) {
    assert(op.valid());
    m_data = op;
  }
  void fill_block_and_unlock(const opndcoll_rfu_t::op_t &op) {
    m_data = op;
    unlock();
  }
  bool is_dirty() { return m_dirty; }
  opndcoll_rfu_t::op_t get_data() { return m_data; }
  void dump() {
    std::cout << (m_locked ? 'l':'u') << ' ' << (m_dirty ? 'd': 'c') << ' ' << (m_valid ? 'v': 'i') << ' ';
    std::cout << m_missed_oc_slots << ' ';
    if(m_data.valid())
      m_data.dump(stdout);
    else
    {
      std::cout << " empty ";
    }
    
    std::cout << std::endl;
  }
 private:
  bool m_locked;
  bool m_dirty;
  opndcoll_rfu_t::op_t m_data;
  bool m_valid;
  missed_oc_slots_t m_missed_oc_slots;
};

enum CACHE_ACCESS_STAT_t { Hit, Miss, Reservation, Invalid_Access };
class RFCache {
 public:
  void init(shader_core_ctx *shader) {
    m_shader = shader;
    m_rfcache_stats = shader->m_stats->m_rfcache_stats;
  }
  virtual CACHE_ACCESS_STAT_t access(const tag_t &) = 0;
  virtual CACHE_ACCESS_STAT_t read_access(const tag_t &, unsigned oc_slot) = 0;
  virtual CACHE_ACCESS_STAT_t write_access(const tag_t &) = 0;

  virtual missed_oc_slots_t service_read_miss(const tag_t &) = 0;
  virtual void allocate_for_write(const tag_t &) = 0;

 protected:
  shader_core_ctx *m_shader;
  RFCacheStats *m_rfcache_stats;
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
  RFWithCache(const shader_core_config *config);

  virtual ~RFWithCache();
  virtual void init(unsigned num_banks, shader_core_ctx *shader);
  virtual void add_cu_set(unsigned cu_set, unsigned num_cu,
                          unsigned num_dispatch);
  virtual void allocate_reads();
  virtual bool writeback(warp_inst_t &warp);

 protected:
  const RFCacheConfig &m_rfcache_config;
  class modified_arbiter_t : public opndcoll_rfu_t::arbiter_t {
   public:
    modified_arbiter_t(shader_core_ctx *shader) : m_shader{shader} {}

   protected:
   private:
    shader_core_ctx *m_shader;
    virtual void add_read_requests(collector_unit_t *cu);
  };
  shader_core_ctx *m_shader;  // shader core
  RFCache *m_cache;

 protected:
  RFCacheStats *m_rfcache_stats;
  class modified_collector_unit_t : public opndcoll_rfu_t::collector_unit_t {
   public:
    modified_collector_unit_t(RFCache *cache) : m_cache(cache) {}
    virtual bool allocate(register_set *pipeline_reg, register_set *output_reg);
    virtual void collect_operand(unsigned op);

   protected:
    RFCache *m_cache;
  };
};

class RFWithCacheFirst : public RFWithCache {
 public:
  RFWithCacheFirst(const shader_core_config *config);
  virtual void init(unsigned num_banks, shader_core_ctx *shader);

  bool lock_srcopnd_slots_in_cache(const warp_inst_t &inst) { return true; }
  bool lock_dstopnd_slots_in_cache(const warp_inst_t &inst);
  bool dstopnd_can_allocate_or_update(const warp_inst_t &inst,
                                      std::vector<tag_t> &dsts);
  bool srcopnds_can_allocate_or_update(const warp_inst_t &inst,
                                       std::vector<tag_t> &srcs);

  virtual bool writeback(warp_inst_t &inst);

  void fill_srcop(const tag_t &tag, const op_t& op);
  void allocate_cu(unsigned port_num) {
    DDPRINTF(std::cout << "First: allocate_cu" << std::endl;)
    input_port_t &inp = m_in_ports[port_num];
    for (unsigned i = 0; i < inp.m_in.size(); i++) {
      if ((*inp.m_in[i]).has_ready()) {
        // find a free cu
        for (unsigned j = 0; j < inp.m_cu_sets.size(); j++) {
          std::vector<collector_unit_t *> &cu_set = m_cus[inp.m_cu_sets[j]];
          bool allocated = false;
          for (unsigned k = 0; k < cu_set.size(); k++) {
            if (cu_set[k]->is_free()) {
              collector_unit_t *cu = cu_set[k];
              allocated = cu->allocate(inp.m_in[i], inp.m_out[i]);
              m_arbiter->add_read_requests(cu);
              break;
            }
          }
          if (allocated)
            break;  // cu has been allocated, no need to search more.
        }
        break;  // can only service a single input, if it failed it will fail
                // for others.
      }
    }
  }

  virtual void add_cu_set(unsigned set_id, unsigned num_cu,
                          unsigned num_dispatch) {
    m_cus[set_id].reserve(num_cu);  // this is necessary to stop pointers in
                                    // m_cu from being invalid do to a resize;
    for (unsigned i = 0; i < num_cu; i++) {
      m_cus[set_id].push_back(
          new first_collector_unit_t(&m_rfcache_config, this, m_cache));
      m_cu.push_back(m_cus[set_id].back());
    }
    // for now each collector set gets dedicated dispatch units.
    for (unsigned i = 0; i < num_dispatch; i++) {
      m_dispatch_units.push_back(dispatch_unit_t(&m_cus[set_id]));
    }
  }
  virtual void step();

  void allocate_read_writes();
  class first_arbiter_t : public modified_arbiter_t {
   public:
    first_arbiter_t(shader_core_ctx *shader);
    virtual ~first_arbiter_t() {
      delete m_write_queue;
      m_write_queue = nullptr;
    }
    virtual void init(unsigned num_cu, unsigned num_banks);
    virtual void add_write_request(const op_t &op);
    virtual std::list<op_t> allocate_reads();

   protected:
    std::list<op_t> *m_write_queue;
  };
  class first_collector_unit_t : public RFWithCache::modified_collector_unit_t {
   public:
    first_collector_unit_t(const RFCacheConfig *rfcache_config,
                           RFWithCacheFirst *rf, RFCache *cache)
        : m_rfcache_config(rfcache_config),
          m_rf(rf),
          modified_collector_unit_t(cache) {
      DPRINTF(std::cout << "First Collector Unit Initialized" << std::endl;)
      assert(m_rfcache_config->is_first());
    }

    virtual bool allocate(register_set *pipeline_reg_set,
                          register_set *output_reg_set);
    virtual void collect_operand(unsigned operand_offset, const tag_t &tag, const op_t &op) {
      assert(m_not_ready.test(operand_offset));
      DDPRINTF(std::cout << "sid: " << m_shader->get_sid() << " op_id: " << operand_offset
                         << " cid: " << get_id()
                         << " V: " << m_src_op[operand_offset].valid() << " Res: "
                         << m_src_op[operand_offset].valid() << " not_r" << std::endl;)
      m_rf->fill_srcop(tag, op);
      m_not_ready.reset(operand_offset);
      m_src_op[operand_offset].unreserve();
    }

    virtual void dispatch();

   private:
    const RFCacheConfig *m_rfcache_config;
    RFWithCacheFirst *m_rf;
  };
};

class ReplacementController {
 public:
  ReplacementController(size_t n_entries)
      : m_n_entries{n_entries}, m_n_locked{0} {}
  virtual void allocate_and_lock(const tag_t &) = 0;
  virtual void allocate(const tag_t &) = 0;
  virtual void update_and_lock(const tag_t &) = 0;
  virtual void access(replacement_order_list_iter item) = 0;

  void lock(const tag_t &tag) {
    assert(m_ordered_replace_candidates_refs.find(tag) !=
           m_ordered_replace_candidates_refs.end());
    if (!m_ordered_replace_candidates_refs[tag]->second) {
      m_ordered_replace_candidates_refs[tag]->second = true;
      m_n_locked++;
    }
    DPRINTF(std::cout << "First: lock wid: " << tag.first
                      << " regid: " << tag.second
                      << " m_n_locked: " << m_n_locked << " locked <wid: "
                      << m_ordered_replace_candidates_refs[tag]->first.first
                      << ", regid: "
                      << m_ordered_replace_candidates_refs[tag]->first.second
                      << " locked: "
                      << m_ordered_replace_candidates_refs[tag]->second
                      << std::endl;)

    assert(check_valid_sizes());
  }
  void unlock(const tag_t &tag) {
    if (m_ordered_replace_candidates_refs[tag]->second) {
      m_ordered_replace_candidates_refs[tag]->second = false;

      assert(m_n_locked > 0);
      m_n_locked--;
    }
    DPRINTF(std::cout << "First: unlock wid: " << tag.first
                      << " regid: " << tag.second
                      << " m_n_locked: " << m_n_locked << std::endl;)
    assert(check_valid_sizes());
  }
  virtual std::pair<bool, tag_t> do_replacement() = 0;
  std::pair<bool, tag_t> get_last_replaced() const { return m_last_replaced; }

  bool check_valid_sizes() {
    size_t allocated_entries_list = m_ordered_replace_candidates.size();
    size_t allocated_entries_hash_table =
        m_ordered_replace_candidates_refs.size();
    assert(allocated_entries_hash_table == allocated_entries_list);
    size_t unlocked_allocated_enties = allocated_entries_list - m_n_locked;
    assert(unlocked_allocated_enties >= 0);
    return unlocked_allocated_enties + m_n_locked <= m_n_entries;
  }

 protected:
  size_t m_n_entries;
  size_t m_n_locked;

  replacement_order_list_t m_ordered_replace_candidates;
  std::unordered_map<tag_t, replacement_order_list_iter, hash_pair>
      m_ordered_replace_candidates_refs;
  std::pair<bool, tag_t> m_last_replaced;
};

class FIFOReplacementController : public ReplacementController {
 public:
  FIFOReplacementController(size_t n_entries)
      : ReplacementController(n_entries) {
    DPRINTF(std::cout << "FIFO Replacement Policy Added" << std::endl;)
  }
  virtual void allocate(const tag_t &tag) {
    m_ordered_replace_candidates.push_back(std::pair<tag_t, bool>(tag, false));
    auto last_iter = ++m_ordered_replace_candidates.rbegin();
    m_ordered_replace_candidates_refs[tag] = last_iter.base();

    m_last_replaced = do_replacement();
    DPRINTF(
        // dump();
        std::cout << "First allocate tag<" << tag.first << ", " << tag.second
                  << "> wid: "
                  << m_ordered_replace_candidates_refs[tag]->first.first
                  << " regid: "
                  << m_ordered_replace_candidates_refs[tag]->first.second
                  << " locked: "
                  << m_ordered_replace_candidates_refs[tag]->second
                  << " m_n_locked: " << m_n_locked << std::endl;)
    assert(check_valid_sizes());
  }
  void dump() {
    for (auto order : m_ordered_replace_candidates) {
      unsigned wid = order.first.first;
      unsigned regid = order.first.second;
      std::cout << "order_candidates: tag<" << wid << ", " << regid
                << ">, locked: " << order.second << std::endl;
    }
    for (auto order_ref : m_ordered_replace_candidates_refs) {
      std::cout << "order_refs: tag<" << order_ref.second->first.first << ", "
                << order_ref.second->first.second
                << ">, locked: " << order_ref.second->second << std::endl;
    }
  }
  void lock_end() {
    assert(!m_ordered_replace_candidates.empty());
    auto end_tag = m_ordered_replace_candidates.back().first;

    lock(end_tag);
  }

  virtual void allocate_and_lock(const tag_t &tag) {
    allocate(tag);
    lock_end();
  }
  virtual std::pair<bool, tag_t> do_replacement() {
    bool have_replacement = false;
    tag_t tag;
    if (m_ordered_replace_candidates.size() > m_n_entries) {
      assert(m_ordered_replace_candidates.size() == m_n_entries + 1);
      have_replacement = true;
      for (replacement_order_list_iter iter =
               m_ordered_replace_candidates.begin();
           iter != m_ordered_replace_candidates.end(); iter++)
        if (!iter->second) {
          tag = iter->first;
          m_ordered_replace_candidates.erase(iter);
          assert(m_ordered_replace_candidates_refs.find(tag) !=
                 m_ordered_replace_candidates_refs.end());
          m_ordered_replace_candidates_refs.erase(tag);
          break;
        }
    }
    return std::pair<bool, tag_t>(have_replacement, tag);
  }
  virtual void update_and_lock(const tag_t &tag) { lock(tag); }
  virtual void access(replacement_order_list_iter item){};
};
class FirstCache : public IdealRFCache {
 public:
  FirstCache(const RFCacheConfig &rfcache_config)
      : IdealRFCache(),
        m_n_entries{rfcache_config.get_n_blocks()},
        m_n_locked{0},
        m_arbiter{nullptr},
        m_replacement_controller{
            rfcache_config.get_replacement_policy() == RF_FIFO
                ? new FIFOReplacementController(rfcache_config.get_n_blocks())
                : nullptr} {
    DPRINTF(std::cout << "First Cache initizlied number of entries: "
                      << m_n_entries << std::endl);
  }
  virtual ~FirstCache() {
    if (m_replacement_controller) delete m_replacement_controller;
  }
  bool can_release_src(const tag_t &tag) {
    return m_lockable_cache_table[tag].is_locked() && !m_lockable_cache_table[tag].get_missed_oc_slots_mask().any();
  }
  
  void release_src(const tag_t &tag) {
    assert(!m_lockable_cache_table[tag].get_missed_oc_slots_mask().any());
    unlock(tag);
  }

  bool can_allocate_or_update(const std::vector<tag_t> &tags) {
    assert(check_valid_sizes());

    size_t total_available_space = m_n_entries - m_n_locked;

    for (auto tag : tags) {
      bool hit =
          m_lockable_cache_table.find(tag) != m_lockable_cache_table.end();
      if (hit) {
        continue;
      }
      // if miss we Ave to allocate a new entry
      if (!total_available_space--)
        return false;  // if we have pending writes and no room
    }
    return true;
  }

  void lock(const tag_t &tag) {
    assert(!m_lockable_cache_table[tag].is_locked());
    // lock existing unlocked
    m_lockable_cache_table[tag].lock();
    m_replacement_controller->update_and_lock(tag);
    m_n_locked++;
    assert(check_valid_sizes());
  }
  bool has_data(const tag_t &tag) {
    bool has_data = m_lockable_cache_table[tag].get_data().valid();
    DDPRINTF(std::cout << "<" << tag.first << ", " << tag.second << "> " << (has_data ? "has": "doesn't have") << " data" << std::endl);
    return has_data;
  } 
  bool allocate_and_lock_for_read(const tag_t &tag, unsigned oc_slot,
                                  bool &was_reserve) {
    DPRINTF(std::cout << "First: allocate_and_lock_for_read sid: "
                      << m_shader->get_sid() << " wid: " << tag.first
                      << " reg_id: " << tag.first << std::endl;)
    bool miss =
        m_lockable_cache_table.find(tag) == m_lockable_cache_table.end();
    if (miss) {
      // allocate and lock
      m_lockable_cache_table[tag].allocate_and_lock_for_read(
          oc_slot);  // may lock an already locked entry
      m_replacement_controller->allocate_and_lock(tag);
      handle_replacement();
      m_n_locked++;
      assert(check_valid_sizes());
      was_reserve = false;
      return false;
    } else if (!has_data(tag)) {
      // already locked for other source reservation
      m_lockable_cache_table[tag].reserve(oc_slot);
      was_reserve = true;
      return false;
    } else {
      // hit case
      lock(tag);
      m_lockable_cache_table[tag].reserve(oc_slot);
      was_reserve = true;
      assert(check_valid_sizes());
      return true;
    }
    
  }
  void allocate_and_lock_for_write(const tag_t &tag) {
    DPRINTF(std::cout << "First: allocate_and_lock_for_write sid: "
                      << m_shader->get_sid() << " wid: " << tag.first
                      << " reg_id: " << tag.first << std::endl;)
    bool miss =
        m_lockable_cache_table.find(tag) == m_lockable_cache_table.end();
    if (miss) {
      // allocate and lock
      m_lockable_cache_table[tag]
          .allocate_and_lock_for_write();  // may lock an already locked entry
      m_replacement_controller->allocate_and_lock(tag);
      handle_replacement();
      m_n_locked++;
      assert(check_valid_sizes());
    } else if (m_lockable_cache_table[tag].is_locked()) {
      // already locked

    } else {
      lock(tag);
    }
  }

  void handle_replacement() {
    auto replaced_item = m_replacement_controller->get_last_replaced();
    if (replaced_item.first) {
      DDPRINTF(std::cout << "First replacement sid: " << m_shader->get_sid()
                         << " wid: " << replaced_item.second.first
                         << " regid: " << replaced_item.second.second;)
      auto replaced_tag = replaced_item.second;
      assert(m_lockable_cache_table.find(replaced_tag) !=
             m_lockable_cache_table.end());
      assert(!m_lockable_cache_table[replaced_tag].is_locked());
      if (m_lockable_cache_table[replaced_tag].is_dirty()) {
        // replacing a dirty block
        auto data = m_lockable_cache_table[replaced_tag].get_data();
        DPRINTF(std::cout << " write added " << std::endl;)
        m_arbiter->add_write_request(data);
      }
      m_lockable_cache_table.erase(replaced_tag);
    }
  }

  void allocate_or_update_for_write(const tag_t &tag) {
    DPRINTF(std::cout << "First: allocate_or_update_for_write sid: "
                      << m_shader->get_sid() << " wid: " << tag.first
                      << " reg_id: " << tag.first << std::endl;)

    bool hit = m_lockable_cache_table.find(tag) != m_lockable_cache_table.end();
    if (hit) {
      // update
      assert(!m_lockable_cache_table[tag].is_locked());
    }

    m_lockable_cache_table[tag] = first_cache_block_t();
    m_lockable_cache_table[tag].allocate_for_write();
    assert(check_valid_sizes());
  }
  void unlock(const tag_t &tag) {
    DPRINTF(std::cout << "First: unlock sid: " << m_shader->get_sid()
                      << " wid: " << tag.first << " reg_id: " << tag.first
                      << std::endl;)

    assert(m_lockable_cache_table.find(tag) != m_lockable_cache_table.end());
    assert(m_lockable_cache_table[tag].is_locked());
    m_lockable_cache_table[tag].unlock();
    m_replacement_controller->unlock(tag);
    assert(m_n_locked > 0);
    m_n_locked--;
    assert(check_valid_sizes());
  }

  void write(const RFWithCacheFirst::op_t &op) {
    DPRINTF(std::cout << "First: write sid: " << m_shader->get_sid()
                      << " wid: " << op.get_wid() << " regid: " << op.get_reg()
                      << std::endl;)
    tag_t tag(op.get_wid(), op.get_reg());
    bool hit = m_lockable_cache_table.find(tag) != m_lockable_cache_table.end();
    assert(hit && !m_lockable_cache_table[tag].is_locked());

    fill_block_and_unlock(tag, op);
  }
  void fill_block_and_unlock(const tag_t& tag, const RFWithCacheFirst::op_t& op) {
    assert(op.valid());
    DDPRINTF(std::cout << "Fill block and unlock <" << tag.first << ", " << tag.second << "> " << std::endl;)
    m_lockable_cache_table[tag].fill_block_and_unlock(op);
  }
  void fill_block(const tag_t &tag, const RFWithCacheFirst::op_t &op) {
    DDPRINTF(std::cout << "Fill block <" << tag.first << ", " << tag.second << ">" << std::endl;)
    assert(op.valid());
    m_lockable_cache_table[tag].fill_block(op);
  }
  void set_arbiter(opndcoll_rfu_t::arbiter_t *arbiter) {
    m_arbiter = static_cast<RFWithCacheFirst::first_arbiter_t *>(arbiter);
  }
  virtual missed_oc_slots_t service_read_miss(const tag_t &tag) {
    DDPRINTF(
        std::cout
            << "sid: " << m_shader->get_sid() << " service read miss: "
            << "wid: " << tag.first << " reg_id: " << tag.second
            << " table_size: " << m_lockable_cache_table.size()
            << " miss slots: "
            << m_lockable_cache_table[tag].get_missed_oc_slots_mask().count()
            << std::endl;)
    assert(m_lockable_cache_table[tag].get_missed_oc_slots_mask().count());
    assert(m_lockable_cache_table[tag].is_locked());
    missed_oc_slots_t tmp =
        m_lockable_cache_table[tag].get_missed_oc_slots_mask();
    m_lockable_cache_table[tag].reset_missed_oc_slots_mask();
    return tmp;
  }
  void dump(){
    for (auto entry :m_lockable_cache_table) {
      std::cout << "<" << entry.first.first << ',' << entry.first.second << "> ";
      entry.second.dump();
    }
    auto replacement_controller = static_cast<FIFOReplacementController *>(m_replacement_controller);
    replacement_controller->dump();
  }
 private:
  bool check_valid_sizes() const {
    size_t allocated_entries = m_lockable_cache_table.size();
    size_t unlocked_allocated_enties = allocated_entries - m_n_locked;
    assert(unlocked_allocated_enties >= 0);
    return unlocked_allocated_enties + m_n_locked <= m_n_entries &&
           m_replacement_controller->check_valid_sizes();
  }
  std::unordered_map<tag_t, first_cache_block_t, hash_pair>
      m_lockable_cache_table;
  std::list<tag_t> replacement_ordering;
  size_t m_n_entries;
  size_t m_n_locked;
  RFWithCacheFirst::first_arbiter_t *m_arbiter;
  ReplacementController *m_replacement_controller;
};