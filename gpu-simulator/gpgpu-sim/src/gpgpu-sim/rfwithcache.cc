#include <algorithm>
#include <utility>

#include "shader.h"

RFWithCache::RFWithCache(const shader_core_config *config)
    : m_rfcache_config(config->m_rfcache_config),
      m_oc_allocator(config->gpgpu_operand_collector_num_units_gen,
                     config->max_warps_per_shader) {}

RFWithCache::modified_collector_unit_t::modified_collector_unit_t(
    std::size_t sz)
    : m_rfcache(sz) {
  DPRINTF("modified collector unit constructed")
}

void RFWithCache::add_cu_set(unsigned set_id, unsigned num_cu,
                             unsigned num_dispatch) {
  m_cus[set_id].reserve(num_cu);  // this is necessary to stop pointers in m_cu
                                  // from being invalid do to a resize;
  for (unsigned i = 0; i < num_cu; i++) {
    m_cus[set_id].push_back(
        std::make_shared<modified_collector_unit_t>(m_rfcache_config.size()));
    m_cu.push_back(m_cus[set_id].back());
  }
  // for now each collector set gets dedicated dispatch units.
  for (unsigned i = 0; i < num_dispatch; i++) {
    m_dispatch_units.push_back(opndcoll_rfu_t::dispatch_unit_t(&m_cus[set_id]));
  }
}

RFWithCache::OCAllocator::OCAllocator(std::size_t num_ocs,
                                      std::size_t num_warps_per_shader)
    : m_n_ocs{num_ocs},
      m_lru_policy{num_ocs},
      m_n_warps_per_shader{num_warps_per_shader} {
  DDPRINTF("OCAllocator Constructed Size:" << num_ocs)
}

void RFWithCache::OCAllocator::add_oc(
    RFWithCache::modified_collector_unit_t &oc) {
  DDPRINTF("OCAllocator Add OC : " << oc.get_id())
  m_n_available++;
  assert(m_n_available <= m_n_ocs);
  assert(m_n_warps_per_shader + m_n_ocs - 1 < -1U);
  unsigned invalid_unique_wid =
      -m_n_available;  // initialized info table with invalid ids
  m_info_table.insert(
      {invalid_unique_wid,
       {oc, true}});  // initialize ref to oc and make it available
  unsigned replaced_wid;
  auto has_replacement = m_lru_policy.refer(invalid_unique_wid, replaced_wid);
  assert(!has_replacement);
}

void RFWithCache::init(unsigned num_banks, shader_core_ctx *shader) {
  m_shader = shader;
  m_arbiter = new modified_arbiter_t();
  m_arbiter->init(m_cu.size(), num_banks);
  // for( unsigned n=0; n<m_num_ports;n++ )
  //    m_dispatch_units[m_output[n]].init( m_num_collector_units[n] );
  m_num_banks = num_banks;
  m_bank_warp_shift = 0;
  m_warp_size = shader->get_config()->warp_size;
  m_bank_warp_shift = (unsigned)(int)(log(m_warp_size + 0.5) / log(2.0));
  assert((m_bank_warp_shift == 5) || (m_warp_size != 32));

  sub_core_model = shader->get_config()->sub_core_model;
  m_num_warp_sceds = shader->get_config()->gpgpu_num_sched_per_core;
  if (sub_core_model)
    assert(num_banks % shader->get_config()->gpgpu_num_sched_per_core == 0);
  m_num_banks_per_sched =
      num_banks / shader->get_config()->gpgpu_num_sched_per_core;

  for (unsigned j = 0; j < m_cu.size(); j++) {
    m_cu[j]->init(j, num_banks, m_bank_warp_shift, shader->get_config(), this,
                  sub_core_model, m_num_banks_per_sched);
    m_oc_allocator.add_oc(
        *static_cast<RFWithCache::modified_collector_unit_t *>(m_cu[j].get()));
  }
  m_initialized = true;
}

void RFWithCache::allocate_cu(unsigned port_num) {
  input_port_t &inp = m_in_ports[port_num];
  for (unsigned i = 0; i < inp.m_in.size(); i++) {
    if ((*inp.m_in[i]).has_ready()) {
      DDPRINTF("OCAllocator Allocate New OC")
      warp_inst_t **pipeline_reg = inp.m_in[i]->get_ready();
      warp_inst_t inst = **pipeline_reg;
      auto allocated = m_oc_allocator.allocate(inst);
      if (allocated.first) {
        allocated.second.allocate(inp.m_in[i], inp.m_out[i]);
        m_arbiter->add_read_requests(&allocated.second);
      } else {
        DDDPRINTF("OCAllocator stalled")
      }
      break;  // can only service a single input, if it failed it will fail for
              // others.
    }
  }
}

std::pair<bool, RFWithCache::modified_collector_unit_t &>
RFWithCache::OCAllocator::allocate(const warp_inst_t &inst) {
  if (m_n_available == 0) {
    return std::pair<bool, modified_collector_unit_t &>{
        false, m_info_table.begin()->second.m_oc};
  }
  auto wid = inst.warp_id();
  if (m_info_table.find(wid) == m_info_table.end()) {  // warp_id is new
    unsigned replaced_candidate_wid;
    bool has_replacement =
        m_lru_policy.get_replacement_candidate(wid, replaced_candidate_wid);
    assert(has_replacement);
    auto &replaced_oc = m_info_table.find(replaced_candidate_wid)->second.m_oc;

    if (!replaced_oc.can_be_allocated(inst)) {
      return std::pair<bool, modified_collector_unit_t &>{
          false, m_info_table.begin()->second.m_oc};
    }

    unsigned replaced_wid;
    auto had_replacement = m_lru_policy.refer(wid, replaced_wid);
    assert(had_replacement && (replaced_wid == replaced_candidate_wid));
    auto replaced_info_iter = m_info_table.find(replaced_wid);
    assert(replaced_info_iter != m_info_table.end() &&
           replaced_info_iter->second.m_availble == true);

    m_info_table.insert({wid, {replaced_info_iter->second.m_oc, false}});
    m_info_table.erase(replaced_info_iter);
    auto new_element_iter = m_info_table.find(wid);
    new_element_iter->second.m_availble = false;
    m_n_available--;
    m_lru_policy.lock(wid);
    dump();
    return std::pair<bool, modified_collector_unit_t &>(
        true, new_element_iter->second.m_oc);
  } else {
    auto iter = m_info_table.find(wid);

    if (iter->second.m_availble == false ||
        !(iter->second.m_oc.can_be_allocated(inst))) {
      return std::pair<bool, modified_collector_unit_t &>(false,
                                                          iter->second.m_oc);
    }
    iter->second.m_availble = false;
    m_n_available--;
    unsigned replaced_wid;
    auto had_replacement = m_lru_policy.refer(wid, replaced_wid);
    assert(!had_replacement);
    m_lru_policy.lock(wid);
    dump();
    return std::pair<bool, modified_collector_unit_t &>(true,
                                                        iter->second.m_oc);
  }
}

void RFWithCache::dispatch_ready_cu() {
  for (unsigned p = 0; p < m_dispatch_units.size(); ++p) {
    dispatch_unit_t &du = m_dispatch_units[p];
    collector_unit_t *cu = du.find_ready();
    if (cu) {
      cu->dispatch();
      m_oc_allocator.dispatch(cu->get_warp_id());
    }
  }
}

void RFWithCache::OCAllocator::dispatch(unsigned wid) {
  DDPRINTF("OCAllocator Dispatch " << wid)
  assert(m_n_available < m_n_ocs);
  m_n_available++;
  auto iter = m_info_table.find(wid);
  assert(iter != m_info_table.end() && iter->second.m_availble == false);
  iter->second.m_availble = true;
  dump();
  m_lru_policy.unlock(wid);
}

void RFWithCache::OCAllocator::dump() {
  DDPRINTF("OCAllocator Dump")
  std::stringstream ss;
  ss << "Info Table\n";
  for (auto el : m_info_table) {
    ss << "<" << el.first << ", " << (el.second.m_availble ? 'F' : 'O') << ", "
       << el.second.m_oc.get_id() << "> ";
  }
  DDPRINTF(ss.str());
}

RFWithCache::modified_collector_unit_t::RFCache::RFCache(std::size_t sz)
    : m_n_available{sz}, m_rpolicy(sz), m_size(sz), m_n_locked{0} {
  DDDPRINTF("Constructed Size: " << sz)
}

bool RFWithCache::modified_collector_unit_t::allocate(
    register_set *pipeline_reg_set, register_set *output_reg_set) {
  DDDPRINTF("New OC Allocation")
  warp_inst_t **pipeline_reg = pipeline_reg_set->get_ready();
  assert((pipeline_reg) && !((*pipeline_reg)->empty()));
  warp_inst_t &inst = **pipeline_reg;
  auto srcs = get_src_ops(inst);  // extract all source ops

  assert(m_rfcache.can_allocate(
      srcs, inst.warp_id()));  // it should be accepted by cache
  auto last_warp_id = m_warp_id;
  m_warp_id = inst.warp_id();
  if (m_warp_id != last_warp_id) {  // new warp allocated to this oc
    DDDPRINTF("OC Preempted lwid: " << last_warp_id << " nwid: " << m_warp_id)
    // flush everyting
    m_rfcache.flush();
  }
  m_free = false;
  m_output_register = output_reg_set;
  unsigned op = 0;
  for (auto src : srcs) {
    auto access_status = m_rfcache.read_access(src, m_warp_id);
    if (access_status ==
        RFWithCache::modified_collector_unit_t::access_t::Miss) {
      // only for missed values we need to wait
      m_src_op[op] =
          op_t(this, op, src, m_num_banks, m_bank_warp_shift, m_sub_core_model,
               m_num_banks_per_sched, inst.get_schd_id());
      m_not_ready.set(op);
    } else {
      m_src_op[op] = op_t();
    }
    op++;
  }

  pipeline_reg_set->move_out_to(m_warp);
  return true;
}

RFWithCache::modified_collector_unit_t::access_t
RFWithCache::modified_collector_unit_t::RFCache::read_access(unsigned regid,
                                                             unsigned wid) {
  tag_t tag(wid, regid);
  if (m_cache_table.find(tag) == m_cache_table.end()) {
    DDDPRINTF("Miss in cache table <" << tag.first << ", " << tag.second << ">")
    // not present in the cache
    if (m_fill_buffer.find(tag)) {
      // present in fill_buffer
      // everything in the fill buffer has data
      DDDPRINTF("Hit in fill buffer <" << tag.first << ", " << tag.second
                                       << ">")
      return Hit;
    }
    // not present in cache and fill buffer
    // allocate an entry
    if (m_n_available > 0) {
      // there is space in the table
      m_cache_table.insert({tag, {}});
      tag_t replaced_tag;
      auto had_replacement = m_rpolicy.refer(tag, replaced_tag);
      assert(!had_replacement);
      m_n_available--;
      // m_rpolicy.lock(tag);
    } else {
      // should replace an entry
      tag_t replaced_tag;
      bool had_replaced = m_rpolicy.refer(tag, replaced_tag);
      assert(had_replaced);
      DDDPRINTF("Replaced PREG: <" << replaced_tag.first << ", "
                                   << replaced_tag.second << "> ")
      assert(!m_cache_table[replaced_tag].is_locked());
      m_cache_table.erase(replaced_tag);
      m_cache_table.insert({tag, {}});
    }
    
    lock(tag); // wait for data from banks
    
    assert(check_size());
    dump();
    return Miss;
  } else {
    // present in cache
    DDDPRINTF("Hit in cache table <" << tag.first << ", " << tag.second << ">")
    tag_t replaced_tag;
    auto had_replacement = m_rpolicy.refer(tag, replaced_tag);
    assert(!had_replacement);
    if (m_cache_table[tag].is_locked()) {
      // RF read pending
      // do nothing
    } else {
      // should wait for data
    }
    dump();
    return Hit;
  }
}

void RFWithCache::modified_arbiter_t::add_read_requests(collector_unit_t *cu) {
  auto oc = static_cast<modified_collector_unit_t *>(cu);
  const op_t *src = oc->get_operands();
  for (unsigned i = 0; i < MAX_REG_OPERANDS * 2; i++) {
    const op_t &op = src[i];
    if (oc->is_not_ready(i)) {
      DDDPRINTF("add_read_requests: <" << op.get_wid() << ", " << op.get_reg()
                                       << ">")
      assert(op.valid());
      unsigned bank = op.get_bank();
      m_queue[bank].push_back(op);
    }
  }
}

std::vector<unsigned> RFWithCache::modified_collector_unit_t::get_src_ops(
    const warp_inst_t &inst) const {
  std::vector<unsigned> srcs;
  // get src ops and remove replicas
  for (unsigned op = 0; op < MAX_REG_OPERANDS; op++) {
    int reg_num = inst.arch_reg.src[op];  // this math needs to match that used
                                          // in function_info::ptx_decode_inst
    if (reg_num < 0) {                    // valid register
      m_src_op[op] = op_t();
    } else {
      srcs.push_back(reg_num);
    }
  }

  return srcs;
}

bool RFWithCache::modified_collector_unit_t::RFCache::can_allocate(
    const std::vector<unsigned> &ops, unsigned wid) const {
  assert(m_size >=
         ops.size());       // at least we need enough space for one instruction
  assert(m_n_locked == 0);  // we allocate when all source ops had been read
  auto available_tmp = m_n_available;

  std::vector<unsigned> allocated_ops;

  for (auto op : ops) {
    tag_t tag(wid, op);
    if (std::find(allocated_ops.begin(), allocated_ops.end(), op) !=
        allocated_ops.end()) {
      // already allocated op
      continue;
    }
    if (m_cache_table.find(tag) != m_cache_table.end()) {
      // already in the table
      continue;
    }
    if (m_fill_buffer.find(tag)) {
      // already in the fill buffer
      continue;
    }
    // !table && !fill_buffer && !allocated
    if (available_tmp != 0) {
      // there is space in table for new entries (no replacement)
      available_tmp--;
    }
    // there wasn't available space in table
    // we replace entries
    allocated_ops.push_back(op);
  }
  return true;
}

bool RFWithCache::modified_collector_unit_t::can_be_allocated(
    const warp_inst_t &inst) const {
  if (inst.empty() || !m_free || !m_not_ready.none()) return false;
  auto srcs = get_src_ops(inst);
  auto wid = inst.warp_id();
  return m_rfcache.can_allocate(srcs, wid);
}

bool RFWithCache::modified_collector_unit_t::RFCache::FillBuffer::find(
    const tag_t &tag) const {
  if (std::find_if(m_buffer.begin(), m_buffer.end(),
                   [&tag](const std::pair<unsigned, unsigned> &p) -> bool {
                     return (tag.first == p.first) && (tag.second == p.second);
                   }) == m_buffer.end()) {
    return false;
  }
  return true;
}

bool RFWithCache::modified_collector_unit_t::RFCache::check_size() const {
  return (m_n_available <= m_size) && (m_n_locked <= m_size) &&
         (m_n_locked + m_n_available <= m_size);
}

bool RFWithCache::modified_collector_unit_t::RFCache::FillBuffer::
    has_pending_writes() const {
  return m_buffer.size() > 0;
}

void RFWithCache::modified_collector_unit_t::RFCache::lock(const tag_t &tag) {
  assert(m_cache_table.find(tag) != m_cache_table.end());
  assert(!m_cache_table[tag].is_locked());
  m_cache_table[tag].lock();
  m_rpolicy.lock(tag);
  m_n_locked++;
  assert(check_size());
}

void RFWithCache::modified_collector_unit_t::RFCache::unlock(const tag_t &tag) {
  assert(m_cache_table.find(tag) != m_cache_table.end());
  assert(m_cache_table[tag].is_locked());
  m_cache_table[tag].unlock();
  m_rpolicy.unlock(tag);
  m_n_locked--;
  assert(check_size());
  dump();
}

void RFWithCache::modified_collector_unit_t::RFCache::dump() {
  DDDPRINTF("Cache Table S: " << m_size << " A: " << m_n_available
                              << " L: " << m_n_locked)
  for (auto b : m_cache_table) {
    auto &block = b.second;

    block.dump(b.first);
  }
  m_rpolicy.dump();
}