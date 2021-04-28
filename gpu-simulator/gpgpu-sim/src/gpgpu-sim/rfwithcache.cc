#include <algorithm>
#include <utility>

#include "shader.h"

RFWithCache::RFWithCache(const shader_core_config *config,
                         RFCacheStats &rfcache_stats,
                         const shader_core_ctx *shader)
    : m_rfcache_config(config->m_rfcache_config),
      m_oc_allocator(config->gpgpu_operand_collector_num_units_gen,
                     config->max_warps_per_shader),
      m_rfcache_stats(rfcache_stats),
      m_shdr(shader),
      m_schedulers(nullptr) {}

RFWithCache::modified_collector_unit_t::modified_collector_unit_t(
    std::size_t sz, unsigned sid, unsigned num_oc_per_core, unsigned oc_id,
    RFCacheStats &rfcache_stats)
    : m_rfcache(sz, rfcache_stats, sid * num_oc_per_core + oc_id), m_sid(sid) {
  DPRINTF("modified collector unit constructed")
  opndcoll_rfu_t::collector_unit_t::m_cuid = oc_id;
}

void RFWithCache::add_cu_set(unsigned set_id, unsigned num_cu,
                             unsigned num_dispatch) {
  m_cus[set_id].reserve(num_cu);  // this is necessary to stop pointers in m_cu
                                  // from being invalid do to a resize;
  for (unsigned i = 0; i < num_cu; i++) {
    m_cus[set_id].push_back(std::make_shared<modified_collector_unit_t>(
        m_rfcache_config.size(), m_shdr->get_sid(),
        m_shdr->get_config()->gpgpu_operand_collector_num_units_gen, i,
        m_rfcache_stats));
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

void RFWithCache::init(unsigned num_banks, unsigned num_scheds,
                       std::vector<scheduler_unit *> *schedulers,
                       shader_core_ctx *shader) {
  m_shader = shader;
  m_n_schedulers = num_scheds;  // number of schedulers
  m_schedulers = schedulers;    // pointer to all schedulers
  assert(m_schedulers && m_n_schedulers > 0);
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

void RFWithCache::signal_schedulers(unsigned last_wid, unsigned new_wid) {
  for (unsigned i = 0; i < m_n_schedulers; i++) {
    auto scheduler = static_cast<GTOCTOScheduler *>((*m_schedulers)[i]);
    scheduler->allocate_oc(last_wid, new_wid);
  }
}

void RFWithCache::allocate_cu(unsigned port_num) {
  input_port_t &inp = m_in_ports[port_num];
  sort_next_input_port(
      port_num,
      inp);  // first we should sort inst in the input port
             // to priritize warps that are currently in the ocs
             // for (auto iter = m_prioritized_input_port.begin(); iter !=
  if (m_prioritized_input_port.size() > 0) {
    DDPRINTF("OCAllocator Allocate New OC")
#ifdef D8BUG
    dump_in_latch(port_num);
#endif
    auto next_ready_inst_outp =
        m_prioritized_input_port.front();  // the first priority in the list
    m_prioritized_input_port
        .pop_front();  // if one ready failed don't consider it again

    // warp_inst_t **pipeline_reg = inp.m_in[i]->get_ready();
    warp_inst_t **next_ready_inst =
        next_ready_inst_outp.first;  //**pipeline_reg;
    register_set *next_outp =
        next_ready_inst_outp.second;  // output port for next ready instruction
    assert(next_ready_inst && !(*next_ready_inst)->empty());
    auto allocated = m_oc_allocator.allocate(**next_ready_inst);
    if (allocated.first) {
      auto last_wid = allocated.second.get_warp_id();  // previous wid in the oc
      allocated.second.allocate(next_ready_inst, next_outp);
      auto new_wid = allocated.second.get_warp_id();  // new wid in the cache
      signal_schedulers(last_wid, new_wid);
      m_arbiter->add_read_requests(&allocated.second);
    } else {
      DDDPRINTF("OCAllocator stalled")  // can only service one input
    }
  }
}

void RFWithCache::sort_next_input_port(unsigned port_num,
                                       const input_port_t &inp) {
  std::list<std::pair<warp_inst_t **, register_set *>>
      temp;             // temp list of ready insts in the input latches 
                        // and target out ports
  if (port_num == 0) {  // only do sorting for port 0, other ports are replicas
    auto warps_in_ocs = m_oc_allocator.get_warps_in_ocs();
    unsigned reg_set_num = 0;
    for (auto in_reg_set : inp.m_in) {
      auto out_reg_set = inp.m_out[reg_set_num];
      while(warp_inst_t **next_ready_inst = in_reg_set->get_next_ready()) {
        auto ready_outp = std::make_pair(next_ready_inst, out_reg_set);
        temp.push_back(ready_outp);
      }
      assert(in_reg_set->traversed_all_regs());
      reg_set_num++;
    }
    // temp has list of ready insts
    // now we should sort them
    temp.sort([this, warps_in_ocs](
                  const std::pair<warp_inst_t **, register_set *> &lhs,
                  const std::pair<warp_inst_t **, register_set *> &rhs) -> bool {
      return this->priority_func(**lhs.first, **rhs.first, warps_in_ocs);
    });
    m_prioritized_input_port =
        temp;  // set the next priritized list of ready insts and output ports
  }
}

bool RFWithCache::priority_func(
    const warp_inst_t &lhs,
    const warp_inst_t &rhs,
    std::list<unsigned> warps_in_ocs) const {
  assert(!lhs.empty() && !rhs.empty());

  auto lhs_uid = lhs.get_uid();
  auto rhs_uid = rhs.get_uid();
  auto lhs_wid = lhs.warp_id();
  auto lhs_dwid = lhs.dynamic_warp_id();
  auto rhs_wid = rhs.warp_id();
  auto rhs_dwid = rhs.dynamic_warp_id();
  auto lhs_is_in_oc = std::find(warps_in_ocs.begin(), warps_in_ocs.end(),
                                lhs_wid) != warps_in_ocs.end();
  auto rhs_is_in_oc = std::find(warps_in_ocs.begin(), warps_in_ocs.end(),
                                rhs_wid) != warps_in_ocs.end();

  if (lhs_wid == rhs_wid) {
    if (lhs_dwid != rhs_dwid) return lhs_dwid < rhs_dwid;  
    return lhs_uid < rhs_uid;
  } else if (lhs_is_in_oc && rhs_is_in_oc) {
    return lhs_dwid < rhs_dwid;
  } else if (!lhs_is_in_oc && rhs_is_in_oc) {
    return false;
  } else if (lhs_is_in_oc && !rhs_is_in_oc) {
    return true;
  } else {
    // both are not in ocs
    return lhs_dwid < rhs_dwid;
  }
}

void RFWithCache::dump_in_latch(unsigned port_num) const {
  D8PRINTF("prioritized input latch")
  std::stringstream ss;
  auto warps_in_ocs = m_oc_allocator.get_warps_in_ocs();
  if (port_num == 0 && !warps_in_ocs.empty()) {
    ss << "warps in ocs: ";
    for (auto warp : warps_in_ocs) {
      ss << warp << " ";
    }
    D8PRINTF(ss.str());
    ss.str("");
  }
  if (m_prioritized_input_port.size() > 0) {
    ss << "list: ";
    for (auto in_latch_inst : m_prioritized_input_port) {
      auto &inst = **in_latch_inst.first;
      auto wid = inst.warp_id();
      auto dwid = inst.dynamic_warp_id();
      assert(!inst.empty());
      ss << "<" << wid << ", " << dwid <<  ", " << inst.pc << "> ";
    }
    D8PRINTF(ss.str())
  }
}

std::pair<bool, RFWithCache::modified_collector_unit_t &>
RFWithCache::OCAllocator::allocate(const warp_inst_t &inst) {
  assert(!inst.empty());
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

RFWithCache::modified_collector_unit_t::RFCache::RFCache(
    std::size_t sz, RFCacheStats &rfcache_stats, unsigned global_oc_id)
    : m_n_available{sz},
      m_rpolicy(sz),
      m_size(sz),
      m_n_lives(0),
      m_n_deads(0),
      m_n_locked{0},
      m_rfcache_stats(rfcache_stats),
      m_global_oc_id(global_oc_id) {
  DDDPRINTF("Constructed Size: " << sz)
}

bool RFWithCache::modified_collector_unit_t::allocate(
    warp_inst_t **inst_in_inp_reg, register_set *output_reg_set) {
  DDDPRINTF("New OC Allocation")
  // warp_inst_t **pipeline_reg = pipeline_reg_set->get_ready();
  assert(inst_in_inp_reg && !(*inst_in_inp_reg)->empty());
  // warp_inst_t &inst = **pipeline_reg;
  auto srcs = get_src_ops(**inst_in_inp_reg);  // extract all source ops

  assert(m_rfcache.can_allocate(
      srcs, (*inst_in_inp_reg)->warp_id()));  // it should be accepted by cache
  auto last_warp_id = m_warp_id;
  m_warp_id = (*inst_in_inp_reg)->warp_id();
  if (m_warp_id != last_warp_id) {  // new warp allocated to this oc
    DDDPRINTF("OC " << get_id() << " Preempted lwid: " << last_warp_id
                    << " nwid: " << m_warp_id)
    // flush everyting
    m_rfcache.flush();
  }
  assert(m_free);
  m_free = false;
  m_output_register = output_reg_set;
  unsigned op = 0;
  for (auto src : srcs) {
    auto pending_reuses = (*inst_in_inp_reg)->arch_reg_pending_reuses.src[op]; // get the pending reuses to this source operand
    auto access_status = m_rfcache.read_access(src, pending_reuses, m_warp_id);
    if (access_status ==
        RFWithCache::modified_collector_unit_t::access_t::Miss) {
      // only for missed values we need to wait
      m_src_op[op] =
          op_t(this, op, src, m_num_banks, m_bank_warp_shift, m_sub_core_model,
               m_num_banks_per_sched, (*inst_in_inp_reg)->get_schd_id());
      m_not_ready.set(op);
    } else {
      m_src_op[op] = op_t();
    }
    op++;
  }

  assert(m_warp->empty());
  move_warp(m_warp /*dst*/,
            *inst_in_inp_reg /*src*/);  // transfer inst from input reg to oc
  assert(!m_warp->empty());
  assert((*inst_in_inp_reg)->empty());
  m_warp->set_dst_oc_id(get_id());

  return true;
}

RFWithCache::modified_collector_unit_t::access_t
RFWithCache::modified_collector_unit_t::RFCache::read_access(unsigned regid,
                                                             int pending_reuses,
                                                             unsigned wid) {
  DDDPRINTF("Read Access Wid: " << wid << " Regid: " << regid << " PR: " << pending_reuses)
  assert(pending_reuses >= 0);
  tag_t tag(wid, regid);
  if (m_cache_table.find(tag) == m_cache_table.end()) {
    DDDPRINTF("Miss in cache table <" << tag.first << ", " << tag.second << ">")
    // not present in the cache
    if (m_fill_buffer.find(tag)) {
      // present in fill_buffer
      // everything in the fill buffer has data
      DDDPRINTF("Hit in fill buffer <" << tag.first << ", " << tag.second
                                       << ">")
      m_rfcache_stats.inc_read_hits(m_global_oc_id);
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
      allocate(tag, pending_reuses);
      // m_rpolicy.lock(tag);
    } else {
      // should replace an entry
      tag_t replaced_tag;
      bool had_replaced = m_rpolicy.refer(tag, replaced_tag);
      assert(had_replaced);
      DDDPRINTF("Replaced PREG: <" << replaced_tag.first << ", "
                                   << replaced_tag.second << "> ")
      assert(!m_cache_table[replaced_tag].is_locked());
      replace(replaced_tag);
      m_cache_table.erase(replaced_tag);
      m_cache_table.insert({tag, {}});
      allocate(tag, pending_reuses);
    }

    lock(tag);  // wait for data from banks

    assert(check_size());
    dump();
    m_rfcache_stats.inc_read_misses(m_global_oc_id);
    return Miss;
  } else {
    // present in cache
    DDDPRINTF("Hit in cache table <" << tag.first << ", " << tag.second << ">")
    tag_t replaced_tag;
    auto had_replacement = m_rpolicy.refer(tag, replaced_tag);
    assert(!had_replacement);
    assert(m_cache_table[tag].m_pending_reuses > 0);
    assert(--m_cache_table[tag].m_pending_reuses == pending_reuses);
    if(pending_reuses == 0) turn_live_to_dead(tag);
    if (m_cache_table[tag].is_locked()) {
      // RF read pending
      // do nothing
    } else {
      // should wait for data
    }
    dump();
    m_rfcache_stats.inc_read_hits(m_global_oc_id);
    assert(check_size());
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
      assert(inst.arch_reg_pending_reuses.src[op] == -1); // sanity check for pending reuse traces
    } else {
      srcs.push_back(reg_num);
      assert(inst.arch_reg_pending_reuses.src[op] >= 0); // sanity check for pending reuse traces
    }
    // ------ sanity check for consistent pending reuse traces
    if(inst.arch_reg.dst[op] <0){
      assert(inst.arch_reg_pending_reuses.dst[op] == -1);
    } else {
      assert(inst.arch_reg_pending_reuses.dst[op] >=0);
    }
    // -----------
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
 
  return m_redundant_write_tracker.find(tag) != m_redundant_write_tracker.end();
}

bool RFWithCache::modified_collector_unit_t::RFCache::check_size() const {
  return (m_n_available <= m_size) && (m_n_locked <= m_size) &&
         (m_n_locked + m_n_available <= m_size) &&
         (m_n_lives + m_n_deads + m_n_available == m_size);
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
                              << " L: " << m_n_locked << " Li: " << m_n_lives <<" De: " << m_n_deads)
  for (auto b : m_cache_table) {
    auto &block = b.second;

    block.dump(b.first);
  }
  m_rpolicy.dump();
  m_fill_buffer.dump();
}

void RFWithCache::step() {
  DDDPRINTF("Step")
  dispatch_ready_cu();
  allocate_reads();
  for (unsigned p = 0; p < m_in_ports.size(); p++) allocate_cu(p);
  m_prioritized_input_port.clear();
  cache_cycle();
  process_banks();
}

bool RFWithCache::writeback(warp_inst_t &inst) {
  assert(!inst.empty());
  std::list<unsigned> regs = m_shader->get_regs_written(inst);
  unsigned req_per_inst = 0;
  for (unsigned op = 0; op < MAX_REG_OPERANDS; op++) {
    int reg_num = inst.arch_reg.dst[op];  // this math needs to match that used
                                          // in function_info::ptx_decode_inst
    if (reg_num >= 0) {                   // valid register
      unsigned bank = register_bank(reg_num, inst.warp_id(), m_num_banks,
                                    m_bank_warp_shift, sub_core_model,
                                    m_num_banks_per_sched, inst.get_schd_id());
      if (m_arbiter->bank_idle(bank)) {
        DDDPRINTF("Writeback " << (inst.op_pipe == MEM__OP ? 'M' : 'A'))
        m_arbiter->allocate_bank_for_write(
            bank,
            op_t(&inst, reg_num, m_num_banks, m_bank_warp_shift, sub_core_model,
                 m_num_banks_per_sched, inst.get_schd_id()));
        assert(inst.arch_reg_pending_reuses.dst[op] >= 0);
        m_write_reqs.push(inst.get_dst_oc_id(), inst.warp_id(), reg_num, inst.arch_reg_pending_reuses.dst[op]);
        inst.arch_reg.dst[op] = -1;
        inst.arch_reg_pending_reuses.dst[op] = -1;
        req_per_inst++;
      } else {
        return false;
      }
    }
  }
//  assert(req_per_inst == 1);
  return true;
}
void RFWithCache::cache_cycle() {
  DDDPRINTF("RF Cache Cycle")
  // process fill buffer
  process_fill_buffer();
  // process this cycle writes
  process_writes();
}

void RFWithCache::process_writes() {
  for (auto cu : m_cu) {
    auto oc = static_cast<modified_collector_unit_t *>(cu.get());
    auto oc_id = cu->get_id();
    auto global_oc_id =
        m_shdr->get_config()->gpgpu_operand_collector_num_units_gen *
            m_shdr->get_sid() +
        oc_id;
    auto curr_wid = oc->get_warp_id();
    size_t oc_wreqs = 0;
    while (m_write_reqs.has_req(oc_id)) {
    bool wreq_to_preempted_oc = false;
      oc_wreqs++;
      m_rfcache_stats.inc_writes(m_shdr->get_sid());
      auto req = m_write_reqs.pop(oc_id);
      if (req.wid() != curr_wid) {
        wreq_to_preempted_oc = true;
        continue;
      }

      oc->process_write(req.regid(), req.pending_reuses());
    }
    DDDPRINTF("RF Writes: OCID: " << oc_id << " tot_reqs: " << oc_wreqs << " ")
  }
  assert(m_write_reqs.size() == 0);
}

void RFWithCache::WriteReqs::push(unsigned oc_id, unsigned wid,
                                  unsigned regid, int pending_reuse) {
  
  m_write_reqs[oc_id].push_back(WriteReq(wid, regid, pending_reuse));
  m_size++;
  DDDPRINTF("Push WRQ: <" << oc_id << ", " << wid << ", " << regid << ">"
                          << " S: " << m_size << " PR: " << pending_reuse)
}

bool RFWithCache::WriteReqs::has_req(unsigned oc_id) const {
  return (m_write_reqs.find(oc_id) != m_write_reqs.end());
}

void RFWithCache::WriteReqs::flush(unsigned oc_id) {
  assert(m_write_reqs.find(oc_id) != m_write_reqs.end());
  m_size -= m_write_reqs[oc_id].size();
  m_write_reqs.erase(oc_id);
}

RFWithCache::WriteReqs::WriteReq RFWithCache::WriteReqs::pop(unsigned oc_id) {
  assert(m_write_reqs.find(oc_id) != m_write_reqs.end());
  auto result = m_write_reqs[oc_id].back();
  m_write_reqs[oc_id].pop_back();
  if (m_write_reqs[oc_id].size() == 0) {
    m_write_reqs.erase(oc_id);
  }
  m_size--;
  DDPRINTF("Pop WRQ: <" << oc_id << ", " << result.m_wid << ", "
                        << result.m_regid << "> S: " << m_size << " PR: " << result.m_pendign_reuse)
  return result;
}

void RFWithCache::modified_collector_unit_t::collect_operand(unsigned op) {
  DDDPRINTF("Collect Operand <" << m_warp_id << ", " << m_src_op[op].get_reg()
                                << ">")
  m_not_ready.reset(op);
  auto wid = m_src_op[op].get_wid();
  assert(m_warp_id == wid);
  auto regid = m_src_op[op].get_reg();
  tag_t tag(wid, regid);
  m_rfcache.unlock(tag);
}

void RFWithCache::modified_collector_unit_t::RFCache::flush() {
  /*remove everything in cache and fill buffer*/
  assert(m_n_locked == 0);
  m_n_available = m_size;
  m_n_lives = 0;
  m_n_deads = 0;
  m_rpolicy.reset();
  m_cache_table.clear();
  m_fill_buffer.flush();
}

void RFWithCache::modified_collector_unit_t::process_write(unsigned regid, int pending_reuse) {
  assert(pending_reuse >= 0);
  m_rfcache.write_access(m_warp_id, regid, pending_reuse);
}

void RFWithCache::modified_collector_unit_t::RFCache::write_access(
    unsigned wid, unsigned regid, int pending_reuse) {
  DDDPRINTF("Write Access " << "Wid: " << wid << " Regid: " << regid << " PR: " << pending_reuse)
  tag_t tag(wid, regid);
  RFWithCache::modified_collector_unit_t::RFCache::FillBuffer::Entry entry;
  entry.m_tag = tag;
  entry.m_pending_reuse = pending_reuse;
  if (m_cache_table.find(tag) != m_cache_table.end()) {
    // register found in the table
    auto &block = m_cache_table[tag];
    // check value is unlocked (scoreboard should catch)
    // assert(!block.is_locked());
    // update the value
    // update the replacement policy
    tag_t replaced_tag;
    bool had_replacement = m_rpolicy.refer(tag, replaced_tag);
    assert(!had_replacement);
    // rewriting a register
    assert(block.m_pending_reuses == 0 || block.m_pending_reuses == pending_reuse); // wheather rewrite in two different insts or multiple writes because of one memory inst
    replace(tag); // value is dead and it could be replaced with a live or dead value depending on the pending_reuse value
    allocate(tag, pending_reuse);
    assert(check_size());
  } else {
    // register not found in the table
    // if fill buffer has entries try to put the value in the fill buffer
    // else try to put it in the table
    if (m_fill_buffer.has_pending_writes()) {
      // fill buffer has entries
      // search in the fill buffer
      if (m_fill_buffer.find(tag)) {
        // fill buffer has pending write to the same reg
        auto was_not_first = m_fill_buffer.push_back(entry);
        assert(was_not_first);
      } else {
        // there is no pending write to the same reg
        auto was_not_first = m_fill_buffer.push_back(entry);
        assert(!was_not_first);
      }
    } else {
      // fill buffer is empty
      // try to put the value in the table
      if (m_n_available > 0) {
        // there is space in the table
        // add the value in the table, there is no replacement
        m_cache_table.insert({tag, {}});
        m_n_available--;
        tag_t replaced_tag;
        auto had_replacement = m_rpolicy.refer(tag, replaced_tag);
        assert(!had_replacement);
        allocate(tag, pending_reuse);
        assert(check_size());
      } else {
        // all entries are locked -> add to fill buffer
        // there is unlocked entries in the table -> replace one
        if (m_n_locked == m_size) {
          // all entries are locked
          // add to the fill buffer
          auto was_not_first = m_fill_buffer.push_back(entry);
          assert(!was_not_first);
        } else {
          // there is replacement available
          // replace one entry
          tag_t replaced_tag;
          bool had_replacement = m_rpolicy.refer(tag, replaced_tag);
          assert(had_replacement);
          assert((m_cache_table.find(replaced_tag) != m_cache_table.end()) &&
                 !m_cache_table[replaced_tag].is_locked());
          replace(replaced_tag);
          m_cache_table.erase(
              replaced_tag);  // remove replaced entry from table
          m_cache_table.insert({tag, {}});
          allocate(tag, pending_reuse);
          assert(check_size());
        }
      }
    }
  }
  dump();
  assert(check_size());
}

void RFWithCache::modified_collector_unit_t::RFCache::FillBuffer::dump() {
  if (m_buffer.size() > 0) {
    DDDPRINTF("Fill Buffer")
    std::stringstream ss;
    for (auto pending_writes : m_buffer) {
      ss << "<" << pending_writes.m_tag.first << ", " << pending_writes.m_tag.second
         << "> ";
    }
    ss << std::endl;
    ss << "Redundant Write Tracker";
    for (auto el : m_redundant_write_tracker) {
      ss << "<" << el.first.first << ", " << el.first.second
         << "> : " << el.second << std::endl;
    }
    DDPRINTF(ss.str())
  }
}

bool RFWithCache::modified_collector_unit_t::RFCache::FillBuffer::push_back(
    const RFWithCache::modified_collector_unit_t::RFCache::FillBuffer::Entry &entry) {
  m_buffer.push_back(entry);
  m_redundant_write_tracker[entry.m_tag]++;
  return (m_redundant_write_tracker[entry.m_tag] > 1);
}

void RFWithCache::process_fill_buffer() {
  for (auto cu : m_cu) {
    auto oc = static_cast<modified_collector_unit_t *>(cu.get());
    oc->process_fill_buffer();
  }
}

void RFWithCache::modified_collector_unit_t::process_fill_buffer() {
  DDDPRINTF("Process Fill Buffer in OC<" << get_id() << ">")
  m_rfcache.process_fill_buffer();
}

void RFWithCache::modified_collector_unit_t::RFCache::process_fill_buffer() {
  if (m_fill_buffer.has_pending_writes()) {
    assert(1);
    RFWithCache::modified_collector_unit_t::RFCache::FillBuffer::Entry entry;
    auto has_no_pending_updates = m_fill_buffer.pop_front(entry);
    if (has_no_pending_updates) {
      assert(m_cache_table.find(entry.m_tag) == m_cache_table.end());
      // there is pending write in the fill buffer
      if (m_n_available > 0) {
        // there is available entry in the cache
        // replacement not required
        tag_t replaced_tag;
        auto had_replacement = m_rpolicy.refer(entry.m_tag, replaced_tag);
        assert(!had_replacement);
        m_cache_table.insert({entry.m_tag, {}});
        allocate(entry.m_tag, entry.m_pending_reuse);
        m_n_available--;
        assert(check_size());
      } else {
        // no available entry
        if (m_n_locked == m_size) {
          // all entries are locked
          // we should stall
        } else {
          // replacement should happen
          tag_t replaced_tag;
          auto had_replacement = m_rpolicy.refer(entry.m_tag, replaced_tag);
          assert(had_replacement);
          assert(m_cache_table.find(replaced_tag) != m_cache_table.end());
          assert(!m_cache_table[replaced_tag].is_locked());
          replace(replaced_tag);
          m_cache_table.erase(replaced_tag);
          m_cache_table.insert({entry.m_tag, {}});
          allocate(entry.m_tag, entry.m_pending_reuse);
          assert(check_size());
        }
      }
    } else {
      // register has been updated by next writes
    }
  } else {
    // there is no pending write in the fill buffer
  }
  assert(check_size());
  m_rfcache_stats.reg_fill_buffer_size(m_fill_buffer.size());
}

bool RFWithCache::modified_collector_unit_t::RFCache::FillBuffer::pop_front(
    Entry &front_entry) {
  front_entry = m_buffer.front();
  m_buffer.pop_front();
  assert(m_redundant_write_tracker.find(front_entry.m_tag) !=
         m_redundant_write_tracker.end());
  auto pending_updates = --m_redundant_write_tracker[front_entry.m_tag];
  if (!pending_updates) {
    m_redundant_write_tracker.erase(front_entry.m_tag);
  }
  return (pending_updates == 0);
}