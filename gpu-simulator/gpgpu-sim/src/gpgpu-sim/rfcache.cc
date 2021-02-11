#include "rfcache.h"

CACHE_ACCESS_STAT_t IdealRFCache::read_access(const tag_t &tag,
                                              unsigned oc_slot) {
  DPRINTF(std::cout << "sid: " << m_shader->get_sid()
                    << " read access: table_size: " << m_cache_table.size()
                    << " ";)

  unsigned sid = m_shader->get_sid();
  if (m_cache_table.find(tag) == m_cache_table.end()) {
    DPRINTF(std::cout << "Miss: <" << tag.first << ", " << tag.second
                      << ">, oc_slot: " << m_cache_table[tag].count()
                      << std::endl;)
    m_cache_table[tag].set(oc_slot);
    m_rfcache_stats->incread_miss(sid);
    return Miss;
  } else if (m_cache_table[tag].any()) {
    DPRINTF(std::cout << "Reservation: <" << tag.first << ", " << tag.second
                      << ">, oc_slot: " << m_cache_table[tag].count()
                      << std::endl;)
    m_cache_table[tag].set(oc_slot);
    m_rfcache_stats->incread_reservation(sid);
    return Reservation;
  } else {
    DPRINTF(std::cout << "Hit: <" << tag.first << ", " << tag.second
                      << ">, oc_slot: " << m_cache_table[tag].count()
                      << std::endl;)
    m_rfcache_stats->incread_hit(sid);
    return Hit;
  }
}

missed_oc_slots_t IdealRFCache::service_read_miss(const tag_t &tag) {
  DPRINTF(std::cout << "sid: " << m_shader->get_sid() << " service read miss: "
                    << "wid: " << tag.first << " reg_id: " << tag.second
                    << " table_size: " << m_cache_table.size()
                    << " miss slots: " << m_cache_table[tag].count()
                    << std::endl;)
  assert(m_cache_table[tag].count());
  missed_oc_slots_t tmp = m_cache_table[tag];
  m_cache_table[tag].reset();
  return tmp;
}

CACHE_ACCESS_STAT_t IdealRFCache::write_access(const tag_t &tag) {
  DPRINTF(std::cout << "sid: " << m_shader->get_sid()
                    << " cache write access: wid: " << tag.first
                    << " reg_id: " << tag.second;)
  unsigned sid = m_shader->get_sid();
  if (m_cache_table.find(tag) == m_cache_table.end()) {
    DPRINTF(std::cout << " Miss: table_size: " << m_cache_table.size()
                      << std::endl;)
    m_rfcache_stats->incwrite_miss(sid);
    return Miss;
  } else {
    DPRINTF(std::cout << " Hit: table_size: " << m_cache_table.size()
                      << " miss_oc_slots: " << m_cache_table[tag].count()
                      << std::endl;)
    assert(!m_cache_table[tag].any());
    m_rfcache_stats->incwrite_hit(sid);
    return Hit;
  }
}

void IdealRFCache::allocate_for_write(const tag_t &tag) {
  DPRINTF(std::cout << "sid: " << m_shader->get_sid()
                    << " Allocate for write: wid: " << tag.first
                    << " reg_id: " << tag.second << std::endl;)
  assert(m_cache_table.find(tag) == m_cache_table.end());
  m_cache_table[tag] = missed_oc_slots_t();
}

RFWithCache::~RFWithCache() { delete m_cache; }

void RFWithCache::init(unsigned num_banks, shader_core_ctx *shader) {
  m_shader = shader;
  assert(m_shader->get_config()->gpgpu_operand_collector_num_units_gen <=
         MAX_OC_COUNT);
  m_rfcache_stats = m_shader->m_stats->m_rfcache_stats;
  opndcoll_rfu_t::init(num_banks, shader);
  m_cache->init(m_shader);

  delete m_arbiter;
  m_arbiter = new modified_arbiter_t(m_shader);
  m_arbiter->init(m_cu.size(), num_banks);
}

void RFWithCache::add_cu_set(unsigned set_id, unsigned num_cu,
                             unsigned num_dispatch) {
  m_cus[set_id].reserve(num_cu);  // this is necessary to stop pointers in m_cu
                                  // from being invalid do to a resize;
  for (unsigned i = 0; i < num_cu; i++) {
    m_cus[set_id].push_back(new modified_collector_unit_t(m_cache));
    m_cu.push_back(m_cus[set_id].back());
  }
  // for now each collector set gets dedicated dispatch units.
  for (unsigned i = 0; i < num_dispatch; i++) {
    m_dispatch_units.push_back(dispatch_unit_t(&m_cus[set_id]));
  }
}
void RFWithCache::allocate_reads() {
  // process read requests that do not have conflicts
  std::list<op_t> allocated = m_arbiter->allocate_reads();
  std::map<unsigned, op_t> read_ops;
  for (std::list<op_t>::iterator r = allocated.begin(); r != allocated.end();
       r++) {
    const op_t &rr = *r;
    unsigned reg = rr.get_reg();
    unsigned wid = rr.get_wid();
    unsigned bank =
        register_bank(reg, wid, m_num_banks, m_bank_warp_shift, sub_core_model,
                      m_num_banks_per_sched, rr.get_sid());
    m_arbiter->allocate_for_read(bank, rr);
    read_ops[bank] = rr;
  }
  std::map<unsigned, op_t>::iterator r;
  for (r = read_ops.begin(); r != read_ops.end(); ++r) {
    op_t &op = r->second;
    unsigned wid = op.get_wid();
    unsigned reg = op.get_reg();
    tag_t tag(wid, reg);
    missed_oc_slots_t missed_oc_slots = m_cache->service_read_miss(tag);
    for (unsigned oc_slot = 0; missed_oc_slots.any();
         oc_slot++, missed_oc_slots >>= 1) {
      if (!missed_oc_slots.test(0)) continue;
      unsigned cu = oc_slot / MAX_SLOTS_PER_OC;
      unsigned operand = oc_slot % MAX_SLOTS_PER_OC;
      assert(!m_cu[cu]->is_free());
      DPRINTF(std::cout << "m_sid: " << m_shader->get_sid()
                        << " Collect Operand: wid: " << wid << " reg_id: "
                        << reg << " missed_count: " << missed_oc_slots.count();)
      m_cu[cu]->collect_operand(operand);
    }
  }
}

bool RFWithCache::writeback(warp_inst_t &inst) {
  DPRINTF(std::cout << "sid: " << m_shader->get_sid() << " WB: ";)
  assert(!inst.empty());
  for (unsigned op = 0; op < MAX_REG_OPERANDS; op++) {
    int reg_num = inst.arch_reg.dst[op];  // this math needs to match that used
                                          // in function_info::ptx_decode_inst
    if (reg_num >= 0) {                   // valid register
      unsigned wid = inst.warp_id();
      tag_t tag(wid, reg_num);
      if (m_cache->write_access(tag) == Miss) {
        unsigned bank = register_bank(
            reg_num, wid, m_num_banks, m_bank_warp_shift, sub_core_model,
            m_num_banks_per_sched, inst.get_schd_id());
        if (m_arbiter->bank_idle(bank)) {
          m_arbiter->allocate_bank_for_write(
              bank,
              op_t(&inst, reg_num, m_num_banks, m_bank_warp_shift,
                   sub_core_model, m_num_banks_per_sched, inst.get_schd_id()));
          m_cache->allocate_for_write(tag);
          inst.arch_reg.dst[op] = -1;
        } else {
          return false;
        }
      } else {  // Hit access serviced by cache
        inst.arch_reg.dst[op] = -1;
      }
    }
  }

  return true;
}

bool RFWithCache::modified_collector_unit_t::allocate(
    register_set *pipeline_reg_set, register_set *output_reg_set) {
  DPRINTF(std::cout << "sid: " << m_shader->get_sid() << " OC Allocated"
                    << std::endl;)
  assert(m_free);
  assert(m_not_ready.none());
  m_free = false;
  m_output_register = output_reg_set;
  warp_inst_t **pipeline_reg = pipeline_reg_set->get_ready();
  if ((pipeline_reg) and !((*pipeline_reg)->empty())) {
    m_warp_id = (*pipeline_reg)->warp_id();
    for (unsigned op = 0; op < MAX_REG_OPERANDS; op++) {
      int reg_num =
          (*pipeline_reg)
              ->arch_reg.src[op];  // this math needs to match that used in
                                   // function_info::ptx_decode_inst
      if (reg_num >= 0) {          // valid register
        m_src_op[op] = op_t(this, op, reg_num, m_num_banks, m_bank_warp_shift,
                            m_sub_core_model, m_num_banks_per_sched,
                            (*pipeline_reg)->get_schd_id());

        assert(op < MAX_SLOTS_PER_OC);
        unsigned oc_id = get_id();
        assert(oc_id < MAX_OC_COUNT);
        unsigned oc_slot = oc_id * MAX_SLOTS_PER_OC + op;
        tag_t tag(m_warp_id, reg_num);
        CACHE_ACCESS_STAT_t access_status = m_cache->read_access(tag, oc_slot);
        if (access_status == Miss) {
          m_not_ready.set(op);
        } else if (access_status == Reservation) {
          m_not_ready.set(op);
          m_src_op[op].reserve();
        }

      } else
        m_src_op[op] = op_t();
    }
    // move_warp(m_warp,*pipeline_reg);
    pipeline_reg_set->move_out_to(m_warp);
    return true;
  }
  return false;
}

void RFWithCache::modified_arbiter_t::add_read_requests(collector_unit_t *cu) {
  DDPRINTF(std::cout << "add read reqs sid: " << m_shader->get_sid();)
  const op_t *src = cu->get_operands();
  for (unsigned i = 0; i < MAX_REG_OPERANDS * 2; i++) {
    const op_t &op = src[i];
    if (op.valid() && !op.is_reserved() && cu->is_not_ready_op(i)) {
      DDPRINTF(std::cout << " warp_id: " << op.get_wid()
                         << " regid: " << op.get_reg();)
      unsigned bank = op.get_bank();
      m_queue[bank].push_back(op);
    }
  }
  DDPRINTF(std::cout << std::endl;)
}

RFWithCache::RFWithCache(const shader_core_config *config)
    : m_rfcache_config(config->m_rfcache_config) {
  if (m_rfcache_config.is_ideal())
    m_cache = new IdealRFCache();
  else if (m_rfcache_config.is_first())
    m_cache = new FirstCache(m_rfcache_config);
  else {
    std::cerr << "Error: undefined RF Cache type" << std::endl;
    exit(1);
  }
}

void RFWithCache::modified_collector_unit_t::collect_operand(unsigned op) {
  assert(m_not_ready.test(op));
  DPRINTF(std::cout << "sid: " << m_shader->get_sid() << " op_id: " << op
                    << " cid: " << get_id() << " V: " << m_src_op[op].valid()
                    << " Res: " << m_src_op[op].valid() << " not_r"
                    << std::endl;)
  m_not_ready.reset(op);
  m_src_op[op].unreserve();
}

RFWithCacheFirst::RFWithCacheFirst(const shader_core_config *config)
    : RFWithCache(config) {
  DPRINTF(std::cout << "RF Cache With First Initialized" << std::endl;)
  assert(m_rfcache_config.is_first());
}

void RFWithCacheFirst::init(unsigned num_banks, shader_core_ctx *shader) {
  assert(m_rfcache_config.is_first());
  RFWithCache::init(num_banks, shader);
  delete m_arbiter;
  m_arbiter = new first_arbiter_t(m_shader);
  m_arbiter->init(m_cu.size(), num_banks);

  auto cache = static_cast<FirstCache *>(m_cache);
  cache->set_arbiter(m_arbiter);
}

bool RFWithCacheFirst::lock_dstopnd_slots_in_cache(const warp_inst_t &inst) {
  assert(!inst.empty());
  std::vector<tag_t> dsts;
  auto cache = static_cast<FirstCache *>(m_cache);
  if (dstopnd_can_allocate_or_update(inst, dsts)) {
    assert(dsts.size() <= 2);
    assert(m_rfcache_config.is_first());
    for (auto dst : dsts) {
      cache->allocate_and_lock_for_write(dst);
    }
    return true;
  }

  return false;
}

bool RFWithCacheFirst::writeback(warp_inst_t &inst) {
  DPRINTF(std::cout << "First WB: "
                    << "sid: " << m_shader->get_sid() << " WB: ";)
  assert(!inst.empty());
  assert(m_rfcache_config.is_first());
  auto cache = static_cast<FirstCache *>(m_cache);
  std::vector<tag_t> dsts;

  if (dstopnd_can_allocate_or_update(inst, dsts)) {
    for (auto dst : dsts) {
      auto op = op_t(&inst, dst.second, m_num_banks, m_bank_warp_shift,
                     sub_core_model, m_num_banks_per_sched, inst.get_schd_id());

      if (inst.op_pipe == MEM__OP) {
        assert(dsts.size() <= 1);
        assert(m_rfcache_config.is_first());
        cache->allocate_and_lock_for_write(dst);
        cache->unlock(dst);
        cache->write(op);
        // auto arbiter =
        //     static_cast<RFWithCacheFirst::first_arbiter_t *>(m_arbiter);
        // arbiter->add_write_request(op);
      } else {
        cache->unlock(dst);
        cache->write(op);
      }
    }

    for (unsigned op = 0; op < MAX_REG_OPERANDS; op++) {
      inst.arch_reg.dst[op] = -1;  // this is needed in LDST writeback
    }
    return true;
  }

  return false;
}

void RFWithCacheFirst::step() {
  DPRINTF(std::cout << "First RF step" << std::endl;)
  dispatch_ready_cu();
  allocate_read_writes();
  for (unsigned p = 0; p < m_in_ports.size(); p++) allocate_cu(p);
  process_banks();
  DDPRINTF(dump(stdout); auto cache = static_cast<FirstCache *>(m_cache);
           cache->dump();)
}

void RFWithCacheFirst::allocate_read_writes() {
  // process read requests that do not have conflicts
  std::list<op_t> allocated = m_arbiter->allocate_reads();
  std::map<unsigned, op_t> read_ops;
  for (std::list<op_t>::iterator r = allocated.begin(); r != allocated.end();
       r++) {
    const op_t &rr = *r;
    unsigned reg = rr.get_reg();
    unsigned wid = rr.get_wid();
    unsigned bank =
        register_bank(reg, wid, m_num_banks, m_bank_warp_shift, sub_core_model,
                      m_num_banks_per_sched, rr.get_sid());
    DPRINTF(std::cout << "First: allocate for read bank_id: " << bank
                      << std::endl;)
    m_arbiter->allocate_for_read(bank, rr);
    read_ops[bank] = rr;
  }
  std::map<unsigned, op_t>::iterator r;
  for (r = read_ops.begin(); r != read_ops.end(); ++r) {
    op_t &op = r->second;
    unsigned wid = op.get_wid();
    unsigned reg = op.get_reg();
    tag_t tag(wid, reg);
    missed_oc_slots_t missed_oc_slots = m_cache->service_read_miss(tag);
    for (unsigned oc_slot = 0; missed_oc_slots.any();
         oc_slot++, missed_oc_slots >>= 1) {
      if (!missed_oc_slots.test(0)) continue;
      unsigned cu = oc_slot / MAX_SLOTS_PER_OC;
      unsigned operand_offset = oc_slot % MAX_SLOTS_PER_OC;
      assert(!m_cu[cu]->is_free());
      DDPRINTF(std::cout << "m_sid: " << m_shader->get_sid()
                         << " Collect Operand: wid: " << wid
                         << " reg_id: " << reg
                         << " missed_count: " << missed_oc_slots.count();)
      auto fcu =
          static_cast<RFWithCacheFirst::first_collector_unit_t *>(m_cu[cu]);
      fcu->collect_operand(operand_offset, tag, op);
    }
    assert(missed_oc_slots.count() == 0);
  }
}

RFWithCacheFirst::first_arbiter_t::first_arbiter_t(shader_core_ctx *shader)
    : modified_arbiter_t(shader), m_write_queue{nullptr} {
  DPRINTF(std::cout << "First rf arbiter added" << std::endl;)
}

void RFWithCacheFirst::first_arbiter_t::init(unsigned num_cu,
                                             unsigned num_banks) {
  DPRINTF(std::cout << "First arbiter initialized" << std::endl;)
  modified_arbiter_t::init(num_cu, num_banks);
  m_write_queue = new std::list<op_t>[num_banks];
}

void RFWithCacheFirst::first_arbiter_t::add_write_request(const op_t &op) {
  unsigned bank = op.get_bank();
  DPRINTF(std::cout << "First op added to write queue bank_id: " << bank
                    << std::endl;)
  m_write_queue[bank].push_back(op);
}

std::list<opndcoll_rfu_t::op_t>
RFWithCacheFirst::first_arbiter_t::allocate_reads() {
  DPRINTF(std::cout << "First arbiter allocate reads" << std::endl;)
  std::list<op_t>
      result;  // a list of registers that (a) are in different register
               // banks, (b) do not go to the same operand collector

  int input;
  int output;
  int _inputs = m_num_banks;
  int _outputs = m_num_collectors;
  int _square = (_inputs > _outputs) ? _inputs : _outputs;
  assert(_square > 0);
  int _pri = (int)m_last_cu;

  // Clear matching
  for (int i = 0; i < _inputs; ++i) _inmatch[i] = -1;
  for (int j = 0; j < _outputs; ++j) _outmatch[j] = -1;

  for (unsigned i = 0; i < m_num_banks; i++) {
    for (unsigned j = 0; j < m_num_collectors; j++) {
      assert(i < (unsigned)_inputs);
      assert(j < (unsigned)_outputs);
      _request[i][j] = 0;
    }
    if (!m_queue[i].empty()) {
      const op_t &op = m_queue[i].front();
      int oc_id = op.get_oc_id();
      assert(i < (unsigned)_inputs);
      assert(oc_id < _outputs);
      _request[i][oc_id] = 1;
    }
    if (!m_write_queue[i].empty()) {
      assert(i < (unsigned)_inputs);
      op_t &op = m_write_queue[i].front();
      DPRINTF(std::cout << "First: allocate for write bank_id: " << i
                        << std::endl;)
      allocate_bank_for_write(i, op);
      m_write_queue[i].pop_front();
      _inmatch[i] = 0;  // write gets priority
    }
  }

  ///// wavefront allocator from booksim... --->

  // Loop through diagonals of request matrix
  // printf("####\n");

  for (int p = 0; p < _square; ++p) {
    output = (_pri + p) % _outputs;

    // Step through the current diagonal
    for (input = 0; input < _inputs; ++input) {
      assert(input < _inputs);
      assert(output < _outputs);
      if ((output < _outputs) && (_inmatch[input] == -1) &&
          //( _outmatch[output] == -1 ) &&   //allow OC to read multiple reg
          // banks at the same cycle
          (_request[input][output] /*.label != -1*/)) {
        // Grant!
        _inmatch[input] = output;
        _outmatch[output] = input;
        // printf("Register File: granting bank %d to OC %d, schedid %d,
        // warpid %d, Regid %d\n", input, output,
        // (m_queue[input].front()).get_sid(),
        // (m_queue[input].front()).get_wid(),
        // (m_queue[input].front()).get_reg());
      }

      output = (output + 1) % _outputs;
    }
  }

  // Round-robin the priority diagonal
  _pri = (_pri + 1) % _outputs;

  /// <--- end code from booksim

  m_last_cu = _pri;
  for (unsigned i = 0; i < m_num_banks; i++) {
    if (_inmatch[i] != -1) {
      if (!m_allocated_bank[i].is_write()) {
        unsigned bank = (unsigned)i;
        assert(m_write_queue[bank].empty());
        op_t &op = m_queue[bank].front();
        result.push_back(op);
        m_queue[bank].pop_front();
      }
    }
  }

  return result;
}

bool RFWithCacheFirst::dstopnd_can_allocate_or_update(
    const warp_inst_t &inst, std::vector<tag_t> &dsts) {
  assert(!inst.empty());
  auto cache = static_cast<FirstCache *>(m_cache);

  for (unsigned op = 0; op < MAX_REG_OPERANDS;
       op++) {  // generte tags for all dst regs
    int reg_num = inst.arch_reg.dst[op];
    if (reg_num >= 0) {
      unsigned wid = inst.warp_id();
      tag_t tag(wid, reg_num);
      dsts.push_back(tag);
    }
  }

  if (!cache->can_allocate_or_update(dsts))
    return false;  // in case cannot allocate or update all dsts

  return true;
}

bool RFWithCacheFirst::srcopnds_can_allocate_or_update(
    const warp_inst_t &inst, std::vector<tag_t> &srcs) {
  assert(!inst.empty());
  auto cache = static_cast<FirstCache *>(m_cache);

  for (unsigned op = 0; op < MAX_REG_OPERANDS;
       op++) {  // generte tags for all dst regs
    int reg_num = inst.arch_reg.src[op];
    if (reg_num >= 0) {
      unsigned wid = inst.warp_id();
      tag_t tag(wid, reg_num);
      srcs.push_back(tag);
    }
  }

  if (!cache->can_allocate_or_update(srcs))
    return false;  // in case cannot allocate or update all srcs

  return true;
}

bool RFWithCacheFirst::first_collector_unit_t::allocate(
    register_set *pipeline_reg_set, register_set *output_reg_set) {
  DPRINTF(std::cout << "First Collector Unit Allocate called" << std::endl;)
  DPRINTF(std::cout << "sid: " << m_shader->get_sid() << std::endl;)
  assert(m_free);
  assert(m_not_ready.none());

  warp_inst_t **pipeline_reg = pipeline_reg_set->get_ready();
  if ((pipeline_reg) and !((*pipeline_reg)->empty())) {
    warp_inst_t inst = **pipeline_reg;
    std::vector<tag_t> srcs;
    if (m_rf->srcopnds_can_allocate_or_update(inst, srcs)) {
      DDPRINTF(std::cout << "can allocate and lock source operands sid: "
                         << m_shader->get_sid() << " wid: " << inst.warp_id()
                         << std::endl;)
      assert(m_rfcache_config->is_first());
      assert(srcs.size() < MAX_SLOTS_PER_OC);
      auto cache = static_cast<FirstCache *>(m_cache);

      unsigned oc_id = get_id();
      assert(oc_id < MAX_OC_COUNT);
      size_t src_id = 0;
      for (auto src : srcs) {
        DDPRINTF(std::cout << " src[" << src_id << "]: " << src.second
                           << std::endl;)
        m_src_op[src_id] =
            op_t(this, src_id, src.second, m_num_banks, m_bank_warp_shift,
                 m_sub_core_model, m_num_banks_per_sched, inst.get_schd_id());
        unsigned oc_slot = oc_id * MAX_SLOTS_PER_OC + src_id;
        bool was_reserve = false;
        bool was_hit =
            cache->allocate_and_lock_for_read(src, oc_slot, was_reserve);
        if (!was_hit) {
          m_not_ready.set(src_id);
        } else {
          m_not_ready.reset(src_id);
        }
        if (was_reserve) {
          m_src_op[src_id].reserve();
        }
        src_id++;
      }
      DDPRINTF(std::cout << "nr: " << m_not_ready << std::endl;)
      for (unsigned op = src_id; op < MAX_REG_OPERANDS; op++) {
        m_src_op[op] = op_t();
      }
      m_free = false;
      m_output_register = output_reg_set;
      m_warp_id = inst.warp_id();
      pipeline_reg_set->move_out_to(m_warp);
      return true;
    }
  }
  return false;
}
void RFWithCacheFirst::first_collector_unit_t::dispatch() {
  DDPRINTF(std::cout << "Dispatch OC " << get_id() << std::endl;)
  assert(m_not_ready.none());
  unsigned wid = m_warp->warp_id();
  m_output_register->move_in(m_warp);
  m_free = true;
  m_output_register = NULL;

  auto cache = static_cast<FirstCache *>(m_cache);

  for (unsigned i = 0; i < MAX_REG_OPERANDS * 2; i++) {
    auto op = m_src_op[i];
    if (op.valid()) {
      auto tag = tag_t(wid, op.get_reg());
      if (cache->can_release_src(tag)) {
        cache->release_src(tag);
      }
    }
    op.reset();
  }
}

void RFWithCacheFirst::fill_srcop(const tag_t &tag, const op_t &op) {
  assert(op.valid());
  assert(m_rfcache_config.is_first());
  DDPRINTF(std::cout << "Fill src ops <" << tag.first << ", " << tag.second
                     << "> " << std::endl;)
  auto cache = static_cast<FirstCache *>(m_cache);
  cache->fill_block(tag, op);
}