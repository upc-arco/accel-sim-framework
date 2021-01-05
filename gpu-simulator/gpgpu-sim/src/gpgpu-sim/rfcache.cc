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
  DPRINTF(std::cout << m_shader->get_sid() << " Reqs are adding to read queues"
                    << std::endl;)
  const op_t *src = cu->get_operands();
  for (unsigned i = 0; i < MAX_REG_OPERANDS * 2; i++) {
    const op_t &op = src[i];
    if (op.valid() && !op.is_reserved() && cu->is_not_ready_op(i)) {
      unsigned bank = op.get_bank();
      m_queue[bank].push_back(op);
    }
  }
}