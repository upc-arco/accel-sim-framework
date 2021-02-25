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
    m_cus[set_id].push_back(modified_collector_unit_t(m_rfcache_config.size()));
    m_cu.push_back(&m_cus[set_id].back());
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

  m_lru_policy.refer(invalid_unique_wid);
}

void RFWithCache::init(unsigned num_banks, shader_core_ctx *shader) {
  m_shader = shader;
  m_arbiter.init(m_cu.size(), num_banks);
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
        *static_cast<RFWithCache::modified_collector_unit_t *>(m_cu[j]));
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
      unsigned wid = inst.warp_id();
      auto allocated = m_oc_allocator.allocate(wid);
      if (allocated.first) {
        bool successfull_allocation =
            allocated.second.allocate(inp.m_in[i], inp.m_out[i]);
        assert(successfull_allocation);
        m_arbiter.add_read_requests(&allocated.second);
      } else {
        DDPRINTF("OCAllocator stalled")
      }
      break;  // can only service a single input, if it failed it will fail for
              // others.
    }
  }
}

std::pair<bool, RFWithCache::modified_collector_unit_t &> RFWithCache::OCAllocator::allocate(unsigned wid) {
  if (m_n_available == 0) {
    return std::pair<bool, modified_collector_unit_t &>{
        false, m_info_table.begin()->second.m_oc};
  }
  if (m_info_table.find(wid) == m_info_table.end()) {  // warp_id is new
    auto replaced_element = m_lru_policy.refer(wid);
    auto replaced_info_iter = m_info_table.find(replaced_element);
    assert(replaced_info_iter != m_info_table.end() && replaced_info_iter->second.m_availble == true);
    m_info_table.insert({wid, replaced_info_iter->second});
    m_info_table.erase(replaced_info_iter);
    auto new_element_iter = m_info_table.find(wid);
    new_element_iter->second.m_availble = false;
    m_n_available--;
    m_lru_policy.lock(wid);
    dump();
    return std::pair<bool, modified_collector_unit_t &>(true,
                                                        new_element_iter->second.m_oc);
  } else {
    auto iter = m_info_table.find(wid);
    if(iter->second.m_availble == false) {
      return std::pair<bool, modified_collector_unit_t &>(false,
                                                        iter->second.m_oc);
    }
    iter->second.m_availble = false;
    m_n_available--;
    m_lru_policy.refer(wid);
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
  assert( iter != m_info_table.end() && iter->second.m_availble == false);
  iter->second.m_availble = true;
  dump();
  m_lru_policy.unlock(wid);
}

void RFWithCache::OCAllocator::dump() {
  DDPRINTF("OCAllocator Dump")
  std::stringstream ss;
  ss << "Info Table\n";
  for (auto el : m_info_table)
  {
    ss << "<" << el.first << ", " << (el.second.m_availble ? 'F' : 'O') << ", " << el.second.m_oc.get_id() << "> "; 
  }
  DDPRINTF(ss.str());
}