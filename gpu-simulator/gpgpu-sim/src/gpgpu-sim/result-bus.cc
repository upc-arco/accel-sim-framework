#include <cassert>
#include <iostream>

#include "result-bus.h"
#include "shader.h"

ResultBus::~ResultBus() {
  for (auto bus : m_res_busses) delete bus;
}

void ResultBus::cycle() {
  for (auto bus : m_res_busses) (*bus) >>= 1;
}

unsigned ResultBus::num_free_slots(unsigned latency) const {
  unsigned free_slots_remained = m_width;
  assert(free_slots_remained > 0);
  for (auto bus : m_res_busses)
    if (bus->test(latency))
      if (!--free_slots_remained) return 0;
  return free_slots_remained;
}

void ResultBus::allocate_noreg_insts(const warp_inst_t *inst) const {
  unsigned latency = inst->latency;
  bool allocated = false;
  for (size_t i = m_num_banks; i < m_res_busses.size(); i++)
    if (!m_res_busses[i]->test(latency)) {
      m_res_busses[i]->set(latency);
      allocated = true;
    }
  assert(allocated);
}

int ResultBus::test(const warp_inst_t *inst) {
  unsigned latency = inst->latency;
  assert(latency < MAX_ALU_LATENCY);
  unsigned fs_count = num_free_slots(latency);
  if (!fs_count) return -1;  // there is no latch available

  int regbank1, regbank2;
  find_reg_banks(inst, regbank1, regbank2);
  if (regbank1 == -1) {
    allocate_noreg_insts(inst);
    return 1;
  }

  if (m_res_busses[regbank1]->test(latency))
    return -1;                   // conflict with first access
  if (regbank2 != -1) {          // there is a second access
    if (regbank1 == regbank2) {  // conflict within the instruction accesses
      if (m_res_busses[regbank2]->test(latency + 1) ||
          !num_free_slots(latency + 1))
        return -1;  // conflict or no room for second access
      m_res_busses[regbank2]->set(latency + 1);
    } else {
      if (m_res_busses[regbank2]->test(latency) || fs_count <= 1)
        return -1;  // conflict with second access or no room for two accesses

      m_res_busses[regbank2]->set(latency);
    }
  }

  m_res_busses[regbank1]->set(latency);

  return 1;
}

void ResultBus::find_reg_banks(const warp_inst_t *inst, int &regbank1,
                               int &regbank2) const {
  regbank1 = regbank2 = -1;
  for (unsigned op = 0; op < MAX_REG_OPERANDS; ++op) {
    int reg_num = inst->arch_reg.dst[op];
    if (reg_num >= 0) {
      unsigned bank =
          register_bank(reg_num, inst->warp_id(), m_num_banks,
                        m_rf->m_bank_warp_shift, m_rf->sub_core_model,
                        m_rf->m_num_banks_per_sched, inst->get_schd_id());
      assert(regbank2 == -1);
      if (regbank1 == -1) {
        regbank1 = bank;
        continue;
      }
      regbank2 = bank;
    }
  }
}
void ResultBus::init(unsigned width, unsigned num_banks, opndcoll_rfu_t *rf) {
  assert(width > 0);
  assert(num_banks > 0);
  m_width = width;
  m_num_banks = num_banks;
  m_rf = rf;

  for (size_t i = 0; i < m_num_banks + m_width; i++) {
    m_res_busses.push_back(new std::bitset<MAX_ALU_LATENCY>());
  }
}
