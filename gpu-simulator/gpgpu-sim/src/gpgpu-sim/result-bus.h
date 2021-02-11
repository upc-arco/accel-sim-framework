#pragma once

#include <bitset>
#include <vector>

#include "../abstract_hardware_model.h"
#include "rfcache_configs.h"

class opndcoll_rfu_t;

class ResultBus {
public:
    ResultBus(const RFCacheConfig &config);
    ~ResultBus();
    void init(unsigned width, unsigned num_banks, opndcoll_rfu_t *rf);
    void cycle();
    int test(const warp_inst_t *inst);
    int first_test(const warp_inst_t *inst);
    int ideal_test(const warp_inst_t *inst);
private:
    size_t get_num_regs(const warp_inst_t &inst) const;
    void find_reg_banks(const warp_inst_t *inst, int &regbank1, int &regbank2) const;
    unsigned num_free_slots(unsigned latency) const;
    void allocate_noreg_insts(const warp_inst_t * inst) const;

    unsigned m_width; //max writebacks = m_width
    unsigned m_num_banks; // #RF_banks
    opndcoll_rfu_t *m_rf; //reference to register file
    const RFCacheConfig &m_rfcache_config;
    unsigned m_rfcache_type;

    static const unsigned MAX_ALU_LATENCY = 512;

    std::vector<std::bitset<MAX_ALU_LATENCY> *> m_res_busses;
};