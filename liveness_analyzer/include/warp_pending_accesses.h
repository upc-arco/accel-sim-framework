#pragma once

#include <map>
#include "inst_reg_accesses.h"

class WarpPendingAccesses
{
public:
    WarpPendingAccesses(unsigned dwarp_id) : m_dwarp_id{dwarp_id}, m_next_dval_id{0}, m_n_untracked_reuses{0} {}
    void add_reg_use(const std::vector<unsigned> &dsts, const std::vector<unsigned> &srcs);
    std::string update_pending_reuses(const std::vector<unsigned> &dsts, const std::vector<unsigned> &srcs);
    void done_adding_reg_use();
    bool are_all_reuses_tracked() { return m_n_untracked_reuses == 0; }
private:
    unsigned long long m_n_untracked_reuses;
    unsigned m_dwarp_id; // dynamic warp id
    unsigned long long m_next_dval_id;
    std::map<unsigned, unsigned long long> m_reg_map;        // <reg_id, val_id>
    std::map<unsigned long long, unsigned> m_pending_reuses; // <val_id, pending_reuse>
};