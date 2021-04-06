#include <cassert>
#include <sstream>
#include <string>

#include "warp_pending_accesses.h"

void WarpPendingAccesses::add_reg_use(const std::vector<unsigned> &dsts, const std::vector<unsigned> &srcs)
{
    for (auto &src : srcs)
    {
        if (m_reg_map.find(src) == m_reg_map.end())
        {
            // reading a register never written
            m_reg_map[src] = m_next_dval_id; // allocate new dynamic value
            m_pending_reuses[m_next_dval_id++] = 0; // initialized pending reuse and update next_value index
        }
        else
        {
            // reuse value
            auto val_id = m_reg_map[src]; // get val id
            m_pending_reuses[val_id]++; // inc reuse
            m_n_untracked_reuses++;
        }
    }

    for (auto &dst : dsts)
    {
        m_reg_map[dst] = m_next_dval_id; // allocate new id
        m_pending_reuses[m_next_dval_id++] = 0; // initialized pending reuse and update next id
    }
}

void WarpPendingAccesses::done_adding_reg_use()
{
    m_next_dval_id = 0; // next dynamic value id
    m_reg_map.clear();
}

std::string WarpPendingAccesses::update_pending_reuses(const std::vector<unsigned> &dsts, const std::vector<unsigned> &srcs)
{
    std::stringstream ss;
    std::string result;
    for (auto &src : srcs)
    {
        if (m_reg_map.find(src) == m_reg_map.end())
        {
            // reading value for the first time
            m_reg_map[src] = m_next_dval_id; // give it a dynamic name
            assert(m_pending_reuses.find(m_next_dval_id) != m_pending_reuses.end()); // There should be accesses pending to this value
            assert(m_pending_reuses[m_next_dval_id] >= 0); // pending reqs should be in correct range
            ss << ", " << m_pending_reuses[m_next_dval_id++]; // write pending and update next id
        }
        else
        {
            // this is a reuse to a value
            auto val_id = m_reg_map[src]; // get the dynamic val id
            assert(m_pending_reuses[val_id] > 0);
            ss << ", " << --m_pending_reuses[val_id]; // exclude current use and print the remaining
            m_n_untracked_reuses--;
        }
    }
    result = ss.str();
    ss.str("");

    assert(dsts.size() <= 1);

    for (auto &dst : dsts)
    {
        if (m_reg_map.find(dst) != m_reg_map.end())
        {
            // over write a register
            auto val_id = m_reg_map[dst]; // get previous dynamic val
            assert(m_pending_reuses[val_id] == 0); // there should not be pending reuse remained to this value
        }
        assert(m_pending_reuses.find(m_next_dval_id) != m_pending_reuses.end()); // there should be pending reuses for new value
        m_reg_map[dst] = m_next_dval_id; // update new dynamic id
        ss << m_pending_reuses[m_next_dval_id++]; // print pending reuses to this value and update dynamic id
    }
    if(srcs.empty()) {
        ss << ",";
    }

    result = ss.str() + result;
    return result;
}