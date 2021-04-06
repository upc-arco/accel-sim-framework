#include <iostream>
#include <cassert>

#include "inst_reg_accesses.h"

InstRegAccesses::InstRegAccesses(const std::vector<unsigned> &dsts, const std::vector<unsigned> &srcs)
    : m_srcs{srcs}, m_dsts{dsts}
{
}

void InstRegAccesses::print(std::ostream &of) const
{
    auto num_dsts = m_dsts.size();
    of << num_dsts << ", ";
    for (auto dst : m_dsts)
    {
        of << dst << ", ";
    }
    auto num_srcs = m_srcs.size();
    of << num_srcs;
    for (auto src : m_srcs)
    {
        of << ", " << src;
    }
    of << std::endl;
}


