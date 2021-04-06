#pragma once

#include <vector>
#include <string>
#include <map>

class InstRegAccesses
{
public:
    InstRegAccesses(const std::vector<unsigned> &dsts, const std::vector<unsigned> &srcs);
    void print(std::ostream &of) const;
private:
    std::vector<unsigned> m_srcs;
    std::vector<unsigned> m_dsts;
};