#pragma once
#include <map>
#include <list>
#include <vector>

class WarpReuseDistance {
public:
    WarpReuseDistance(unsigned dwid) : m_dwid{dwid}, m_ip{0}, total_insts{0} {}
    ~WarpReuseDistance();
    void add_reg_reuse(const std::vector<unsigned> &dsts, const std::vector<unsigned> &srcs);
    void done_adding_reg_use() { 
        total_insts = m_ip;
        m_ip = 0; 
    }
    bool all_insts_covered() { return m_ip == total_insts; }
    std::string update_reuse_distance(const std::vector<unsigned> &dsts, const std::vector<unsigned> &srcs);
private:
    unsigned m_dwid;
    unsigned long long m_ip; // instruction pointer
    unsigned long long total_insts; 
    std::map<unsigned, unsigned> m_last_use; // <regid, last_inst>
    std::map<unsigned, std::list<int>> m_reuse_distances; // <regid, {reuse_distance1, reuse_distance2, ...}>
};
