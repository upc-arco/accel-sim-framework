#include <cassert>
#include <sstream>
#include <iostream>

#include "warp_reuse_distance.h"

WarpReuseDistance::~WarpReuseDistance() {
    m_last_use.clear();
    for (auto &reuse_distance: m_reuse_distances) {
        reuse_distance.second.clear();
    }
    m_reuse_distances.clear();
}

void WarpReuseDistance::add_reg_reuse(const std::vector<unsigned> &dsts, const std::vector<unsigned> &srcs) {
    // new inst has come
    for (auto &src: srcs) {

        if (m_last_use.find(src) != m_last_use.end()) {
            // it could be because of previous write or previous read
            // there was a previous occurence
            auto distance = m_ip - m_last_use[src]; // compute distance
            m_reuse_distances[src].push_back(distance); // store distance
        } 
        m_last_use[src] = m_ip; // update last occurence
    }
    assert(dsts.size() <= 1);
    for (auto &dst : dsts) {
        if(m_last_use.find(dst) != m_last_use.end()) {
            // last occurence could be write or read
            // in both cases this is not a reuse
            m_reuse_distances[dst].push_back(-1);
            
        }
        // this is a new dst
        m_last_use[dst] = m_ip; // store the last use pointer
    }
    m_ip++; // point to the next inst
    // we could left last distance uncaptured
}

std::string WarpReuseDistance::update_reuse_distance(const std::vector<unsigned> &dsts, const std::vector<unsigned> &srcs) {
    std::stringstream ss;
    std::string result;
    for (auto &src : srcs) {
        ss << ',';
        if(!m_reuse_distances[src].empty() && m_reuse_distances[src].front() == -2) {
            std::cout << "There is untracked register reuse" << std::endl;
            exit(1);
        }
        else if(!m_reuse_distances[src].empty()) {
            ss << m_reuse_distances[src].front();
            m_reuse_distances[src].pop_front();
        } else {
            // the last uncaptured occurence
            ss << -1;
            m_reuse_distances[src].push_back(-2);
        }
    }
    result = ss.str();
    ss.str("");
    assert(dsts.size() <= 1);
    for (auto &dst : dsts) {
        if(!m_reuse_distances[dst].empty() && m_reuse_distances[dst].front() == -2) {
            std::cout << "There is untracked register reuse" << std::endl;
            exit(1);
        }
        else if(!m_reuse_distances[dst].empty()) {
            ss << m_reuse_distances[dst].front();
            m_reuse_distances[dst].pop_front();
        } else {
            // the last uncaptured occurence
            ss << -1;
            m_reuse_distances[dst].push_back(-2);
        }
    }
    if(srcs.size() == 0) result = ",";
    result = ss.str() + result;
    m_ip++;
    return result;
}