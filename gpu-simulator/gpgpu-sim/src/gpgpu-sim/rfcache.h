#pragma once

#include "rfcache_rep_policy.h"
class RFCache {
public:
    RFCache(std::size_t sz);
private:
    std::size_t m_size;
    ReplacementPolicy<unsigned> m_rpolicy;
};