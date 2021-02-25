#include "rfcache.h"
#include <cstddef>

RFCache::RFCache(std::size_t sz) : m_rpolicy(sz), m_size(sz) {
    DPRINTF("RF Cache Constructed Size: " << sz)
}