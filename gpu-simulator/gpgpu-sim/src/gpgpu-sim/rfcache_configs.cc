#include <cassert>
#include <cstring>
#include <sstream>

#include "rfcache_configs.h"

void RFCacheConfig::init() {
  std::stringstream cfg_str(m_config_string);
  std::string token;

  for (size_t i = 0; std::getline(cfg_str, token, ':'); i++) {
    switch (i) {
      case 0:
        set_type(token);
        break;
      case 1:
        set_n_blocks(std::stoi(token));
        break;
      case 2:
        assert(token.length() == 1);
        set_replacement_policy(token[0]);
        break;

      default:
        std::cerr << "Error: undefiend rf config string" << std::endl;
    }
  }
}

void RFCacheConfig::set_replacement_policy(char policy_chr) {
  if (policy_chr == 'f') {
    m_rpolicy = RF_FIFO;
  } else if (policy_chr == 'l') {
    m_rpolicy = RF_LRU;
  } else {
    std::cerr << "Error: undefined replacement policy" << std::endl;
    exit(1);
  }
}