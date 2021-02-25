
#include <cstring>

#include "rfcache_config.h"
#include "debug.h"
#include "../option_parser.h"

RFCacheConfig::RFCacheConfig() { DPRINTF("RFCacheConfig Constructed") }

void RFCacheConfig::reg_options(class OptionParser* opp) {
  DPRINTF("RFCacheConfig Register Options")
  option_parser_register(opp, "-gpgpu-rfcache", OPT_CSTR, &m_config_str,
                         "Per shader rf cache config "
                         "{<size>}",
                         "10");
}

void RFCacheConfig::init(){
    DPRINTF("RFCacheConfig Initialization ")
    char *token = std::strtok(m_config_str, ",");
    size_t i = 0;
    while(token) {
      switch (i)
      {
      case 0:
        m_size = std::atoi(token);
        break;
      default:
        std::cerr << "Error: Invalid RF Cache Config Detected" << std::endl;
        exit(1);
        break;
      }
      token = std::strtok(NULL, ",");
    }
    DPRINTF("RFCacheConfig Size: " << m_size)
}