#pragma once

#include <iostream>
#include <string>

enum cache_type_t { IDEAL, FIRST, UNDEFINED };
enum REPLACEMENT_POLICY { RF_FIFO, RF_LRU, UNDEFINED_REPLACEMENT_POLICY };
class RFCacheConfig {
 public:
  char *m_config_string;
  void init();
  bool is_ideal() const { return m_type == IDEAL; }
  bool is_first() const { return m_type == FIRST; }
  size_t get_n_blocks() const { return m_n_blocks; }
  REPLACEMENT_POLICY get_replacement_policy() const { return m_rpolicy; }

 private:
  void set_type(const std::string &type) {
    m_type = type == "1" ? IDEAL : type == "2" ? FIRST : UNDEFINED;
  }
  void set_n_blocks(size_t n_blocks) { m_n_blocks = n_blocks; }
  void set_replacement_policy(char policy_chr);
  cache_type_t m_type;
  size_t m_n_blocks;
  REPLACEMENT_POLICY m_rpolicy;
};
