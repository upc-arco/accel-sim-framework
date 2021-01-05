#pragma once

#include <cassert>
#include <cstdio>
#include <vector>

class shader_core_config;

class RFCacheStats {
 public:
  void init(const shader_core_config *config);
  void incread_hit(unsigned sid) {
    ++m_n_read_hits.at(sid);
  }
  void incread_miss(unsigned sid) {
    ++m_n_read_misses.at(sid);
  }
  void incread_reservation(unsigned sid) {
    ++m_n_read_reservation.at(sid);
  }
  void incwrite_hit(unsigned sid) {
    ++m_n_write_hits.at(sid);
  }
  void incwrite_miss(unsigned sid) {
    ++m_n_write_misses.at(sid);
  }
  void print(FILE *fout) const;

 private:
  const shader_core_config *m_config;
  unsigned m_n_shaders;
  std::vector<unsigned long long> m_n_read_hits;
  std::vector<unsigned long long> m_n_read_misses;
  std::vector<unsigned long long> m_n_read_reservation;
  std::vector<unsigned long long> m_n_write_hits;
  std::vector<unsigned long long> m_n_write_misses;
};