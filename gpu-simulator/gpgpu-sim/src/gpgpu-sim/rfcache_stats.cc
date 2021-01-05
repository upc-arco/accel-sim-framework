#include "rfcache_stats.h"
#include <cstdio>
#include "shader.h"

void RFCacheStats::init(const shader_core_config *config) {
  m_config = config;
  m_n_shaders = config->num_shader();

  m_n_shaders = m_n_shaders;
  m_n_read_hits.resize(m_n_shaders);
  m_n_read_misses.resize(m_n_shaders);
  m_n_read_reservation.resize(m_n_shaders);
  m_n_write_hits.resize(m_n_shaders);
  m_n_write_misses.resize(m_n_shaders);
}

void RFCacheStats::print(FILE *fout) const {
  unsigned long long rfcache_n_tot_read_hits = 0, rfcache_n_tot_read_misses = 0,
                     rfcache_n_tot_read_reservation = 0,
                     rfcache_n_tot_write_hits = 0,
                     rfcache_n_tot_write_misses = 0;
  for (unsigned i = 0; i < m_config->num_shader(); i++) {
    rfcache_n_tot_read_hits += m_n_read_hits.at(i);
    rfcache_n_tot_read_misses += m_n_read_misses.at(i);
    rfcache_n_tot_read_reservation += m_n_read_reservation.at(i);
    rfcache_n_tot_write_hits += m_n_write_hits.at(i);
    rfcache_n_tot_write_misses += m_n_write_misses.at(i);
  }

  fprintf(fout, "rfcache_n_tot_read_hits = %lld\n", rfcache_n_tot_read_hits);
  fprintf(fout, "rfcache_n_tot_read_misses = %lld\n",
          rfcache_n_tot_read_misses);
  fprintf(fout, "rfcache_n_tot_read_reservation = %lld\n",
          rfcache_n_tot_read_reservation);
  fprintf(fout, "rfcache_n_tot_write_hits = %lld\n", rfcache_n_tot_write_hits);
  fprintf(fout, "rfcache_n_tot_write_misses = %lld\n",
          rfcache_n_tot_write_misses);
}