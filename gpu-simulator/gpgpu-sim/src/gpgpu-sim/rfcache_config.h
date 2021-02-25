#pragma once
class RFCacheConfig {
 public:
  RFCacheConfig();
  void init();
  void reg_options(class OptionParser* opp);
  size_t size() const { return m_size; }
  char* m_config_str;

 private:
  std::size_t m_size;
};