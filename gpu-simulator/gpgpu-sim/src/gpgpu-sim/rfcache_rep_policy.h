#pragma once

#include <cstddef>
#include <functional>
#include <list>
#include <unordered_map>
#include <optional>
template <typename T, typename U = std::hash<T>>
class ReplacementPolicy {
 public:
  ReplacementPolicy(std::size_t sz);
  bool refer(const T &el, T &replaced_el);
  bool get_replacement_candidate(const T &, T &);
  void lock(const T &el);
  void unlock(const T &el);
  bool can_allocate() const { return m_n_locked_entries < m_size; }
  void dump() const;

 private:
  bool check_size() const {
    return (m_refs.size() == m_order_q.size()) &&
           (m_order_q.size() <= m_size) && (m_n_locked_entries <= m_size);
  }
  size_t m_size;
  size_t m_n_locked_entries;
  std::list<T> m_order_q;
  std::unordered_map<T, std::pair<bool, typename std::list<T>::iterator>, U>
      m_refs;
};

#include "rfcache_rep_policy.tpp"
