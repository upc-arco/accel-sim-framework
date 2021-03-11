#include <cassert>
#include <iostream>
#include <optional>
#include <sstream>
#include "debug.h"

inline std::stringstream &operator<<(std::stringstream &ss,
                                     const std::pair<unsigned, unsigned> &p) {
  ss << "<";
  ss << p.first;
  ss << ", ";
  ss << p.second;
  ss << ">";
  return ss;
}

template <typename T, typename U>
ReplacementPolicy<T, U>::ReplacementPolicy(std::size_t sz)
    : m_size{sz} {
          DPRINTF("RFCache Replacement Policy Constructed Size: " << m_size)};

template <typename T, typename U>
bool ReplacementPolicy<T, U>::refer(const T &el, T &replaced_el) {
  DPRINTF("RFCache Replacement Policy Refer Called")
  DDPRINTF("OCAllocator Refer Called " << el)
  //assert(this->m_n_locked_entries < this->m_size);
  // T replaced_el;
  if (m_refs.find(el) == m_refs.end()) {  // try to allocate new warp id
    assert(m_n_locked_entries <= m_size);
    if (m_n_locked_entries == m_size) return false;
    m_order_q.push_front(el);
    m_refs[el] = std::pair<bool, typename std::list<T>::iterator>{
        false, m_order_q.begin()};
    if (m_order_q.size() > m_size) {
      // should pop the last unlocked element
      for (auto iter = m_order_q.rbegin(); iter != m_order_q.rend(); iter++) {
        if (m_refs[*iter].first) continue;
        replaced_el = *iter;
        m_order_q.erase(m_refs[replaced_el].second);
        m_refs.erase(replaced_el);
        //dump();
        assert(check_size());
        return true;
      }
    }
  } else {  // allocate available warp
    //assert(m_refs[el].first == false);
    auto old_ref = m_refs[el].second;
    m_order_q.push_front(el);
    m_refs[el] = std::pair<bool, typename std::list<T>::iterator>(
        {m_refs[el].first, m_order_q.begin()});
    m_order_q.erase(old_ref);
  }
  //dump();
  assert(check_size());
  return false;
}

template <typename T, typename U>
bool ReplacementPolicy<T, U>::get_replacement_candidate(const T &el,
                                                        T &rep_el) {
  DDPRINTF("get_replacement_candidate called " << el)
  if (m_refs.find(el) == m_refs.end()) {
    if (m_order_q.size() < m_size) return false;
    for (auto iter = m_order_q.rbegin(); iter != m_order_q.rend(); iter++) {
      if (m_refs[*iter].first) continue;
      rep_el = *iter;
      return true;
    }
  } else {
    assert(*(m_refs[el].second) == el);
  }
  return false;
}

template <typename T, typename U>
void ReplacementPolicy<T, U>::dump() const {
  std::stringstream ss;
  ss << "order_list\n";
  for (auto el : m_order_q) {
    ss << el << " ";
  }
  ss << "\nref_list\n";
  for (auto el : m_refs) {
    ss << "<";
    ss << el.first << ", ";
    ss << (el.second.first ? 'L' : 'U') << "> ";
  }
  if (!std::is_unsigned<T>::value) {

    DDDPRINTF("Replacement Policy")
    DDDPRINTF(ss.str())
  }

  //  DDPRINTF("OCAllocator Replacement Policy")
  //  DDPRINTF(ss.str());
}

template <typename T, typename U>
void ReplacementPolicy<T, U>::lock(const T &el) {
  DDPRINTF("Replacement Policy Lock " << el)
  m_n_locked_entries++;
  assert(m_refs[el].first == false);
  m_refs[el].first = true;
  assert(check_size());
}

template <typename T, typename U>
void ReplacementPolicy<T, U>::unlock(const T &el) {
  DDPRINTF("Replacement Policy Unlock " << el)
  m_n_locked_entries--;
  assert(m_refs[el].first == true);
  m_refs[el].first = false;
  assert(check_size());
}