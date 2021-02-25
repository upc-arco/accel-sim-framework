#include <iostream>
#include <sstream>
#include <cassert>
#include "debug.h"

template <typename T>
ReplacementPolicy<T>::ReplacementPolicy(std::size_t sz)
    : m_size{sz} {
          DPRINTF("RFCache Replacement Policy Constructed Size: " << m_size)};

template <typename T>
T ReplacementPolicy<T>::refer(const T &el) {
  DPRINTF("RFCache Replacement Policy Refer Called")
  DDPRINTF("OCAllocator Refer Called " << el)
  assert(this->m_n_locked_entries < this->m_size);
  T replaced_el;
  if (m_refs.find(el) == m_refs.end()) { // try to allocate new warp id
    m_order_q.push_front(el);
    m_refs[el] = std::pair<bool, typename std::list<T>::iterator>{false, m_order_q.begin()};
    if (m_order_q.size() > m_size) {
      // should pop the last unlocked element
      for (auto iter = m_order_q.rbegin(); iter != m_order_q.rend(); iter++){
        if(m_refs[*iter].first)
          continue;
        replaced_el = *iter;
        m_order_q.erase(m_refs[replaced_el].second);
        m_refs.erase(replaced_el);
        break;
      }
    }
  } else { //allocate available warp
    assert(m_refs[el].first == false);
    auto old_ref = m_refs[el].second;
    m_order_q.push_front(el);
    m_refs[el] = std::pair<bool, typename std::list<T>::iterator>({false, m_order_q.begin()});
    m_order_q.erase(old_ref);
  }
  dump();
  assert(check_size());
  return replaced_el;
}

template <typename T>
void ReplacementPolicy<T>::dump() const {
  DDPRINTF("OCAllocator Replacement Policy")
  std::stringstream ss;
  ss << "order_list\n";
  for (auto el: m_order_q)
  {
    ss << el << " ";
  }
  ss << "\nref_list\n";
  for (auto el : m_refs) {
    ss << "<" << el.first << ", " << (el.second.first ? 'L' : 'U') << "> ";
  }
  DDPRINTF(ss.str());
}

template <typename T>
void ReplacementPolicy<T>::lock(const T &el) {
  DDPRINTF("Replacement Policy Lock " << el)
  m_n_locked_entries++;
  assert(m_refs[el].first == false);
  m_refs[el].first = true;
  assert(check_size());
}

template <typename T>
void ReplacementPolicy<T>::unlock(const T &el) {
  DDPRINTF("Replacement Policy Unlock " << el)
  m_n_locked_entries--;
  assert(m_refs[el].first == true);
  m_refs[el].first = false;
  assert(check_size());
}