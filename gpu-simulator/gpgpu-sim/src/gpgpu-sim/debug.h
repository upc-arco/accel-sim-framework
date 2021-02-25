#pragma once
#include <iostream>
#ifdef DEBUG
#define DPRINTF(x) std::cout << x << std::endl;
#else
#define DPRINTF(x) 
#endif
#define DDEBUG
#ifdef DDEBUG
#define DDPRINTF(x) std::cout << x << std::endl;
#else
#define DDPRINTF(x) 
#endif