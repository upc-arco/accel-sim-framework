#pragma once
#include <iostream>

#ifdef DEBUG
#define DPRINTF(x) std::cout << x << std::endl;
#else
#define DPRINTF(x) 
#endif
#ifdef DDEBUG
#define DDPRINTF(x) std::cout << x << std::endl;
#else
#define DDPRINTF(x) 
#endif
#ifdef DDDEBUG
#define DDDPRINTF(x) std::cout << "RFCache: " << x << std::endl;
#else
#define DDDPRINTF(x) 
#endif