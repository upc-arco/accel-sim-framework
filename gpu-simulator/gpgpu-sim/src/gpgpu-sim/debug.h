#pragma once
#include <iostream>

#ifdef D8BUG
#define D8PRINTF(x) std::cout << "RFCache INP: " << x << std::endl;
#else
#define D8PRINTF(x) 
#endif

#ifdef D7BUG
#define D7PRINTF(x) std::cout << "RFScheduler: " << x << std::endl;
#else
#define D7PRINTF(x) 
#endif

#ifdef D6BUG
#define D6PRINTF(x) std::cout << "RFScheduler: " << x << std::endl;
#else
#define D6PRINTF(x) 
#endif

#ifdef DDDDBUG
#define DDDDPRINTF(x) std::cout << "RFCacheStats: " << x << std::endl;
#else
#define DDDDPRINTF(x) 
#endif

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