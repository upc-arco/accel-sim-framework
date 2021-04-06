#pragma once

#define DEBUG
#ifdef DEBUG
#define DPRINTF(x) std::cout << "Liveness Analyzer: " << x << std::endl;
#elif
#define DPRINTF(x) 
#endif