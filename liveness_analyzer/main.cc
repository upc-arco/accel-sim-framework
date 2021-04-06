#include <iostream>
#include <filesystem>
#include <stdexcept>
#include <vector>

#include "traces.h"

using namespace std;

namespace fs = std::filesystem;

int main(int argc, char *argv[])
{
    if (argc != 3)
    {
        cerr << "Invalid arguments" << endl;
        cerr << "./liveness_analyzer trace_dir benchmark_suite" << endl;
        exit(1);
    }

    fs::path bsuite_bpath = std::string(argv[1]) + "/" + std::string(argv[2]);
    try
    {
        Traces traces(bsuite_bpath);
        traces.parse();
    }
    catch (const exception &e)
    {
        std::cerr << e.what() << std::endl;
        exit(1);
    }
    return 0;
}