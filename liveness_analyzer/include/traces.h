#pragma once

#include <filesystem>
#include <vector>
#include <list>

#include "trace_file.h"

const std::string cuda_number = "11.0";

namespace fs = std::filesystem;
class Traces {
public:
    Traces(const fs::path &bsuite_bpath);
    void parse();
private:
    std::vector<fs::path> get_dir_content(const fs::path &p) const;
    std::list<TraceFile*> m_trace_files; // data paths of each benchmark
};