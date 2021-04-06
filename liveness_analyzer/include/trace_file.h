#pragma once

#include <map>
#include <filesystem>

#include "inst_reg_accesses.h"
#include "warp_pending_accesses.h"

namespace fs = std::filesystem;

class TraceFile {
public:
    TraceFile(const fs::path &path);
    void parse();
private:
    void parse_line(const std::string &line, std::vector<unsigned> &dsts, std::vector<unsigned> &srcs);
    fs::path m_filename;
    fs::path m_path;
};