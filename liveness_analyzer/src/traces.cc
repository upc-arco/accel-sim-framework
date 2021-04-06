#include <iostream>
#include <stdexcept>

#include "traces.h"
#include "debug.h"

Traces::Traces(const fs::path &bsuite_bpath)
{
    if (!fs::is_directory(bsuite_bpath))
    {
        throw std::runtime_error(bsuite_bpath.string() + " is not a benchmark suite directory");
    }

    auto bsuite_dir_content = get_dir_content(bsuite_bpath); // files and folders in bsuite root dir

    bool found_traces_dir = false;
    fs::path traces_root_path;
    for (auto dir : bsuite_dir_content)
    {
        const auto bsuit_root_path = bsuite_bpath / cuda_number; // we expect traces insede a folder with cuda number name

        if ((dir == bsuit_root_path) && fs::is_directory(dir))
        {
            found_traces_dir = true;
            traces_root_path = dir;
        }
    }

    if (!found_traces_dir)
    {
        throw std::runtime_error("Traces Root DIR Not Found");
    }

    auto bench_root_dirs = get_dir_content(traces_root_path);

    for (const auto &bench_root : bench_root_dirs)
    {
        if (!fs::is_directory(bench_root))
        {
            throw std::runtime_error("Not Found Directory " + bench_root.string());
        }

        auto bench_name = bench_root.filename().string();  // each benchmark should have a folder
        auto data_root_dirs = get_dir_content(bench_root); // each benchmark has mutiple data
        for (const auto &bench_data_root : data_root_dirs)
        {
            if (!fs::is_directory(bench_root))
            {
                throw std::runtime_error("Not Found Directory " + bench_root.string());
            }
            auto traces_root_dir = bench_data_root / "traces"; // each benchmark has a traces folder
            if (!fs::is_directory(traces_root_dir))
            {
                throw std::runtime_error("Not Found Directory " + traces_root_dir.string());
            }
            auto traces_dir_content = get_dir_content(traces_root_dir);
            for (const auto &trace_dir_file : traces_dir_content)
            {
                auto extension = trace_dir_file.extension().string();
                if (extension != ".traceg")
                    continue;
                // we found a trace file
                //auto trace_file_name = trace_dir_file.filename().string();
                m_trace_files.push_back(new TraceFile(trace_dir_file));
            }
        }
    }

    DPRINTF(m_trace_files.size() << " Trace Files Found")
}

std::vector<fs::path> Traces::get_dir_content(const fs::path &p) const
{
    std::vector<fs::path> result;
    for (const auto &entry : fs::directory_iterator(p))
    {
        result.push_back(entry.path());
    }
    return result;
}

void Traces::parse()
{
    unsigned num_analyzed = 0;
    for (auto it = m_trace_files.begin(); it != m_trace_files.end();)
    {
        auto &trace = *it;
        //auto &trace = m_trace_files.front();
        trace->parse();
        //trace.reuse_analysis();
        //trace.write_reuse_trace();
        delete trace;
        trace = nullptr;
        it = m_trace_files.erase(it);
        num_analyzed++;
    }
    DPRINTF("Analyzed Files = " << num_analyzed)
}