#include <iostream>
#include <fstream>
#include <string>
#include <vector>
#include <cassert>
#include <memory>

#include "trace_file.h"
#include "debug.h"
#include "warp_reuse_distance.h"

TraceFile::TraceFile(const fs::path &path)
    : m_path{path.parent_path()}, m_filename{path.filename()}
{
    DPRINTF("Trace File " << m_path.string() << "/" << m_filename.string() << " Created")
}

void TraceFile::parse()
{
    /*-----------------Openning Files------------------*/
    auto full_path = m_path / m_filename;
    DPRINTF("Trace File " << full_path << " Parsing Started")
    std::ifstream trace_file(full_path);
    if (!trace_file)
    {
        throw std::runtime_error("Cannot Open File " + full_path.string());
    }

    auto reuse_trace_fname = m_filename.stem();
    reuse_trace_fname += ".rtrace";
    DPRINTF("Writing Reuse Trace " << (m_path / reuse_trace_fname) << " Started")
    std::ofstream reuse_trace_file(m_path / reuse_trace_fname);
    if (!reuse_trace_file)
    {
        throw std::runtime_error("cannot open file " + (m_path / reuse_trace_fname).string());
    }

    /*---------------------Open reuse distance file------------------------*/
    auto rdistance_trace_fname = m_filename.stem();
    rdistance_trace_fname += ".rdtrace";
    DPRINTF("Writing Reuse Distance Trace " << (m_path / rdistance_trace_fname) << " Started")
    std::ofstream rdist_trace_file(m_path / rdistance_trace_fname);
    if (!rdist_trace_file)
    {
        throw std::runtime_error("cannot open file " + (m_path / rdistance_trace_fname).string());
    }
    /*---------------------------------------------*/

    std::string line;                    // each line read from trace file
    unsigned long long parsed_lines = 0; // num lines in trace file
    unsigned long long writen_lines = 0;
    unsigned chunk_id = 0; 

    const unsigned chunck_size = 100;
    unsigned next_warp_id = 0;

    std::vector<std::vector<std::vector<unsigned>>> warps_inst_srcs; // Inst srcs
    std::vector<std::vector<std::vector<unsigned>>> warps_inst_dsts; // Inst dsts
    std::vector<WarpPendingAccesses *> warps_pending_reuses;         // warps pending tracker
    std::vector<WarpReuseDistance *> warps_reuse_distances; // Warps reuse distance tracker
    
    while (std::getline(trace_file, line))
    {
        // read trace file line by line
        if (line.empty() || line[0] == '-' || line[0] == '#' || isspace(line[0]) || line[0] == 't' || line[0] == 'w')
        {
            continue; // skip trace configs and params
        }
        if (line[0] == 'i') // as soon as we observed insts it means that this is a new warp execution
        {
            parsed_lines++;
            // new warp started

            if (next_warp_id == chunck_size)
            {
                assert(warps_pending_reuses.size() == warps_inst_srcs.size());
                assert(warps_reuse_distances.size() == warps_pending_reuses.size());
                assert(warps_inst_srcs.size() == warps_inst_dsts.size());
                assert(warps_inst_dsts.size() == chunck_size);
                auto warp_inst_srcs = warps_inst_srcs.begin();
                auto warp_inst_dsts = warps_inst_dsts.begin();
                unsigned warp_offset = 0;
                for (auto &warp_pending_reuses : warps_pending_reuses)
                {
                    assert(warp_pending_reuses);
                    // if new warp started we should write last warp reuse info
                    assert(warp_inst_srcs->size() == warp_inst_dsts->size()); // n_srcs = n_dsts
                    auto &warp_reuse_distance = warps_reuse_distances[warp_offset];
                    assert(warp_reuse_distance);
                    warp_pending_reuses->done_adding_reg_use();
                    warp_reuse_distance->done_adding_reg_use();
                    reuse_trace_file << "DWID = " << chunk_id * chunck_size + warp_offset << std::endl;
                    rdist_trace_file << "DWID = " << chunk_id * chunck_size + warp_offset << std::endl;
                    writen_lines++;
                    for (auto inst_reg_srcs = warp_inst_srcs->begin(), inst_reg_dsts = warp_inst_dsts->begin(); inst_reg_srcs != warp_inst_srcs->end(); inst_reg_srcs++, inst_reg_dsts++)
                    {
                        reuse_trace_file << warp_pending_reuses->update_pending_reuses(*inst_reg_dsts, *inst_reg_srcs) << std::endl;
                        rdist_trace_file << warp_reuse_distance->update_reuse_distance(*inst_reg_dsts, *inst_reg_srcs) << std::endl;
                        writen_lines++;
                    }
                    warp_inst_srcs++;
                    warp_inst_dsts++;
                    warp_offset++;
                    delete warp_pending_reuses;
                    assert(warp_reuse_distance->all_insts_covered());
                    delete warp_reuse_distance;
                    warp_pending_reuses = nullptr;
                    warp_reuse_distance = nullptr;
                }

                assert(warp_inst_srcs == warps_inst_srcs.end());
                assert(warp_inst_dsts == warps_inst_dsts.end());

                next_warp_id = 0;
                chunk_id++;
                warps_inst_srcs.clear();
                warps_inst_dsts.clear();
                warps_pending_reuses.clear();
                warps_reuse_distances.clear();
            }
            warps_inst_srcs.push_back({});
            warps_inst_dsts.push_back({});
            warps_pending_reuses.push_back(new WarpPendingAccesses(chunk_id * chunck_size + next_warp_id)); // add per warp access tracker
            warps_reuse_distances.push_back(new WarpReuseDistance(chunk_id * chunck_size + next_warp_id)); // add per warp reuse distance tracker
            next_warp_id++;
            continue;
        }
        parsed_lines++;
        std::vector<unsigned> srcs;
        std::vector<unsigned> dsts;
        parse_line(line, dsts, srcs);
        warps_pending_reuses.back()->add_reg_use(dsts, srcs);
        warps_reuse_distances.back()->add_reg_reuse(dsts,srcs);
        warps_inst_dsts.back().push_back(dsts);
        warps_inst_srcs.back().push_back(srcs);
        //m_reg_accesses[dwid].push_back(reg_accesses);
    }
    if (parsed_lines > 0)
    {
        // for the last chunk
        assert(warps_pending_reuses.size() == warps_inst_srcs.size());
        assert(warps_reuse_distances.size() == warps_pending_reuses.size());
        assert(warps_inst_srcs.size() == warps_inst_dsts.size());
        assert(warps_inst_dsts.size() <= chunck_size);

        auto warp_inst_srcs = warps_inst_srcs.begin();
        auto warp_inst_dsts = warps_inst_dsts.begin();
        unsigned warp_offset = 0;
        for (auto &warp_pending_reuses : warps_pending_reuses)
        {
            assert(warp_pending_reuses);
            auto &warp_reuse_distance = warps_reuse_distances[warp_offset];
            assert(warp_reuse_distance);
            // if new warp started we should write last warp reuse info
            assert(warp_inst_srcs->size() == warp_inst_dsts->size()); // n_srcs = n_dsts
            warp_pending_reuses->done_adding_reg_use();
            warp_reuse_distance->done_adding_reg_use();
            reuse_trace_file << "DWID = " << chunk_id * chunck_size + warp_offset << std::endl;
            rdist_trace_file << "DWID = " << chunk_id * chunck_size + warp_offset << std::endl;
            writen_lines++;
            for (auto inst_reg_srcs = warp_inst_srcs->begin(), inst_reg_dsts = warp_inst_dsts->begin(); inst_reg_srcs != warp_inst_srcs->end(); inst_reg_srcs++, inst_reg_dsts++)
            {
                reuse_trace_file << warp_pending_reuses->update_pending_reuses(*inst_reg_dsts, *inst_reg_srcs) << std::endl;
                rdist_trace_file << warp_reuse_distance->update_reuse_distance(*inst_reg_dsts, *inst_reg_srcs) << std::endl;
                writen_lines++;
            }
            warp_inst_srcs++;
            warp_inst_dsts++;
            warp_offset++;
            
            delete warp_pending_reuses;
            warp_pending_reuses = nullptr;
            assert(warp_reuse_distance->all_insts_covered());
            delete warp_reuse_distance;
            warp_reuse_distance = nullptr;
        }
        assert(warp_inst_srcs == warps_inst_srcs.end());
        assert(warp_inst_dsts == warps_inst_dsts.end());
        warps_inst_dsts.clear();
        warps_inst_srcs.clear();
        warps_pending_reuses.clear();
        warps_reuse_distances.clear();
    }

    /*----------------------closing files--------------------*/
    trace_file.close();
    reuse_trace_file.close();
    rdist_trace_file.close(); // close reuse distance trace file
    /*-------------------------------------------------------*/

    DPRINTF(parsed_lines << " lines parsed " << writen_lines << " lines written")
    assert(parsed_lines == writen_lines);
}

void TraceFile::parse_line(const std::string &line, std::vector<unsigned> &dsts, std::vector<unsigned> &srcs)
{

    std::stringstream ss(line);
    std::string token;
    size_t token_num = 0;
    unsigned num_dsts = 0;
    unsigned num_srcs = 0;
    while (std::getline(ss, token, ' '))
    {
        if (token_num == 0)
        {
            token_num++;
            continue;
        }
        if (token_num == 1)
        {
            token_num++;
            continue; // Skip Active Mask
        }
        if (token_num == 2)
        {
            // we reached destnum token
            num_dsts = std::stoi(token);
            assert(num_dsts < 2);
            token_num++;
            continue;
        }
        if (token_num > 2 && token_num < 3 + num_dsts)
        {
            //we are reading each dst
            assert(token[0] == 'R');
            auto reg_num = std::stoi(token.substr(1));
            dsts.push_back(reg_num);
            token_num++;
            continue;
        }
        if (token_num == 3 + num_dsts)
        {
            // this is the inst
            token_num++;
            continue;
        }
        if (token_num == 4 + num_dsts)
        {
            // we reached source count in the instruction
            num_srcs = std::stoi(token);
            assert(num_srcs < 5);
            token_num++;
            continue;
        }
        if (token_num > 4 + num_dsts && token_num < 5 + num_dsts + num_srcs)
        {
            // we are reading source operands
            assert(token[0] == 'R');
            auto reg_num = std::stoi(token.substr(1));
            srcs.push_back(reg_num);
            token_num++;
            continue;
        }
    }
    assert(dsts.size() == num_dsts);
    assert(srcs.size() == num_srcs);
}
