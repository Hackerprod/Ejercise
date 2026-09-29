#include "kq_candidate2.hpp"

#include <iostream>
#include <stdexcept>

int main() {
    using namespace omega_v2_1;
    try {
        constexpr DWORD cpu_set_ids[] = {266, 264, 258, 270};
        HardwareInfo hardware;
        std::string error;
        if (!query_hardware(hardware, error)) throw std::runtime_error(error);
        std::vector<CpuSetRecord> selected;
        for (DWORD id : cpu_set_ids) {
            const CpuSetRecord* found = nullptr;
            for (const CoreRecord& core : hardware.cores) {
                if (!core.classified_p_core || core.intel_core_type != 0x40) continue;
                for (const CpuSetRecord& cpu : core.cpu_sets) if (cpu.id == id) found = &cpu;
            }
            if (!found) throw std::runtime_error("frozen P-core CPU-set missing");
            selected.push_back(*found);
        }
        WorkerPool pool(selected);
        const std::string report = omega_v2_1b_candidate_02::full_block_scalar_reference_test_candidate2(pool);
        const bool affinity_ok = pool.affinity_intact();
        pool.stop();
        std::cout << report << "\n{\"worker_affinity_ok\":" << (affinity_ok ? "true" : "false") << "}\n";
        return affinity_ok && report.find("\"pass\":true") != std::string::npos ? 0 : 1;
    } catch (const std::exception& exception) {
        std::cerr << "candidate_02 offline correctness error: " << exception.what() << '\n';
        return 2;
    }
}
