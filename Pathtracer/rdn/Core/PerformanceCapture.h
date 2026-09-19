#pragma once
#include <cstdlib>
#include <fstream>
#include <iomanip>
#include "../Common.h"
#include "../minecraft/voxel_streamer.h"

// Opt-in capture for repeatable scene comparisons, including the actual pass
// timings and published geometry/light counts rather than pool allocations.
class PerformanceCapture {
    std::ofstream output;
    uint64_t frame = 0;
public:
    PerformanceCapture() {
        const char* path = std::getenv("RT_PERF_CSV");
        if (!path || !*path) return;
        output.open(path);
        output << "frame,kind,pass,ms,triangles,blas,light_triangles,lod_factor,pending,light_selection_reused\n";
        output << std::setprecision(8);
    }
    void record(const FrameStats& stats, const mc::StreamerStats* mc) {
        if (!output) return;
        auto row = [&](const char* kind, const std::string& pass, double ms) {
            output << frame << ',' << kind << ",\"" << pass << "\"," << ms << ','
                << (mc ? mc->trianglesInTlas : 0) << ',' << (mc ? mc->geometryBlasRendered : 0) << ','
                << (mc ? mc->lightTrisInTree : 0) << ',' << (mc ? mc->lodFactorNow : 0) << ','
                << (mc ? mc->pending+mc->meshing+mc->meshed+mc->uploading : 0) << ','
                << (mc && mc->lightSelectionReused ? 1 : 0) << '\n';
        };
        row("cpu","Frame",stats.cpuFrameMs); row("cpu","Streaming",stats.cpuStreamingMs);
        if(mc)row("cpu","Light selection",mc->lightSelectionMs);
        if(stats.gpuTimingsValid) {
            row("gpu","Frame",stats.gpuFrameMs);
            for(const auto& pass : stats.gpuPasses)row("gpu",pass.name,pass.gpuMs);
        }
        ++frame; output.flush();
    }
};
