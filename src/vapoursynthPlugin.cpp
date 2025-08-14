#include "VapourSynth4.h"
#include "VSHelper4.h"
//#include "butter/vapoursynth.hpp"
#include "ssimu2/vapoursynth.hpp"
#include "util/gpuhelper.hpp"

static void VS_CC GpuInfo(const VSMap *in, VSMap *out, void *userData, VSCore *core, const VSAPI *vsapi) {
    std::stringstream ss;
    int count, device;

    //we don't need a full check at that point
    try{
        count = helper::checkGpuCount();
    } catch (const VshipError& e){
        vsapi->mapSetError(out, e.getErrorMessage().c_str());
        return;
    }

    int error;
    int gpuid = vsapi->mapGetInt(in, "gpu_id", 0, &error);
    if (error != peSuccess){
        gpuid = 0;
    }
    
    if (count <= gpuid || gpuid < 0){
        vsapi->mapSetError(out, VshipError(BadDeviceArgument, __FILE__, __LINE__).getErrorMessage().c_str());
        return;
    }

    auto devices = sycl::device::get_devices(sycl::info::device_type::gpu);

    if (error != peSuccess){
        //no gpu_id was selected
        for (int i = 0; i < count; i++){
            const auto& dev = devices[i];
            ss << "GPU " << i << ": " << dev.get_info<sycl::info::device::name>() << std::endl;
        }
    } else {
        const auto& dev = devices[gpuid];
        ss << "Name: " << dev.get_info<sycl::info::device::name>() << std::endl;
        ss << "MaxComputeUnits: " << dev.get_info<sycl::info::device::max_compute_units>() << std::endl;
        ss << "MaxWorkGroupSize: " << dev.get_info<sycl::info::device::max_work_group_size>() << std::endl;
        ss << "LocalMemSize: " << dev.get_info<sycl::info::device::local_mem_size>() << " bytes" << std::endl;
        ss << "GlobalMemSize: " << dev.get_info<sycl::info::device::global_mem_size>() / 1e9 << " GB" << std::endl;
        //ss << "MemoryBusWidth: " << dev.get_info<sycl::info::device::global_mem_cache_line_size>()*8 << " bits" << std::endl;
        ss << "Integrated: " << dev.is_cpu() << std::endl; // True if integrated (CPU) device
        try {
            sycl::queue q{sycl::queue{sycl::gpu_selector_v}};
            int res = helper::gpuKernelCheck(q);
            ss << "PassKernelCheck : " << res << std::endl;
        } catch (const VshipError&) {
            printf("Didn't pass kernel check");
            ss << "PassKernelCheck : 0" << std::endl;
        }
    }
    vsapi->mapSetData(out, "gpu_human_data", ss.str().data(), ss.str().size(), dtUtf8, maReplace);
}

VS_EXTERNAL_API(void) VapourSynthPluginInit2(VSPlugin *plugin, const VSPLUGINAPI *vspapi) {
    vspapi->configPlugin("com.swarejonge.vscycle", "vscycle", "VapourSynth SSIMULACRA2 on GPU", VS_MAKE_VERSION(3, 2), VAPOURSYNTH_API_VERSION, 0, plugin);
    vspapi->registerFunction("SSIMULACRA2", "reference:vnode;distorted:vnode;numStream:int:opt;gpu_id:int:opt;", "clip:vnode;", ssimu2::ssimulacra2Create, NULL, plugin);
    //vspapi->registerFunction("BUTTERAUGLI", "reference:vnode;distorted:vnode;intensity_multiplier:float:opt;distmap:int:opt;numStream:int:opt;gpu_id:int:opt;", "clip:vnode;", butter::butterCreate, NULL, plugin);
    vspapi->registerFunction("GpuInfo", "gpu_id:int:opt;", "gpu_human_data:data;", GpuInfo, NULL, plugin);
}
