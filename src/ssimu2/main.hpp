#pragma once
#include "../util/preprocessor.hpp"
#include "../util/VshipExceptions.hpp"
#include "../util/gpuhelper.hpp"
#include "../util/float3operations.hpp"
#include "../util/concurrency.hpp"
#include "makeXYB.hpp"
#include "downsample.hpp"
#include "gaussianblur.hpp"
#include "score.hpp"

namespace ssimu2{

template <InputMemType T>
static void memoryorganizer(TVec3<f32>* out,
                    const uint8_t* srcp0,
                    const uint8_t* srcp1,
                    const uint8_t* srcp2,
                    int64_t stride,
                    int64_t width,
                    int64_t height,
                    sycl::queue& q)
{
    const int64_t total = width * height;

    // Keep your 256-thread work-group size
    const size_t local_size = std::min<int64_t>(256, total);
    const size_t global_size = ((total + local_size - 1) / local_size) * local_size;

    q.submit([&](sycl::handler& h) {
        h.parallel_for(
            sycl::nd_range<1>(sycl::range<1>(global_size),
                            sycl::range<1>(local_size)),
            [=](sycl::nd_item<1> it) {
                const int64_t x = it.get_global_id(0);
                if (x >= total) return;

                const int j = x % width;
                const int i = x / width;

                out[i * width + j].x() = convertPointer<T>(srcp0, i, j, stride);
                out[i * width + j].y() = convertPointer<T>(srcp1, i, j, stride);
                out[i * width + j].z() = convertPointer<T>(srcp2, i, j, stride);
            }
        );
    });
}

int64_t getTotalScaleSize(int64_t width, int64_t height){
    int64_t result = 0;
    for (int scale = 0; scale < 6; scale++){
        result += width*height;
        width = (width-1)/2+1;
        height = (height-1)/2+1;
    }
    return result;
}

//expects packed linear RGB input. Beware that each src1_d, src2_d and temp_d must be of size "totalscalesize" even if the actual image is contained in a width*height format
// src_1_d src_2_d and temp_d all are on the GPU
double ssimu2GPUProcess(TVec3<f32>* src1_d, TVec3<f32>* src2_d, TVec3<f32>* temp_d, TVec3<f32>* pinned, int64_t width, int64_t height, GaussianHandle& gaussianhandle, int64_t maxshared, sycl::queue& q){
    const int64_t totalscalesize = getTotalScaleSize(width, height);
    //step 1 : fill the downsample part
    int64_t nw = width;
    int64_t nh = height;
    int64_t index = 0;
    for (int scale = 1; scale <= 5; scale++){
        downsample(src1_d+index, src1_d+index+nw*nh, nw, nh, q);
        downsample(src2_d+index, src2_d+index+nw*nh, nw, nh, q);
        index += nw*nh;
        nw = (nw -1)/2 + 1;
        nh = (nh - 1)/2 + 1;
    }

    //step 2 : positive XYB transition
    rgb_to_positive_xyb(src1_d, totalscalesize, q);
    rgb_to_positive_xyb(src2_d, totalscalesize, q);

    //step 4 : ssim map
    
    //step 5 : edge diff map    
    std::vector<TVec3<f32>> allscore_res = allscore_map(src1_d, src2_d, temp_d, pinned, width, height, maxshared, gaussianhandle, q);
    

    //step 6 : format the vector
    std::vector<f32> measure_vec(108);

    for (int plane = 0; plane < 3; plane++) {
        for (int scale = 0; scale < 6; scale++) {
            for (int n = 0; n < 2; n++) {
                for (int i = 0; i < 3; i++) {
                    if (plane == 0) measure_vec[plane*6*2*3 + scale*2*3 + n*3 + i] = allscore_res[scale*2*3 + i*2 + n].x();
                    if (plane == 1) measure_vec[plane*6*2*3 + scale*2*3 + n*3 + i] = allscore_res[scale*2*3 + i*2 + n].y();
                    if (plane == 2) measure_vec[plane*6*2*3 + scale*2*3 + n*3 + i] = allscore_res[scale*2*3 + i*2 + n].z();
                }
            }
        }
    }

    //step 7 : enjoy !
    f32 res = final_score(measure_vec);

    //hipEventRecord(event_d, stream); //place an event in the stream at the end of all our operations
    //hipEventSynchronize(event_d); //when the event is complete, we know our gpu result is ready!

    return res;
}

template <InputMemType T>
double ssimu2process(const uint8_t *srcp1[3], const uint8_t *srcp2[3], TVec3<f32>* pinned, int64_t stride, int64_t width, int64_t height, GaussianHandle& gaussianhandle, int64_t maxshared, sycl::queue& stream){
    // bytes needed for the three-plane staging area vs. a float3 buffer of totalscalesize
    const int64_t totalscalesize = getTotalScaleSize(width, height);
    const size_t plane_bytes = static_cast<size_t>(stride) * static_cast<size_t>(height);
    const size_t three_planes = plane_bytes * 3;
    const size_t float3_block = sizeof(TVec3<f32>) * static_cast<size_t>(totalscalesize);

    // single big block: [ src1_d | src2_d | temp_scratch ]
    const size_t total_bytes =
        2 * float3_block + sycl::max(float3_block, three_planes);

    // Allocate device USM (throws on OOM → convert to your error)
    unsigned char* mem = nullptr;
    try {
        mem = sycl::malloc_device<unsigned char>(total_bytes, stream);
        if (!mem) throw std::bad_alloc{};
    } catch (...) {
        VSHIP_THROW(OutOfVRAM);
    }

    auto* src1_d = reinterpret_cast<TVec3<f32>*>(mem);
    auto* src2_d = reinterpret_cast<TVec3<f32>*>(mem + float3_block);
    unsigned char* temp_bytes = mem + 2 * float3_block;              // scratch base (bytes)
    void* temp_scratch_for_gpu = static_cast<void*>(temp_bytes);     // pass-through scratch

    // Stage the three host planes for src1 into device scratch
    {
        uint8_t* p0 = temp_bytes;
        uint8_t* p1 = temp_bytes + 1 * plane_bytes;
        uint8_t* p2 = temp_bytes + 2 * plane_bytes;

        stream.memcpy(p0, srcp1[0], plane_bytes);
        stream.memcpy(p1, srcp1[1], plane_bytes);
        stream.memcpy(p2, srcp1[2], plane_bytes);
        
        // Convert staged planes → interleaved/float3 RGB into src1_d
        memoryorganizer<T>(src1_d, p0, p1, p2, stride, width, height, stream);
    }

    // Stage the three host planes for src2 into the same device scratch (reused)
    {
        uint8_t* p0 = temp_bytes + 0 * plane_bytes;
        uint8_t* p1 = temp_bytes + 1 * plane_bytes;
        uint8_t* p2 = temp_bytes + 2 * plane_bytes;

        stream.memcpy(p0, srcp2[0], plane_bytes);
        stream.memcpy(p1, srcp2[1], plane_bytes);
        stream.memcpy(p2, srcp2[2], plane_bytes);

        memoryorganizer<T>(src2_d, p0, p1, p2, stride, width, height, stream);
    }

    // Colorspace
    rgb_to_linear(src1_d, totalscalesize, stream);
    rgb_to_linear(src2_d, totalscalesize, stream);

    double res;
    try {
        res = ssimu2GPUProcess(src1_d, src2_d, (TVec3<f32>*)(temp_bytes), pinned, width, height, gaussianhandle, maxshared, stream);
    } catch (const VshipError& e){
        sycl::free(temp_bytes, stream);
        throw e;
    }

    // Make sure all enqueued ops that might touch 'mem' are done before free
    stream.wait_and_throw();
    sycl::free(mem, stream);
    
    return res;
}

class SSIMU2ComputingImplementation{
public:
    SSIMU2ComputingImplementation() : stream(sycl::queue{sycl::gpu_selector_v, sycl::property::queue::in_order{}}) {
    }

    void init(int64_t w, int64_t h) {
        width = w;
        height = h;

        gaussianhandle.init(stream);

        // Query “shared memory per block” equivalent
        auto dev = stream.get_device();
        maxshared = dev.get_info<sycl::info::device::local_mem_size>();

        // Allocate pinned host memory (USM host). Many backends pin this.
        const int64_t pinnedsize = allocsizeScore(width, height, maxshared);
        pinned = sycl::malloc_host<TVec3<f32>>(static_cast<size_t>(pinnedsize), stream);
        if (!pinned) {
            gaussianhandle.destroy(stream);
            VSHIP_THROW(OutOfRAM);
        }
    }

    void destroy() {
        gaussianhandle.destroy(stream);
        sycl::free(pinned, stream);
    }

    template <InputMemType T>
    double run(const uint8_t* srcp1[3], const uint8_t* srcp2[3], int64_t stride){
        return ssimu2process<T>(srcp1, srcp2, pinned, stride, width, height, gaussianhandle, maxshared, stream);
    }

private:
    sycl::queue stream;
    GaussianHandle gaussianhandle;
    TVec3<f32>* pinned;
    int64_t width;
    int64_t height;
    int maxshared;
};

}