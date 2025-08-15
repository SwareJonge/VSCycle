#include <math.h>

namespace ssimu2{

int64_t allocsizeScore(int64_t width, int64_t height, int maxshared){
    int64_t w = width;
    int64_t h = height;
    int64_t th_x, th_y, bl_x, bl_y;
    int64_t pinnedsize = 0;
    for (int i = 0; i < 6; i++){
        th_x = 16;
        th_y = 16;
        bl_x = (w-1)/th_x + 1;
        bl_y = (h-1)/th_y + 1;
        bl_x = bl_x*bl_y; //convert to linear
        th_x = 0;
        if (maxshared != 0) {
            th_x = sycl::min<int64_t>(maxshared/(6*sizeof(sycl::float3)/32*32), (int64_t)sycl::min((int64_t)1024, bl_x));    
        }
        
        while (bl_x >= 256){
            bl_x = (bl_x -1)/th_x + 1;
        }
        pinnedsize += 6*bl_x;

        w = (w-1)/2 + 1;
        h = (h-1)/2 + 1;
    }
    return pinnedsize;
}

void allscore_map_Kernel(
    sycl::queue &q,
    sycl::float3* dst,                         // device USM pointer where per-block outputs go
    sycl::float3* im1,                         // device USM input 1 (base + index offset handled by caller)
    sycl::float3* im2,                         // device USM input 2
    int64_t width,
    int64_t height,
    float* gaussiankernel,                   // device USM pointer
    float* gaussiankernel_integral,
    int64_t bl_x,                            // number of blocks in X (as computed by caller)
    int64_t bl_y,                            // number of blocks in Y
    int64_t th_x,                            // local threads X
    int64_t th_y                             // local threads Y
) {
    // local (Y,X) and global ranges for SYCL
    sycl::range<2> local_range((size_t)th_y, (size_t)th_x);
    sycl::range<2> global_range((size_t)bl_y * (size_t)th_y, (size_t)bl_x * (size_t)th_x);

    // number of threads per block (1 block = th_x * th_y)
    const int threadnum = th_x * th_y;

    // how many local elements we need for shared usage:
    const size_t local_elems_for_reduce = (size_t)6 * (size_t)threadnum;
    const size_t local_elems_for_sharedmem = 32 * 32;
    const size_t shared_elems = sycl::max(local_elems_for_reduce, local_elems_for_sharedmem);

    q.submit([&](sycl::handler &h) {
        // one local buffer used both for the 6*threadnum reduction arrays
        // and (at the same time) as a 32x32 sharedmem for GaussianSmart* helpers.
        sycl::local_accessor<sycl::float3, 1> sharedmem(sycl::range<1>(shared_elems), h);

        h.parallel_for(
            sycl::nd_range<2>(global_range, local_range),
            [=](sycl::nd_item<2> it) {
                // --- indexes (NOTE: SYCL ranges are (Y,X)) ---
                const int64_t gx = (int64_t)it.get_global_id(1); // x
                const int64_t gy = (int64_t)it.get_global_id(0); // y
                const int64_t x = gx;
                const int64_t y = gy;
                const int64_t id = y * width + x;

                const int64_t lx = it.get_local_id(1); // thread x within block
                const int64_t ly = it.get_local_id(0); // thread y within block
                const int64_t blid = ly * it.get_local_range(1) + lx; // local flattened thread id
                const int64_t threadnum_local = it.get_local_range(0) * it.get_local_range(1);

                // local pointer
                sycl::float3* smem = sharedmem.get_multi_ptr<sycl::access::decorated::no>().get();

                // carve reduction sections (each of size threadnum_local)
                sycl::float3* sumssim1 = smem;                              // [0 .. threadnum-1]
                sycl::float3* sumssim4 = sumssim1 + threadnum_local;        // [threadnum .. 2*threadnum-1]
                sycl::float3* suma1    = sumssim4 + threadnum_local;        // ...
                sycl::float3* suma4    = suma1 + threadnum_local;
                sycl::float3* sumd1    = suma4 + threadnum_local;
                sycl::float3* sumd4    = sumd1 + threadnum_local;

                // --- compute m1 ---
                GaussianSmartSharedLoad(sharedmem, im1, x, y, width, height, it);
                GaussianSmart_Device(sharedmem, x, y, width, height, gaussiankernel, gaussiankernel_integral, it);
                const sycl::float3 m1 = sharedmem[(ly + 8) * 32 + (lx + 8)];
                it.barrier(sycl::access::fence_space::local_space);

                // --- compute m2 ---
                GaussianSmartSharedLoad(sharedmem, im2, x, y, width, height, it);
                GaussianSmart_Device(sharedmem, x, y, width, height, gaussiankernel, gaussiankernel_integral, it);
                const sycl::float3 m2 = sharedmem[(ly + 8) * 32 + (lx + 8)];
                it.barrier(sycl::access::fence_space::local_space);

                // --- su11 ---
                GaussianSmartSharedLoadProduct(sharedmem, im1, im1, x, y, width, height, it);
                GaussianSmart_Device(sharedmem, x, y, width, height, gaussiankernel, gaussiankernel_integral, it);
                const sycl::float3 su11 = sharedmem[(ly + 8) * 32 + (lx + 8)];
                it.barrier(sycl::access::fence_space::local_space);

                // --- su22 ---
                GaussianSmartSharedLoadProduct(sharedmem, im2, im2, x, y, width, height, it);
                GaussianSmart_Device(sharedmem, x, y, width, height, gaussiankernel, gaussiankernel_integral, it);
                const sycl::float3 su22 = sharedmem[(ly + 8) * 32 + (lx + 8)];
                it.barrier(sycl::access::fence_space::local_space);

                // --- su12 ---
                GaussianSmartSharedLoadProduct(sharedmem, im1, im2, x, y, width, height, it);
                GaussianSmart_Device(sharedmem, x, y, width, height, gaussiankernel, gaussiankernel_integral, it);
                const sycl::float3 su12 = sharedmem[(ly + 8) * 32 + (lx + 8)];
                it.barrier(sycl::access::fence_space::local_space);

                // --- compute d0, d1, d2 (same math as before) ---
                sycl::float3 d0, d1, d2;
                if (x < width && y < height) {
                    const sycl::float3 m11 = m1 * m1;
                    const sycl::float3 m22 = m2 * m2;
                    const sycl::float3 m12 = m1 * m2;
                    const sycl::float3 m_diff = m1 - m2;
                    const sycl::float3 num_m = fma(m_diff, m_diff * -1.0f, 1.0f);
                    const sycl::float3 num_s = fma(su12 - m12, 2.0f, 0.0009f);

                    const sycl::float3 denom_s = (su11 - m11) + (su22 - m22) + 0.0009f;
                    d0 = sycl::max(1.0f - ((num_m * num_s) / denom_s), 0.0f);

                    const sycl::float3 v1 = (sycl::fabs(im2[id] - m2) + 1.0f) /
                                          (sycl::fabs(im1[id] - m1) + 1.0f) - 1.0f;
                    d1 = sycl::max(v1, 0.0f); // artifact
                    d2 = sycl::max(v1 * -1.0f, 0.0f); //detailloss
                } else {
                    zeroVec(d0); zeroVec(d1); zeroVec(d2);
                }

                // write per-thread temporary accumulators into the shared arrays
                sumssim1[blid] = d0;
                sumssim4[blid] = tothe4th(d0);     // assume tothe4th available
                suma1[blid]    = d1;
                suma4[blid]    = tothe4th(d1);
                sumd1[blid]    = d2;
                sumd4[blid]    = tothe4th(d2);
                it.barrier(sycl::access::fence_space::local_space);

                // --- in-block pointer-jumping reduction ---
                int next = 1;
                while (next < threadnum_local) {
                    if (blid + next < threadnum_local && (blid % (next * 2) == 0)) {
                        sumssim1[blid] += sumssim1[blid + next];
                        sumssim4[blid] += sumssim4[blid + next];
                        suma1[blid]    += suma1[blid + next];
                        suma4[blid]    += suma4[blid + next];
                        sumd1[blid]    += sumd1[blid + next];
                        sumd4[blid]    += sumd4[blid + next];
                    }
                    next *= 2;
                    it.barrier(sycl::access::fence_space::local_space);
                }

                // if thread 0 within block, write block result to dst
                if (blid == 0) {
                    // compute linear block index = blockIdx.y * gridDim.x + blockIdx.x
                    const int64_t block_x = it.get_group(1);
                    const int64_t block_y = it.get_group(0);
                    const int64_t gridDim_x = it.get_group_range(1);
                    const int64_t block_linear = block_y * gridDim_x + block_x;

                    const float norm = 1.0f / (float)(width * height);
                    dst[0 * (bl_x * bl_y) + block_linear] = sumssim1[0] * norm;
                    dst[1 * (bl_x * bl_y) + block_linear] = sumssim4[0] * norm;
                    dst[2 * (bl_x * bl_y) + block_linear] = suma1[0]    * norm;
                    dst[3 * (bl_x * bl_y) + block_linear] = suma4[0]    * norm;
                    dst[4 * (bl_x * bl_y) + block_linear] = sumd1[0]    * norm;
                    dst[5 * (bl_x * bl_y) + block_linear] = sumd4[0]    * norm;
                }
            } // end parallel_for
        ); // end submit
    }); // end q.submit
}

std::vector<sycl::float3> allscore_map(sycl::float3* im1, sycl::float3* im2, sycl::float3* temp, sycl::float3* pinned, int64_t basewidth, int64_t baseheight, int64_t maxshared, GaussianHandle& gaussianhandle, sycl::queue& stream){
     // output is {normssim1scale1, normssim4scale1, ..., normd4scale3} (18 vec3 pairs)
    std::vector<sycl::float3> result(2 * 6 * 3);
    for (auto& v : result) { zeroVec(v); }

    constexpr int reduce_up_to = 256;
    int64_t w = basewidth;
    int64_t h = baseheight;
    int64_t th_x, th_y;
    int64_t bl_x, bl_y;
    int64_t index = 0;
    std::vector<int> scaleoutdone(7);
    scaleoutdone[0] = 0;
    for (int scale = 0; scale < 6; scale++){
        th_x = 16;
        th_y = 16;
        bl_x = (w-1)/th_x + 1;
        bl_y = (h-1)/th_y + 1;
        int64_t blr_x = bl_x*bl_y;
        
        allscore_map_Kernel(stream,
                           temp + scaleoutdone[scale] + ((blr_x >= reduce_up_to) ? 6*bl_x*bl_y : 0),
                           im1 + index,
                           im2 + index,
                           w, h,
                           gaussianhandle.gaussiankernel_d,
                           gaussianhandle.gaussiankernel_integral_d,
                           bl_x, bl_y, th_x, th_y);
        
        //printf("I got %s with %ld %ld %ld\n", hipGetErrorString(hipGetLastError()), 6*sizeof(sycl::float3)*th_x*th_y, bl_x, bl_y);
        //GPU_CHECK(hipGetLastError());

        th_x = sycl::min((int64_t)(maxshared/(6*sizeof(sycl::float3)))/32*32, sycl::min((int64_t)1024, blr_x));
        int oscillate = 0; //3 sets of memory: real destination at 0, first at 6*bl_x for oscillate 0 and last at 12*bl_x for oscillate 1;
        int64_t oldblr_x = blr_x;
        while (blr_x >= reduce_up_to){
            blr_x = (blr_x - 1) / th_x + 1;

            sycl::float3* dst_ptr =
                temp + scaleoutdone[scale] +
                ((blr_x >= reduce_up_to) ? ((oscillate ^ 1) + 1) * 6 * bl_x * bl_y : 0);
            sycl::float3* src_ptr =
                temp + scaleoutdone[scale] + (oscillate + 1) * 6 * bl_x * bl_y;

            // launch sumreduce for this stage
            {
                sycl::range<1> local(th_x);
                sycl::range<1> global(blr_x * th_x);

                stream.submit([&](sycl::handler& h) {
                    sycl::local_accessor<sycl::float3, 1> smem(sycl::range<1>(6 * th_x), h);

                    h.parallel_for(
                        sycl::nd_range<1>(global, local),
                        [=](sycl::nd_item<1> it) {
                            const int64_t x       = it.get_global_linear_id();
                            const int64_t th      = it.get_local_linear_id();
                            const int64_t threads = it.get_local_range(0);
                            const int64_t block   = it.get_group_linear_id();
                            const int64_t blocks  = it.get_group_range(0);

                            sycl::float3* shm = smem.get_multi_ptr<sycl::access::decorated::no>().get();
                            sycl::float3* s1 = shm;
                            sycl::float3* s4 = s1 + threads;
                            sycl::float3* a1 = s4 + threads;
                            sycl::float3* a4 = a1 + threads;
                            sycl::float3* d1 = a4 + threads;
                            sycl::float3* d4 = d1 + threads;

                            if (x >= oldblr_x) {
                                zeroVec(s1[th]); zeroVec(s4[th]);
                                zeroVec(a1[th]); zeroVec(a4[th]);
                                zeroVec(d1[th]); zeroVec(d4[th]);
                            } else {
                                s1[th] = src_ptr[x];
                                s4[th] = src_ptr[x + oldblr_x];
                                a1[th] = src_ptr[x + 2 * oldblr_x];
                                a4[th] = src_ptr[x + 3 * oldblr_x];
                                d1[th] = src_ptr[x + 4 * oldblr_x];
                                d4[th] = src_ptr[x + 5 * oldblr_x];
                            }
                            it.barrier(sycl::access::fence_space::local_space);

                            for (int step = 1; step < threads; step <<= 1) {
                                if (th + step < threads && (th % (step * 2) == 0)) {
                                    s1[th] += s1[th + step];
                                    s4[th] += s4[th + step];
                                    a1[th] += a1[th + step];
                                    a4[th] += a4[th + step];
                                    d1[th] += d1[th + step];
                                    d4[th] += d4[th + step];
                                }
                                it.barrier(sycl::access::fence_space::local_space);
                            }

                            if (th == 0) {
                                dst_ptr[block] = s1[0];
                                dst_ptr[1 * blocks + block] = s4[0];
                                dst_ptr[2 * blocks + block] = a1[0];
                                dst_ptr[3 * blocks + block] = a4[0];
                                dst_ptr[4 * blocks + block] = d1[0];
                                dst_ptr[5 * blocks + block] = d4[0];
                            }
                        }
                    );
                });
            }

            oscillate ^= 1;
            oldblr_x = blr_x;
        }

        scaleoutdone[scale+1] = scaleoutdone[scale]+6*blr_x;
        index += w*h;
        w = (w-1)/2+1;
        h = (h-1)/2+1;
    }
    sycl::float3* hostback = pinned;
    //printf("I am sending : %llu %llu %lld %d", hostback, temp, sizeof(sycl::float3)*scaleoutdone[6], stream);
    //GPU_CHECK(hipMemcpyDtoHAsync(hostback, (hipDeviceptr_t)temp, sizeof(sycl::float3)*scaleoutdone[6], stream));
    stream.memcpy(hostback, temp,  sizeof(sycl::float3)*scaleoutdone[6]).wait();

    //let s reduce!
    for (int scale = 0; scale < 6; scale++){
        bl_x = (scaleoutdone[scale+1] - scaleoutdone[scale])/6;
        for (int i = 0; i < 6*bl_x; i++){
            if (i < bl_x){
                result[6*scale] += hostback[scaleoutdone[scale] + i];
            } else if (i < 2*bl_x) {
                result[6*scale+1] += hostback[scaleoutdone[scale] + i];
            } else if (i < 3*bl_x) {
                result[6*scale+2] += hostback[scaleoutdone[scale] + i];
            } else if (i < 4*bl_x) {
                result[6*scale+3] += hostback[scaleoutdone[scale] + i];
            } else if (i < 5*bl_x) {
                result[6*scale+4] += hostback[scaleoutdone[scale] + i];
            } else {
                result[6*scale+5] += hostback[scaleoutdone[scale] + i];
            }
        }
    }

    for (int i = 0; i < 18; i++){
        result[2*i+1].x() = sycl::sqrt(sycl::sqrt(result[2*i+1].x()));
        result[2*i+1].y() = sycl::sqrt(sycl::sqrt(result[2*i+1].y()));
        result[2*i+1].z() = sycl::sqrt(sycl::sqrt(result[2*i+1].z()));
    } //completing 4th norm

    return result;
}

const float weights[108] = {
    0.0f,
    0.0007376606707406586f,
    0.0f,
    0.0f,
    0.0007793481682867309f,
    0.0f,
    0.0f,
    0.0004371155730107379f,
    0.0f,
    1.1041726426657346f,
    0.00066284834129271f,
    0.00015231632783718752f,
    0.0f,
    0.0016406437456599754f,
    0.0f,
    1.8422455520539298f,
    11.441172603757666f,
    0.0f,
    0.0007989109436015163f,
    0.000176816438078653f,
    0.0f,
    1.8787594979546387f,
    10.94906990605142f,
    0.0f,
    0.0007289346991508072f,
    0.9677937080626833f,
    0.0f,
    0.00014003424285435884f,
    0.9981766977854967f,
    0.00031949755934435053f,
    0.0004550992113792063f,
    0.0f,
    0.0f,
    0.0013648766163243398f,
    0.0f,
    0.0f,
    0.0f,
    0.0f,
    0.0f,
    7.466890328078848f,
    0.0f,
    17.445833984131262f,
    0.0006235601634041466f,
    0.0f,
    0.0f,
    6.683678146179332f,
    0.00037724407979611296f,
    1.027889937768264f,
    225.20515300849274f,
    0.0f,
    0.0f,
    19.213238186143016f,
    0.0011401524586618361f,
    0.001237755635509985f,
    176.39317598450694f,
    0.0f,
    0.0f,
    24.43300999870476f,
    0.28520802612117757f,
    0.0004485436923833408f,
    0.0f,
    0.0f,
    0.0f,
    34.77906344483772f,
    44.835625328877896f,
    0.0f,
    0.0f,
    0.0f,
    0.0f,
    0.0f,
    0.0f,
    0.0f,
    0.0f,
    0.0008680556573291698f,
    0.0f,
    0.0f,
    0.0f,
    0.0f,
    0.0f,
    0.0005313191874358747f,
    0.0f,
    0.00016533814161379112f,
    0.0f,
    0.0f,
    0.0f,
    0.0f,
    0.0f,
    0.0004179171803251336f,
    0.0017290828234722833f,
    0.0f,
    0.0020827005846636437f,
    0.0f,
    0.0f,
    8.826982764996862f,
    23.19243343998926f,
    0.0f,
    95.1080498811086f,
    0.9863978034400682f,
    0.9834382792465353f,
    0.0012286405048278493f,
    171.2667255897307f,
    0.9807858872435379f,
    0.0f,
    0.0f,
    0.0f,
    0.0005130064588990679f,
    0.0f,
    0.00010854057858411537f,
};

double final_score(const std::vector<float> &scores){
    //score has to be of size 108
    float ssim = 0.0f;
    for (int i = 0; i < 108; i++){
        ssim = sycl::fma(weights[i], scores[i], ssim);
    }
    ssim *= 0.9562382616834844;
    ssim = (6.248496625763138e-5 * ssim * ssim) * ssim +
        2.326765642916932 * ssim -
        0.020884521182843837 * ssim * ssim;
    
    if (ssim > 0.0) {
        ssim = sycl::pow((double)ssim, 0.6276336467831387) * -10.0 + 100.0;
    } else {
        ssim = 100.0f;
    }

    return ssim;
}

}