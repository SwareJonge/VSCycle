namespace ssimu2{

class GaussianHandle {
public:
    void init(sycl::queue& q) {
        float gaussiankernel[4*GAUSSIANSIZE+3];

        for (int i = 0; i < 2*GAUSSIANSIZE+1; i++) {
            gaussiankernel[i] = std::exp(-(GAUSSIANSIZE-i)*(GAUSSIANSIZE-i) /
                                        (2*SIGMA*SIGMA)) /
                                (std::sqrt(TAU*SIGMA*SIGMA));
            gaussiankernel[2*GAUSSIANSIZE+2+i] =
                gaussiankernel[2*GAUSSIANSIZE+1+i] + gaussiankernel[i];
        }
        gaussiankernel_d = sycl::malloc_device<f32>(4*GAUSSIANSIZE+3, q);

        q.memcpy(gaussiankernel_d, gaussiankernel,
                sizeof(f32)*(4*GAUSSIANSIZE+3)).wait();
        
        gaussiankernel_integral_d = gaussiankernel_d + 2*GAUSSIANSIZE+1;
    }

    void destroy(sycl::queue& q) {
        sycl::free(gaussiankernel_d, q);
    }

    f32* gaussiankernel_d;
    f32* gaussiankernel_integral_d;
};

inline void GaussianSmartSharedLoadProduct(sycl::local_accessor<sycl::float3, 1> tampon,
                                           sycl::global_ptr<const sycl::float3> src1,
                                           sycl::global_ptr<const sycl::float3> src2,
                                           int64_t x, int64_t y,
                                           int64_t width, int64_t height,
                                           sycl::nd_item<2> item) {
    const int thx = item.get_local_id(1);
    const int thy = item.get_local_id(0);
    const int tampon_base_x = x - thx - 8;
    const int tampon_base_y = y - thy - 8;

    auto makeZero = [](){ return sycl::float3({0.0f}); };

    auto idx = [&](int yy, int xx) { return (yy)*32 + (xx); };

    // fill tampon
    tampon[idx(thy, thx)] =
        (tampon_base_x + thx >= 0 && tampon_base_x + thx < width &&
         tampon_base_y + thy >= 0 && tampon_base_y + thy < height)
        ? src1[(tampon_base_y+thy)*width + tampon_base_x+thx] *
          src2[(tampon_base_y+thy)*width + tampon_base_x+thx]
        : makeZero();

    tampon[idx(thy+16, thx)] =
        (tampon_base_x + thx >= 0 && tampon_base_x + thx < width &&
         tampon_base_y + thy + 16 >= 0 && tampon_base_y + thy + 16 < height)
        ? src1[(tampon_base_y+thy+16)*width + tampon_base_x+thx] *
          src2[(tampon_base_y+thy+16)*width + tampon_base_x+thx]
        : makeZero();

    tampon[idx(thy, thx+16)] =
        (tampon_base_x + thx + 16 >= 0 && tampon_base_x + thx + 16 < width &&
         tampon_base_y + thy >= 0 && tampon_base_y + thy < height)
        ? src1[(tampon_base_y+thy)*width + tampon_base_x+thx+16] *
          src2[(tampon_base_y+thy)*width + tampon_base_x+thx+16]
        : makeZero();

    tampon[idx(thy+16, thx+16)] =
        (tampon_base_x + thx + 16 >= 0 && tampon_base_x + thx + 16 < width &&
            tampon_base_y + thy + 16 >= 0 && tampon_base_y + thy + 16 < height)
        ? src1[(tampon_base_y+thy+16)*width + tampon_base_x+thx+16] *
            src2[(tampon_base_y+thy+16)*width + tampon_base_x+thx+16]
        : makeZero();

    item.barrier(sycl::access::fence_space::local_space);
}

inline void GaussianSmartSharedLoad(sycl::local_accessor<sycl::float3, 1> tampon,
                                           sycl::global_ptr<const sycl::float3> src,
                                           int64_t x, int64_t y,
                                           int64_t width, int64_t height,
                                           sycl::nd_item<2> item) {
    const int thx = item.get_local_id(1);
    const int thy = item.get_local_id(0);
    const int tampon_base_x = x - thx - 8;
    const int tampon_base_y = y - thy - 8;

    auto makeZero = [](){ return sycl::float3({0.0f, 0.0f, 0.0f}); };

    auto idx = [&](int yy, int xx) { return (yy)*32 + (xx); };

    // fill tampon
    tampon[idx(thy, thx)] = 
        (tampon_base_x + thx >= 0 && tampon_base_x + thx < width && 
            tampon_base_y + thy >= 0 && tampon_base_y + thy < height) 
            ? src[(tampon_base_y+thy)*width + tampon_base_x+thx] : makeZero();
    
    tampon[idx(thy+16, thx)] = 
    (tampon_base_x + thx >= 0 && tampon_base_x + thx < width && 
        tampon_base_y + thy + 16 >= 0 && tampon_base_y + thy + 16 < height) 
        ? src[(tampon_base_y+thy+16)*width + tampon_base_x+thx] : makeZero();
    
    tampon[idx(thy, thx+16)] = 
    (tampon_base_x + thx +16 >= 0 && tampon_base_x + thx +16 < width && 
        tampon_base_y + thy >= 0 && tampon_base_y + thy < height) 
        ? src[(tampon_base_y+thy)*width + tampon_base_x+thx+16] : makeZero();
    
    tampon[idx(thy+16, thx+16)] = 
    (tampon_base_x + thx +16 >= 0 && tampon_base_x + thx +16 < width && 
        tampon_base_y + thy + 16 >= 0 && tampon_base_y + thy + 16 < height) 
        ? src[(tampon_base_y+thy+16)*width + tampon_base_x+thx+16] : makeZero();

    item.barrier(sycl::access::fence_space::local_space);
}

inline void GaussianSmart_Device(sycl::local_accessor<sycl::float3, 1> tampon,
                                 int64_t x, int64_t y,
                                 int64_t width, int64_t height,
                                 sycl::global_ptr<const f32> gaussiankernel,
                                 sycl::global_ptr<const f32> gaussiankernel_integral,
                                 sycl::nd_item<2> item) {
    const int thx = item.get_local_id(1);
    const int thy = item.get_local_id(0);

    auto idx = [&](int yy, int xx) { return yy * 32 + xx; };

    // --- Horizontal Blur ---
    sycl::float3 out = tampon[idx(thy, thx)] * gaussiankernel[0];
    sycl::float3 out2 = tampon[idx(thy + 16, thx)] * gaussiankernel[0];

    int beg = sycl::max<int64_t>(0, x - 8) - (x - 8);
    int end2 = sycl::min<int64_t>(width, x + 9) - (x - 8);
    f32 tot = gaussiankernel_integral[end2] - gaussiankernel_integral[beg];

    for (int i = 1; i < 17; i++) {
        out += tampon[idx(thy, thx + i)] * gaussiankernel[i];
        out2 += tampon[idx(thy + 16, thx + i)] * gaussiankernel[i];
    }

    item.barrier(sycl::access::fence_space::local_space);
    tampon[idx(thy, thx + 8)] = out / tot;
    tampon[idx(thy + 16, thx + 8)] = out2 / tot;
    item.barrier(sycl::access::fence_space::local_space);

    // --- Vertical Blur ---
    out = tampon[idx(thy, thx + 8)] * gaussiankernel[0];
    beg = sycl::max<int64_t>(0, y - 8) - (y - 8);
    end2 = sycl::min<int64_t>(height, y + 9) - (y - 8);
    tot = gaussiankernel_integral[end2] - gaussiankernel_integral[beg];

    for (int i = 1; i < 17; i++) {
        out += tampon[idx(thy + i, thx + 8)] * gaussiankernel[i];
    }

    item.barrier(sycl::access::fence_space::local_space);
    tampon[idx(thy + 8, thx + 8)] = out / tot;
    item.barrier(sycl::access::fence_space::local_space);
}

}