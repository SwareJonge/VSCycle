#ifndef FLOAT3OPHPP
#define FLOAT3OPHPP

#include "sycl/sycl.hpp"

inline float tothe4th(float x){
    x = x*x;
    return x*x;
}

inline sycl::float3 tothe4th(sycl::float3 in){
    sycl::float3 y = in*in;
    return y*y;
}

inline void zeroVec(sycl::float3 &in) {
    in.x() = in.y() = in.z() = 0.0f;
}

inline sycl::float3 fma(const sycl::float3& a, const sycl::float3& b, float c){
    return { 
        sycl::fma(a.x(), b.x(), c),
        sycl::fma(a.y(), b.y(), c),
        sycl::fma(a.z(), b.z(), c)
    };
}

inline sycl::float3 fma(const sycl::float3& a, float b, float c){
    return { 
        sycl::fma(a.x(), b, c),
        sycl::fma(a.y(), b, c),
        sycl::fma(a.z(), b, c)
    };
}

static inline sycl::float3 fabs(const sycl::float3& a) {
    return { 
        sycl::fabs(a.x()), 
        sycl::fabs(a.y()), 
        sycl::fabs(a.z()) 
    };
}

void multarray(sycl::queue& q, float* src1, float* src2, float* dst, int64_t width) {
    int64_t th_x = std::min((int64_t)256, width);
    int64_t bl_x = (width - 1) / th_x + 1;
          
    q.submit([&](sycl::handler& h) {
        h.parallel_for(
            sycl::nd_range<1>(sycl::range<1>(bl_x * th_x),
                            sycl::range<1>(th_x)),
            [=](sycl::nd_item<1> item) {
                auto x = item.get_global_id(0);
                if (x < width) {
                    dst[x] = src1[x] * src2[x];
                }
            }
        );
    });

}

void subarray(sycl::queue& q, float* src1, float* src2, float* dst, int64_t width) {
    int64_t th_x = std::min((int64_t)256, width);
    int64_t bl_x = (width - 1) / th_x + 1;
          
    q.submit([&](sycl::handler& h) {
        h.parallel_for(
            sycl::nd_range<1>(sycl::range<1>(bl_x * th_x),
                            sycl::range<1>(th_x)),
            [=](sycl::nd_item<1> item) {
                auto x = item.get_global_id(0);
                if (x < width) {
                    dst[x] = src1[x] - src2[x];
                }
            }
        );
    });
}

template <InputMemType T>
inline float convertPointer(const uint8_t* src, int i, int j, int64_t stride);

template <>
inline float convertPointer<FLOAT>(const uint8_t* src, int i, int j, int64_t stride){
    return ((float*)(src + i*stride))[j];
}

template <>
inline float convertPointer<HALF>(const uint8_t* src, int i, int j, int64_t stride){
    return ((sycl::half*)(src + i*stride))[j];
}

template <>
inline float convertPointer<UINT16>(const uint8_t* src, int i, int j, int64_t stride){
    return ((float)((uint16_t*)(src + i*stride))[j])/((1 << 16)-1);
}

#endif