#ifndef FLOAT3OPHPP
#define FLOAT3OPHPP

#include "sycl/sycl.hpp"

template <typename T>
class TVec3 : public sycl::float3 {
public:
    TVec3() { zero(); }
    TVec3(T X) { 
        x() = y() = z() = X;  
    }

    TVec3(T X, T Y, T Z) { 
        x() = X;
        y() = Y;
        z() = Z;
    }

    TVec3(const sycl::float3 &lhs) { 
        x() = lhs.x();
        y() = lhs.y();
        z() = lhs.z();  
    }

    void zero() {
        x() = y() = z() = (T)0;
    }

    void scale(T b) {
        x() *= b;
        y() *= b;
        z() *= b;
    }

    void scale(TVec3<T> b) {
        x() *= b.x();
        y() *= b.y();
        z() *= b.z();
    }

    inline void fma(TVec3<T> a, TVec3<T> b, T c){
        x() = sycl::fma(a.x(), b.x(), c);
        y() = sycl::fma(a.y(), b.y(), c);
        z() = sycl::fma(a.z(), b.z(), c);
    }

    inline void fma(TVec3<T> a, T b, T c){
        x() = sycl::fma(a.x(), b, c);
        y() = sycl::fma(a.y(), b, c);
        z() = sycl::fma(a.z(), b, c);
    }

    TVec3<T> pow2() const {
        return { x()*x(), y()*y(), z()*z() };
    }

    TVec3<T> toThe4th() const {
        const TVec3<T> &tmp = pow2();
        return tmp * tmp;
    }

    //static const TVec3<T> ZERO = TVec3<T>(0.0f);
};

typedef TVec3<f32> TVec3f;

inline float tothe4th(float x){
    float y = x*x;
    return y*y;
}

inline sycl::float3 tothe4th(sycl::float3 in){
    sycl::float3 y = in*in;
    return y*y;
}

inline void zeroVec(sycl::float3 &in) {
    in.x() = in.y() = in.z() = 0.0f;
}

inline sycl::float3 fma(sycl::float3 a, sycl::float3 b, float c){
    return { 
        sycl::fma(a.x(), b.x(), c),
        sycl::fma(a.y(), b.y(), c),
        sycl::fma(a.z(), b.z(), c)
    };
}

inline sycl::float3 fma(sycl::float3 a, float b, float c){
    return { 
        sycl::fma(a.x(), b, c),
        sycl::fma(a.y(), b, c),
        sycl::fma(a.z(), b, c)
    };
}

static inline sycl::float3 fabs(sycl::float3 a) {
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