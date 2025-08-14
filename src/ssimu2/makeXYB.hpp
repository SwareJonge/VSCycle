#ifndef MAKEXYBHPP
#define MAKEXYBHPP

inline void opsin_absorbance(TVec3<f32>& a){
    const f32 x = a.x();
    const f32 y = a.y();
    const f32 z = a.z();
    const f32 opsin_bias = 0.0037930734f;

    a.x() = sycl::fma(0.30f, x,
    sycl::fma(0.622f, y,
    sycl::fma(0.078f, z,
    opsin_bias)));

    a.y() = sycl::fma(0.23f, x,
    sycl::fma(0.692f, y,
    sycl::fma(0.078f, z,
    opsin_bias)));

    a.z() = sycl::fma(0.24342269f, x,
    sycl::fma(0.20476745f, y,
    sycl::fma(0.55180986f, z,
    opsin_bias)));
}

inline void mixed_to_xyb(TVec3<f32>& a){
    a.y() += (a.x() = 0.5f * (a.x() - a.y()));
}

inline void linear_rgb_to_xyb(TVec3<f32>& a){
    const float abs_bias = -0.1559542025327239f;
    opsin_absorbance(a);
    //printf("from %f to %f\n", a.x, cbrtf(a.x*((int)(a.x() >= 0))));
    a.x() = sycl::cbrt(a.x() * ((int)(a.x() >= 0))) + abs_bias;
    a.y() = sycl::cbrt(a.y() * ((int)(a.y() >= 0))) + abs_bias;
    a.z() = sycl::cbrt(a.z() * ((int)(a.z() >= 0))) + abs_bias;
    //printf("got %f, %f, %f\n", a.x, a.y, a.z);
    mixed_to_xyb(a);
}

inline void make_positive_xyb(TVec3<f32>& a) {
    a.z() = (a.z() - a.y()) + 0.55f;
    a.y() += 0.01f;
    a.x() = a.x() * 14.0f + 0.42f;
}

inline void rgb_to_positive_xyb_d(TVec3<f32>& a){
    linear_rgb_to_xyb(a);
    make_positive_xyb(a);
}

inline void rgb_to_linrgbfunc(f32& a) {
    if (a < 0.f){
        if (a < -0.04045f){
            a = -sycl::pow(((-a+0.055f)*(1.0f/1.055f)), 2.4f);
        } else {
            a *= 1.0f/12.92f;
        }
    } else if (a > 0.04045f){
        a = sycl::pow(((a+0.055f)*(1.0f/1.055f)), 2.4f);
    } else {
        a *= 1.0f/12.92f;
    }
}

inline void rgb_to_linrgb(TVec3<f32>& a){
    rgb_to_linrgbfunc(a.x());
    rgb_to_linrgbfunc(a.y());
    rgb_to_linrgbfunc(a.z());
}

void rgb_to_positive_xyb(TVec3<f32>* array, int64_t width, sycl::queue& q) {
    int64_t th_x = std::min<int64_t>(256, width);
    int64_t bl_x = (width - 1) / th_x + 1;

    sycl::range<1> local(th_x);          // threads per work-group
    sycl::range<1> global(bl_x * th_x);  // total threads

    q.submit([&](sycl::handler& h) {
        h.parallel_for(
            sycl::nd_range<1>(global, local),
            [=](sycl::nd_item<1> item) {
                int64_t x = item.get_global_id(0);
                if (x >= width) return;

                rgb_to_positive_xyb_d(array[x]);
            });
    });
}

inline void rgb_to_linear(TVec3<f32>* array, int64_t width, sycl::queue &stream){
    int64_t th_x = std::min<int64_t>(256, width);
    int64_t bl_x = (width - 1) / th_x + 1;

    sycl::range<1> local(th_x);          // threads per work-group
    sycl::range<1> global(bl_x * th_x);  // total threads

    stream.submit([&](sycl::handler& h) {
        h.parallel_for(
            sycl::nd_range<1>(global, local),
            [=](sycl::nd_item<1> item) {
                int64_t x = item.get_global_id(0);
                if (x >= width) return;

                rgb_to_linrgb(array[x]);
            });
    });
}
#endif