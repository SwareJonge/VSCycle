namespace ssimu2{

void downsample(sycl::float3* src, sycl::float3* dst, int64_t width, int64_t height, sycl::queue& q) {
        int64_t newh = (height - 1) / 2 + 1;
        int64_t neww = (width - 1) / 2 + 1;

        int64_t th_x = sycl::min((int64_t)16, neww);
        int64_t th_y = sycl::min((int64_t)16, newh);
        int64_t bl_x = (neww - 1) / th_x + 1;
        int64_t bl_y = (newh - 1) / th_y + 1;

        sycl::range<2> local(th_y, th_x);                     // threads per work-group
        sycl::range<2> global(bl_y * th_y, bl_x * th_x);      // total threads

        q.submit([&](sycl::handler& h) {
            h.parallel_for(
                sycl::nd_range<2>(global, local),
                [=](sycl::nd_item<2> item) {
                    int64_t y = item.get_global_id(0);
                    int64_t x = item.get_global_id(1);

                    if (x >= neww || y >= newh) return;

                    const int64_t idx = y * neww + x;
                    dst[idx] =
                        src[sycl::min(2 * y, height - 1) * width + sycl::min(2 * x, width - 1)];
                    dst[idx] +=
                        src[sycl::min(2 * y + 1, height - 1) * width + sycl::min(2 * x, width - 1)];
                    dst[idx] +=
                        src[sycl::min(2 * y, height - 1) * width + sycl::min(2 * x + 1, width - 1)];
                    dst[idx] +=
                        src[sycl::min(2 * y + 1, height - 1) * width + sycl::min(2 * x + 1, width - 1)];

                    dst[idx] *= 0.25f;
                });
        });
    }

}