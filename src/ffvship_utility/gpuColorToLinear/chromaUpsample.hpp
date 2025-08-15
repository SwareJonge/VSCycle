#pragma once

namespace VshipColorConvert{

class CubicHermitSplineInterpolator{
    float v1; float v2; float v3; float v4;
public:
    CubicHermitSplineInterpolator(const float p0, const float m0, const float p1, const float m1){
        v1 = 2*p0 + m0 - 2*p1 + m1;
        v2 = -3*p0 + 3*p1 - 2*m0 - m1;
        v3 = m0;
        v4 = p0;
    }
    float get(const float t){ //cubic uses t between 0 and 1
        float res = v1;
        res *= t;
        res += v2;
        res *= t;
        res += v3;
        res *= t;
        res += v4;
        return res;
    }
};

CubicHermitSplineInterpolator getHorizontalInterpolator_device(float* src, int64_t x, int64_t y, int64_t width, int64_t height){ //width and height must be the one of source!!!!
    const float el0 = src[sycl::min(y, height-1)*width + sycl::min(x, width-1)];
    const float elm1 = src[sycl::min(y, height-1)*width + sycl::min(sycl::max(x-1, (int64_t)0), width-1)]; //left element
    const float el1 = src[sycl::min(y, height-1)*width + sycl::min(x+1, width-1)];
    const float el2 = src[sycl::min(y, height-1)*width + sycl::min(x+2, width-1)];
    return CubicHermitSplineInterpolator(el0, (el1 - elm1)/2, el1, (el2 - el0)/2);
}

CubicHermitSplineInterpolator getVerticalInterpolator_device(float* src, int64_t x, int64_t y, int64_t width, int64_t height){ //width and height must be the one of source!!!!
    const float el0 = src[sycl::min(y, height-1)*width + sycl::min(x, width-1)];
    const float elm1 = src[sycl::min(sycl::max(y-1, (int64_t)0), height-1)*width + sycl::min(x, width-1)]; //left element
    const float el1 = src[sycl::min(y+1, height-1)*width + sycl::min(x, width-1)];
    const float el2 = src[sycl::min(y+2, height-1)*width + sycl::min(x, width-1)];
    return CubicHermitSplineInterpolator(el0, (el1 - elm1)/2, el1, (el2 - el0)/2);
}

//block x should range from 0 to width INCLUDED
//dst of size (2*width)*height while src is of size width*height
void bicubicHorizontalCenterUpscaleX2_Kernel(float* dst, float* src, int64_t width, int64_t height, sycl::nd_item<2>& it){
    int64_t x = it.get_global_id(0) - 1;
    int64_t y = it.get_global_id(1);
    if (y < height && x < width) {
        CubicHermitSplineInterpolator interpolator = getHorizontalInterpolator_device(src, x, y, width, height);
        if (x != -1)       dst[y*2*width + 2*x + 1] = interpolator.get(0.25f);
        if (x != width-1)  dst[y*2*width + 2*x + 2] = interpolator.get(0.75f);
    }
}

//block x should range from 0 to width-1 INCLUDED
//dst of size (2*width)*height while src is of size width*height
void bicubicHorizontalLeftUpscaleX2_Kernel(float* dst, float* src, int64_t width, int64_t height, sycl::nd_item<2>& it){
    int64_t x = it.get_global_id(0) - 1;
    int64_t y = it.get_global_id(1);
    //this interpolator is valid on interval [0, 1] representing [x, x+1]
    //we are Left so we are interested in values: 0 and 0.5 (0 is directly our value)
    if (y < height && x < width){
        CubicHermitSplineInterpolator interpolator = getHorizontalInterpolator_device(src, x, y, width, height);
        dst[y*2*width + 2*x] = src[y*width + x];
        dst[y*2*width + 2*x+1] = interpolator.get(0.5f);
    }
}

//block x should range from 0 to width INCLUDED
//dst of size (4*width)*height while src is of size width*height
void bicubicHorizontalCenterUpscaleX4_Kernel(float* dst, float* src, int64_t width, int64_t height, sycl::nd_item<2>& it){
    int64_t x = it.get_global_id(0) - 1;
    int64_t y = it.get_global_id(1);
    //this interpolator is valid on interval [0, 1] representing [x, x+1]
    //we are Center so we are interested in values: 0.125, 0.375, 0.625 and 0.875
    if (y < height && x < width){
        CubicHermitSplineInterpolator interpolator = getHorizontalInterpolator_device(src, x, y, width, height);
        if (x != -1) dst[y*4*width + 4*x+2] = interpolator.get(0.125f);
        if (x != -1) dst[y*4*width + 4*x+3] = interpolator.get(0.375f);
        if (x != width-1) dst[y*4*width + 4*x+4] = interpolator.get(0.625f);
        if (x != width-1) dst[y*4*width + 4*x+5] = interpolator.get(0.875f);
    }
}

//block x should range from 0 to width-1 INCLUDED
//dst of size (4*width)*height while src is of size width*height
void bicubicHorizontalLeftUpscaleX4_Kernel(float* dst, float* src, int64_t width, int64_t height, sycl::nd_item<2>& it){
    int64_t x = it.get_global_id(0);
    int64_t y = it.get_global_id(1);
    //this interpolator is valid on interval [0, 1] representing [x, x+1]
    //we are Left so we are interested in values: 0, 0.25, 0.5 and 0.75 (0 is directly our value)
    if (y < height && x < width){
        CubicHermitSplineInterpolator interpolator = getHorizontalInterpolator_device(src, x, y, width, height);
        dst[y*4*width + 4*x] = src[y*width + x];
        dst[y*4*width + 4*x+1] = interpolator.get(0.25f);
        dst[y*4*width + 4*x+2] = interpolator.get(0.5f);
        dst[y*4*width + 4*x+3] = interpolator.get(0.75f);
    }
}

//block y should range from 0 to width INCLUDED
//dst of size width*(2*height) while src is of size width*height
void bicubicVerticalCenterUpscaleX2_Kernel(float* dst, float* src, int64_t width, int64_t height, sycl::nd_item<2>& it){
    int64_t x = it.get_global_id(0);
    int64_t y = it.get_global_id(1) - 1;
    //this interpolator is valid on interval [0, 1] representing [y, y+1]
    //we are Center so we are interested in values: 0.25 and 0.75
    if (y < height && x < width){
        CubicHermitSplineInterpolator interpolator = getVerticalInterpolator_device(src, x, y, width, height);
        if (y != -1) dst[(2*y +1)*width + x] = interpolator.get(0.25f);
        if (y != height-1) dst[(2*y+2)*width + x] = interpolator.get(0.75f);
    }
}

//block y should range from 0 to width-1 INCLUDED
//dst of size width*(2*height) while src is of size width*height
void bicubicVerticalTopUpscaleX2_Kernel(float* dst, float* src, int64_t width, int64_t height, sycl::nd_item<2>& it){
    int64_t x = it.get_global_id(0);
    int64_t y = it.get_global_id(1);
    //this interpolator is valid on interval [0, 1] representing [y, y+1]
    //we are Top so we are interested in values: 0 and 0.5
    if (y < height && x < width){
        CubicHermitSplineInterpolator interpolator = getVerticalInterpolator_device(src, x, y, width, height);
        dst[(2*y)*width + x] = src[y*width + x];
        dst[(2*y+1)*width + x] = interpolator.get(0.5f);
    }
}

//source is of size width * height possibly chroma downsampled
int inline upsample(float* dst[3], float* src[3], int64_t width, int64_t height, AVChromaLocation location, int subw, int subh, sycl::queue& q){
    if (subw == 0 && subh == 0) return 0;
    width >>= subw; //get chroma plane size
    height >>= subh; 
    sycl::range<2> local(16, 16); // thx, thy
    const int blx1 = ((width + local[0]-1)/local[0]) * local[0];
    const int blx2 = ((width+1 + local[0]-1)/local[0]) * local[0];
    const int bly1 = ((height + local[1]-1)/local[1])  * local[1];
    const int bly2 = ((height+1 + local[1]-1)/local[1])  * local[1];

    sycl::range<2> globalLeft(blx1, bly1);
    sycl::range<2> globalHCenter(blx2, bly1);
    sycl::range<2> globalVCenter(blx1, bly2);

    switch (location){
        case (AVCHROMA_LOC_LEFT):
        case (AVCHROMA_LOC_TOPLEFT):
            if (subw == 0){
            } else if (subw == 1){                
                q.submit([&](sycl::handler& cgh) {
                    cgh.parallel_for<class BicubicHorizontalLeftUpscaleX2Single>(
                        sycl::nd_range<2>(globalLeft, local),
                        [=](sycl::nd_item<2> it) {
                            bicubicHorizontalLeftUpscaleX2_Kernel(dst[1], src[1], width, height, it);
                            bicubicHorizontalLeftUpscaleX2_Kernel(dst[2], src[2], width, height, it);
                        }
                    );
                });
                width *= 2;                
            } else if (subw == 2) {
                q.submit([&](sycl::handler& cgh) {
                    cgh.parallel_for<class BicubicHorizontalLeftUpscaleX4Single>(
                        sycl::nd_range<2>(globalLeft, local),
                        [=](sycl::nd_item<2> it) {
                            bicubicHorizontalLeftUpscaleX4_Kernel(dst[1], src[1], width, height, it);
                            bicubicHorizontalLeftUpscaleX4_Kernel(dst[2], src[2], width, height, it);
                        }
                    );
                });
                width *= 4;
            } else {
                return 1; //not implemented
            }
            break;
        case (AVCHROMA_LOC_CENTER):
        case (AVCHROMA_LOC_TOP):
            if (subw == 0){
            } else if (subw == 1){
                q.submit([&](sycl::handler& cgh) {
                    cgh.parallel_for<class BicubicHorizontalCenterUpscaleX2Single>(
                        sycl::nd_range<2>(globalHCenter, local),
                        [=](sycl::nd_item<2> it) {
                            bicubicHorizontalCenterUpscaleX2_Kernel(dst[1], src[1], width, height, it);
                            bicubicHorizontalCenterUpscaleX2_Kernel(dst[2], src[2], width, height, it);
                        }
                    );
                });
                width *= 2;
            } else if (subw == 2){
                q.submit([&](sycl::handler& cgh) {
                    cgh.parallel_for<class BicubicHorizontalCenterUpscaleX4Single>(
                        sycl::nd_range<2>(globalHCenter, local),
                        [=](sycl::nd_item<2> it) {
                            bicubicHorizontalCenterUpscaleX4_Kernel(dst[1], src[1], width, height, it);
                            bicubicHorizontalCenterUpscaleX4_Kernel(dst[2], src[2], width, height, it);
                        }
                    );
                });
                width *= 4;
            } else {
                return 1; //not implemented
            }
            break;
        default:
            if (subw != 0) return 1; //not implemented
    }

    switch (location){
        case (AVCHROMA_LOC_TOP):
        case (AVCHROMA_LOC_TOPLEFT):
            if (subh == 0){
            } else if (subh == 1){
                q.submit([&](sycl::handler& cgh) {
                    cgh.parallel_for<class BicubicVertialLeftUpscaleX2Single>(
                        sycl::nd_range<2>(globalLeft, local),
                        [=](sycl::nd_item<2> it) {
                            bicubicVerticalTopUpscaleX2_Kernel(dst[1], src[1], width, height, it);
                            bicubicVerticalTopUpscaleX2_Kernel(dst[2], src[2], width, height, it);
                        }
                    );
                });
                height *= 2;
            } else {
                return 1; //not implemented
            }
            break;
        case (AVCHROMA_LOC_CENTER):
        case (AVCHROMA_LOC_LEFT):
            if (subh == 0){
            } else if (subh == 1){
                q.submit([&](sycl::handler& cgh) {
                    cgh.parallel_for<class BicubicVerticalCenterUpscaleX4Single>(
                        sycl::nd_range<2>(globalVCenter, local),
                        [=](sycl::nd_item<2> it) {
                            bicubicVerticalCenterUpscaleX2_Kernel(dst[1], src[1], width, height, it);
                            bicubicVerticalCenterUpscaleX2_Kernel(dst[2], src[2], width, height, it);
                        }
                    );
                });
                height *= 2;
            } else {
                return 1; //not implemented
            }
            break;
        default:
            if (subh != 0) return 1; //not implemented
    }

    return 0;
}

}