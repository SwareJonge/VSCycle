#pragma once

namespace VshipColorConvert{

template<AVColorPrimaries T>
void inline PrimariesToBT709(TVec3<f32> a);

template<>
void inline PrimariesToBT709<AVCOL_PRI_BT709>(TVec3<f32> a){

}

//https://www.itu.int/dms_pub/itu-r/opb/rep/R-REP-BT.2407-2017-PDF-E.pdf
template<>
void inline PrimariesToBT709<AVCOL_PRI_BT2020>(TVec3<f32> a){
    const f32 x = a.x();
    const f32 y = a.y();
    const f32 z = a.z();
    a.x() = sycl::fma(1.6605f, x,  sycl::fma(-0.5876f, y, -z*0.0728f));
    a.y() = sycl::fma(-0.1246f, x, sycl::fma(1.1329f,  y, -z*0.0083f));
    a.z() = sycl::fma(-0.0182f, x, sycl::fma(-0.1006f, y,  z*1.1187f));
}

}