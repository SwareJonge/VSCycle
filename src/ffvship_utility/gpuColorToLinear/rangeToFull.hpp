#pragma once

namespace VshipColorConvert{

template <FFMS_ColorRanges RANGE_TYPE>
void inline RangeLinearize(float& a);

template <FFMS_ColorRanges RANGE_TYPE>
void inline RangeLinearize(sycl::float3& a){
    RangeLinearize<RANGE_TYPE>(a.x());
    RangeLinearize<RANGE_TYPE>(a.y());
    RangeLinearize<RANGE_TYPE>(a.z());
}

template<>
void inline RangeLinearize<FFMS_CR_JPEG>(float& a){

}

template<>
void inline RangeLinearize<FFMS_CR_MPEG>(float& a){
    a = (255.f*a - 16.f)/235.f;
}

}