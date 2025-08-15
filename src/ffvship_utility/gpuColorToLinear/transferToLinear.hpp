#pragma once

/*
List of currently implemented transfer functions

Linear
sRGB
BT709
GAMMA22
GAMMA28
ST428
PQ
(HLG)
*/

namespace VshipColorConvert{

template <AVColorTransferCharacteristic TRANSFER_TYPE>
void inline transferLinearize(float& a);

//apply linear on all 3 components
template <AVColorTransferCharacteristic TRANSFER_TYPE>
void inline transferLinearize(sycl::float3 a){
    transferLinearize<TRANSFER_TYPE>(a.x());
    transferLinearize<TRANSFER_TYPE>(a.y());
    transferLinearize<TRANSFER_TYPE>(a.z());
}


//define transferLinearize

template <>
void inline transferLinearize<AVCOL_TRC_LINEAR>(float& a){
}

//source Wikipedia
template <>
void inline transferLinearize<AVCOL_TRC_IEC61966_2_1>(float& a){
    if (a < 0){
        if (a < -0.04045f){
            a = -sycl::pow(((-a+0.055f)*(1.0f/1.055f)), 2.4f);
        } else {
            a *= 1.0f/12.92f;
        }
    } else {
        if (a > 0.04045f){
            a = sycl::pow(((a+0.055f)*(1.0f/1.055f)), 2.4f);
        } else {
            a *= 1.0f/12.92f;
        }
    }
}

//source https://www.image-engineering.de/library/technotes/714-color-spaces-rec-709-vs-srgb
//I inversed the function myself
template <>
void inline transferLinearize<AVCOL_TRC_BT709>(float& a){
    if (a < 0){
        if (a < -0.081f){
            a = -sycl::pow(((-a+0.099f)/1.099f), 2.2f);
        } else {
            a *= 1.0f/4.5f;
        }
    } else {
        if (a > 0.081f){
            a = sycl::pow(((a+0.099f)/1.099f), 2.2f);
        } else {
            a *= 1.0f/4.5f;
        }
    }
}

inline void gamma_to_linrgbfunc(float& a, float gamma){
    if (a < 0.0f){
        a = -sycl::pow(-a, gamma);
    } else {
        a = sycl::pow(a, gamma);
    }
}

template <>
void inline transferLinearize<AVCOL_TRC_GAMMA22>(float& a){
    gamma_to_linrgbfunc(a, 2.2f);
}

template <>
void inline transferLinearize<AVCOL_TRC_GAMMA28>(float& a){
    gamma_to_linrgbfunc(a, 2.8f);
}

//source https://github.com/haasn/libplacebo/blob/master/src/shaders/colorspace.c (14/05/2025 line 670)
template <>
void inline transferLinearize<AVCOL_TRC_SMPTE428>(float& a){
    gamma_to_linrgbfunc(a, 2.6f);
    a *= 52.37f/48.f;
}

//source https://fr.wikipedia.org/wiki/Perceptual_Quantizer
//Note: this is PQ
template<>
void inline transferLinearize<AVCOL_TRC_SMPTE2084>(float& a){
    const float c1 = 107.f/128.f;
    const float c2 = 2413.f/128.f;
    const float c3 = 2392.f/128.f;
    a = sycl::pow(a, 32.f/2523.f);
    a = sycl::fmax(a - c1, 0.f)/(c2 - c3*a);
    a = sycl::pow(a, 8192.f/1305.f);
    a *= 10000;
}

/*
//https://en.wikipedia.org/wiki/Hybrid_log%E2%80%93gamma
//Note: this is HLG
template<>
void inline transferLinearize<AVCOL_TRC_ARIB_STD_B67>(float& a){

}
*/

}