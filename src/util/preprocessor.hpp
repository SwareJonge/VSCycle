#ifndef PREPROCESSHPP
#define PREPROCESSHPP

#include <string>
#include <sstream>
#include <iostream>
#include <stdlib.h>
#include <stdio.h>
#include<math.h>
#include<vector>
#include <chrono>
#include <thread>
#include<exception>
#include<set>
#include <mutex>
#include <condition_variable>
#include <cassert>
#include "sycl/sycl.hpp"

#ifdef _WIN32
    #define aligned_alloc(a, b) malloc(b)
#endif


#define GAUSSIANSIZE 8
#define SIGMA 1.5f
#define PI  3.14159265359
#define TAU 6.28318530718

enum InputMemType {UINT16, HALF, FLOAT};
typedef float f32;

#endif