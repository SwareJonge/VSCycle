#ifndef GPUHELPERHPP
#define GPUHELPERHPP

#include "preprocessor.hpp"
#include "VshipExceptions.hpp"

//here is the format of the answer:

//case where gpu_id is not specified:

//GPU 0: {GPU Name}
//...

//case where gpu_id is specified:

//Name: {GPU Name string}
//MultiProcessorCount: {multiprocessor count integer}
//ClockRate: {clockRate float} Ghz
//MaxSharedMemoryPerBlock: {Max Shared Memory Per Block integer} bytes
//WarpSize: {Warp Size integer}
//VRAMCapacity: {GPU VRAM Capacity float} GB
//MemoryBusWidth: {memory bus width integer} bits
//MemoryClockRate: {memory clock rate float} Ghz
//Integrated: {0|1}
//PassKernelCheck : {0|1}

namespace helper{

    int checkGpuCount(){
        auto devices = sycl::device::get_devices(sycl::info::device_type::gpu);
        int count = static_cast<int>(devices.size());
        if (count == 0) {
            VSHIP_THROW(NoDeviceDetected);
        }
        return count;
    }

    void kernelTest(sycl::queue& q, int* inputtest) {
        q.submit([&](sycl::handler &cgh) {
            cgh.parallel_for(sycl::range<1>{1}, [=](sycl::id<1> idx) {
                inputtest[0] = 4320984;
            });
        });
    }

    bool gpuKernelCheck(sycl::queue& q) {
        int inputtest = 0;

        // Allocate USM device memory
        int* inputtest_d = sycl::malloc_device<int>(1, q);
        if (!inputtest_d) {
            std::cerr << "Device memory allocation failed\n";
            return false;
        }

        // Initialize to 0
        q.memset(inputtest_d, 0, sizeof(int)).wait();

        // Submit kernel
        q.submit([&](sycl::handler &cgh) {
            cgh.parallel_for(sycl::range<1>{1}, [=](sycl::id<1> idx) {
                inputtest_d[0] = 4320984;  // write the magic number
            });
        }).wait();  // Must wait so kernel finishes before reading

        // Copy back to host
        q.memcpy(&inputtest, inputtest_d, sizeof(int)).wait();

        sycl::free(inputtest_d, q);

        return inputtest == 4320984;
    }

    void gpuFullCheck(int gpuid = 0){
        auto devices = sycl::device::get_devices(sycl::info::device_type::gpu);
        int count = checkGpuCount();

        if (count <= gpuid || gpuid < 0){
            VSHIP_THROW(BadDeviceArgument);
        }
        sycl::queue q(devices[gpuid]);
        if (!gpuKernelCheck(q)){
            VSHIP_THROW(BadDeviceCode);
        }
    }

    std::string listGPU() {
        std::stringstream ss;
        auto devices = sycl::device::get_devices(sycl::info::device_type::gpu);

        for (size_t i = 0; i < devices.size(); i++) {
            ss << "GPU " << i << ": " << devices[i].get_info<sycl::info::device::name>() << std::endl;
        }

        return ss.str();
    }
}

#endif