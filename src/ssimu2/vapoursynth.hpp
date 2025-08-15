#pragma once

#include "torgbs.hpp"
#include "main.hpp"
#include "../util/gpuhelper.hpp"

namespace ssimu2{

typedef struct Ssimulacra2Data{
    VSNode *reference;
    VSNode *distorted;
    SSIMU2ComputingImplementation* ssimu2Streams;
    threadSet<int>* streamSet;
    int streamnum = 0;
} Ssimulacra2Data;

static const VSFrame *VS_CC ssimulacra2GetFrame(int n, int activationReason, void *instanceData, void **frameData, VSFrameContext *frameCtx, VSCore *core, const VSAPI *vsapi) {
    Ssimulacra2Data *d = (Ssimulacra2Data *)instanceData;

    if (activationReason == arInitial) {
        vsapi->requestFrameFilter(n, d->reference, frameCtx);
        vsapi->requestFrameFilter(n, d->distorted, frameCtx);
    } else if (activationReason == arAllFramesReady) {
        const VSFrame *src1 = vsapi->getFrameFilter(n, d->reference, frameCtx);
        const VSFrame *src2 = vsapi->getFrameFilter(n, d->distorted, frameCtx);
        
        int64_t height = vsapi->getFrameHeight(src1, 0);
        int64_t width = vsapi->getFrameWidth(src1, 0);
        int64_t stride = vsapi->getStride(src1, 0);

        VSFrame *dst = vsapi->copyFrame(src2, core);

        const uint8_t *srcp1[3] = {
            vsapi->getReadPtr(src1, 0),
            vsapi->getReadPtr(src1, 1),
            vsapi->getReadPtr(src1, 2),
        };

        const uint8_t *srcp2[3] = {
            vsapi->getReadPtr(src2, 0),
            vsapi->getReadPtr(src2, 1),
            vsapi->getReadPtr(src2, 2),
        };

        double val;
        const int stream = d->streamSet->pop();
        SSIMU2ComputingImplementation& ssimu2Stream = d->ssimu2Streams[stream];
        try{
            val = ssimu2Stream.run<FLOAT>(srcp1, srcp2, stride);
        } catch (const VshipError& e){
            vsapi->setFilterError(e.getErrorMessage().c_str(), frameCtx);
            d->streamSet->insert(stream);
            vsapi->freeFrame(src1);
            vsapi->freeFrame(src2);
            return NULL;
        }
        d->streamSet->insert(stream);

        vsapi->mapSetFloat(vsapi->getFramePropertiesRW(dst), "_SSIMULACRA2", val, maReplace);

        // Release the source frame
        vsapi->freeFrame(src1);
        vsapi->freeFrame(src2);

        // A reference is consumed when it is returned, so saving the dst reference somewhere
        // and reusing it is not allowed.
        return dst;
    }

    return NULL;
}

// Free all allocated data on filter destruction
static void VS_CC ssimulacra2Free(void *instanceData, VSCore *core, const VSAPI *vsapi) {
    Ssimulacra2Data *d = (Ssimulacra2Data *)instanceData;
    vsapi->freeNode(d->reference);
    vsapi->freeNode(d->distorted);

    for (int i = 0; i < d->streamnum; i++){
        d->ssimu2Streams[i].destroy();
    }
    free(d->ssimu2Streams);
    delete d->streamSet;

    free(d);
}

// This function is responsible for validating arguments and creating a new filter  
static void VS_CC ssimulacra2Create(const VSMap *in, VSMap *out, void *userData, VSCore *core, const VSAPI *vsapi) {
    Ssimulacra2Data d;
    Ssimulacra2Data *data;

    // Get a clip reference from the input arguments. This must be freed later.
    d.reference = toRGBS(vsapi->mapGetNode(in, "reference", 0, 0), core, vsapi, false);
    d.distorted = toRGBS(vsapi->mapGetNode(in, "distorted", 0, 0), core, vsapi, false);
    const VSVideoInfo *viref = vsapi->getVideoInfo(d.reference);
    const VSVideoInfo *vidis = vsapi->getVideoInfo(d.distorted);

    if (!(vsh::isSameVideoInfo(viref, vidis))){
        vsapi->mapSetError(out, VshipError(DifferingInputType, __FILE__, __LINE__).getErrorMessage().c_str());
        vsapi->freeNode(d.reference);
        vsapi->freeNode(d.distorted);
        return;
    }

    if ((viref->format.bitsPerSample != 32) || (viref->format.colorFamily != cfRGB) || viref->format.sampleType != stFloat){
        vsapi->mapSetError(out, VshipError(NonRGBSInput, __FILE__, __LINE__).getErrorMessage().c_str());
        vsapi->freeNode(d.reference);
        vsapi->freeNode(d.distorted);
        return;
    }

    int error;
    int gpuid = vsapi->mapGetInt(in, "gpu_id", 0, &error);
    if (error != peSuccess){
        gpuid = 0;
    }

    try{
        //if succeed, this function also does hipSetDevice
        helper::gpuFullCheck(gpuid);
    } catch (const VshipError& e){
        vsapi->mapSetError(out, e.getErrorMessage().c_str());
        return;
    }

    //int videowidth = viref->width;
    //int videoheight = viref->height;
    //put optimal thread number
    VSCoreInfo infos;
    vsapi->getCoreInfo(core, &infos);
    //d.oldthreadnum = infos.numThreads;
    //int64_t freemem, totalmem;
    //hipMemGetInfo (&freemem, &totalmem);

    d.streamnum = vsapi->mapGetInt(in, "numStream", 0, &error);
    if (error != peSuccess){
        d.streamnum = infos.numThreads;
    }

    d.streamnum = std::min(d.streamnum, infos.numThreads); // vs threads < numStream would make no sense
    d.streamnum = std::max(d.streamnum, 1); //at least one stream to not just wait indefinitely

    try{
        auto devices = sycl::device::get_devices(sycl::info::device_type::gpu);
        d.ssimu2Streams = (SSIMU2ComputingImplementation*)malloc(sizeof(SSIMU2ComputingImplementation)*d.streamnum);
        for (int i = 0; i < d.streamnum; i++){
            new(&d.ssimu2Streams[i]) SSIMU2ComputingImplementation(viref->width, viref->height, 0);
        }
        
    } catch (const VshipError& e){
        vsapi->mapSetError(out, e.getErrorMessage().c_str());
        return;
    }

    std::set<int> newstreamset;
    for (int i = 0; i < d.streamnum; i++){
        newstreamset.insert(i);
    }
    d.streamSet = new threadSet(newstreamset);

    data = (Ssimulacra2Data *)malloc(sizeof(d));
    *data = d;

    VSFilterDependency deps[] = {{d.reference, rpStrictSpatial}, {d.distorted, rpStrictSpatial}};

    vsapi->createVideoFilter(out, "vscycle", viref, ssimulacra2GetFrame, ssimulacra2Free, fmParallel, deps, 2, data, core);
}

}