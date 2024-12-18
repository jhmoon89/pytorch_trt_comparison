#ifndef INFERENCE_H
#define INFERENCE_H

#include "logger.h"
#include <NvInfer.h>
#include <cuda_runtime.h>
#include <vector>

class InferenceEngine {
public:
    InferenceEngine(const std::string& modelPath);
    ~InferenceEngine();

    void loadModel(const std::string& modelPath);
    void infer(int numRuns);

private:
    nvinfer1::IRuntime* runtime;
    nvinfer1::ICudaEngine* engine;
    nvinfer1::IExecutionContext* context;

    void* d_input;
    void* d_output;

    const int inputSize = 1 * 3 * 384 * 384;
    const int outputSize = 1 * 512 * 192 * 192;
};

#endif // INFERENCE_H
