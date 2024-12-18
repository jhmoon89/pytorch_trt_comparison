#include "logger.h"      // include/logger.h
#include "inference.h"   // include/inference.h

#include <fstream>
#include <iostream>
#include <chrono>
#include <algorithm>
#include <numeric>

Logger gLogger; // Global Logger Instance

InferenceEngine::InferenceEngine(const std::string& modelPath) 
    : runtime(nullptr), engine(nullptr), context(nullptr), d_input(nullptr), d_output(nullptr) {
    loadModel(modelPath);
}

InferenceEngine::~InferenceEngine() {
    if (d_input) cudaFree(d_input);
    if (d_output) cudaFree(d_output);
    if (context) context->destroy();
    if (engine) engine->destroy();
    if (runtime) runtime->destroy();
}

void InferenceEngine::loadModel(const std::string& modelPath) {
    std::ifstream file(modelPath, std::ios::binary | std::ios::ate);
    std::streamsize size = file.tellg();
    file.seekg(0, std::ios::beg);

    std::vector<char> buffer(size);
    if (!file.read(buffer.data(), size)) {
        throw std::runtime_error("Failed to read model file");
    }

    runtime = nvinfer1::createInferRuntime(gLogger);
    engine = runtime->deserializeCudaEngine(buffer.data(), size, nullptr);
    context = engine->createExecutionContext();

    cudaMalloc(&d_input, inputSize * sizeof(float));
    cudaMalloc(&d_output, outputSize * sizeof(float));
}

void InferenceEngine::infer(int numRuns) {
    float* input = new float[inputSize];
    std::fill(input, input + inputSize, 1.0f);

    cudaMemcpy(d_input, input, inputSize * sizeof(float), cudaMemcpyHostToDevice);

    // Warm-up run
    context->enqueueV2(&d_input, 0, nullptr);

    double totalDuration = 0.0;
    for (int i = 0; i < numRuns; i++) {
        auto start = std::chrono::high_resolution_clock::now();
        context->enqueueV2(&d_input, 0, nullptr);
        auto end = std::chrono::high_resolution_clock::now();
        totalDuration += std::chrono::duration_cast<std::chrono::microseconds>(end - start).count();
    }

    std::cout << "Average time per inference: " << (totalDuration / numRuns) / 1000.0 << " milliseconds\n";

    delete[] input;
}
