#ifndef LOGGER_H
#define LOGGER_H

#include <NvInfer.h>
#include <iostream>

// TensorRT Custom Logger
class Logger : public nvinfer1::ILogger {
    void log(Severity severity, const char* msg) noexcept override {
        if (severity == Severity::kERROR || severity == Severity::kINTERNAL_ERROR) {
            std::cerr << "TensorRT error: " << msg << std::endl;
        }
    }
};

// Global Logger Instance
extern Logger gLogger;

#endif // LOGGER_H
