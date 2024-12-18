#include "inference.h"   // include/inference.h
#include <iostream>

int main() {
    try {
        // const std::string modelPath = "../engine_files/lseg_model_241115.trt";
        const std::string modelPath = "../engine_files/visual_component.trt";
        const int numRuns = 1000;

        InferenceEngine engine(modelPath);
        engine.infer(numRuns);
    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << std::endl;
        return -1;
    }

    return 0;
}
