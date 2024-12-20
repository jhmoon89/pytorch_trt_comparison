#include "inference.h"   // include/inference.h
#include <iostream>
#include <string>

int main(int argc, char* argv[]) {
    try {
        if (argc < 2) {
            std::cerr << "Usage: " << argv[0] << " <model_path>" << std::endl;
            return -1;
        }

        const std::string modelPath = argv[1]; // Get model path from command-line argument
        const int numRuns = 1000;

        // const std::string modelPath = "../engine_files/lseg_model_241115.trt";
        // const std::string modelPath = "../engine_files/visual_component.trt";
        // const int numRuns = 1000;

        InferenceEngine engine(modelPath);
        engine.infer(numRuns);
    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << std::endl;
        return -1;
    }

    return 0;
}
