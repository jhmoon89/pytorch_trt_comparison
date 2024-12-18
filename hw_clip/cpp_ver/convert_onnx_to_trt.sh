#!/bin/bash

# ONNX -> TRT Conversion Script
# Usage: ./convert_onnx_to_trt.sh /path/to/model.onnx /path/to/output.trt

# Check for correct number of arguments
if [ $# -ne 2 ]; then
    echo "Usage: $0 <ONNX model file path> <TRT output file path>"
    exit 1
fi

ONNX_FILE=$1  # Input ONNX file
TRT_FILE=$2   # Output TRT file

# Check if the input file exists
if [ ! -f "$ONNX_FILE" ]; then
    echo "Error: Input file $ONNX_FILE does not exist."
    exit 1
fi

# Check if the output directory exists (create it if not)
OUTPUT_DIR=$(dirname "$TRT_FILE")
if [ ! -d "$OUTPUT_DIR" ]; then
    echo "Output directory $OUTPUT_DIR does not exist. Creating it..."
    mkdir -p "$OUTPUT_DIR"
fi

# Start the conversion
echo "Converting: $ONNX_FILE -> $TRT_FILE"

/usr/src/tensorrt/bin/trtexec --onnx="$ONNX_FILE" --saveEngine="$TRT_FILE" --explicitBatch

# Check if the conversion was successful
if [ $? -eq 0 ]; then
    echo "Conversion successful: $TRT_FILE"
else
    echo "Conversion failed: $ONNX_FILE"
    exit 1
fi
