#pragma once

#include <onnxruntime_cxx_api.h>
#include <opencv2/opencv.hpp>

#include <algorithm>
#include <fstream>
#include <iostream>
#include <numeric>
#include <string>
#include <vector>
#include <memory>
#include <chrono>
#include <random>
#include <unordered_map>
#include <thread>
#include <filesystem>
#include <memory>
#include <string>
#include <vector>

#include "detection.h"
#include "geometry.h"

namespace nx_meta_plugin {
    class YOLO11Classifier {
    public:
        explicit YOLO11Classifier();

        void ensureInitialized(std::filesystem::path modelPath);

        bool isTerminated() const;

        void terminate();

        std::string run(const cv::Mat &frame);

    private:
        void loadModel(std::filesystem::path modelPath);

        std::string runImpl(const cv::Mat &frame);

        cv::Mat preprocess(const cv::Mat &image, float *&blob, std::vector<int64_t> &inputTensorShape);

        std::string
        postprocess(const cv::Size &originalImageSize, const cv::Size &resizedImageShape,
                    const std::vector<Ort::Value> &outputTensors, float confThreshold = 0.25f,
                    float iouThreshold = 0.45f);

    private:
        bool m_netLoaded = false;
        bool m_terminated = false;
        bool useGPU = false;

        Ort::Env env{nullptr};                         // ONNX Runtime environment
        Ort::SessionOptions sessionOptions{nullptr};   // Session options for ONNX Runtime
        Ort::Session session{nullptr};                 // ONNX Runtime session for running inference
        bool isDynamicInputShape{};                    // Flag indicating if input shape is dynamic
        cv::Size inputImageShape;                      // Expected input image shape for the model

        // Vectors to hold allocated input and output node names
        std::vector<Ort::AllocatedStringPtr> inputNodeNameAllocatedStrings;
        std::vector<const char *> inputNames;
        std::vector<Ort::AllocatedStringPtr> outputNodeNameAllocatedStrings;
        std::vector<const char *> outputNames;

        size_t numInputNodes, numOutputNodes;          // Number of input and output nodes in the model
    };
}
