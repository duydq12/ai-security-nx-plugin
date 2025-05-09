#pragma once

#include "yolo11_classifier.h"
#include "yolo11_detector.h"
#include "detection.h"
#include "frame.h"
#include "geometry.h"

namespace nx_meta_plugin {
    class Classifier {
    public:
        explicit Classifier(std::filesystem::path modelPath);

        void ensureInitialized();

        bool isTerminated() const;

        void terminate();

        DetectionList run(const Frame &frame);
//        DetectionList run(const cv::Mat &frame);

    private:
        DetectionList runImpl(const Frame &frame);
//        DetectionList runImpl(const cv::Mat &frame);

    private:
        YOLO11Detector m_yolo11_detector;
        YOLO11Classifier m_yolo11_classifier;
        bool m_netLoaded = false;
        bool m_terminated = false;
        bool useGPU = false;
        std::filesystem::path m_modelPath;
    };
}
