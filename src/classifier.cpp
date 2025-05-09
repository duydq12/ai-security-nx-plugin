// Copyright 2018-present Network Optix, Inc. Licensed under MPL 2.0: www.mozilla.org/MPL/2.0/


#include <opencv2/core.hpp>

#include "classifer.h"
#include "exceptions.h"

namespace nx_meta_plugin {
    using namespace std::string_literals;
    using namespace cv;

    Classifier::Classifier(std::filesystem::path modelPath) :
            m_modelPath(std::move(modelPath)) {
    }

    void Classifier::ensureInitialized() {
        if (isTerminated()) {
            throw ObjectDetectorIsTerminatedError(
                    "Object detector initialization error: object detector is terminated.");
        }
        if (m_netLoaded)
            return;

        try {
            m_yolo11_detector.ensureInitialized(m_modelPath);
            m_yolo11_classifier.ensureInitialized(m_modelPath);
        }
        catch (const cv::Exception &e) {
            terminate();
            throw ObjectDetectorInitializationError("Loading model: " + cvExceptionToStdString(e));
        }
        catch (const std::exception &e) {
            terminate();
            throw ObjectDetectorInitializationError("Loading model: Error: "s + e.what());
        }
    }

    bool Classifier::isTerminated() const {
        return m_terminated;
    }

    void Classifier::terminate() {
        m_terminated = true;
    }

    DetectionList Classifier::run(const Frame &frame) {
//    DetectionList Classifier::run(const cv::Mat &frame) {
        if (isTerminated())
            throw ObjectDetectorIsTerminatedError("Detection error: object detector is terminated.");

        try {
            return runImpl(frame);
        }
        catch (const cv::Exception &e) {
            terminate();
            throw ObjectDetectionError(cvExceptionToStdString(e));
        }
        catch (const std::exception &e) {
            terminate();
            throw ObjectDetectionError("Error: "s + e.what());
        }
    }

        DetectionList Classifier::runImpl(const Frame &frame) {
//    DetectionList Classifier::runImpl(const cv::Mat &frame) {
        if (isTerminated()) {
            throw ObjectDetectorIsTerminatedError(
                    "Object detection error: object detector is terminated.");
        }

        const Mat image = frame.cvMat;
//        const Mat image = frame;

        DetectionList detections = m_yolo11_detector.run(image);
        const cv::Size originalImageSize = image.size();
        if (!detections.empty()){
            cv::Rect boundingBox = nxRectToCvRect(detections[0]->boundingBox, originalImageSize.width, originalImageSize.height);
            cv::Mat cropped_image = image(boundingBox).clone();
            cv::Mat cropped_image_resize;
            cv::resize(cropped_image, cropped_image_resize, cv::Size(640, 640));
            std::string classLabel = m_yolo11_classifier.run(image);
            detections[0]->classLabel = classLabel;
            return detections;
        }
        return {};
    }
}