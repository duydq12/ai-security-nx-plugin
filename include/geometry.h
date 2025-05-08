// Copyright 2018-present Network Optix, Inc. Licensed under MPL 2.0: www.mozilla.org/MPL/2.0/

#pragma once

#include <opencv2/opencv.hpp>

#include <nx/kit/debug.h>
#include <nx/sdk/analytics/rect.h>

namespace nx_meta_plugin {
    inline cv::Rect nxRectToCvRect(nx::sdk::analytics::Rect rect, int width, int height) {
        if (!NX_KIT_ASSERT(width > 0) || !NX_KIT_ASSERT(height > 0))
            return {};

        return cv::Rect(
                (int) (rect.x * width),
                (int) (rect.y * height),
                (int) (rect.width * width),
                (int) (rect.height * height)
        ) & cv::Rect(0, 0, width, height); //< Ensure that the result is inside the frame rect.
    }

    inline nx::sdk::analytics::Rect cvRectToNxRect(cv::Rect rect, int width, int height) {
        if (!NX_KIT_ASSERT(width > 0) || !NX_KIT_ASSERT(height > 0))
            return {};

        return nx::sdk::analytics::Rect(
                (float) rect.x / width,
                (float) rect.y / height,
                (float) rect.width / width,
                (float) rect.height / height);
    }

    template<typename T>
    typename std::enable_if<std::is_arithmetic<T>::value, T>::type
    inline clamp(const T &value, const T &low, const T &high) {
        // Ensure the range [low, high] is valid; swap if necessary
        T validLow = low < high ? low : high;
        T validHigh = low < high ? high : low;

        // Clamp the value to the range [validLow, validHigh]
        if (value < validLow)
            return validLow;
        if (value > validHigh)
            return validHigh;
        return value;
    }

    inline void letterBox(const cv::Mat &image, cv::Mat &outImage,
                          const cv::Size &newShape,
                          const cv::Scalar &color = cv::Scalar(114, 114, 114),
                          bool auto_ = true,
                          bool scaleFill = false,
                          bool scaleUp = true,
                          int stride = 32) {
        // Calculate the scaling ratio to fit the image within the new shape
        float ratio = std::min(static_cast<float>(newShape.height) / image.rows,
                               static_cast<float>(newShape.width) / image.cols);

        // Prevent scaling up if not allowed
        if (!scaleUp) {
            ratio = std::min(ratio, 1.0f);
        }

        // Calculate new dimensions after scaling
        int newUnpadW = static_cast<int>(std::round(image.cols * ratio));
        int newUnpadH = static_cast<int>(std::round(image.rows * ratio));

        // Calculate padding needed to reach the desired shape
        int dw = newShape.width - newUnpadW;
        int dh = newShape.height - newUnpadH;

        if (auto_) {
            // Ensure padding is a multiple of stride for model compatibility
            dw = (dw % stride) / 2;
            dh = (dh % stride) / 2;
        } else if (scaleFill) {
            // Scale to fill without maintaining aspect ratio
            newUnpadW = newShape.width;
            newUnpadH = newShape.height;
            ratio = std::min(static_cast<float>(newShape.width) / image.cols,
                             static_cast<float>(newShape.height) / image.rows);
            dw = 0;
            dh = 0;
        } else {
            // Evenly distribute padding on both sides
            // Calculate separate padding for left/right and top/bottom to handle odd padding
            int padLeft = dw / 2;
            int padRight = dw - padLeft;
            int padTop = dh / 2;
            int padBottom = dh - padTop;

            // Resize the image if the new dimensions differ
            if (image.cols != newUnpadW || image.rows != newUnpadH) {
                cv::resize(image, outImage, cv::Size(newUnpadW, newUnpadH), 0, 0, cv::INTER_LINEAR);
            } else {
                // Avoid unnecessary copying if dimensions are the same
                outImage = image;
            }

            // Apply padding to reach the desired shape
            cv::copyMakeBorder(outImage, outImage, padTop, padBottom, padLeft, padRight, cv::BORDER_CONSTANT,
                               color);
            return; // Exit early since padding is already applied
        }

        // Resize the image if the new dimensions differ
        if (image.cols != newUnpadW || image.rows != newUnpadH) {
            cv::resize(image, outImage, cv::Size(newUnpadW, newUnpadH), 0, 0, cv::INTER_LINEAR);
        } else {
            // Avoid unnecessary copying if dimensions are the same
            outImage = image;
        }

        // Calculate separate padding for left/right and top/bottom to handle odd padding
        int padLeft = dw / 2;
        int padRight = dw - padLeft;
        int padTop = dh / 2;
        int padBottom = dh - padTop;

        // Apply padding to reach the desired shape
        cv::copyMakeBorder(outImage, outImage, padTop, padBottom, padLeft, padRight, cv::BORDER_CONSTANT,
                           color);
    }

    size_t vectorProduct(const std::vector<int64_t> &vector);

    void NMSBoxes(const std::vector<cv::Rect> &boundingBoxes,
                  const std::vector<float> &scores,
                  float scoreThreshold,
                  float nmsThreshold,
                  std::vector<int> &indices);

    cv::Rect scaleCoords(const cv::Size &imageShape, cv::Rect coords,
                         const cv::Size &imageOriginalShape, bool p_Clip);
}