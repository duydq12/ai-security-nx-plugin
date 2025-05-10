// Copyright 2018-present Network Optix, Inc. Licensed under MPL 2.0: www.mozilla.org/MPL/2.0/

#pragma once

#include <opencv2/opencv.hpp>

#include <nx/kit/debug.h>
#include <nx/sdk/analytics/rect.h>
#include "detection.h"
#include "frame.h"
#include "geometry.h"

namespace nx_meta_plugin {
    inline void drawBoundingBox(const cv::Mat &image, std::shared_ptr<Detection> &detection) {
        // Draw the bounding box rectangle
        cv::Size frameSize = image.size();
        const cv::Rect boundingBox = nxRectToCvRect(detection->boundingBox, frameSize.width, frameSize.height);
        cv::rectangle(image, cv::Point(boundingBox.x, boundingBox.y),
                      cv::Point(boundingBox.x + boundingBox.width,
                                boundingBox.y + boundingBox.height),
                      cv::Scalar(0, 0, 0), 2, cv::LINE_AA);

        // Prepare label text with class name and confidence percentage

        // Define text properties for labels
        int fontFace = cv::FONT_HERSHEY_SIMPLEX;
        double fontScale = std::min(image.rows, image.cols) * 0.0008;
        const int thickness = std::max(1, static_cast<int>(std::min(image.rows, image.cols) * 0.002));
        int baseline = 0;

        // Calculate text size for background rectangles
        cv::Size textSize = cv::getTextSize(detection->classLabel, fontFace, fontScale, thickness, &baseline);

        // Define positions for the label
        int labelY = std::max(static_cast<int>(detection->boundingBox.y), textSize.height + 5);
        cv::Point labelTopLeft(detection->boundingBox.x, labelY - textSize.height - 5);
        cv::Point labelBottomRight(detection->boundingBox.x + textSize.width + 5, labelY + baseline - 5);

        // Draw background rectangle for label
        cv::rectangle(image, labelTopLeft, labelBottomRight, cv::Scalar(0, 0, 0), cv::FILLED);

        // Put label text
        cv::putText(image, detection->classLabel, cv::Point(detection->boundingBox.x + 2, labelY - 2), fontFace,
                    fontScale,
                    cv::Scalar(255, 255, 255), thickness, cv::LINE_AA);
    }

    void pprint_float_array(const float *data, int total_items_to_print, int items_per_line = 10, int precision = 4,
                            int width = 12) {
        if (!data) {
            std::cerr << "Error: Null pointer provided." << std::endl;
            return;
        }

        // Set up formatting for cout
        std::cout << std::fixed << std::setprecision(precision);

        for (int i = 0; i < total_items_to_print; ++i) {
            std::cout << std::setw(width) << data[i];
            if ((i + 1) % items_per_line == 0 || i == total_items_to_print - 1) {
                std::cout << std::endl;
            } else {
                std::cout << " "; // Separator between numbers on the same line
            }
        }
    }

    void printFirst5Rows(const cv::Mat &mat, const std::string &matName = "Matrix") {
        std::cout << "===== First 5 rows of " << matName << " =====" << std::endl;
        std::cout << "Size: " << mat.rows << " x " << mat.cols << std::endl;
        std::cout << "Type: " << cv::typeToString(mat.type()) << std::endl;

        // Determine number of rows to print (min of 5 or mat.rows)
        int rowsToPrint = std::min(20, mat.rows);

        for (int i = 0; i < rowsToPrint; i++) {
            std::cout << "Row " << i << ": ";

            // Handle different matrix types
            switch (mat.type()) {
                case CV_8U:
                    for (int j = 0; j < mat.cols; j++) {
                        std::cout << static_cast<int>(mat.at<uchar>(i, j)) << " ";
                    }
                    break;
                case CV_8UC3:
                    for (int j = 0; j < mat.cols; j++) {
                        cv::Vec3b pixel = mat.at<cv::Vec3b>(i, j);
                        std::cout << "(" << static_cast<int>(pixel[0]) << ","
                                  << static_cast<int>(pixel[1]) << ","
                                  << static_cast<int>(pixel[2]) << ") ";
                    }
                    break;
                case CV_32FC3:
                    for (int j = 0; j < mat.cols; j++) {
                        std::cout << mat.at<float>(i, j) << " ";
                    }
                    break;
                case CV_64F:
                    for (int j = 0; j < mat.cols; j++) {
                        std::cout << mat.at<double>(i, j) << " ";
                    }
                    break;
                default:
                    std::cout << "Unsupported matrix type for printing";
            }
            std::cout << std::endl;
        }
        std::cout << "=============================" << std::endl;
    }

}