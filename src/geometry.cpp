//
// Created by ubuntu on 08/05/2025.
//

#include <iostream>
#include <algorithm>
#include <numeric>

#include "geometry.h"

namespace nx_meta_plugin {
    using namespace std::string_literals;
    using namespace cv;

    size_t vectorProduct(const std::vector<int64_t> &vector) {
        return std::accumulate(vector.begin(), vector.end(), 1ull, std::multiplies<size_t>());
    }

    cv::Rect scaleCoords(const cv::Size &imageShape, cv::Rect coords,
                         const cv::Size &imageOriginalShape, bool p_Clip) {
        cv::Rect result;
        float gain = std::min(
                static_cast<float>(imageShape.height) / static_cast<float>(imageOriginalShape.height),
                static_cast<float>(imageShape.width) / static_cast<float>(imageOriginalShape.width));

        int padX = static_cast<int>(std::round((imageShape.width - imageOriginalShape.width * gain) / 2.0f));
        int padY = static_cast<int>(std::round((imageShape.height - imageOriginalShape.height * gain) / 2.0f));

        result.x = static_cast<int>(std::round((coords.x - padX) / gain));
        result.y = static_cast<int>(std::round((coords.y - padY) / gain));
        result.width = static_cast<int>(std::round(coords.width / gain));
        result.height = static_cast<int>(std::round(coords.height / gain));

        if (p_Clip) {
            result.x = clamp(result.x, 0, imageOriginalShape.width);
            result.y = clamp(result.y, 0, imageOriginalShape.height);
            result.width = clamp(result.width, 0, imageOriginalShape.width - result.x);
            result.height = clamp(result.height, 0, imageOriginalShape.height - result.y);
        }
        return result;
    }

// Optimized Non-Maximum Suppression Function
    void NMSBoxes(const std::vector<cv::Rect> &boundingBoxes,
                  const std::vector<float> &scores,
                  float scoreThreshold,
                  float nmsThreshold,
                  std::vector<int> &indices) {
        indices.clear();

        const size_t numBoxes = boundingBoxes.size();
        if (numBoxes == 0) {
            std::cout << "No bounding boxes to process in NMS" << std::endl;
            return;
        }

        // Step 1: Filter out boxes with scores below the threshold
        // and create a list of indices sorted by descending scores
        std::vector<int> sortedIndices;
        sortedIndices.reserve(numBoxes);
        for (size_t i = 0; i < numBoxes; ++i) {
            if (scores[i] >= scoreThreshold) {
                sortedIndices.push_back(static_cast<int>(i));
            }
        }

        // If no boxes remain after thresholding
        if (sortedIndices.empty()) {
            std::cout << "No bounding boxes above score threshold" << std::endl;
            return;
        }

        // Sort the indices based on scores in descending order
        std::sort(sortedIndices.begin(), sortedIndices.end(),
                  [&scores](int idx1, int idx2) {
                      return scores[idx1] > scores[idx2];
                  });

        // Step 2: Precompute the areas of all boxes
        std::vector<float> areas(numBoxes, 0.0f);
        for (size_t i = 0; i < numBoxes; ++i) {
            areas[i] = boundingBoxes[i].width * boundingBoxes[i].height;
        }

        // Step 3: Suppression mask to mark boxes that are suppressed
        std::vector<bool> suppressed(numBoxes, false);

        // Step 4: Iterate through the sorted list and suppress boxes with high IoU
        for (size_t i = 0; i < sortedIndices.size(); ++i) {
            int currentIdx = sortedIndices[i];
            if (suppressed[currentIdx]) {
                continue;
            }

            // Select the current box as a valid detection
            indices.push_back(currentIdx);

            const cv::Rect &currentBox = boundingBoxes[currentIdx];
            const float x1_max = currentBox.x;
            const float y1_max = currentBox.y;
            const float x2_max = currentBox.x + currentBox.width;
            const float y2_max = currentBox.y + currentBox.height;
            const float area_current = areas[currentIdx];

            // Compare IoU of the current box with the rest
            for (size_t j = i + 1; j < sortedIndices.size(); ++j) {
                int compareIdx = sortedIndices[j];
                if (suppressed[compareIdx]) {
                    continue;
                }

                const cv::Rect &compareBox = boundingBoxes[compareIdx];
                const float x1 = std::max(x1_max, static_cast<float>(compareBox.x));
                const float y1 = std::max(y1_max, static_cast<float>(compareBox.y));
                const float x2 = std::min(x2_max, static_cast<float>(compareBox.x + compareBox.width));
                const float y2 = std::min(y2_max, static_cast<float>(compareBox.y + compareBox.height));

                const float interWidth = x2 - x1;
                const float interHeight = y2 - y1;

                if (interWidth <= 0 || interHeight <= 0) {
                    continue;
                }

                const float intersection = interWidth * interHeight;
                const float unionArea = area_current + areas[compareIdx] - intersection;
                const float iou = (unionArea > 0.0f) ? (intersection / unionArea) : 0.0f;

                if (iou > nmsThreshold) {
                    suppressed[compareIdx] = true;
                }
            }
        }
    }
}