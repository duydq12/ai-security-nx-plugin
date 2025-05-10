// Copyright 2018-present Network Optix, Inc. Licensed under MPL 2.0: www.mozilla.org/MPL/2.0/

#pragma once

#include <memory>
#include <vector>
#include <string>
#include <opencv2/opencv.hpp>
#include <nx/sdk/analytics/rect.h>
#include <nx/sdk/uuid.h>

namespace nx_meta_plugin {
    extern const std::vector<std::string> kClasses;
    extern const std::vector<std::string> kClassesToDetect;
    extern const std::vector<std::string> kClassesToClassification;

    struct Detection {
        const nx::sdk::analytics::Rect boundingBox;
        std::string classLabel;
        const float confidence;
        const nx::sdk::Uuid trackId;
    };

    using DetectionList = std::vector<std::shared_ptr<Detection>>;
}