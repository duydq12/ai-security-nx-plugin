// Copyright 2018-present Network Optix, Inc. Licensed under MPL 2.0: www.mozilla.org/MPL/2.0/

#pragma once

#include <map>
#include <memory>
#include <set>

#include <opencv2/tracking/tracking_by_matching.hpp>

#include <nx/sdk/helpers/uuid_helper.h>
#include <nx/sdk/uuid.h>

#include "detection.h"
#include "frame.h"

namespace nx_meta_plugin {

/**
 * Provides conversion from int ids coming from the tracker to Uuid ids that are needed by the
 * Server.
 */
    class IdMapper {
    public:
        nx::sdk::Uuid get(int64_t id);

        void removeAllExcept(const std::set<int64_t> &idsToKeep);

    private:
        std::map<int64_t, nx::sdk::Uuid> m_map;
    };

    struct DetectionInternal {
        std::shared_ptr<Detection> detection;
        int64_t cvTrackId;
    };

    using DetectionInternalList = std::vector<std::shared_ptr<DetectionInternal>>;

    struct CompositeDetectionId {
        const int64_t frameIndex;
        const cv::Rect rect;
    };

    using ClassLabelMap = std::map<const CompositeDetectionId, std::string>;

    cv::tbm::TrackedObjects convertDetectionsToTrackedObjects(
            const Frame &frame,
            const DetectionList &detections,
            ClassLabelMap *inOutClassLabels);

    std::shared_ptr<Detection> convertTrackedObjectToDetection(
            const Frame &frame,
            const cv::tbm::TrackedObject &trackedDetection,
            const std::string &classLabel,
            IdMapper *idMapper);

    DetectionList convertTrackedObjectsToDetections(
            const Frame &frame,
            const cv::tbm::TrackedObjects &trackedDetections,
            const ClassLabelMap &classLabels,
            IdMapper *idMapper);

    DetectionList extractDetectionList(const DetectionInternalList &detectionsInternal);

}

namespace std {
    using namespace nx_meta_plugin;

    template<>
    struct less<const CompositeDetectionId> {
        bool operator()(const CompositeDetectionId &lhs, const CompositeDetectionId &rhs) const {
            if (lhs.frameIndex != rhs.frameIndex)
                return lhs.frameIndex < rhs.frameIndex;
            if (lhs.rect.x != rhs.rect.x)
                return lhs.rect.x < rhs.rect.x;
            if (lhs.rect.y != rhs.rect.y)
                return lhs.rect.y < rhs.rect.y;
            if (lhs.rect.width != rhs.rect.width)
                return lhs.rect.width < rhs.rect.width;
            return lhs.rect.width < rhs.rect.width;
        }
    };
}
