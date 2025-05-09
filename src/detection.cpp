//
// Created by ubuntu on 08/05/2025.
//
#include "detection.h"

namespace nx_meta_plugin {
    const std::vector<std::string> kClasses{"person", "bicycle", "car", "motorbike", "aeroplane", "bus",
                                            "train", "truck", "boat", "traffic light", "fire hydrant",
                                            "stop sign", "parking meter", "bench", "bird", "cat", "dog",
                                            "horse", "sheep", "cow", "elephant", "bear", "zebra", "giraffe",
                                            "backpack", "umbrella", "handbag", "tie", "suitcase", "frisbee",
                                            "skis", "snowboard", "sports ball", "kite", "baseball bat",
                                            "baseball glove", "skateboard", "surfboard", "tennis racket",
                                            "bottle", "wine glass", "cup", "fork", "knife", "spoon", "bowl",
                                            "banana", "apple", "sandwich", "orange", "broccoli", "carrot",
                                            "hot dog", "pizza", "donut", "cake", "chair", "sofa",
                                            "pottedplant", "bed", "diningtable", "toilet", "tvmonitor",
                                            "laptop", "mouse", "remote", "keyboard", "cell phone", "microwave",
                                            "oven", "toaster", "sink", "refrigerator", "book", "clock", "vase",
                                            "scissors", "teddy bear", "hair drier", "toothbrush"};
    const std::vector<std::string> kClassesToDetect{"person"};
    const std::vector<std::string> kClassesToClassification{"CA", "PN"};
}
