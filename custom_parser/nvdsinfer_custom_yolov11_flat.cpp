#include <algorithm>
#include <cmath>
#include <iostream>
#include <vector>

#include "nvdsinfer_custom_impl.h"

// Helper function to clamp coordinate values within frame boundaries
static inline float clampf(float v, float lo, float hi) {
    return std::max(lo, std::min(v, hi));
}

// Struct representing a parsed bounding box from the model output
struct ParsedBox {
    float x1;
    float y1;
    float x2;
    float y2;
    float score;
    int class_id;
};

// Parser for Row-Major (box-by-box layout: [num_boxes, 6])
static bool parse_row_major_box(const float *output, int offset, ParsedBox &box) {
    box.x1       = output[offset + 0];
    box.y1       = output[offset + 1];
    box.x2       = output[offset + 2];
    box.y2       = output[offset + 3];
    box.score    = output[offset + 4];
    box.class_id = static_cast<int>(std::round(output[offset + 5]));
    
    return std::isfinite(box.x1) && std::isfinite(box.y1) &&
           std::isfinite(box.x2) && std::isfinite(box.y2) &&
           std::isfinite(box.score);
}

// Parser for Column-Major (transposed layout: [6, num_boxes])
static bool parse_col_major_box(const float *output, int num_boxes, int i, ParsedBox &box) {
    box.x1       = output[0 * num_boxes + i];
    box.y1       = output[1 * num_boxes + i];
    box.x2       = output[2 * num_boxes + i];
    box.y2       = output[3 * num_boxes + i];
    box.score    = output[4 * num_boxes + i];
    box.class_id = static_cast<int>(std::round(output[5 * num_boxes + i]));
    
    return std::isfinite(box.x1) && std::isfinite(box.y1) &&
           std::isfinite(box.x2) && std::isfinite(box.y2) &&
           std::isfinite(box.score);
}

extern "C" bool NvDsInferParseCustomYoloV11Flat(
    std::vector<NvDsInferLayerInfo> const &outputLayersInfo,
    NvDsInferNetworkInfo const &networkInfo,
    NvDsInferParseDetectionParams const &detectionParams,
    std::vector<NvDsInferObjectDetectionInfo> &objectList) {
    
    if (outputLayersInfo.empty()) {
        std::cerr << "ERROR: No output layer found." << std::endl;
        return false;
    }

    const NvDsInferLayerInfo &layer = outputLayersInfo[0];
    const float *output = reinterpret_cast<const float *>(layer.buffer);
    const NvDsInferDims &dims = layer.inferDims;

    int num_boxes = 0;
    int elements_per_box = 0;
    bool col_major = false;

    /*
      Expected shape formats:
      - Row-Major: [1, 8400, 6] or [8400, 6]
      - Column-Major (transposed): [1, 6, 8400] or [6, 8400]

      Each box elements format:
      [x1, y1, x2, y2, confidence, class_id]
    */

    if (dims.numDims == 3) {
        if (dims.d[2] == 6) {
            num_boxes = dims.d[1];
            elements_per_box = dims.d[2];
        } else if (dims.d[1] == 6) {
            num_boxes = dims.d[2];
            elements_per_box = dims.d[1];
            col_major = true;
        }
    } else if (dims.numDims == 2) {
        if (dims.d[1] == 6) {
            num_boxes = dims.d[0];
            elements_per_box = dims.d[1];
        } else if (dims.d[0] == 6) {
            num_boxes = dims.d[1];
            elements_per_box = dims.d[0];
            col_major = true;
        }
    } else {
        std::cerr << "ERROR: Unsupported output dimensions. numDims=" << dims.numDims << std::endl;
        return false;
    }

    // Safety guard — the branches above only set elements_per_box when dims.d[x]==6,
    // so this should never trigger in practice.
    if (elements_per_box != 6) {
        std::cerr << "ERROR: Expected 6 values per box, but got " << elements_per_box << std::endl;
        return false;
    }

    // Pre-allocate memory to avoid multiple reallocations during push_back
    objectList.reserve(objectList.size() + std::min(num_boxes, 300));

    for (int i = 0; i < num_boxes; ++i) {
        ParsedBox box = {0};
        const bool parsed_successfully = col_major
            ? parse_col_major_box(output, num_boxes, i, box)
            : parse_row_major_box(output, i * elements_per_box, box);
            
        if (!parsed_successfully) {
            continue;
        }

        // Validate class ID limits
        if (box.class_id < 0 || box.class_id >= static_cast<int>(detectionParams.numClassesConfigured)) {
            continue;
        }

        // Determine pre-cluster score threshold for the current class
        float threshold = 0.25f;
        if (box.class_id < static_cast<int>(detectionParams.perClassPreclusterThreshold.size())) {
            threshold = detectionParams.perClassPreclusterThreshold[box.class_id];
        }

        if (box.score < threshold) {
            continue;
        }

        // Standardize coordinate boundaries
        float x1 = std::min(box.x1, box.x2);
        float y1 = std::min(box.y1, box.y2);
        float x2 = std::max(box.x1, box.x2);
        float y2 = std::max(box.y1, box.y2);
        
        float width  = x2 - x1;
        float height = y2 - y1;

        // Ignore zero-area or tiny boxes
        if (width <= 1.0f || height <= 1.0f) {
            continue;
        }

        NvDsInferObjectDetectionInfo obj = {0}; // Zero-initialize to fix DeepStream 9.0 OBB random rotation bug
        obj.classId             = box.class_id;
        obj.detectionConfidence = box.score;

        // Clip boxes to network input bounds
        obj.left   = clampf(x1, 0.0f, static_cast<float>(networkInfo.width - 1));
        obj.top    = clampf(y1, 0.0f, static_cast<float>(networkInfo.height - 1));
        obj.width  = clampf(width, 0.0f, static_cast<float>(networkInfo.width) - obj.left);
        obj.height = clampf(height, 0.0f, static_cast<float>(networkInfo.height) - obj.top);

        if (obj.width > 0.0f && obj.height > 0.0f) {
            objectList.push_back(obj);
        }
    }

    return true;
}

// Suppress compiler warning about infinite recursion inside the NVIDIA checking macro
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Winfinite-recursion"
CHECK_CUSTOM_PARSE_FUNC_PROTOTYPE(NvDsInferParseCustomYoloV11Flat);
#pragma GCC diagnostic pop

extern "C" bool NvDsInferParseCustomYoloChar(
    std::vector<NvDsInferLayerInfo> const &outputLayersInfo,
    NvDsInferNetworkInfo const &networkInfo,
    NvDsInferParseDetectionParams const &detectionParams,
    std::vector<NvDsInferObjectDetectionInfo> &objectList) {
    
    if (outputLayersInfo.empty()) {
        std::cerr << "ERROR: No output layer found." << std::endl;
        return false;
    }

    const NvDsInferLayerInfo &layer = outputLayersInfo[0];
    const float *output = reinterpret_cast<const float *>(layer.buffer);
    const NvDsInferDims &dims = layer.inferDims;

    // Expected shape: [1, 25200, 41] or [25200, 41]
    int num_boxes = 0;
    int elements_per_box = 0;

    if (dims.numDims == 3) {
        num_boxes = dims.d[1];
        elements_per_box = dims.d[2];
    } else if (dims.numDims == 2) {
        num_boxes = dims.d[0];
        elements_per_box = dims.d[1];
    } else {
        std::cerr << "ERROR: Unsupported output dimensions. numDims=" << dims.numDims << std::endl;
        return false;
    }

    if (elements_per_box != 35) {
        std::cerr << "ERROR: Expected 35 values per box for Char YOLO, but got " << elements_per_box << std::endl;
        return false;
    }

    objectList.reserve(objectList.size() + 50);

    for (int i = 0; i < num_boxes; ++i) {
        int offset = i * elements_per_box;
        float obj_conf = output[offset + 4];

        float max_cls_score = 0.0f;
        int class_id = -1;
        for (int c = 0; c < 30; ++c) {
            if (output[offset + 5 + c] > max_cls_score) {
                max_cls_score = output[offset + 5 + c];
                class_id = c;
            }
        }

        float score = obj_conf * max_cls_score;

        float threshold = 0.25f;
        if (class_id >= 0 && class_id < static_cast<int>(detectionParams.perClassPreclusterThreshold.size())) {
            threshold = detectionParams.perClassPreclusterThreshold[class_id];
        }

        if (score < threshold) continue;

        float cx = output[offset + 0];
        float cy = output[offset + 1];
        float w  = output[offset + 2];
        float h  = output[offset + 3];

        float x1 = cx - w / 2.0f;
        float y1 = cy - h / 2.0f;

        NvDsInferObjectDetectionInfo obj = {0};
        obj.classId = class_id;
        obj.detectionConfidence = score;

        obj.left   = clampf(x1, 0.0f, static_cast<float>(networkInfo.width - 1));
        obj.top    = clampf(y1, 0.0f, static_cast<float>(networkInfo.height - 1));
        obj.width  = clampf(w, 0.0f, static_cast<float>(networkInfo.width) - obj.left);
        obj.height = clampf(h, 0.0f, static_cast<float>(networkInfo.height) - obj.top);

        if (obj.width > 1.0f && obj.height > 1.0f) {
            objectList.push_back(obj);
        }
    }

    return true;
}

#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Winfinite-recursion"
CHECK_CUSTOM_PARSE_FUNC_PROTOTYPE(NvDsInferParseCustomYoloChar);
#pragma GCC diagnostic pop
