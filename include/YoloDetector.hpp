#pragma once

#include <opencv2/opencv.hpp>
#include <opencv2/dnn.hpp>
#include <string>
#include <vector>

/* ---------------- model description ---------------- */
struct ModelInfo
{
    std::string cfg;
    std::string weights;
    std::string names;
    int         input_width  = 416;
    int         input_height = 416;
    float       conf_thresh  = 0.25f;
    float       nms_thresh   = 0.4f;
};
/* --------------------------------------------------- */

class YoloDetector
{
public:
    struct Detection
    {
        int       classId;
        float     confidence;
        cv::Rect  box;
    };

    /* legacy constructor (kept for compatibility) */
    YoloDetector(const std::string& cfg,
                 const std::string& weights,
                 const std::string& names,
                 float confThresh = 0.25f,
                 float nmsThresh  = 0.4f,
                 int   inpWidth   = 416,
                 int   inpHeight  = 416,
                 bool  debug      = false);

    /* NEW convenient constructor */
    YoloDetector(const ModelInfo& mi, bool debug = false);

    /** Run inference and return final detections (after NMS). */
    std::vector<Detection> detect(const cv::Mat& frame);

    /** Draw rectangles + labels in-place. */
    void drawDetections(cv::Mat& frame,
                        const std::vector<Detection>& detections) const;

private:
    cv::dnn::Net             net_;
    std::vector<std::string> classNames_;
    float                    confThresh_;
    float                    nmsThresh_;
    int                      inpWidth_;
    int                      inpHeight_;
    bool                     debug_;
};