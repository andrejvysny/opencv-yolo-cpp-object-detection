#include "YoloDetector.hpp"
#include <fstream>
#include <sstream>
#include <iostream>
#include <algorithm>

// Helper to load class names
static std::vector<std::string> loadClassNames(const std::string& path)
{
    std::vector<std::string> names;
    std::ifstream ifs(path);
    std::string line;
    while (std::getline(ifs, line))
        names.push_back(line);
    return names;
}

/* --------- new ctor accepting ModelInfo ------------------ */
YoloDetector::YoloDetector(const ModelInfo& m, bool debug)
    : confThresh_{m.conf_thresh},
      nmsThresh_{m.nms_thresh},
      inpWidth_{m.input_width},
      inpHeight_{m.input_height},
      debug_{debug},
      classNames_{loadClassNames(m.names)}
{
    net_ = cv::dnn::readNetFromDarknet(m.cfg, m.weights);
    net_.setPreferableBackend(cv::dnn::DNN_BACKEND_OPENCV);
    net_.setPreferableTarget(cv::dnn::DNN_TARGET_CPU);
}
/* --------------------------------------------------------- */

YoloDetector::YoloDetector(const std::string& cfg,
                           const std::string& weights,
                           const std::string& names,
                           float confThresh,
                           float nmsThresh,
                           int   inpW,
                           int   inpH,
                           bool  debug)
    : confThresh_{confThresh},
      nmsThresh_{nmsThresh},
      inpWidth_{inpW},
      inpHeight_{inpH},
      debug_{debug},
      classNames_{loadClassNames(names)}
{
    net_ = cv::dnn::readNetFromDarknet(cfg, weights);
    net_.setPreferableBackend(cv::dnn::DNN_BACKEND_OPENCV);
    net_.setPreferableTarget(cv::dnn::DNN_TARGET_CPU);
}

std::vector<YoloDetector::Detection>
YoloDetector::detect(const cv::Mat& frame)
{
    if (debug_)
        std::cout << "[YoloDetector] Frame size: " << frame.cols << "x" << frame.rows << "\n";

    int origW = frame.cols;
    int origH = frame.rows;

    float scale = std::min(static_cast<float>(inpWidth_) / origW,
                           static_cast<float>(inpHeight_) / origH);
    int newW = static_cast<int>(origW * scale);
    int newH = static_cast<int>(origH * scale);

    int padX = (inpWidth_ - newW) / 2;
    int padY = (inpHeight_ - newH) / 2;

    if (debug_)
        std::cout << "[YoloDetector] scale=" << scale
                  << ", new=" << newW << "x" << newH
                  << ", pad=(" << padX << "," << padY << ")\n";

    cv::Mat resized;
    cv::resize(frame, resized, cv::Size(newW, newH));
    cv::Mat canvas(cv::Size(inpWidth_, inpHeight_), CV_8UC3, cv::Scalar(114, 114, 114));
    resized.copyTo(canvas(cv::Rect(padX, padY, newW, newH)));

    cv::Mat blob;
    cv::dnn::blobFromImage(canvas, blob, 1/255.0,
                           cv::Size(inpWidth_, inpHeight_),
                           cv::Scalar(), true, false);
    net_.setInput(blob);

    std::vector<cv::Mat> outputs;
    net_.forward(outputs, net_.getUnconnectedOutLayersNames());

    std::vector<int> classIds;
    std::vector<float> confidences;
    std::vector<cv::Rect> boxes;

    for (const auto& output : outputs)
    {
        const float* data = reinterpret_cast<const float*>(output.data);
        for (int i = 0; i < output.rows; ++i, data += output.cols)
        {
            float objConf = data[4];
            cv::Mat scores(1, output.cols - 5, CV_32F, (void*)(data + 5));
            cv::Point classIdPt;
            double maxClassScore;
            cv::minMaxLoc(scores, nullptr, &maxClassScore, nullptr, &classIdPt);

            float conf = objConf * static_cast<float>(maxClassScore);
            if (conf < confThresh_)
                continue;

            float cx = data[0] * inpWidth_;
            float cy = data[1] * inpHeight_;
            float w  = data[2] * inpWidth_;
            float h  = data[3] * inpHeight_;

            float x0 = (cx - w / 2.0f - padX) / scale;
            float y0 = (cy - h / 2.0f - padY) / scale;
            float x1 = x0 + w / scale;
            float y1 = y0 + h / scale;

            x0 = std::max(0.f, std::min(x0, static_cast<float>(origW - 1)));
            y0 = std::max(0.f, std::min(y0, static_cast<float>(origH - 1)));
            x1 = std::max(0.f, std::min(x1, static_cast<float>(origW - 1)));
            y1 = std::max(0.f, std::min(y1, static_cast<float>(origH - 1)));

            int left   = static_cast<int>(x0);
            int top    = static_cast<int>(y0);
            int width  = static_cast<int>(x1 - x0);
            int height = static_cast<int>(y1 - y0);

            classIds.push_back(classIdPt.x);
            confidences.push_back(conf);
            boxes.emplace_back(left, top, width, height);

            if (debug_)
                std::cout << "  raw box [" << i << "]: class=" << classIdPt.x
                          << ", conf=" << conf << "\n";
        }
    }

    std::vector<int> indices;
    cv::dnn::NMSBoxes(boxes, confidences, confThresh_, nmsThresh_, indices);

    if (debug_)
        std::cout << "Boxes before NMS: " << boxes.size()
                  << ", after NMS: " << indices.size() << "\n";

    std::vector<Detection> detections;
    detections.reserve(indices.size());
    for (int idx : indices)
    {
        detections.push_back({classIds[idx], confidences[idx], boxes[idx]});
    }

    if (debug_)
        std::cout << "Final detections: " << detections.size() << "\n";

    return detections;
}

void YoloDetector::drawDetections(cv::Mat& frame,
                                  const std::vector<Detection>& dets) const
{
    for (const auto& d : dets)
    {
        cv::rectangle(frame, d.box, cv::Scalar(0, 255, 0), 2);
        std::ostringstream label;
        label << classNames_.at(d.classId) << ": "
              << std::fixed << std::setprecision(2) << d.confidence;
        int baseLine = 0;
        cv::Size sz = cv::getTextSize(label.str(),
                                      cv::FONT_HERSHEY_SIMPLEX, 0.5, 1, &baseLine);
        int top = std::max(d.box.y, sz.height);
        cv::rectangle(frame,
                      cv::Point(d.box.x, top - sz.height),
                      cv::Point(d.box.x + sz.width, top + baseLine),
                      cv::Scalar(0, 255, 0), cv::FILLED);
        cv::putText(frame, label.str(), cv::Point(d.box.x, top),
                    cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(0, 0, 0), 1);
    }
}