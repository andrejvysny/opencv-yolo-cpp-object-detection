#include "YoloDetector.hpp"
#include <opencv2/opencv.hpp>
#include <opencv2/core.hpp>
#include <iostream>
#include <iomanip>
#include <filesystem>

/* ---------------------------------------------------------------
   YAML schema example:

   active_model: yolov4-tiny
   models:
     yolov4-tiny:
       cfg:     "../models/yolov4-tiny.cfg"
       weights: "../models/yolov4-tiny.weights"
       names:   "../models/coco.names"
       input_width:  416
       input_height: 416
       conf_thresh:  0.25
       nms_thresh:   0.4
     yolov3-tiny:
       cfg:     "../models/yolov3-tiny.cfg"
       ...
---------------------------------------------------------------- */

static bool loadModelInfo(const std::string& yamlPath,
                          ModelInfo& outInfo,
                          bool debug = false)
{
    std::cout << "[Config] opening file: " << yamlPath << '\n';
    cv::FileStorage fs(yamlPath,
                       cv::FileStorage::READ | cv::FileStorage::FORMAT_YAML);
    if (!fs.isOpened()) {
        std::cerr << "Cannot open config file: " << yamlPath << '\n';
        return false;
    }

    std::string active;
    fs["active_model"] >> active;
    if (active.empty()) {
        std::cerr << "active_model not specified in " << yamlPath << '\n';
        return false;
    }
    if (debug)
        std::cout << "[Config] active_model = " << active << '\n';

    bool save_detections = false;
    fs["save_detections"] >> save_detections;
    if (debug)
        std::cout << "[Config] save_detections = " << save_detections << '\n';

    cv::FileNode modelsNode = fs["models"];
    cv::FileNode modelNode  = modelsNode[active];
    if (modelNode.empty()) {
        std::cerr << "Model '" << active << "' not found in config.\n";
        return false;
    }

    modelNode["cfg"]     >> outInfo.cfg;
    modelNode["weights"] >> outInfo.weights;
    modelNode["names"]   >> outInfo.names;
    modelNode["input_width"]   >> outInfo.input_width;
    modelNode["input_height"]  >> outInfo.input_height;
    modelNode["conf_thresh"]   >> outInfo.conf_thresh;
    modelNode["nms_thresh"]    >> outInfo.nms_thresh;

    if (debug) {
        std::cout << "[Config] cfg=" << outInfo.cfg
                  << ", weights=" << outInfo.weights
                  << ", names=" << outInfo.names << '\n';
    }

    // Validate that model files actually exist
    if (!std::filesystem::exists(outInfo.cfg)) {
        std::cerr << "Config file not found: " << outInfo.cfg << '\n';
        return false;
    }
    if (!std::filesystem::exists(outInfo.weights)) {
        std::cerr << "Weights file not found: " << outInfo.weights << '\n';
        return false;
    }
    if (!std::filesystem::exists(outInfo.names)) {
        std::cerr << "Names file not found: " << outInfo.names << '\n';
        return false;
    }
    return true;
}

constexpr bool DISPLAY_RESULTS = false;

// ---------------------------------------------------------------
static std::string deriveOutputName(const std::string& path)
{
    auto dot = path.find_last_of('.');
    return path.substr(0, dot) + "_det" + path.substr(dot);
}

// ---------------------------------------------------------------
class ImageProcessor
{
public:
    ImageProcessor(const std::string& configPath, bool debug = false)
    {
        if (!loadModelInfo(configPath, mi_, debug)) {
            throw std::runtime_error("Failed to load model config.");
        }
        detector_ = std::make_unique<YoloDetector>(mi_, debug);
    }

    void run(const std::string& imgPath)
    {
        cv::Mat img = cv::imread(imgPath);
        if (img.empty())
        {
            std::cerr << "Cannot read " << imgPath << '\n';
            return;
        }

        auto detections = detector_->detect(img);
        std::cout << "[ImageProcessor] Detections found: "
                  << detections.size() << '\n';

        detector_->drawDetections(img, detections);

        if (DISPLAY_RESULTS)
        {
            cv::imshow("Detections", img);
            cv::waitKey(0);
        }
        else
        {
            std::string out = deriveOutputName(imgPath);
            cv::imwrite(out, img);
            std::cout << "Written " << out << '\n';
        }
    }

private:
    ModelInfo mi_;
    std::unique_ptr<YoloDetector> detector_;
};

// ---------------------------------------------------------------
int main(int argc, char** argv)
{
    if (argc != 3)
    {
        std::cerr << "Usage: " << argv[0]
                  << " <config.yml> <image>\n";
        return 1;
    }

    try {
        ImageProcessor proc(argv[1], true /*debug*/);
        proc.run(argv[2]);
    } catch (const std::exception& ex) {
        std::cerr << ex.what() << '\n';
        return 1;
    }
    return 0;
}