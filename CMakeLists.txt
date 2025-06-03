cmake_minimum_required(VERSION 3.10)
project(YOLOv4TinyDetector LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

find_package(OpenCV 4.6 REQUIRED COMPONENTS core imgproc highgui dnn)

# library
add_library(yolo_detector
    src/YoloDetector.cpp
)
target_include_directories(yolo_detector PUBLIC include)
target_link_libraries(yolo_detector PUBLIC ${OpenCV_LIBS})

# demo executable
add_executable(image_inference
    src/ImageProcessor.cpp
)
target_link_libraries(image_inference PRIVATE yolo_detector)