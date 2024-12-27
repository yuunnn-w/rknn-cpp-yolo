#include <iostream>
#include <vector>
#include <cstring>
#include <chrono> // 用于计时
#include <iomanip> // 用于 setprecision
#include <opencv2/opencv.hpp> // 包含 OpenCV 头文件
#include "rknn_model.h"
#include "rga_utils.h"

int main() {
    // 初始化模型
    std::string model_path = "yolo11s.rknn"; // yolov9c.rknn
    rknn_model model(model_path);

    int ctx_index = 0; // 使用第一个上下文

    std::string image_path = "bus.jpg";
    cv::Mat image = cv::imread(image_path);
    if (image.empty()) {
        std::cerr << "Failed to read the image: " << image_path << std::endl;
        return -1;
    }
    cv::cvtColor(image, image, cv::COLOR_BGR2RGB);

    // 打印图像的尺寸和数据类型
    std::cout << "Image size: " << image.size() << std::endl;
    std::cout << "Image type: " << image.type() << std::endl;

    // 打印图像的数据类型名称
    std::string type_name;
    switch (image.type()) {
    case CV_8U:   type_name = "CV_8U";   break;
    case CV_8S:   type_name = "CV_8S";   break;
    case CV_16U:  type_name = "CV_16U";  break;
    case CV_16S:  type_name = "CV_16S";  break;
    case CV_32S:  type_name = "CV_32S";  break;
    case CV_32F:  type_name = "CV_32F";  break;
    case CV_64F:  type_name = "CV_64F";  break;
    case CV_8UC3: type_name = "CV_8UC3"; break;
    default:      type_name = "Unknown"; break;
    }
    std::cout << "Image type name: " << type_name << std::endl;

    const int num_inferences = 100;
    double total_time_ms = 0.0;
    object_detect_result_list od_results;

    for (int i = 0; i < num_inferences; ++i) {
        auto start = std::chrono::high_resolution_clock::now();

        // 运行推理
        int ret = model.run_inference(image, ctx_index, &od_results);
        auto end = std::chrono::high_resolution_clock::now();
        if (ret < 0) {
            printf("rknn_run fail! ret=%d\n", ret);
            return -1;
        }
        std::chrono::duration<double, std::milli> elapsed = end - start;
        total_time_ms += elapsed.count();
    }

    double avg_time_ms = total_time_ms / num_inferences;
    double fps = num_inferences / (total_time_ms / 1000.0);

    std::cout << std::fixed << std::setprecision(10); // 设置小数点后十位
    std::cout << "\nAverage inference time over " << num_inferences << " runs: "
        << avg_time_ms << " ms" << std::endl;

    std::cout << std::fixed << std::setprecision(2); // 设置小数点后两位
    std::cout << "Frames per second (FPS): " << fps << std::endl;

    /* //打印最后一次推理的结果（如果你需要）
    for (int i = 0; i < od_results.count; ++i) {
        object_detect_result result = od_results.results[i];
        printf("Object %d:\n", i + 1);
        printf("  Box: (%d, %d, %d, %d)\n",
            result.box.left,
            result.box.top,
            result.box.right,
            result.box.bottom);
        printf("  Class ID: %d\n", result.cls_id);
        printf("  Confidence: %.2f\n", result.prop);
    }
    */

    // 创建一个 RGB 格式的副本用于绘制
    cv::Mat image_rgb = image.clone();
    cv::cvtColor(image_rgb, image_rgb, cv::COLOR_RGB2BGR); // 转换回 BGR 格式以便显示正确颜色


    // 定义向下偏移量
    int offset = 50;
    // 绘制检测框
    for (int i = 0; i < od_results.count; ++i) {
        object_detect_result result = od_results.results[i];

        // 绘制矩形框
        cv::Rect rect(result.box.left, result.box.top, result.box.right - result.box.left, result.box.bottom - result.box.top);
        cv::rectangle(image_rgb, rect, cv::Scalar(0, 255, 0), 2); // 绿色框线

        // 添加文本标签
        std::ostringstream label;
        label << "ID: " << result.cls_id << " Conf: " << std::fixed << std::setprecision(2) << result.prop;
        int baseLine = 0;
        cv::Size label_size = cv::getTextSize(label.str(), cv::FONT_HERSHEY_SIMPLEX, 0.5, 1, &baseLine);
        cv::rectangle(image_rgb, cv::Point(result.box.left, result.box.top - label_size.height),
            cv::Point(result.box.left + label_size.width, result.box.top + baseLine),
            cv::Scalar(0, 255, 0), -1); // 填充背景
        cv::putText(image_rgb, label.str(), cv::Point(result.box.left, result.box.top),
            cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(0, 0, 0), 1, cv::LINE_AA); // 黑色文字
    }

    // 保存结果图像
    cv::imwrite("result.jpg", image_rgb);

    return 0;
}