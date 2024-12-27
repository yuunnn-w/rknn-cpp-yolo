#ifndef RKNN_MODEL_H
#define RKNN_MODEL_H

#include <rknn_api.h>
#include <vector>
#include <string>
#include <opencv2/opencv.hpp> // 包含 OpenCV 头文件
//#include "../runtime/Linux/librknn_api/include/rknn_api.h"
#include <iostream>
#include <cstdio>
#include <cstring>
#include <execution> // 需要 C++17 支持
#include "common.h"
#include <omp.h> // OpenMP 头文件
#include <chrono>
#include <iomanip> // for std::setw and std::setprecision


class rknn_model {
public:
    rknn_model(const std::string& model_path);
    ~rknn_model();

    void query_model_info(int& ctx_index);

    // 获取输入输出格式
    const std::vector<rknn_tensor_attr>& get_input_attrs() const { return input_attrs; }
    const std::vector<rknn_tensor_attr>& get_output_attrs() const { return output_attrs; }
    rknn_context& get_context(int index) { return ctxs[index]; }
    // 打印量化和反量化信息
    void print_quantization_info() const;
    // 新增方法：封装推理过程
    int run_inference(cv::Mat& input_image, int& ctx_index, object_detect_result_list* od_results);
    rknn_tensor_mem* get_input_mem_ptr(int ctx_index, int mem_index);
    // 后处理流程
    int post_process(void* outputs, letterbox_t* letter_box, float conf_threshold, float nms_threshold, object_detect_result_list* od_results);

private:
    rknn_context ctxs[3];

    std::vector<rknn_tensor_attr> input_attrs;
    std::vector<rknn_tensor_attr> input_native_attrs;
    std::vector<rknn_tensor_attr> output_attrs;
	std::vector<rknn_tensor_attr> output_native_attrs;
    rknn_input_output_num io_num;
    int model_height;
	int model_width;
	int model_channel;

    std::vector<std::vector<rknn_tensor_mem*>> input_mems; // 输入内存
    std::vector<std::vector<rknn_tensor_mem*>> output_mems; // 输出内存
    int num_outputs;

    void init_model(const std::string& model_path);
    void release_model();
    void initialize_mems();
    void release_mems();
};




#endif // RKNN_MODEL_H