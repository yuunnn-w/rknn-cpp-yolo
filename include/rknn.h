
/*
#include <iostream>
#include <vector>
#include <cstring>
#include <chrono> // 用于计时
#include <iomanip> // 用于 setprecision
#include "rknn_model.h"

int main() {
    // 初始化模型
    std::string model_path = "/root/.vs/rknn/yolov9c.rknn";
    rknn_model model(model_path);

    // 查询模型信息
    model.query_model_info();

    // 打印量化和反量化信息
    model.print_quantization_info();

    // 获取输入和输出属性
    const std::vector<rknn_tensor_attr>& input_attrs = model.get_input_attrs();
    const std::vector<rknn_tensor_attr>& output_attrs = model.get_output_attrs();



    // 打印输入张量形状
    std::cout << "Input Tensor Shapes:" << std::endl;
    for (const auto& attr : input_attrs) {
        std::cout << "  Name: " << attr.name << ", Shape: ";
        for (uint32_t j = 0; j < attr.n_dims; ++j) {
            std::cout << attr.dims[j] << " ";
        }
        std::cout << std::endl;
    }
    // 设置模型输入参数
    std::vector<rknn_input> inputs(input_attrs.size());
    for (size_t i = 0; i < input_attrs.size(); ++i) {
        inputs[i].index = i;
        inputs[i].buf = new float[input_attrs[i].n_elems]; // 分配内存
        std::memset(inputs[i].buf, 0, input_attrs[i].n_elems * sizeof(float)); // 初始化为0
        inputs[i].size = input_attrs[i].n_elems; // * sizeof(float)
        inputs[i].pass_through = 0; // 用于指定输入数据是否直接传递给模型的输入节点
        inputs[i].type = RKNN_TENSOR_INT8;//input_attrs[i].type; // 输入数据类型 RKNN_TENSOR_INT8
        inputs[i].fmt = RKNN_TENSOR_NHWC;//input_attrs[i].fmt; // 输入数据格式
        // fmt: rknn_tensor_format类型，常见的有RKNN_TENSOR_NCHW、RKNN_TENSOR_NHWC、RKNN_TENSOR_NCHW_VEC、RKNN_TENSOR_UNDEFINED
        std::cout << "inputs[i].size: " << input_attrs[i].n_elems << std::endl;
        // 打印设置的输入参数
        std::cout << "Setting input parameter for tensor " << i << ":" << std::endl;
        std::cout << "  Index: " << inputs[i].index << std::endl;
        std::cout << "  Buffer size: " << inputs[i].size << " bytes" << std::endl;
        std::cout << "  Pass through: " << static_cast<int>(inputs[i].pass_through) << std::endl;
        std::cout << "  Data type: " << inputs[i].type << std::endl;
        std::cout << "  Data format: " << inputs[i].fmt << std::endl; //RKNN_TENSOR_NHWC
    }
    model.set_input(inputs);

    // 运行模型推理100次并计时
    const int num_runs = 20;
    std::chrono::duration<double, std::milli> total_time(0);

    for (int i = 0; i < num_runs; ++i) {
        auto start_time = std::chrono::high_resolution_clock::now();
        model.run();
        auto end_time = std::chrono::high_resolution_clock::now();
        total_time += end_time - start_time;
    }

    // 计算平均推理时间
    double average_time = total_time.count() / num_runs;
    std::cout << "Average inference time over " << num_runs << " runs: " << std::fixed << std::setprecision(10) << average_time << " ms" << std::endl;

    // 查询模型推理的逐层耗时
    rknn_perf_detail perf_detail;
    int ret = rknn_query(model.get_context(), RKNN_QUERY_PERF_DETAIL, &perf_detail, sizeof(perf_detail));
    if (ret == RKNN_SUCC) {
        std::cout << "Model inference layer-wise performance details (in microseconds, 2 decimal places):" << std::endl;
        std::cout << perf_detail.perf_data << std::endl;
    }
    else {
        std::cerr << "Failed to query performance details." << std::endl;
    }

    // 查询模型推理的总耗时
    rknn_perf_run perf_run;
    ret = rknn_query(model.get_context(), RKNN_QUERY_PERF_RUN, &perf_run, sizeof(perf_run));
    if (ret == RKNN_SUCC) {
        std::cout << "Total inference time (in milliseconds, 4 decimal places): " << std::fixed << std::setprecision(4) << static_cast<double>(perf_run.run_duration) / 1000.0 << " ms" << std::endl;
    }
    else {
        std::cerr << "Failed to query total inference time." << std::endl;
    }
    // 获取模型输出
    std::vector<rknn_output> outputs(output_attrs.size());
    for (size_t i = 0; i < output_attrs.size(); ++i) {
        outputs[i].index = i;
        outputs[i].is_prealloc = 0;
        outputs[i].want_float = 1;
    }
    model.get_output(outputs);

    // 打印输出张量形状
    std::cout << "Output Tensor Shapes:" << std::endl;
    for (const auto& attr : output_attrs) {
        std::cout << "  Name: " << attr.name << ", Shape: ";
        for (uint32_t j = 0; j < attr.n_dims; ++j) {
            std::cout << attr.dims[j] << " ";
        }
        std::cout << std::endl;
    }

    // 释放输出资源
    model.release_output(outputs);

    // 释放输入数据内存
    for (auto& input : inputs) {
        delete[] static_cast<float*>(input.buf);
    }
    return 0;
}
*/

#include <iostream>
#include <vector>
#include <cstring>
#include <chrono> // 用于计时
#include <iomanip> // 用于 setprecision
#include <opencv2/opencv.hpp> // 包含 OpenCV 头文件
#include "rknn_model.h"

int main() {
    // 初始化模型
    std::string model_path = "/root/.vs/rknn/yolov11s.rknn"; // yolov9c.rknn
    rknn_model model(model_path);

    // 查询模型信息
    int ctx_index = 0; // 假设使用第一个上下文
    // 打印量化和反量化信息
    // model.print_quantization_info();

    // 生成填充为0的640x640x3的图像
    cv::Mat input_image = cv::Mat::ones(640, 640, CV_8UC3);
    //printf("Start inference...");
    // 预热推理5次
    for (int i = 0; i < 5; ++i) {
        model.run_inference(input_image, ctx_index);
    }

    // 进行10次推理并统计时间
    std::vector<double> inference_times;
    for (int i = 0; i < 1; ++i) {
        auto start = std::chrono::high_resolution_clock::now();
        model.run_inference(input_image, ctx_index);
        auto end = std::chrono::high_resolution_clock::now();
        std::chrono::duration<double, std::milli> elapsed = end - start;
        inference_times.push_back(elapsed.count());
    }

    // 计算平均推理时间
    double total_time = 0.0;
    for (double time : inference_times) {
        total_time += time;
    }
    double average_time = total_time / inference_times.size();

    // 输出推理时间
    std::cout << "Function call time: " << std::fixed << std::setprecision(4) << average_time << " ms" << std::endl;



    // 打印输出向量的形状和大小  
    //std::cout << "Output vector shape: [" << output.size() << "]" << std::endl;

    // 打印平均推理时间，精确到毫秒，保留四位小数
    //std::cout << "Average inference time: " << std::fixed << std::setprecision(4) << average_time << " ms" << std::endl;





    // 打印输出向量的形状和大小
    //std::cout << "Output vector shape: [" << output.size() << "]" << std::endl;

    // 重塑输出数据为 [8400, 84]
    //std::vector<std::vector<float>> reshaped_output(8400, std::vector<float>(84));
    //std::memcpy(reshaped_output.data(), output.data(), 8400 * 84 * sizeof(float));

    // 查询模型推理的逐层耗时
    /*
    rknn_perf_detail perf_detail;
    int ret = rknn_query(model.get_context(ctx_index), RKNN_QUERY_PERF_DETAIL, &perf_detail, sizeof(perf_detail));
    if (ret == RKNN_SUCC) {
        std::cout << "Model inference layer-wise performance details (in microseconds, 2 decimal places):" << std::endl;
        std::cout << perf_detail.perf_data << std::endl;
    }
    else {
        std::cerr << "Failed to query performance details." << std::endl;
    }
    */
    // 查询模型推理的总耗时


    rknn_perf_run perf_run;
    int ret = rknn_query(model.get_context(ctx_index), RKNN_QUERY_PERF_RUN, &perf_run, sizeof(perf_run));
    if (ret == RKNN_SUCC) {
        std::cout << "Real inference time: " << std::fixed << std::setprecision(4) << static_cast<double>(perf_run.run_duration) / 1000.0 << " ms" << std::endl;
    }
    else {
        std::cerr << "Failed to query total inference time." << std::endl;
    }


    return 0;
}