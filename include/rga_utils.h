#ifndef RGA_UTILS_H
#define RGA_UTILS_H

#include <opencv2/opencv.hpp>
#include <cstdint>
#include <iostream>
#include <rga/im2d.hpp>
#include <rga/rga.h>
#include "../librga/include/im2d.hpp"
#include "../librga/include/rga.h"
#include <cstring>
#include "common.h"

#ifdef __cplusplus
extern "C" {
#endif

// 优化后的 memcpy 函数，使用 NEON 指令集
void memcpy_neon1(void* dst, const void* src, size_t size);

// 裁剪图像为正方形并确保尺寸为16的倍数
void crop_image_to_square_and_16_alignment(cv::Mat& image);

// 裁剪图像确保尺寸为16的倍数
void crop_image_to_16_alignment(cv::Mat& image);

// 使用 RGA 进行裁剪，裁剪为正方形并确保尺寸为16的倍数
void crop_to_square_align16(const cv::Mat& src, cv::Mat& dst);

// 自适应裁剪填充函数，直接将结果写入目标虚拟地址
int adaptive_letterbox(const cv::Mat& src, int target_size, uint8_t* dst_virtual_addr, letterbox_t* letterbox, uint8_t fill_color, int interpolation = INTER_LINEAR);

// 使用 OpenCV 实现的自适应裁剪填充函数，直接将结果写入目标虚拟地址
int CV_adaptive_letterbox(const cv::Mat& src, int target_size, uint8_t* dst_virtual_addr, uint8_t fill_color, int interpolation = cv::INTER_LINEAR);


#ifdef __cplusplus
}  // extern "C"
#endif

#endif