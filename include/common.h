#ifndef _COMMON_H_
#define _COMMON_H_


#include <stdint.h>
#include <math.h>
#include <vector>
#include <omp.h>
#include <algorithm> // for std::sort
#include <numeric>
#include <stdlib.h>
#include <set>
//#include <cstdint>
#include <iostream>


#define OBJ_NAME_MAX_SIZE 64
#define OBJ_NUMB_MAX_SIZE 128
#define OBJ_CLASS_NUM 80
#define NMS_THRESH 0.45
#define BOX_THRESH 0.25

typedef struct {
    int x_pad;
    int y_pad;
    float scale;
} letterbox_t;

typedef struct {
    int left;
    int top;
    int right;
    int bottom;
} image_rect_t;

typedef struct {
    image_rect_t box;
    float prop;
    int cls_id;
} object_detect_result;

typedef struct {
    int id;
    int count;
    object_detect_result results[OBJ_NUMB_MAX_SIZE];
} object_detect_result_list;

inline int32_t __clip(float val, float min, float max)
{
    return (int32_t)(val <= min ? min : (val >= max ? max : val));
}

inline int8_t qnt_f32_to_affine(float f32, int32_t zp, float scale)
{
    float dst_val = (f32 / scale) + zp;
    return (int8_t)__clip(dst_val, -128.0f, 127.0f);
}

inline float deqnt_affine_to_f32(int8_t qnt, int32_t zp, float scale) { return ((float)qnt - (float)zp) * scale; }


void compute_dfl(float* tensor, int dfl_len, float* box);

int process_i8(int8_t* box_tensor, int32_t box_zp, float box_scale,
    int8_t* score_tensor, int32_t score_zp, float score_scale,
    int8_t* score_sum_tensor, int32_t score_sum_zp, float score_sum_scale,
    int grid_h, int grid_w, int stride, int dfl_len,
    std::vector<float>& boxes,
    std::vector<float>& objProbs,
    std::vector<int>& classId,
    float threshold);


// int quick_sort_indice_inverse(std::vector<float>& input, int left, int right, std::vector<int>& indices);
void quick_sort_indice_inverse(std::vector<float>& input, std::vector<int>& indices);

inline float CalculateOverlap(float xmin0, float ymin0, float xmax0, float ymax0, float xmin1, float ymin1, float xmax1,
    float ymax1)
{
    float w = fmax(0.f, fmin(xmax0, xmax1) - fmax(xmin0, xmin1) + 1.0);
    float h = fmax(0.f, fmin(ymax0, ymax1) - fmax(ymin0, ymin1) + 1.0);
    float i = w * h;
    float u = (xmax0 - xmin0 + 1.0) * (ymax0 - ymin0 + 1.0) + (xmax1 - xmin1 + 1.0) * (ymax1 - ymin1 + 1.0) - i;
    return u <= 0.f ? 0.f : (i / u);
}

int nms(int validCount, std::vector<float>& outputLocations, std::vector<int> classIds, std::vector<int>& order,
    int filterId, float threshold);

inline int clamp(float val, int min, int max) { return val > min ? (val < max ? val : max) : min; }

// 使用标准库的 clamp 函数，前提是你的编译器支持 C++17 或更高版本
/*
inline static int clamp(float val, int minVal, int maxVal) {
    // 首先检查 NaN 情况
    if (std::isnan(val)) {
        return minVal; // 或者可以选择其他行为
    }
    // 显式转换为 int 类型，并使用 std::clamp 来简化逻辑
    return static_cast<int>(std::clamp(val, static_cast<float>(minVal), static_cast<float>(maxVal)));
}
*/

#endif // 