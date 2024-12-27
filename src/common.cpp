#include "common.h"

/*
static void compute_dfl(float* tensor, int dfl_len, float* box) {
    for (int b = 0; b < 4; b++) {
        float exp_t[dfl_len];
        float exp_sum = 0;
        float acc_sum = 0;
        for (int i = 0; i < dfl_len; i++) {
            exp_t[i] = exp(tensor[i + b * dfl_len]);
            exp_sum += exp_t[i];
        }

        for (int i = 0; i < dfl_len; i++) {
            acc_sum += exp_t[i] / exp_sum * i;
        }
        box[b] = acc_sum;
    }
}
*/
// 优化后的函数
void compute_dfl(float* tensor, int dfl_len, float* box) {
    for (int b = 0; b < 4; b++) {
        float exp_sum = 0.0f;
        float acc_sum = 0.0f;
        // 提前计算 exp(tensor[i + b * dfl_len]) 和 exp_sum
        for (int i = 0; i < dfl_len; i++) {
            float exp_val = exp(tensor[i + b * dfl_len]);
            exp_sum += exp_val;
            acc_sum += exp_val * i; // 直接累加加权值
        }
        box[b] = acc_sum / exp_sum;
    }
}

// 定义一个静态函数 process_i8，其接收多个参数，包括：
// - box_tensor: 存储量化后的边界框数据的指针
// - box_zp, box_scale: 边界框数据的零点和比例因子
// - score_tensor: 存储量化后的得分数据的指针
// - score_zp, score_scale: 得分数据的零点和比例因子
// - score_sum_tensor: 存储量化后的得分总和数据的指针（可选）
// - score_sum_zp, score_sum_scale: 得分总和数据的零点和比例因子
// - grid_h, grid_w: 网格的高度和宽度
// - stride: 特征图相对于原始图像的步幅
// - dfl_len: 分布式焦点损失（Distributed Focal Loss）长度
// - boxes, objProbs, classId: 用于存储解码后的边界框、对象概率和类别ID的向量
// - threshold: 阈值，用于过滤低概率的预测
int process_i8(int8_t* box_tensor, int32_t box_zp, float box_scale,
    int8_t* score_tensor, int32_t score_zp, float score_scale,
    int8_t* score_sum_tensor, int32_t score_sum_zp, float score_sum_scale,
    int grid_h, int grid_w, int stride, int dfl_len,
    std::vector<float>& boxes,
    std::vector<float>& objProbs,
    std::vector<int>& classId,
    float threshold)
{
    // 初始化有效预测计数器
    int validCount = 0;

    // 计算网格单元总数
    int grid_len = grid_h * grid_w;

    // 将阈值转换为量化形式，以便与量化后的得分进行比较
    int8_t score_thres_i8 = qnt_f32_to_affine(threshold, score_zp, score_scale);
    int8_t score_sum_thres_i8 = qnt_f32_to_affine(threshold, score_sum_zp, score_sum_scale);

    // 提前计算 scale 的倒数，以优化后续的反量化操作
    float box_scale_inv = 1.0f / box_scale;
    float score_scale_inv = 1.0f / score_scale;

    // 使用 OpenMP 并行化循环，减少处理时间，并使用 reduction 来安全地累加 validCount
    //#pragma omp parallel for reduction(+:validCount)
    for (int i = 0; i < grid_h; i++)
    {
        for (int j = 0; j < grid_w; j++)
        {
            // 计算当前网格单元的偏移量
            int offset = i * grid_w + j;

            // 初始化最大类别ID为无效值
            int max_class_id = -1;

            // 如果提供了 score_sum_tensor，则利用它快速过滤掉低于阈值的网格单元
            if ((score_sum_tensor != nullptr) && (score_sum_tensor[offset] < score_sum_thres_i8)) {
                continue; // 如果得分总和低于阈值，则跳过此网格单元
            }

            // 初始化最大得分变量，确保首次比较总是成功的
            int8_t max_score = -score_zp;

            // 遍历所有类别，找到最高得分及其对应的类别ID
            for (int c = 0; c < OBJ_CLASS_NUM; c++) {
                // 检查当前类别的得分是否高于阈值且大于当前最大得分
                if ((score_tensor[offset] > score_thres_i8) && (score_tensor[offset] > max_score))
                {
                    // 更新最大得分和类别ID
                    max_score = score_tensor[offset];
                    max_class_id = c;
                }
                // 移动到下一个类别的得分位置
                offset += grid_len;
            }

            // 如果找到了有效的得分，则继续计算边界框
            if (max_score > score_thres_i8) {
                // 重置偏移量回到当前网格单元的位置
                offset = i * grid_w + j;

                // 创建临时数组来存储反量化之前的分布式焦点损失（Distributed Focal Loss）值
                float box[4];
                float before_dfl[dfl_len * 4];

                // 反量化并计算边界框的 DFL 值
                for (int k = 0; k < dfl_len * 4; k++) {
                    before_dfl[k] = deqnt_affine_to_f32(box_tensor[offset], box_zp, box_scale);
                    offset += grid_len;
                }
                compute_dfl(before_dfl, dfl_len, box); // 计算最终的边界框坐标

                // 计算边界框的真实坐标和尺寸
                float x1, y1, x2, y2, w, h;
                x1 = (-box[0] + j + 0.5) * stride;
                y1 = (-box[1] + i + 0.5) * stride;
                x2 = (box[2] + j + 0.5) * stride;
                y2 = (box[3] + i + 0.5) * stride;
                w = x2 - x1;
                h = y2 - y1;

                // 使用 OpenMP critical 区域来确保线程安全地访问共享资源
                //#pragma omp critical
                {
                    // 将计算出的边界框坐标、对象概率和类别ID添加到各自的向量中
                    boxes.push_back(x1);
                    boxes.push_back(y1);
                    boxes.push_back(w);
                    boxes.push_back(h);
                    objProbs.push_back(deqnt_affine_to_f32(max_score, score_zp, score_scale));
                    classId.push_back(max_class_id);

                    // 增加有效预测计数
                    validCount++;
                }
            }
        }
    }
    // 返回有效预测的数量
    return validCount;
}
/*
static int quick_sort_indice_inverse(std::vector<float>& input, int left, int right, std::vector<int>& indices)
{
    float key;
    int key_index;
    int low = left;
    int high = right;
    if (left < right)
    {
        key_index = indices[left];
        key = input[left];
        while (low < high)
        {
            while (low < high && input[high] <= key)
            {
                high--;
            }
            input[low] = input[high];
            indices[low] = indices[high];
            while (low < high && input[low] >= key)
            {
                low++;
            }
            input[high] = input[low];
            indices[high] = indices[low];
        }
        input[low] = key;
        indices[low] = key_index;
        quick_sort_indice_inverse(input, left, low - 1, indices);
        quick_sort_indice_inverse(input, low + 1, right, indices);
    }
    return low;
}
*/

// Function to sort the vector and its indices in descending order
void quick_sort_indice_inverse(std::vector<float>& input, std::vector<int>& indices) {
    size_t n = input.size();
    if (indices.empty()) {
        indices.resize(n);
        std::iota(indices.begin(), indices.end(), 0); // Initialize indices with 0, 1, ..., n-1
    }

    // Create a vector of pairs where each pair is (input[i], indices[i])
    std::vector<std::pair<float, int>> paired(n);
    for (size_t i = 0; i < n; ++i) {
        paired[i] = std::make_pair(input[i], indices[i]);
    }
    // Sort using std::sort with a custom comparator for descending order
    std::sort(paired.begin(), paired.end(), [](const std::pair<float, int>& a, const std::pair<float, int>& b) {
        return a.first > b.first;
        });
    // Extract the sorted elements back into the original vectors
    for (size_t i = 0; i < n; ++i) {
        input[i] = paired[i].first;
        indices[i] = paired[i].second;
    }
}



int nms(int validCount, std::vector<float>& outputLocations, std::vector<int> classIds, std::vector<int>& order,
    int filterId, float threshold)
{
    for (int i = 0; i < validCount; ++i)
    {
        int n = order[i];
        if (n == -1 || classIds[n] != filterId)
        {
            continue;
        }
        // 计算第i个检测框的边界坐标，避免重复计算
        float xmin0 = outputLocations[n * 4 + 0];
        float ymin0 = outputLocations[n * 4 + 1];
        float xmax0 = outputLocations[n * 4 + 0] + outputLocations[n * 4 + 2];
        float ymax0 = outputLocations[n * 4 + 1] + outputLocations[n * 4 + 3];
        
        for (int j = i + 1; j < validCount; ++j)
        {
            int m = order[j];
            if (m == -1 || classIds[m] != filterId)
            {
                continue;
            }

            float xmin1 = outputLocations[m * 4 + 0];
            float ymin1 = outputLocations[m * 4 + 1];
            float xmax1 = outputLocations[m * 4 + 0] + outputLocations[m * 4 + 2];
            float ymax1 = outputLocations[m * 4 + 1] + outputLocations[m * 4 + 3];

            float iou = CalculateOverlap(xmin0, ymin0, xmax0, ymax0, xmin1, ymin1, xmax1, ymax1);

            if (iou > threshold)
            {
                order[j] = -1; // 标记为抑制
            }
        }
    }
    return 0;
}