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
// �Ż���ĺ���
void compute_dfl(float* tensor, int dfl_len, float* box) {
    for (int b = 0; b < 4; b++) {
        float exp_sum = 0.0f;
        float acc_sum = 0.0f;
        // ��ǰ���� exp(tensor[i + b * dfl_len]) �� exp_sum
        for (int i = 0; i < dfl_len; i++) {
            float exp_val = exp(tensor[i + b * dfl_len]);
            exp_sum += exp_val;
            acc_sum += exp_val * i; // ֱ���ۼӼ�Ȩֵ
        }
        box[b] = acc_sum / exp_sum;
    }
}

// ����һ����̬���� process_i8������ն��������������
// - box_tensor: �洢������ı߽�����ݵ�ָ��
// - box_zp, box_scale: �߽�����ݵ����ͱ�������
// - score_tensor: �洢������ĵ÷����ݵ�ָ��
// - score_zp, score_scale: �÷����ݵ����ͱ�������
// - score_sum_tensor: �洢������ĵ÷��ܺ����ݵ�ָ�루��ѡ��
// - score_sum_zp, score_sum_scale: �÷��ܺ����ݵ����ͱ�������
// - grid_h, grid_w: ����ĸ߶ȺͿ��
// - stride: ����ͼ�����ԭʼͼ��Ĳ���
// - dfl_len: �ֲ�ʽ������ʧ��Distributed Focal Loss������
// - boxes, objProbs, classId: ���ڴ洢�����ı߽�򡢶�����ʺ����ID������
// - threshold: ��ֵ�����ڹ��˵͸��ʵ�Ԥ��
int process_i8(int8_t* box_tensor, int32_t box_zp, float box_scale,
    int8_t* score_tensor, int32_t score_zp, float score_scale,
    int8_t* score_sum_tensor, int32_t score_sum_zp, float score_sum_scale,
    int grid_h, int grid_w, int stride, int dfl_len,
    std::vector<float>& boxes,
    std::vector<float>& objProbs,
    std::vector<int>& classId,
    float threshold)
{
    // ��ʼ����ЧԤ�������
    int validCount = 0;

    // ��������Ԫ����
    int grid_len = grid_h * grid_w;

    // ����ֵת��Ϊ������ʽ���Ա���������ĵ÷ֽ��бȽ�
    int8_t score_thres_i8 = qnt_f32_to_affine(threshold, score_zp, score_scale);
    int8_t score_sum_thres_i8 = qnt_f32_to_affine(threshold, score_sum_zp, score_sum_scale);

    // ��ǰ���� scale �ĵ��������Ż������ķ���������
    float box_scale_inv = 1.0f / box_scale;
    float score_scale_inv = 1.0f / score_scale;

    // ʹ�� OpenMP ���л�ѭ�������ٴ���ʱ�䣬��ʹ�� reduction ����ȫ���ۼ� validCount
    //#pragma omp parallel for reduction(+:validCount)
    for (int i = 0; i < grid_h; i++)
    {
        for (int j = 0; j < grid_w; j++)
        {
            // ���㵱ǰ����Ԫ��ƫ����
            int offset = i * grid_w + j;

            // ��ʼ��������IDΪ��Чֵ
            int max_class_id = -1;

            // ����ṩ�� score_sum_tensor�������������ٹ��˵�������ֵ������Ԫ
            if ((score_sum_tensor != nullptr) && (score_sum_tensor[offset] < score_sum_thres_i8)) {
                continue; // ����÷��ܺ͵�����ֵ��������������Ԫ
            }

            // ��ʼ�����÷ֱ�����ȷ���״αȽ����ǳɹ���
            int8_t max_score = -score_zp;

            // ������������ҵ���ߵ÷ּ����Ӧ�����ID
            for (int c = 0; c < OBJ_CLASS_NUM; c++) {
                // ��鵱ǰ���ĵ÷��Ƿ������ֵ�Ҵ��ڵ�ǰ���÷�
                if ((score_tensor[offset] > score_thres_i8) && (score_tensor[offset] > max_score))
                {
                    // �������÷ֺ����ID
                    max_score = score_tensor[offset];
                    max_class_id = c;
                }
                // �ƶ�����һ�����ĵ÷�λ��
                offset += grid_len;
            }

            // ����ҵ�����Ч�ĵ÷֣����������߽��
            if (max_score > score_thres_i8) {
                // ����ƫ�����ص���ǰ����Ԫ��λ��
                offset = i * grid_w + j;

                // ������ʱ�������洢������֮ǰ�ķֲ�ʽ������ʧ��Distributed Focal Loss��ֵ
                float box[4];
                float before_dfl[dfl_len * 4];

                // ������������߽��� DFL ֵ
                for (int k = 0; k < dfl_len * 4; k++) {
                    before_dfl[k] = deqnt_affine_to_f32(box_tensor[offset], box_zp, box_scale);
                    offset += grid_len;
                }
                compute_dfl(before_dfl, dfl_len, box); // �������յı߽������

                // ����߽�����ʵ����ͳߴ�
                float x1, y1, x2, y2, w, h;
                x1 = (-box[0] + j + 0.5) * stride;
                y1 = (-box[1] + i + 0.5) * stride;
                x2 = (box[2] + j + 0.5) * stride;
                y2 = (box[3] + i + 0.5) * stride;
                w = x2 - x1;
                h = y2 - y1;

                // ʹ�� OpenMP critical ������ȷ���̰߳�ȫ�ط��ʹ�����Դ
                //#pragma omp critical
                {
                    // ��������ı߽�����ꡢ������ʺ����ID��ӵ����Ե�������
                    boxes.push_back(x1);
                    boxes.push_back(y1);
                    boxes.push_back(w);
                    boxes.push_back(h);
                    objProbs.push_back(deqnt_affine_to_f32(max_score, score_zp, score_scale));
                    classId.push_back(max_class_id);

                    // ������ЧԤ�����
                    validCount++;
                }
            }
        }
    }
    // ������ЧԤ�������
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
        // �����i������ı߽����꣬�����ظ�����
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
                order[j] = -1; // ���Ϊ����
            }
        }
    }
    return 0;
}