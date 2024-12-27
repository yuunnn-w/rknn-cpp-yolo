#include "rga_utils.h"
//#include <arm_neon.h>

// �Ż���� memcpy ������ʹ�� NEON ָ�
__attribute__((optimize("O3"))) void memcpy_neon1(void* dst, const void* src, size_t size) {
    size_t i;
    size_t simd_size = size / 128;     // ÿ�θ��� 128 ���ֽڣ�16 �ֽ� * 8 ��������
    size_t remainder = size % 128;     // ʣ���ֽ���

    uint8_t* dst_ptr = (uint8_t*)dst;
    const uint8_t* src_ptr = (const uint8_t*)src;

    // Ԥȡ���ݵ�������
    asm volatile("prfm pldl1keep, [%[address]]"
        :
    : [address] "r" (src_ptr)
        : );

    for (i = 0; i < simd_size; i++) {
        uint8x16_t q0, q1, q2, q3, q4, q5, q6, q7;
        // ��Դ��ַ���� 128 ���ֽ�
        asm volatile("ldp %q[q0], %q[q1], [%[src]], #32\n"
            "ldp %q[q2], %q[q3], [%[src]], #32\n"
            "ldp %q[q4], %q[q5], [%[src]], #32\n"
            "ldp %q[q6], %q[q7], [%[src]], #32\n"
            : [q0] "=w"(q0), [q1] "=w"(q1), [q2] "=w"(q2), [q3] "=w"(q3),
            [q4] "=w"(q4), [q5] "=w"(q5), [q6] "=w"(q6), [q7] "=w"(q7),
            [src] "+r"(src_ptr));

        // �� 128 ���ֽڴ洢��Ŀ���ַ
        asm volatile("stp %q[q0], %q[q1], [%[dst]], #32\n"
            "stp %q[q2], %q[q3], [%[dst]], #32\n"
            "stp %q[q4], %q[q5], [%[dst]], #32\n"
            "stp %q[q6], %q[q7], [%[dst]], #32\n"
            : [dst] "+r"(dst_ptr)
            : [q0] "w"(q0), [q1] "w"(q1), [q2] "w"(q2), [q3] "w"(q3),
            [q4] "w"(q4), [q5] "w"(q5), [q6] "w"(q6), [q7] "w"(q7));
    }

    // ����ʣ����ֽ�
    memcpy(dst_ptr, src_ptr, remainder);
}

void crop_image_to_square_and_16_alignment(cv::Mat& image) {
    int origin_width = image.cols; // ��ȡԭʼͼ��Ŀ�Ⱥ͸߶�
    int origin_height = image.rows;

    // ����̱߳ߴ�
    int short_side = std::min(origin_width, origin_height);

    // ȷ���ü���ĳߴ��ܱ�16����
    int crop_width = (short_side / 16) * 16;
    int crop_height = (short_side / 16) * 16;

    // ����ü���ĳߴ���ԭͼ��ͬ��ֱ�ӷ���
    if (crop_width == origin_width && crop_height == origin_height) {
        return;
    }

    // ����ü���������Ͻ�����
    int crop_x = (origin_width - crop_width) / 2;
    int crop_y = (origin_height - crop_height) / 2;

    // �ü�ͼ��
    image = image(cv::Range(crop_y, crop_y + crop_height), cv::Range(crop_x, crop_x + crop_width));

    // ����ü����ͼ���������ģ����¡��ȷ���ڴ�����
    if (!image.isContinuous()) {
        image = image.clone();
    }
}

void crop_image_to_16_alignment(cv::Mat& image) {
    int origin_width = image.cols; // ��ȡԭʼͼ��Ŀ�Ⱥ͸߶�
    int origin_height = image.rows;

    // ��������Ŀ�Ⱥ͸߶ȣ�ȷ����16�ı���
    int aligned_width = (origin_width / 16) * 16;
    int aligned_height = (origin_height / 16) * 16;

    // ��������ĳߴ���ԭͼ��ͬ��ֱ�ӷ���
    if (aligned_width == origin_width && aligned_height == origin_height) {
        return;
    }

    // ����ü���������Ͻ�����
    int crop_x = (origin_width - aligned_width) / 2;
    int crop_y = (origin_height - aligned_height) / 2;

    // �ü�ͼ��
    image = image(cv::Range(crop_y, crop_y + aligned_height), cv::Range(crop_x, crop_x + aligned_width));

    // ����ü����ͼ���������ģ����¡��ȷ���ڴ�����
    if (!image.isContinuous()) {
        image = image.clone();
    }
}

void crop_to_square_align16(const cv::Mat& src, cv::Mat& dst) {
    // ��ȡԴͼ��Ŀ�Ⱥ͸߶�
    int src_width = src.cols;
    int src_height = src.rows;

    // ����̱߳ߴ�
    int short_side = std::min(src_width, src_height);

    // ȷ���ü����ĳߴ��ܱ�16����
    //short_side = (short_side / 16) * 16;

    // ����ü���������Ͻ�����
    int crop_x = (src_width - short_side) / 2;
    int crop_y = (src_height - short_side) / 2;

    // ʹ��RGA���вü�
    rga_buffer_t src_buf = wrapbuffer_virtualaddr(src.data, src_width, src_height, RK_FORMAT_RGB_888);
    rga_buffer_t dst_buf = wrapbuffer_virtualaddr(dst.data, short_side, short_side, RK_FORMAT_RGB_888);

    im_rect crop_rect = { crop_x, crop_y, short_side, short_side };

    IM_STATUS status = imcrop(src_buf, dst_buf, crop_rect);
    if (status != IM_STATUS_SUCCESS) {
        std::cerr << "RGA crop failed: " << imStrError(status) << std::endl;
        return;
    }
}

/**
 * @brief ����Ӧ�ü���亯����ֱ�ӽ����д��Ŀ�������ַ
 * @param src Դͼ��� cv::Mat ָ��
 * @param target_size Ŀ��ߴ磨�����α߳���
 * @param dst_virtual_addr Ŀ��ͼ��������ַָ��
 * @param fill_color �����ɫ��RGB��ʽ������114, 114, 114��
 * @param interpolation ��ֵ������Ĭ��ΪINTER_LINEAR��
 * @return �ɹ�����1��ʧ�ܷ���-1
 */
int adaptive_letterbox(const cv::Mat& src, int target_size, uint8_t* dst_virtual_addr, letterbox_t* letterbox, uint8_t fill_color, int interpolation) {
    if (src.empty() || dst_virtual_addr == nullptr || target_size <= 0 || letterbox == nullptr) {
        std::cerr << "Invalid input parameters." << std::endl;
        return -1;
    }

    int src_width = src.cols;
    int src_height = src.rows;

    // ���ͼ���Ѿ����������ҳߴ���Ŀ��ߴ���ͬ��ֱ�ӷ���
    if (src_width == target_size && src_height == target_size) {
        memcpy(dst_virtual_addr, src.data, target_size * target_size * 3);
        letterbox->scale = 1.0; // ���ű���Ϊ1
        letterbox->x_pad = 0;   // �����
        letterbox->y_pad = 0;   // �����
        return 0;
    }

    // �������ű���
    float scale = std::min(static_cast<float>(target_size) / src_width, static_cast<float>(target_size) / src_height);

    // �������ź�ĳߴ�
    int new_width = static_cast<int>(src_width * scale);
    int new_height = static_cast<int>(src_height * scale);

    // ������ź�ĳߴ���Ŀ��ߴ���ͬ��ֱ�����ţ�����Ҫ���
    if (new_width == target_size && new_height == target_size) {
        rga_buffer_t src_buf = wrapbuffer_virtualaddr(src.data, src_width, src_height, RK_FORMAT_RGB_888);
        rga_buffer_t dst_buf = wrapbuffer_virtualaddr(dst_virtual_addr, target_size, target_size, RK_FORMAT_RGB_888);

        IM_STATUS status = imresize(src_buf, dst_buf, scale, scale, interpolation);
        if (status != IM_STATUS_SUCCESS) {
            std::cerr << "RGA resize failed: " << imStrError(status) << std::endl;
            return -1;
        }
        letterbox->scale = scale; // ���ű���
        letterbox->x_pad = 0;    // �����
        letterbox->y_pad = 0;    // �����
        return 0;
    }

    // �������ĺڱߴ�С
    int pad_left = (target_size - new_width) / 2;
    int pad_top = (target_size - new_height) / 2;

    // ���Ŀ���ڴ�Ϊָ����ɫ
    memset(dst_virtual_addr, fill_color, target_size * target_size * 3);

    // ʹ��RGA��������
    rga_buffer_t src_buf = wrapbuffer_virtualaddr(src.data, src_width, src_height, RK_FORMAT_RGB_888);
    rga_buffer_t dst_buf = wrapbuffer_virtualaddr(dst_virtual_addr + (pad_top * target_size + pad_left) * 3, new_width, new_height, RK_FORMAT_RGB_888);

    IM_STATUS status = imresize(src_buf, dst_buf, scale, scale, interpolation);
    if (status != IM_STATUS_SUCCESS) {
        std::cerr << "RGA resize failed: " << imStrError(status) << std::endl;
        return -1;
    }
    // ��ֵ��letterbox_t�ṹ��
    letterbox->scale = scale;       // ���ű���
    letterbox->x_pad = pad_left;    // ������
    letterbox->y_pad = pad_top;     // �������

    return 0;
}

int CV_adaptive_letterbox(const cv::Mat& src, int target_size, uint8_t* dst_virtual_addr, uint8_t fill_color, int interpolation) {
    if (src.empty() || dst_virtual_addr == nullptr || target_size <= 0) {
        std::cerr << "Invalid input parameters." << std::endl;
        return -1;
    }

    int src_width = src.cols;
    int src_height = src.rows;

    // ���ͼ���Ѿ����������ҳߴ���Ŀ��ߴ���ͬ��ֱ�ӷ���
    if (src_width == target_size && src_height == target_size) {
        memcpy(dst_virtual_addr, src.data, target_size * target_size * 3);
        return 0;
    }

    // �������ű���
    float scale = std::min(static_cast<float>(target_size) / src_width, static_cast<float>(target_size) / src_height);

    // �������ź�ĳߴ�
    int new_width = static_cast<int>(src_width * scale);
    int new_height = static_cast<int>(src_height * scale);

    // ������ź�ĳߴ���Ŀ��ߴ���ͬ��ֱ�����ţ�����Ҫ���
    if (new_width == target_size && new_height == target_size) {
        cv::Mat resized_img;
        cv::resize(src, resized_img, cv::Size(target_size, target_size), 0, 0, interpolation);
        memcpy(dst_virtual_addr, resized_img.data, target_size * target_size * 3);
        return 0;
    }

    // �������ĺڱߴ�С
    int pad_left = (target_size - new_width) / 2;
    int pad_top = (target_size - new_height) / 2;

    // ���Ŀ���ڴ�Ϊָ����ɫ
    cv::Mat dst_img(target_size, target_size, CV_8UC3, cv::Scalar(fill_color, fill_color, fill_color));

    // ʹ�� OpenCV ��������
    cv::Mat resized_img;
    cv::resize(src, resized_img, cv::Size(new_width, new_height), 0, 0, interpolation);

    // �����ź��ͼ���Ƶ�Ŀ��ͼ�������λ��
    resized_img.copyTo(dst_img(cv::Rect(pad_left, pad_top, new_width, new_height)));

    // ��������Ƶ�Ŀ�������ַ
    memcpy(dst_virtual_addr, dst_img.data, target_size * target_size * 3);

    return 0;
}