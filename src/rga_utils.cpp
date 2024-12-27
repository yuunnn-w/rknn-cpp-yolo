#include "rga_utils.h"
//#include <arm_neon.h>

// 优化后的 memcpy 函数，使用 NEON 指令集
__attribute__((optimize("O3"))) void memcpy_neon1(void* dst, const void* src, size_t size) {
    size_t i;
    size_t simd_size = size / 128;     // 每次复制 128 个字节（16 字节 * 8 个向量）
    size_t remainder = size % 128;     // 剩余字节数

    uint8_t* dst_ptr = (uint8_t*)dst;
    const uint8_t* src_ptr = (const uint8_t*)src;

    // 预取数据到缓存中
    asm volatile("prfm pldl1keep, [%[address]]"
        :
    : [address] "r" (src_ptr)
        : );

    for (i = 0; i < simd_size; i++) {
        uint8x16_t q0, q1, q2, q3, q4, q5, q6, q7;
        // 从源地址加载 128 个字节
        asm volatile("ldp %q[q0], %q[q1], [%[src]], #32\n"
            "ldp %q[q2], %q[q3], [%[src]], #32\n"
            "ldp %q[q4], %q[q5], [%[src]], #32\n"
            "ldp %q[q6], %q[q7], [%[src]], #32\n"
            : [q0] "=w"(q0), [q1] "=w"(q1), [q2] "=w"(q2), [q3] "=w"(q3),
            [q4] "=w"(q4), [q5] "=w"(q5), [q6] "=w"(q6), [q7] "=w"(q7),
            [src] "+r"(src_ptr));

        // 将 128 个字节存储到目标地址
        asm volatile("stp %q[q0], %q[q1], [%[dst]], #32\n"
            "stp %q[q2], %q[q3], [%[dst]], #32\n"
            "stp %q[q4], %q[q5], [%[dst]], #32\n"
            "stp %q[q6], %q[q7], [%[dst]], #32\n"
            : [dst] "+r"(dst_ptr)
            : [q0] "w"(q0), [q1] "w"(q1), [q2] "w"(q2), [q3] "w"(q3),
            [q4] "w"(q4), [q5] "w"(q5), [q6] "w"(q6), [q7] "w"(q7));
    }

    // 处理剩余的字节
    memcpy(dst_ptr, src_ptr, remainder);
}

void crop_image_to_square_and_16_alignment(cv::Mat& image) {
    int origin_width = image.cols; // 获取原始图像的宽度和高度
    int origin_height = image.rows;

    // 计算短边尺寸
    int short_side = std::min(origin_width, origin_height);

    // 确保裁剪后的尺寸能被16整除
    int crop_width = (short_side / 16) * 16;
    int crop_height = (short_side / 16) * 16;

    // 如果裁剪后的尺寸与原图相同，直接返回
    if (crop_width == origin_width && crop_height == origin_height) {
        return;
    }

    // 计算裁剪区域的左上角坐标
    int crop_x = (origin_width - crop_width) / 2;
    int crop_y = (origin_height - crop_height) / 2;

    // 裁剪图像
    image = image(cv::Range(crop_y, crop_y + crop_height), cv::Range(crop_x, crop_x + crop_width));

    // 如果裁剪后的图像不是连续的，则克隆以确保内存连续
    if (!image.isContinuous()) {
        image = image.clone();
    }
}

void crop_image_to_16_alignment(cv::Mat& image) {
    int origin_width = image.cols; // 获取原始图像的宽度和高度
    int origin_height = image.rows;

    // 计算对齐后的宽度和高度，确保是16的倍数
    int aligned_width = (origin_width / 16) * 16;
    int aligned_height = (origin_height / 16) * 16;

    // 如果对齐后的尺寸与原图相同，直接返回
    if (aligned_width == origin_width && aligned_height == origin_height) {
        return;
    }

    // 计算裁剪区域的左上角坐标
    int crop_x = (origin_width - aligned_width) / 2;
    int crop_y = (origin_height - aligned_height) / 2;

    // 裁剪图像
    image = image(cv::Range(crop_y, crop_y + aligned_height), cv::Range(crop_x, crop_x + aligned_width));

    // 如果裁剪后的图像不是连续的，则克隆以确保内存连续
    if (!image.isContinuous()) {
        image = image.clone();
    }
}

void crop_to_square_align16(const cv::Mat& src, cv::Mat& dst) {
    // 获取源图像的宽度和高度
    int src_width = src.cols;
    int src_height = src.rows;

    // 计算短边尺寸
    int short_side = std::min(src_width, src_height);

    // 确保裁剪出的尺寸能被16整除
    //short_side = (short_side / 16) * 16;

    // 计算裁剪区域的左上角坐标
    int crop_x = (src_width - short_side) / 2;
    int crop_y = (src_height - short_side) / 2;

    // 使用RGA进行裁剪
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
 * @brief 自适应裁剪填充函数，直接将结果写入目标虚拟地址
 * @param src 源图像的 cv::Mat 指针
 * @param target_size 目标尺寸（正方形边长）
 * @param dst_virtual_addr 目标图像的虚拟地址指针
 * @param fill_color 填充颜色（RGB格式，例如114, 114, 114）
 * @param interpolation 插值方法（默认为INTER_LINEAR）
 * @return 成功返回1，失败返回-1
 */
int adaptive_letterbox(const cv::Mat& src, int target_size, uint8_t* dst_virtual_addr, letterbox_t* letterbox, uint8_t fill_color, int interpolation) {
    if (src.empty() || dst_virtual_addr == nullptr || target_size <= 0 || letterbox == nullptr) {
        std::cerr << "Invalid input parameters." << std::endl;
        return -1;
    }

    int src_width = src.cols;
    int src_height = src.rows;

    // 如果图像已经是正方形且尺寸与目标尺寸相同，直接返回
    if (src_width == target_size && src_height == target_size) {
        memcpy(dst_virtual_addr, src.data, target_size * target_size * 3);
        letterbox->scale = 1.0; // 缩放比例为1
        letterbox->x_pad = 0;   // 无填充
        letterbox->y_pad = 0;   // 无填充
        return 0;
    }

    // 计算缩放比例
    float scale = std::min(static_cast<float>(target_size) / src_width, static_cast<float>(target_size) / src_height);

    // 计算缩放后的尺寸
    int new_width = static_cast<int>(src_width * scale);
    int new_height = static_cast<int>(src_height * scale);

    // 如果缩放后的尺寸与目标尺寸相同，直接缩放，不需要填充
    if (new_width == target_size && new_height == target_size) {
        rga_buffer_t src_buf = wrapbuffer_virtualaddr(src.data, src_width, src_height, RK_FORMAT_RGB_888);
        rga_buffer_t dst_buf = wrapbuffer_virtualaddr(dst_virtual_addr, target_size, target_size, RK_FORMAT_RGB_888);

        IM_STATUS status = imresize(src_buf, dst_buf, scale, scale, interpolation);
        if (status != IM_STATUS_SUCCESS) {
            std::cerr << "RGA resize failed: " << imStrError(status) << std::endl;
            return -1;
        }
        letterbox->scale = scale; // 缩放比例
        letterbox->x_pad = 0;    // 无填充
        letterbox->y_pad = 0;    // 无填充
        return 0;
    }

    // 计算填充的黑边大小
    int pad_left = (target_size - new_width) / 2;
    int pad_top = (target_size - new_height) / 2;

    // 填充目标内存为指定颜色
    memset(dst_virtual_addr, fill_color, target_size * target_size * 3);

    // 使用RGA进行缩放
    rga_buffer_t src_buf = wrapbuffer_virtualaddr(src.data, src_width, src_height, RK_FORMAT_RGB_888);
    rga_buffer_t dst_buf = wrapbuffer_virtualaddr(dst_virtual_addr + (pad_top * target_size + pad_left) * 3, new_width, new_height, RK_FORMAT_RGB_888);

    IM_STATUS status = imresize(src_buf, dst_buf, scale, scale, interpolation);
    if (status != IM_STATUS_SUCCESS) {
        std::cerr << "RGA resize failed: " << imStrError(status) << std::endl;
        return -1;
    }
    // 赋值给letterbox_t结构体
    letterbox->scale = scale;       // 缩放比例
    letterbox->x_pad = pad_left;    // 左侧填充
    letterbox->y_pad = pad_top;     // 顶部填充

    return 0;
}

int CV_adaptive_letterbox(const cv::Mat& src, int target_size, uint8_t* dst_virtual_addr, uint8_t fill_color, int interpolation) {
    if (src.empty() || dst_virtual_addr == nullptr || target_size <= 0) {
        std::cerr << "Invalid input parameters." << std::endl;
        return -1;
    }

    int src_width = src.cols;
    int src_height = src.rows;

    // 如果图像已经是正方形且尺寸与目标尺寸相同，直接返回
    if (src_width == target_size && src_height == target_size) {
        memcpy(dst_virtual_addr, src.data, target_size * target_size * 3);
        return 0;
    }

    // 计算缩放比例
    float scale = std::min(static_cast<float>(target_size) / src_width, static_cast<float>(target_size) / src_height);

    // 计算缩放后的尺寸
    int new_width = static_cast<int>(src_width * scale);
    int new_height = static_cast<int>(src_height * scale);

    // 如果缩放后的尺寸与目标尺寸相同，直接缩放，不需要填充
    if (new_width == target_size && new_height == target_size) {
        cv::Mat resized_img;
        cv::resize(src, resized_img, cv::Size(target_size, target_size), 0, 0, interpolation);
        memcpy(dst_virtual_addr, resized_img.data, target_size * target_size * 3);
        return 0;
    }

    // 计算填充的黑边大小
    int pad_left = (target_size - new_width) / 2;
    int pad_top = (target_size - new_height) / 2;

    // 填充目标内存为指定颜色
    cv::Mat dst_img(target_size, target_size, CV_8UC3, cv::Scalar(fill_color, fill_color, fill_color));

    // 使用 OpenCV 进行缩放
    cv::Mat resized_img;
    cv::resize(src, resized_img, cv::Size(new_width, new_height), 0, 0, interpolation);

    // 将缩放后的图像复制到目标图像的中心位置
    resized_img.copyTo(dst_img(cv::Rect(pad_left, pad_top, new_width, new_height)));

    // 将结果复制到目标虚拟地址
    memcpy(dst_virtual_addr, dst_img.data, target_size * target_size * 3);

    return 0;
}