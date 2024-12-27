#include "rknn_model.h"
#include "rga_utils.h"
#include "common.h"
//

rknn_model::rknn_model(const std::string& model_path){
    init_model(model_path);
    int ctx_index = 0;
    query_model_info(ctx_index);
    initialize_mems();
}

rknn_model::~rknn_model() {
    release_mems();
    release_model();
}

static void dump_tensor_attr(rknn_tensor_attr* attr) {
    char dims[128] = { 0 };
    for (int i = 0; i < attr->n_dims; ++i) {
        int idx = strlen(dims);
        sprintf(&dims[idx], "%d%s", attr->dims[i], (i == attr->n_dims - 1) ? "" : ", ");
    }
    printf("  index=%d, name=%s, n_dims=%d, dims=[%s], n_elems=%d, size=%d, w_stride = %d, size_with_stride = %d, "
        "fmt=%s, type=%s, qnt_type=%s, "
        "zp=%d, scale=%f\n",
        attr->index, attr->name, attr->n_dims, dims, attr->n_elems, attr->size, attr->w_stride, attr->size_with_stride,
        get_format_string(attr->fmt), get_type_string(attr->type), get_qnt_type_string(attr->qnt_type), attr->zp,
        attr->scale);
}

void rknn_model::query_model_info(int& ctx_index) {
    // 1. 查询 SDK 版本并打印
    rknn_sdk_version sdk_version;
    int ret = rknn_query(ctxs[ctx_index], RKNN_QUERY_SDK_VERSION, &sdk_version, sizeof(sdk_version));
    if (ret < 0) {
        std::cerr << "rknn_query RKNN_QUERY_SDK_VERSION error ret=" << ret << std::endl;
        return;
    }
    std::cout << "SDK API Version: " << sdk_version.api_version << std::endl; // 2.3.0 (c949ad889d@2024-11-07T11:35:33)
    std::cout << "Driver Version: " << sdk_version.drv_version << std::endl; // 0.9.7

    // 2. 查询 RKNN 模型里面的用户自定义字符串信息并打印
    rknn_custom_string custom_string;
    ret = rknn_query(ctxs[ctx_index], RKNN_QUERY_CUSTOM_STRING, &custom_string, sizeof(custom_string));
    if (ret < 0) {
        std::cerr << "rknn_query RKNN_QUERY_CUSTOM_STRING error ret=" << ret << std::endl; // yolov9c
        return;
    }
    std::cout << "Custom String: " << custom_string.string << std::endl;

    // 3. 查询模型输入、输出 Tensor 的个数、维度、形状、名称、数据类型、量化类型并打印
    //rknn_input_output_num io_num;
    ret = rknn_query(ctxs[ctx_index], RKNN_QUERY_IN_OUT_NUM, &io_num, sizeof(io_num));
    if (ret < 0) {
        std::cerr << "rknn_query RKNN_QUERY_IN_OUT_NUM error ret=" << ret << std::endl;
        return;
    }
    std::cout << "Model input num: " << io_num.n_input << ", output num: " << io_num.n_output << std::endl; // 1

    // 查询常规输入 Tensor 的属性
    input_attrs.resize(io_num.n_input);
    for (uint32_t i = 0; i < io_num.n_input; ++i) {
        input_attrs[i].index = i;
        ret = rknn_query(ctxs[ctx_index], RKNN_QUERY_INPUT_ATTR, &input_attrs[i], sizeof(rknn_tensor_attr));
        if (ret != RKNN_SUCC) {
            std::cerr << "rknn_query RKNN_QUERY_INPUT_ATTR error ret=" << ret << std::endl;
            return;
        }

    }
	// 查询硬件最优的输入Tensor属性
	input_native_attrs.resize(io_num.n_input);
	for (uint32_t i = 0; i < io_num.n_input; ++i) {
		input_native_attrs[i].index = i;
		ret = rknn_query(ctxs[ctx_index], RKNN_QUERY_NATIVE_INPUT_ATTR, &input_native_attrs[i], sizeof(rknn_tensor_attr));
		if (ret != RKNN_SUCC) {
			std::cerr << "rknn_query RKNN_QUERY_NATIVE_INPUT_ATTR ret=" << ret << std::endl;
			return;
		}
		std::cout << "Input Tensor " << i << " info:" << std::endl;
        dump_tensor_attr(&(input_native_attrs[i]));
	}


    // 查询常规输出 Tensor 的属性
    output_attrs.resize(io_num.n_output);
    for (uint32_t i = 0; i < io_num.n_output; ++i) {
        output_attrs[i].index = i;
        ret = rknn_query(ctxs[ctx_index], RKNN_QUERY_OUTPUT_ATTR, &output_attrs[i], sizeof(rknn_tensor_attr)); //RKNN_QUERY_NATIVE_OUTPUT_ATTR
        if (ret != RKNN_SUCC) {
            std::cerr << "rknn_query RKNN_QUERY_OUTPUT_ATTR ret=" << ret << std::endl;
            return;
        }
    }

    // 查询硬件最优的输出Tensor属性
	output_native_attrs.resize(io_num.n_output);
	for (uint32_t i = 0; i < io_num.n_output; ++i) {
		output_native_attrs[i].index = i;
		ret = rknn_query(ctxs[ctx_index], RKNN_QUERY_NATIVE_OUTPUT_ATTR, &output_native_attrs[i], sizeof(rknn_tensor_attr));
		if (ret < 0) {
			std::cerr << "rknn_query RKNN_QUERY_NATIVE_OUTPUT_ATTR ret=" << ret << std::endl;
			return;
		}
		std::cout << "Output Tensor " << i << " info:" << std::endl;
        dump_tensor_attr(&(output_native_attrs[i]));
    }

    if (input_attrs[0].fmt == RKNN_TENSOR_NCHW) {
        printf("model is NCHW input fmt\n");
        model_channel = input_attrs[0].dims[1];
        model_height = input_attrs[0].dims[2];
        model_width = input_attrs[0].dims[3];
    }
    else {
        printf("model is NHWC input fmt\n");
        model_height = input_attrs[0].dims[1];
        model_width = input_attrs[0].dims[2];
        model_channel = input_attrs[0].dims[3];
    }
    printf("model input height=%d, width=%d, channel=%d\n",
        model_height, model_width, model_channel);


    num_outputs = io_num.n_output;
}

void rknn_model::init_model(const std::string& model_path) {
    
    FILE* fp = fopen(model_path.c_str(), "rb"); // 打开模型文件
    if (fp == nullptr) {
        std::cerr << "Failed to open model file: " << model_path << std::endl;
        return;
    }
    fseek(fp, 0, SEEK_END); // 将文件指针移动到文件末尾，以便获取文件大小
    size_t model_size = ftell(fp); // 获取当前文件指针的位置，即文件大小
    fseek(fp, 0, SEEK_SET); // 将文件指针重新移动到文件开头，以便从头开始读取文件内容

    double model_size_mb = static_cast<double>(model_size) / (1024 * 1024);
    std::cout << "Model size: " << model_size_mb << " MB" << std::endl; // 打印模型参数大小（以MB为单位）

    unsigned char* model_data = new unsigned char[model_size]; // 分配一块内存，大小为文件大小，用于存储模型数据
    if (model_data == nullptr) {
        std::cerr << "Failed to allocate memory for model data." << std::endl;
        fclose(fp);
        return;
    }

    // 读取模型数据
    size_t items_read = fread(model_data, 1, model_size, fp);
    if (items_read != model_size) {
        std::cerr << "Failed to read the entire model file." << std::endl;
        delete[] model_data;
        fclose(fp);
        return;
    }
    fclose(fp);

    // 初始化rknn上下文，并设置多个标识
    int ret = rknn_init(&ctxs[0], model_data, model_size,
        //RKNN_FLAG_MEM_ALLOC_OUTSIDE | // 用于表示模型输入、输出、权重、中间tensor内存全部由用户分配
        RKNN_FLAG_COLLECT_PERF_MASK |  // 用于运行时查询网络各层时间
        // RKNN_FLAG_EXECUTE_FALLBACK_PRIOR_DEVICE_GPU |  // 表示所有NPU不支持的层优先选择运行在GPU上
        RKNN_FLAG_ENABLE_SRAM,  // 表示中间tensor内存尽可能分配在SRAM上
        NULL);
    if (ret < 0) {
        std::cerr << "rknn_init error ret=" << ret << std::endl;
        delete[] model_data;
        return;
    }

    // 复制第一个上下文到其余上下文
    for (int i = 1; i < 3; ++i) {
        ret = rknn_dup_context(&ctxs[0], &ctxs[i]);
        if (ret < 0) {
            std::cerr << "Failed to duplicate context to ctx" << i << ". ret=" << ret << std::endl;
            delete[] model_data;
            return;
        }
    }
    // 设置NPU核心掩码
    rknn_set_core_mask(ctxs[0], RKNN_NPU_CORE_0); //设置模型运行在0核心上
    rknn_set_core_mask(ctxs[1], RKNN_NPU_CORE_1); //设置模型运行在1核心上
    rknn_set_core_mask(ctxs[2], RKNN_NPU_CORE_2); //设置模型运行在2核心上
    /*
    if (ret < 0) {
        std::cerr << "Failed to set core mask. ret=" << ret << std::endl;
        delete[] model_data;
        return;
    }
    */
    delete[] model_data;
    return;
}

void rknn_model::release_model() {
    for (int i = 1; i < 3; ++i) {
        int ret = rknn_destroy(ctxs[i]);
        if (ret < 0) {
            std::cerr << "rknn_destroy error ret=" << ret << std::endl;
        }
    }
}


// 实现打印量化和反量化参数
void rknn_model::print_quantization_info() const {
    std::cout << "Quantization and Dequantization Information:" << std::endl;

    // 打印输入张量的量化和反量化信息
    std::cout << "Input Tensors:" << std::endl;
    for (const auto& attr : input_attrs) {
        std::cout << "  Name: " << attr.name << std::endl;
        if (attr.qnt_type == RKNN_TENSOR_QNT_AFFINE_ASYMMETRIC) {
            std::cout << "    Quantization Type: Affine Asymmetric" << std::endl;
            std::cout << "    Quantization Formula: Q = (R / scale) + zero_point" << std::endl;
            std::cout << "    Dequantization Formula: R = (Q - zero_point) * scale" << std::endl;
            std::cout << "    Scale: " << attr.scale << std::endl;
            std::cout << "    Zero Point: " << attr.zp << std::endl; // 使用 zp 而不是 fl
        }
        else if (attr.qnt_type == RKNN_TENSOR_QNT_DFP) {
            std::cout << "    Quantization Type: Dynamic Fixed Point" << std::endl;
            std::cout << "    Quantization Formula: Q = R * (2^(-fl))" << std::endl;
            std::cout << "    Dequantization Formula: R = Q * (2^(fl))" << std::endl;
            std::cout << "    Fraction Length (fl): " << static_cast<int>(attr.fl) << std::endl;
        }
        else {
            std::cout << "    Quantization Type: None" << std::endl;
        }
    }

    // 打印输出张量的量化和反量化信息
    std::cout << "Output Tensors:" << std::endl;
    for (const auto& attr : output_attrs) {
        std::cout << "  Name: " << attr.name << std::endl;
        if (attr.qnt_type == RKNN_TENSOR_QNT_AFFINE_ASYMMETRIC) {
            std::cout << "    Quantization Type: Affine Asymmetric" << std::endl;
            std::cout << "    Quantization Formula: Q = (R / scale) + zero_point" << std::endl;
            std::cout << "    Dequantization Formula: R = (Q - zero_point) * scale" << std::endl;
            std::cout << "    Scale: " << attr.scale << std::endl;
            std::cout << "    Zero Point: " << attr.zp << std::endl;
        }
        else if (attr.qnt_type == RKNN_TENSOR_QNT_DFP) {
            std::cout << "    Quantization Type: Dynamic Fixed Point" << std::endl;
            std::cout << "    Quantization Formula: Q = R * (2^(-fl))" << std::endl;
            std::cout << "    Dequantization Formula: R = Q * (2^(fl))" << std::endl;
            std::cout << "    Fraction Length (fl): " << static_cast<int>(attr.fl) << std::endl;
        }
        else {
            std::cout << "    Quantization Type: None" << std::endl;
        }
    }
    return;
}

// 根据索引获取指定的 input_mems 成员指针
rknn_tensor_mem* rknn_model::get_input_mem_ptr(int ctx_index,int mem_index) {
    if (ctx_index < 0 || ctx_index >= static_cast<int>(input_mems.size())) {
        // 如果索引超出范围，可以抛出异常或返回 nullptr
        throw std::out_of_range("Index out of range for input_mems vector.");
        // 或者 return nullptr;
    }
    if(mem_index < 0 || mem_index >= static_cast<int>(input_mems[ctx_index].size())) {
		throw std::out_of_range("Index out of range for input_mems[ctx_index] vector.");
	}
    return input_mems[ctx_index][mem_index];
}

void rknn_model::initialize_mems() {
    int ctx_size = 3;
    input_mems.resize(ctx_size);
    output_mems.resize(ctx_size);
    // 这里设置输入输出的属性
    for (size_t i = 0; i < ctx_size; ++i) {
        input_mems[i].resize(io_num.n_input);
        output_mems[i].resize(io_num.n_output);
        //std::cerr << "Input size:" << input_attrs[0].size_with_stride << std::endl;
        for (size_t j = 0; j < io_num.n_input; ++j) {
            // default input type is int8 (normalize and quantize need compute in outside)
            // if set uint8, will fuse normalize and quantize to npu
            //input_attrs[j].pass_through = 0;
            input_native_attrs[j].type = RKNN_TENSOR_UINT8;
            //input_attrs[j].fmt = RKNN_TENSOR_NHWC;

            input_mems[i][j] = rknn_create_mem(ctxs[i], input_native_attrs[j].size_with_stride);
            if (!input_mems[i][j]) {
                throw std::runtime_error("rknn_create_mem failed for input");
            }
            // 设置输入内存
            int ret = rknn_set_io_mem(ctxs[i], input_mems[i][j], &input_native_attrs[j]);
            if (ret < 0) {
                printf("input_mems rknn_set_io_mem fail! ret=%d\n", ret);
            }
        }
        //std::cerr << "OutPut size:" << output_attrs[0].size_with_stride << std::endl;
        //std::cerr << "OutPut n_elems:" << output_attrs[0].n_elems << std::endl;
        for (size_t j = 0; j < io_num.n_output; ++j) {
            //output_attrs[j].type = RKNN_TENSOR_FLOAT32; //RKNN_TENSOR_INT8 输出改成int8，需要手动反量化
            //output_attrs[j].fmt = RKNN_TENSOR_NC1HWC2;
            output_mems[i][j] = rknn_create_mem(ctxs[i], output_native_attrs[j].size_with_stride);//.n_elems); //*sizeof(float)
            if (!output_mems[i][j]) {
                throw std::runtime_error("rknn_create_mem failed for output");
            }
            // 设置输出内存
            rknn_set_io_mem(ctxs[i], output_mems[i][j], &output_native_attrs[j]);
        }
    }
    // RKNN_TENSOR_UINT8;


}

void rknn_model::release_mems() {
    int ctx_size = 3;
    // 释放输入输出内存
    for (size_t i = 0; i < ctx_size; ++i) {
        // 遍历所有输入内存
        for (size_t j = 0; j < io_num.n_input; ++j) {
            if (input_mems[i][j] != nullptr && input_mems[i][j]->virt_addr != nullptr) { // 检查指针和虚拟地址是否为nullptr
                rknn_destroy_mem(ctxs[i], input_mems[i][j]);
                //delete input_mems[i][j]; // 如果是动态分配的，请确保这里使用正确的释放方式
                input_mems[i][j] = nullptr; // 确保释放后将指针设为nullptr
            }
        }

        // 遍历所有输出内存
        for (size_t j = 0; j < io_num.n_output; ++j) {
            if (output_mems[i][j] != nullptr && output_mems[i][j]->virt_addr != nullptr) { // 检查指针和虚拟地址是否为nullptr
                rknn_destroy_mem(ctxs[i], output_mems[i][j]);
                //delete output_mems[i][j]; // 如果是动态分配的，请确保这里使用正确的释放方式
                output_mems[i][j] = nullptr; // 确保释放后将指针设为nullptr
            }
        }
    }
}

// 优化后的函数
int NC1HWC2_i8_to_NCHW_i8(const int8_t* src, int8_t* dst, int* dims, int channel, int h, int w, int zp, float scale) {
    int batch = dims[0];
    int C1 = dims[1];
    int C2 = dims[4];
    int hw_src = dims[2] * dims[3];
    int hw_dst = h * w;

    for (int i = 0; i < batch; i++) {
        const int8_t* src_b = src + i * C1 * hw_src * C2;
        int8_t* dst_b = dst + i * channel * hw_dst;
        for (int c = 0; c < channel; ++c) {
            int           plane = c / C2;
            const int8_t* src_bc = src_b + plane * hw_src * C2;
            int           offset = c % C2;
            // Copy data from src to dst in a contiguous manner
            for (int cur_hw = 0; cur_hw < hw_dst; ++cur_hw) {
                dst_b[c * hw_dst + cur_hw] = src_bc[C2 * cur_hw + offset];
            }
        }
    }
    return 0;
}
/*
int rknn_model::run_inference(cv::Mat& input_image, int& ctx_index, object_detect_result_list* od_results) {
    if (input_image.empty()) {
        std::cerr << "Input image is empty!" << std::endl;
        return {};
    }
    crop_image_to_square_and_16_alignment(input_image);
    uint8_t fill_color = 114;

    letterbox_t letter_box;
    memset(&letter_box, 0, sizeof(letterbox_t));

    // 使用 RGA 进行自适应裁剪填充
    int result = adaptive_letterbox(input_image, 640, (uint8_t*)input_mems[ctx_index][0]->virt_addr, &letter_box, fill_color, INTER_LINEAR);
    if (result != 0) {
        std::cerr << "Failed to letterbox the image!" << std::endl;
        return {};
    }

    result = rknn_run(ctxs[ctx_index], NULL);
    if (result != 0) {
        std::cerr << "rknn_run fail! ret=" << result << std::endl;
        return {};
    }

    rknn_output outputs[io_num.n_output];
    memset(outputs, 0, sizeof(outputs));

    for (uint32_t i = 0; i < io_num.n_output; ++i) {
        int channel = output_attrs[i].dims[1];
        int h = output_attrs[i].n_dims > 2 ? output_attrs[i].dims[2] : 1;
        int w = output_attrs[i].n_dims > 3 ? output_attrs[i].dims[3] : 1;
        int hw = h * w;
        int zp = output_native_attrs[i].zp;
        float scale = output_native_attrs[i].scale;

        outputs[i].size = output_native_attrs[i].n_elems * sizeof(int8_t);
        outputs[i].buf = (int8_t*)malloc(outputs[i].size);

        // NC1HWC2 格式转换为 NCHW 格式
        if (output_native_attrs[i].fmt == RKNN_TENSOR_NC1HWC2) {
            NC1HWC2_i8_to_NCHW_i8((int8_t*)output_mems[ctx_index][i]->virt_addr, (int8_t*)outputs[i].buf,
                (int*)output_native_attrs[i].dims, channel, h, w, zp, scale);
        }
        else {
            memcpy(outputs[i].buf, output_mems[ctx_index][i]->virt_addr, outputs[i].size);
        }
    }
    const float nms_threshold = NMS_THRESH;      // 默认的NMS阈值
    const float box_conf_threshold = BOX_THRESH; // 默认的置信度阈值

    memset(od_results, 0x00, sizeof(*od_results));

    // Post Process
    post_process(outputs, &letter_box, box_conf_threshold, nms_threshold, od_results);

    for (int i = 0; i < io_num.n_output; i++) {
        free(outputs[i].buf);
    }

    return 0;
}
*/

int rknn_model::run_inference(cv::Mat& input_image, int& ctx_index, object_detect_result_list* od_results) {
    using namespace std::chrono;

    auto start = high_resolution_clock::now();
    if (input_image.empty()) {
        std::cerr << "Input image is empty!" << std::endl;
        return -1;
    }
    uint8_t fill_color = 114;

    letterbox_t letter_box;
    memset(&letter_box, 0, sizeof(letterbox_t));

    // 使用 RGA 进行自适应裁剪填充
    auto crop_start = high_resolution_clock::now();
    crop_image_to_square_and_16_alignment(input_image);
    auto crop_end = high_resolution_clock::now();

    auto letterbox_start = high_resolution_clock::now();
    int result = adaptive_letterbox(input_image, 640, (uint8_t*)input_mems[ctx_index][0]->virt_addr, &letter_box, fill_color, INTER_LINEAR);
    auto letterbox_end = high_resolution_clock::now();
    if (result != 0) {
        std::cerr << "Failed to letterbox the image!" << std::endl;
        return -1;
    }

    auto inference_start = high_resolution_clock::now();
    result = rknn_run(ctxs[ctx_index], NULL);
    auto inference_end = high_resolution_clock::now();
    if (result != 0) {
        std::cerr << "rknn_run fail! ret=" << result << std::endl;
        return -1;
    }

    rknn_output outputs[io_num.n_output];
    memset(outputs, 0, sizeof(outputs));

    auto post_process_start = high_resolution_clock::now();
    for (uint32_t i = 0; i < io_num.n_output; ++i) {
        int channel = output_attrs[i].dims[1];
        int h = output_attrs[i].n_dims > 2 ? output_attrs[i].dims[2] : 1;
        int w = output_attrs[i].n_dims > 3 ? output_attrs[i].dims[3] : 1;
        int hw = h * w;
        int zp = output_native_attrs[i].zp;
        float scale = output_native_attrs[i].scale;

        outputs[i].size = output_native_attrs[i].n_elems * sizeof(int8_t);
        outputs[i].buf = (int8_t*)malloc(outputs[i].size);

        // NC1HWC2 格式转换为 NCHW 格式
        if (output_native_attrs[i].fmt == RKNN_TENSOR_NC1HWC2) {
            NC1HWC2_i8_to_NCHW_i8((int8_t*)output_mems[ctx_index][i]->virt_addr, (int8_t*)outputs[i].buf,
                (int*)output_native_attrs[i].dims, channel, h, w, zp, scale);
        }
        else {
            memcpy(outputs[i].buf, output_mems[ctx_index][i]->virt_addr, outputs[i].size);
        }
    }

    const float nms_threshold = NMS_THRESH;      // 默认的NMS阈值
    const float box_conf_threshold = BOX_THRESH; // 默认的置信度阈值

    memset(od_results, 0x00, sizeof(*od_results));

    // Post Process
    post_process(outputs, &letter_box, box_conf_threshold, nms_threshold, od_results);

    auto post_process_end = high_resolution_clock::now();

    auto end = high_resolution_clock::now();
    duration<double, std::milli> total_time_ms = end - start;
    duration<double, std::milli> crop_time_ms = crop_end - crop_start;
    duration<double, std::milli> letterbox_time_ms = letterbox_end - letterbox_start;
    duration<double, std::milli> inference_time_ms = inference_end - inference_start;
    duration<double, std::milli> post_process_time_ms = post_process_end - post_process_start;

    // 打印每个步骤的时间和所占比例，精确到毫秒
    std::cout << std::fixed << std::setprecision(6);

    std::cout << "\nPerformance Analysis:" << std::endl;
    std::cout << "Total Time: " << total_time_ms.count() << " ms" << std::endl;

    // 保存当前精度设置
    auto old_precision = std::cout.precision();

    // 输出每个步骤的时间和所占比例，其中时间保留10位小数，百分比保留两位小数
    std::cout << "Crop Image: " << crop_time_ms.count() << " ms ("
        << std::setprecision(2) << (crop_time_ms / total_time_ms * 100.0) << "%"
        << std::setprecision(old_precision) << ")" << std::endl;

    std::cout << "Letterbox Image: " << letterbox_time_ms.count() << " ms ("
        << std::setprecision(2) << (letterbox_time_ms / total_time_ms * 100.0) << "%"
        << std::setprecision(old_precision) << ")" << std::endl;

    std::cout << "Inference: " << inference_time_ms.count() << " ms ("
        << std::setprecision(2) << (inference_time_ms / total_time_ms * 100.0) << "%"
        << std::setprecision(old_precision) << ")" << std::endl;

    std::cout << "Post Process: " << post_process_time_ms.count() << " ms ("
        << std::setprecision(2) << (post_process_time_ms / total_time_ms * 100.0) << "%"
        << std::setprecision(old_precision) << ")" << std::endl;

    for (int i = 0; i < io_num.n_output; i++) {
        free(outputs[i].buf);
    }

    return 0;
}

int rknn_model::post_process(void* outputs, letterbox_t* letter_box, float conf_threshold, float nms_threshold, object_detect_result_list* od_results)
{
    // 将传入的输出指针转换为rknn_output类型，方便后续操作
    rknn_output* _outputs = (rknn_output*)outputs;

    // 定义用于存储过滤后的边界框、对象概率和类别ID的向量
    std::vector<float> filterBoxes;
    std::vector<float> objProbs;
    std::vector<int> classId;
    int validCount = 0; // 有效检测框的数量
    int stride = 0; // 步长，用于计算网格大小
    int grid_h = 0; // 网格高度
    int grid_w = 0; // 网格宽度
    int model_in_w = model_width; // 模型输入宽度
    int model_in_h = model_height; // 模型输入高度

    // 初始化od_results结构体，确保其内容为0
    //memset(od_results, 0, sizeof(object_detect_result_list));


    // 默认有3个输出分支
    int dfl_len = output_attrs[0].dims[1] / 4; // 计算每个分支的DFL长度
    int output_per_branch = io_num.n_output / 3; // 每个分支的输出数量

    /*
    // 预估可能的最大有效检测框数量
    int max_valid_boxes_per_branch = 0;
    for (int i = 0; i < 3; i++) {
        int grid_h = output_attrs[i * output_per_branch].dims[2];
        int grid_w = output_attrs[i * output_per_branch].dims[3];
        max_valid_boxes_per_branch += grid_h * grid_w;
    }

    // 预分配空间以容纳最大可能的有效检测框数量
    filterBoxes.reserve(max_valid_boxes_per_branch * 4); // 每个框有4个坐标值
    objProbs.reserve(max_valid_boxes_per_branch);        // 每个框有一个置信度
    classId.reserve(max_valid_boxes_per_branch);         // 每个框有一个类别ID
    */

    // #pragma omp parallel for schedule(auto) reduction(+:validCount)
    for (int i = 0; i < 3; i++)
    {
        void* score_sum = nullptr;
        int32_t score_sum_zp = 0;
        float score_sum_scale = 1.0;
        // 如果每个分支有3个输出，则获取score_sum的缓冲区、零点和缩放因子
        if (output_per_branch == 3) {
            score_sum = _outputs[i * output_per_branch + 2].buf;
            score_sum_zp = output_attrs[i * output_per_branch + 2].zp;
            score_sum_scale = output_attrs[i * output_per_branch + 2].scale;
        }
        int box_idx = i * output_per_branch; // 当前分支的box输出索引
        int score_idx = i * output_per_branch + 1; // 当前分支的score输出索引
        grid_h = output_attrs[box_idx].dims[2]; // 获取当前分支的网格高度
        grid_w = output_attrs[box_idx].dims[3]; // 获取当前分支的网格宽度
        stride = model_in_h / grid_h; // 计算步长

        // 处理当前分支的输出，提取有效的边界框、对象概率和类别ID
        // #pragma omp critical
        validCount += process_i8((int8_t*)_outputs[box_idx].buf, output_attrs[box_idx].zp, output_attrs[box_idx].scale,
            (int8_t*)_outputs[score_idx].buf, output_attrs[score_idx].zp, output_attrs[score_idx].scale,
            (int8_t*)score_sum, score_sum_zp, score_sum_scale,
            grid_h, grid_w, stride, dfl_len,
            filterBoxes, objProbs, classId, conf_threshold);
    }

    // 如果没有检测到任何对象
    if (validCount <= 0)
    {
        return 0; // 直接返回0，表示没有检测到对象
    }

    // 创建一个索引数组，用于后续的排序和NMS操作
    std::vector<int> indexArray;
    std::iota(indexArray.begin(), indexArray.end(), 0); // 初始化索引数组

    // 对对象概率进行快速排序，并保存排序后的索引
    quick_sort_indice_inverse(objProbs, indexArray);

    // 获取所有检测到的类别ID的集合
    std::set<int> class_set(std::begin(classId), std::end(classId));

    // 对每个类别进行非极大值抑制（NMS）
    //#pragma omp parallel for
    for (int c : class_set)
    {
        nms(validCount, filterBoxes, classId, indexArray, c, nms_threshold);
    }

    int last_count = 0; // 用于记录最终的有效检测框数量
    //od_results->count = 0; // 初始化检测结果的数量

    // 遍历所有有效的检测框，进行最终的处理
    //#pragma omp parallel for schedule(dynamic) reduction(+:last_count)
    for (int i = 0; i < validCount; ++i)
    {
        if (indexArray[i] == -1 || last_count >= OBJ_NUMB_MAX_SIZE)
        {
            continue; // 如果索引无效或已达到最大检测框数量，跳过
        }
        int n = indexArray[i]; // 获取当前检测框的索引

        // 计算检测框的坐标，并减去letterbox的填充
        float x1 = filterBoxes[n * 4 + 0] - letter_box->x_pad;
        float y1 = filterBoxes[n * 4 + 1] - letter_box->y_pad;
        float x2 = x1 + filterBoxes[n * 4 + 2];
        float y2 = y1 + filterBoxes[n * 4 + 3];
        int id = classId[n]; // 获取当前检测框的类别ID
        float obj_conf = objProbs[i]; // 获取当前检测框的对象概率

        // 将检测框的坐标转换为原始图像的坐标，并进行clamp操作，确保坐标在有效范围内
        od_results->results[last_count].box.left = (int)(clamp(x1, 0, model_in_w) / letter_box->scale);
        od_results->results[last_count].box.top = (int)(clamp(y1, 0, model_in_h) / letter_box->scale);
        od_results->results[last_count].box.right = (int)(clamp(x2, 0, model_in_w) / letter_box->scale);
        od_results->results[last_count].box.bottom = (int)(clamp(y2, 0, model_in_h) / letter_box->scale);
        od_results->results[last_count].prop = obj_conf; // 设置对象概率
        od_results->results[last_count].cls_id = id; // 设置类别ID

        // #pragma omp atomic
        last_count++; // 增加有效检测框的数量
    }
    od_results->count = last_count; // 更新检测结果的数量
    return 0; // 返回0，表示处理成功
}