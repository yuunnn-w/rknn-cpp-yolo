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
    // 1. ��ѯ SDK �汾����ӡ
    rknn_sdk_version sdk_version;
    int ret = rknn_query(ctxs[ctx_index], RKNN_QUERY_SDK_VERSION, &sdk_version, sizeof(sdk_version));
    if (ret < 0) {
        std::cerr << "rknn_query RKNN_QUERY_SDK_VERSION error ret=" << ret << std::endl;
        return;
    }
    std::cout << "SDK API Version: " << sdk_version.api_version << std::endl; // 2.3.0 (c949ad889d@2024-11-07T11:35:33)
    std::cout << "Driver Version: " << sdk_version.drv_version << std::endl; // 0.9.7

    // 2. ��ѯ RKNN ģ��������û��Զ����ַ�����Ϣ����ӡ
    rknn_custom_string custom_string;
    ret = rknn_query(ctxs[ctx_index], RKNN_QUERY_CUSTOM_STRING, &custom_string, sizeof(custom_string));
    if (ret < 0) {
        std::cerr << "rknn_query RKNN_QUERY_CUSTOM_STRING error ret=" << ret << std::endl; // yolov9c
        return;
    }
    std::cout << "Custom String: " << custom_string.string << std::endl;

    // 3. ��ѯģ�����롢��� Tensor �ĸ�����ά�ȡ���״�����ơ��������͡��������Ͳ���ӡ
    //rknn_input_output_num io_num;
    ret = rknn_query(ctxs[ctx_index], RKNN_QUERY_IN_OUT_NUM, &io_num, sizeof(io_num));
    if (ret < 0) {
        std::cerr << "rknn_query RKNN_QUERY_IN_OUT_NUM error ret=" << ret << std::endl;
        return;
    }
    std::cout << "Model input num: " << io_num.n_input << ", output num: " << io_num.n_output << std::endl; // 1

    // ��ѯ�������� Tensor ������
    input_attrs.resize(io_num.n_input);
    for (uint32_t i = 0; i < io_num.n_input; ++i) {
        input_attrs[i].index = i;
        ret = rknn_query(ctxs[ctx_index], RKNN_QUERY_INPUT_ATTR, &input_attrs[i], sizeof(rknn_tensor_attr));
        if (ret != RKNN_SUCC) {
            std::cerr << "rknn_query RKNN_QUERY_INPUT_ATTR error ret=" << ret << std::endl;
            return;
        }

    }
	// ��ѯӲ�����ŵ�����Tensor����
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


    // ��ѯ������� Tensor ������
    output_attrs.resize(io_num.n_output);
    for (uint32_t i = 0; i < io_num.n_output; ++i) {
        output_attrs[i].index = i;
        ret = rknn_query(ctxs[ctx_index], RKNN_QUERY_OUTPUT_ATTR, &output_attrs[i], sizeof(rknn_tensor_attr)); //RKNN_QUERY_NATIVE_OUTPUT_ATTR
        if (ret != RKNN_SUCC) {
            std::cerr << "rknn_query RKNN_QUERY_OUTPUT_ATTR ret=" << ret << std::endl;
            return;
        }
    }

    // ��ѯӲ�����ŵ����Tensor����
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
    
    FILE* fp = fopen(model_path.c_str(), "rb"); // ��ģ���ļ�
    if (fp == nullptr) {
        std::cerr << "Failed to open model file: " << model_path << std::endl;
        return;
    }
    fseek(fp, 0, SEEK_END); // ���ļ�ָ���ƶ����ļ�ĩβ���Ա��ȡ�ļ���С
    size_t model_size = ftell(fp); // ��ȡ��ǰ�ļ�ָ���λ�ã����ļ���С
    fseek(fp, 0, SEEK_SET); // ���ļ�ָ�������ƶ����ļ���ͷ���Ա��ͷ��ʼ��ȡ�ļ�����

    double model_size_mb = static_cast<double>(model_size) / (1024 * 1024);
    std::cout << "Model size: " << model_size_mb << " MB" << std::endl; // ��ӡģ�Ͳ�����С����MBΪ��λ��

    unsigned char* model_data = new unsigned char[model_size]; // ����һ���ڴ棬��СΪ�ļ���С�����ڴ洢ģ������
    if (model_data == nullptr) {
        std::cerr << "Failed to allocate memory for model data." << std::endl;
        fclose(fp);
        return;
    }

    // ��ȡģ������
    size_t items_read = fread(model_data, 1, model_size, fp);
    if (items_read != model_size) {
        std::cerr << "Failed to read the entire model file." << std::endl;
        delete[] model_data;
        fclose(fp);
        return;
    }
    fclose(fp);

    // ��ʼ��rknn�����ģ������ö����ʶ
    int ret = rknn_init(&ctxs[0], model_data, model_size,
        //RKNN_FLAG_MEM_ALLOC_OUTSIDE | // ���ڱ�ʾģ�����롢�����Ȩ�ء��м�tensor�ڴ�ȫ�����û�����
        RKNN_FLAG_COLLECT_PERF_MASK |  // ��������ʱ��ѯ�������ʱ��
        // RKNN_FLAG_EXECUTE_FALLBACK_PRIOR_DEVICE_GPU |  // ��ʾ����NPU��֧�ֵĲ�����ѡ��������GPU��
        RKNN_FLAG_ENABLE_SRAM,  // ��ʾ�м�tensor�ڴ澡���ܷ�����SRAM��
        NULL);
    if (ret < 0) {
        std::cerr << "rknn_init error ret=" << ret << std::endl;
        delete[] model_data;
        return;
    }

    // ���Ƶ�һ�������ĵ�����������
    for (int i = 1; i < 3; ++i) {
        ret = rknn_dup_context(&ctxs[0], &ctxs[i]);
        if (ret < 0) {
            std::cerr << "Failed to duplicate context to ctx" << i << ". ret=" << ret << std::endl;
            delete[] model_data;
            return;
        }
    }
    // ����NPU��������
    rknn_set_core_mask(ctxs[0], RKNN_NPU_CORE_0); //����ģ��������0������
    rknn_set_core_mask(ctxs[1], RKNN_NPU_CORE_1); //����ģ��������1������
    rknn_set_core_mask(ctxs[2], RKNN_NPU_CORE_2); //����ģ��������2������
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


// ʵ�ִ�ӡ�����ͷ���������
void rknn_model::print_quantization_info() const {
    std::cout << "Quantization and Dequantization Information:" << std::endl;

    // ��ӡ���������������ͷ�������Ϣ
    std::cout << "Input Tensors:" << std::endl;
    for (const auto& attr : input_attrs) {
        std::cout << "  Name: " << attr.name << std::endl;
        if (attr.qnt_type == RKNN_TENSOR_QNT_AFFINE_ASYMMETRIC) {
            std::cout << "    Quantization Type: Affine Asymmetric" << std::endl;
            std::cout << "    Quantization Formula: Q = (R / scale) + zero_point" << std::endl;
            std::cout << "    Dequantization Formula: R = (Q - zero_point) * scale" << std::endl;
            std::cout << "    Scale: " << attr.scale << std::endl;
            std::cout << "    Zero Point: " << attr.zp << std::endl; // ʹ�� zp ������ fl
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

    // ��ӡ��������������ͷ�������Ϣ
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

// ����������ȡָ���� input_mems ��Աָ��
rknn_tensor_mem* rknn_model::get_input_mem_ptr(int ctx_index,int mem_index) {
    if (ctx_index < 0 || ctx_index >= static_cast<int>(input_mems.size())) {
        // �������������Χ�������׳��쳣�򷵻� nullptr
        throw std::out_of_range("Index out of range for input_mems vector.");
        // ���� return nullptr;
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
    // ���������������������
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
            // ���������ڴ�
            int ret = rknn_set_io_mem(ctxs[i], input_mems[i][j], &input_native_attrs[j]);
            if (ret < 0) {
                printf("input_mems rknn_set_io_mem fail! ret=%d\n", ret);
            }
        }
        //std::cerr << "OutPut size:" << output_attrs[0].size_with_stride << std::endl;
        //std::cerr << "OutPut n_elems:" << output_attrs[0].n_elems << std::endl;
        for (size_t j = 0; j < io_num.n_output; ++j) {
            //output_attrs[j].type = RKNN_TENSOR_FLOAT32; //RKNN_TENSOR_INT8 ����ĳ�int8����Ҫ�ֶ�������
            //output_attrs[j].fmt = RKNN_TENSOR_NC1HWC2;
            output_mems[i][j] = rknn_create_mem(ctxs[i], output_native_attrs[j].size_with_stride);//.n_elems); //*sizeof(float)
            if (!output_mems[i][j]) {
                throw std::runtime_error("rknn_create_mem failed for output");
            }
            // ��������ڴ�
            rknn_set_io_mem(ctxs[i], output_mems[i][j], &output_native_attrs[j]);
        }
    }
    // RKNN_TENSOR_UINT8;


}

void rknn_model::release_mems() {
    int ctx_size = 3;
    // �ͷ���������ڴ�
    for (size_t i = 0; i < ctx_size; ++i) {
        // �������������ڴ�
        for (size_t j = 0; j < io_num.n_input; ++j) {
            if (input_mems[i][j] != nullptr && input_mems[i][j]->virt_addr != nullptr) { // ���ָ��������ַ�Ƿ�Ϊnullptr
                rknn_destroy_mem(ctxs[i], input_mems[i][j]);
                //delete input_mems[i][j]; // ����Ƕ�̬����ģ���ȷ������ʹ����ȷ���ͷŷ�ʽ
                input_mems[i][j] = nullptr; // ȷ���ͷź�ָ����Ϊnullptr
            }
        }

        // ������������ڴ�
        for (size_t j = 0; j < io_num.n_output; ++j) {
            if (output_mems[i][j] != nullptr && output_mems[i][j]->virt_addr != nullptr) { // ���ָ��������ַ�Ƿ�Ϊnullptr
                rknn_destroy_mem(ctxs[i], output_mems[i][j]);
                //delete output_mems[i][j]; // ����Ƕ�̬����ģ���ȷ������ʹ����ȷ���ͷŷ�ʽ
                output_mems[i][j] = nullptr; // ȷ���ͷź�ָ����Ϊnullptr
            }
        }
    }
}

// �Ż���ĺ���
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

    // ʹ�� RGA ��������Ӧ�ü����
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

        // NC1HWC2 ��ʽת��Ϊ NCHW ��ʽ
        if (output_native_attrs[i].fmt == RKNN_TENSOR_NC1HWC2) {
            NC1HWC2_i8_to_NCHW_i8((int8_t*)output_mems[ctx_index][i]->virt_addr, (int8_t*)outputs[i].buf,
                (int*)output_native_attrs[i].dims, channel, h, w, zp, scale);
        }
        else {
            memcpy(outputs[i].buf, output_mems[ctx_index][i]->virt_addr, outputs[i].size);
        }
    }
    const float nms_threshold = NMS_THRESH;      // Ĭ�ϵ�NMS��ֵ
    const float box_conf_threshold = BOX_THRESH; // Ĭ�ϵ����Ŷ���ֵ

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

    // ʹ�� RGA ��������Ӧ�ü����
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

        // NC1HWC2 ��ʽת��Ϊ NCHW ��ʽ
        if (output_native_attrs[i].fmt == RKNN_TENSOR_NC1HWC2) {
            NC1HWC2_i8_to_NCHW_i8((int8_t*)output_mems[ctx_index][i]->virt_addr, (int8_t*)outputs[i].buf,
                (int*)output_native_attrs[i].dims, channel, h, w, zp, scale);
        }
        else {
            memcpy(outputs[i].buf, output_mems[ctx_index][i]->virt_addr, outputs[i].size);
        }
    }

    const float nms_threshold = NMS_THRESH;      // Ĭ�ϵ�NMS��ֵ
    const float box_conf_threshold = BOX_THRESH; // Ĭ�ϵ����Ŷ���ֵ

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

    // ��ӡÿ�������ʱ�����ռ��������ȷ������
    std::cout << std::fixed << std::setprecision(6);

    std::cout << "\nPerformance Analysis:" << std::endl;
    std::cout << "Total Time: " << total_time_ms.count() << " ms" << std::endl;

    // ���浱ǰ��������
    auto old_precision = std::cout.precision();

    // ���ÿ�������ʱ�����ռ����������ʱ�䱣��10λС�����ٷֱȱ�����λС��
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
    // ����������ָ��ת��Ϊrknn_output���ͣ������������
    rknn_output* _outputs = (rknn_output*)outputs;

    // �������ڴ洢���˺�ı߽�򡢶�����ʺ����ID������
    std::vector<float> filterBoxes;
    std::vector<float> objProbs;
    std::vector<int> classId;
    int validCount = 0; // ��Ч���������
    int stride = 0; // ���������ڼ��������С
    int grid_h = 0; // ����߶�
    int grid_w = 0; // ������
    int model_in_w = model_width; // ģ��������
    int model_in_h = model_height; // ģ������߶�

    // ��ʼ��od_results�ṹ�壬ȷ��������Ϊ0
    //memset(od_results, 0, sizeof(object_detect_result_list));


    // Ĭ����3�������֧
    int dfl_len = output_attrs[0].dims[1] / 4; // ����ÿ����֧��DFL����
    int output_per_branch = io_num.n_output / 3; // ÿ����֧���������

    /*
    // Ԥ�����ܵ������Ч��������
    int max_valid_boxes_per_branch = 0;
    for (int i = 0; i < 3; i++) {
        int grid_h = output_attrs[i * output_per_branch].dims[2];
        int grid_w = output_attrs[i * output_per_branch].dims[3];
        max_valid_boxes_per_branch += grid_h * grid_w;
    }

    // Ԥ����ռ������������ܵ���Ч��������
    filterBoxes.reserve(max_valid_boxes_per_branch * 4); // ÿ������4������ֵ
    objProbs.reserve(max_valid_boxes_per_branch);        // ÿ������һ�����Ŷ�
    classId.reserve(max_valid_boxes_per_branch);         // ÿ������һ�����ID
    */

    // #pragma omp parallel for schedule(auto) reduction(+:validCount)
    for (int i = 0; i < 3; i++)
    {
        void* score_sum = nullptr;
        int32_t score_sum_zp = 0;
        float score_sum_scale = 1.0;
        // ���ÿ����֧��3����������ȡscore_sum�Ļ�������������������
        if (output_per_branch == 3) {
            score_sum = _outputs[i * output_per_branch + 2].buf;
            score_sum_zp = output_attrs[i * output_per_branch + 2].zp;
            score_sum_scale = output_attrs[i * output_per_branch + 2].scale;
        }
        int box_idx = i * output_per_branch; // ��ǰ��֧��box�������
        int score_idx = i * output_per_branch + 1; // ��ǰ��֧��score�������
        grid_h = output_attrs[box_idx].dims[2]; // ��ȡ��ǰ��֧������߶�
        grid_w = output_attrs[box_idx].dims[3]; // ��ȡ��ǰ��֧��������
        stride = model_in_h / grid_h; // ���㲽��

        // ����ǰ��֧���������ȡ��Ч�ı߽�򡢶�����ʺ����ID
        // #pragma omp critical
        validCount += process_i8((int8_t*)_outputs[box_idx].buf, output_attrs[box_idx].zp, output_attrs[box_idx].scale,
            (int8_t*)_outputs[score_idx].buf, output_attrs[score_idx].zp, output_attrs[score_idx].scale,
            (int8_t*)score_sum, score_sum_zp, score_sum_scale,
            grid_h, grid_w, stride, dfl_len,
            filterBoxes, objProbs, classId, conf_threshold);
    }

    // ���û�м�⵽�κζ���
    if (validCount <= 0)
    {
        return 0; // ֱ�ӷ���0����ʾû�м�⵽����
    }

    // ����һ���������飬���ں����������NMS����
    std::vector<int> indexArray;
    std::iota(indexArray.begin(), indexArray.end(), 0); // ��ʼ����������

    // �Զ�����ʽ��п������򣬲���������������
    quick_sort_indice_inverse(objProbs, indexArray);

    // ��ȡ���м�⵽�����ID�ļ���
    std::set<int> class_set(std::begin(classId), std::end(classId));

    // ��ÿ�������зǼ���ֵ���ƣ�NMS��
    //#pragma omp parallel for
    for (int c : class_set)
    {
        nms(validCount, filterBoxes, classId, indexArray, c, nms_threshold);
    }

    int last_count = 0; // ���ڼ�¼���յ���Ч��������
    //od_results->count = 0; // ��ʼ�������������

    // ����������Ч�ļ��򣬽������յĴ���
    //#pragma omp parallel for schedule(dynamic) reduction(+:last_count)
    for (int i = 0; i < validCount; ++i)
    {
        if (indexArray[i] == -1 || last_count >= OBJ_NUMB_MAX_SIZE)
        {
            continue; // ���������Ч���Ѵﵽ����������������
        }
        int n = indexArray[i]; // ��ȡ��ǰ���������

        // �����������꣬����ȥletterbox�����
        float x1 = filterBoxes[n * 4 + 0] - letter_box->x_pad;
        float y1 = filterBoxes[n * 4 + 1] - letter_box->y_pad;
        float x2 = x1 + filterBoxes[n * 4 + 2];
        float y2 = y1 + filterBoxes[n * 4 + 3];
        int id = classId[n]; // ��ȡ��ǰ��������ID
        float obj_conf = objProbs[i]; // ��ȡ��ǰ����Ķ������

        // �����������ת��Ϊԭʼͼ������꣬������clamp������ȷ����������Ч��Χ��
        od_results->results[last_count].box.left = (int)(clamp(x1, 0, model_in_w) / letter_box->scale);
        od_results->results[last_count].box.top = (int)(clamp(y1, 0, model_in_h) / letter_box->scale);
        od_results->results[last_count].box.right = (int)(clamp(x2, 0, model_in_w) / letter_box->scale);
        od_results->results[last_count].box.bottom = (int)(clamp(y2, 0, model_in_h) / letter_box->scale);
        od_results->results[last_count].prop = obj_conf; // ���ö������
        od_results->results[last_count].cls_id = id; // �������ID

        // #pragma omp atomic
        last_count++; // ������Ч���������
    }
    od_results->count = last_count; // ���¼����������
    return 0; // ����0����ʾ����ɹ�
}