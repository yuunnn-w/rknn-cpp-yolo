

# rknn-cpp-yolo
This project implements YOLOv11 inference on the RK3588 platform using the RKNN framework. With deep optimization of the official code and RGA hardware acceleration for image preprocessing, it achieves a stable 25 FPS for YOLOv11s without overclocking and core binding, showcasing efficient real-time object detection for embedded applications.

本项目基于RKNN框架，在RK3588平台上实现了YOLOv11推理。通过对官方代码的深度优化和RGA硬件加速预处理，YOLOv11s在未超频和绑定大核的情况下，稳定达到25帧/秒，为嵌入式实时目标检测提供了高效解决方案。
****

# YOLOv11 on RK3588 with RKNN

## Features
- **YOLOv11 Inference**: Optimized implementation for RK3588.
- **RGA Preprocessing**: Utilizes hardware acceleration for image processing.
- **CMake Build System**: Easy to configure and build.
- **High Performance**: Achieves 25 FPS without overclocking or core binding.
- **Zero-Copy API**: Reduces inference overhead for better efficiency.
- **RK3588 Optimization**: Supports concurrent inference across three NPU cores (requires custom thread pool implementation).

## Report: Inference Results and Speed

  ![Result](https://github.com/yuunnn-w/rknn-cpp-yolo/blob/main/img/result.jpg)  

  ![Speed](https://github.com/yuunnn-w/rknn-cpp-yolo/blob/main/img/speed.jpg)  

## Prerequisites
- RK3588 development board
- RKNPU Driver (version >= 0.9.6)
- RKNN SDK
- CMake (version 3.10 or higher)
- OpenCV (for image handling, optional)

## Build Instructions

1. **Clone the repository:**
   ```bash
   git clone https://github.com/yuunnn-w/rknn-cpp-yolo.git
   cd rknn-cpp-yolo
   ```

2. **Install dependencies:**
   ```bash
   sudo apt-get update
   sudo apt-get install -y build-essential gcc g++ gdb cmake ninja-build git libopencv-dev zlib1g-dev librga-dev ninja-build libomp-dev
   ```

3. **Install RKNN SDK:**
   ```bash
   sudo bash install_rknpu.sh
   ```

4. **Create a build directory:**
   ```bash
   mkdir build
   cd build
   ```

5. **Configure the project with CMake:**
   ```bash
   cmake ..
   ```
   *(Alternatively, use the provided CMake presets such as `cmake --preset linux-debug ..` or `cmake --preset orangepi ..`)*

6. **Build the project:**
   ```bash
   make
   ```

7. **Run the inference:**
   ```bash
   ./yolov11_rk3588
   ```

## Usage
After building the project, you can run the inference by executing the generated binary. Ensure that the RKNN model and test images are correctly placed in the specified paths.

## Optimization
The project includes several optimizations to achieve high performance on the RK3588 platform:
- Efficient use of RGA for image preprocessing.
- Memory and computation optimizations in the inference pipeline.
- Zero-Copy API to minimize memory overhead.
- Support for concurrent NPU core utilization (requires thread pool implementation).

## Attention

Please note that this project only provides an example of image-based inference in the `rknn.cpp` file. If you need to perform real-time inference in more complex application scenarios, you will need to implement it yourself.  

Additionally, this project is purely an experimental demo and is not responsible for any products or issues. The final interpretation right belongs to **yuunnn_w**.  

## License
This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.

## Acknowledgments
- Thanks to the RKNN team for their framework and support.
- Inspired by the official YOLOv11 implementation.

For any questions or contributions, feel free to open an issue or submit a pull request.

## Contact Me

If you have any questions, suggestions, or would like to contribute to this project, feel free to reach out! You can contact me through the following channels:

- **Email**: [jiaxinsugar@gmail.com](mailto:jiaxinsugar@gmail.com)
- **GitHub**: [yuunnn-w](https://github.com/yuunnn-w)

I’m always open to discussions, collaborations, and feedback. Let’s make this project even better together! 🚀
