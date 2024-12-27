# CMakeList.txt: rknn 的 CMake 项目，在此处包括源代码并定义
# 项目特定的逻辑。
#
cmake_minimum_required (VERSION 3.8)

# 如果支持，请为 MSVC 编译器启用热重载。
if (POLICY CMP0141)
  cmake_policy(SET CMP0141 NEW)
  set(CMAKE_MSVC_DEBUG_INFORMATION_FORMAT "$<IF:$<AND:$<C_COMPILER_ID:MSVC>,$<CXX_COMPILER_ID:MSVC>>,$<$<CONFIG:Debug,RelWithDebInfo>:EditAndContinue>,$<$<CONFIG:Debug,RelWithDebInfo>:ProgramDatabase>>")
endif()

project ("rknn")

find_package(OpenCV REQUIRED)
find_package(PkgConfig REQUIRED)
find_library(RGA_LIBRARY NAMES rga)
find_package(OpenMP REQUIRED)

find_package(TBB REQUIRED) # 添加对 libtbb 库的依赖
# 将源代码添加到此项目的可执行文件。

file(GLOB SOURCES "src/*.cpp" "rknn.cpp")

add_executable (rknn ${SOURCES})

if (CMAKE_VERSION VERSION_GREATER 3.12)
  set_property(TARGET rknn PROPERTY CXX_STANDARD 20)
endif()

# 添加 /usr/include 到包含路径
include_directories(
    /usr/include
    ${OpenCV_INCLUDE_DIRS}
    ${CMAKE_SOURCE_DIR}/include
    /usr/include/rga
)

# 设置编译选项，启用 OpenMP
if(OpenMP_CXX_FOUND)
    message(STATUS "OpenMP found, enabling OpenMP support.")
    target_link_libraries(rknn OpenMP::OpenMP_CXX)
else()
    message(WARNING "OpenMP not found, compiling without OpenMP.")
endif()

# 链接 /usr/lib 中的库
# 查找库文件
find_library(RKNN_API_LIBRARY NAMES rknnrt PATHS /usr/lib)

# 检查是否找到库文件
if(RKNN_API_LIBRARY)
    target_link_libraries(rknn ${RKNN_API_LIBRARY})
else()
    message(FATAL_ERROR "Could not find librknnrt.so")
endif()

# TODO: 如有需要，请添加测试并安装目标。
# 链接库
target_link_libraries(rknn
    ${OpenCV_LIBS}
    TBB::tbb
    ${RGA_LIBRARY}
)
# sudo apt-get install libtbb-dev

# 设置 C++ 编译选项以启用 O3 优化
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O3")