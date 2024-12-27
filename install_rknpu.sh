#!/bin/bash

# 定义源目录和目标目录
SOURCE_DIR_BIN="./runtime/Linux/rknn_server/aarch64/usr/bin"
DEST_DIR_BIN="/usr/bin"

SOURCE_DIR_INCLUDE="./runtime/Linux/librknn_api/include"
DEST_DIR_INCLUDE="/usr/include"

SOURCE_DIR_LIB="./runtime/Linux/librknn_api/aarch64"
DEST_DIR_LIB="/usr/lib"

# 检查源目录是否存在
check_source_dir() {
  if [ ! -d "$1" ]; then
    echo "源目录 $1 不存在。"
    exit 1
  fi
}

# 检查目标目录是否存在
check_dest_dir() {
  if [ ! -d "$1" ]; then
    echo "目标目录 $1 不存在。"
    exit 1
  fi
}

# 复制文件
copy_files() {
  local source_dir=$1
  local dest_dir=$2
  cp -r "$source_dir"/* "$dest_dir"
  if [ $? -eq 0 ]; then
    echo "文件从 $source_dir 复制到 $dest_dir 成功。"
  else
    echo "文件从 $source_dir 复制到 $dest_dir 失败。"
    exit 1
  fi
}

# 检查并复制 rknn_server 文件
check_source_dir "$SOURCE_DIR_BIN"
check_dest_dir "$DEST_DIR_BIN"
copy_files "$SOURCE_DIR_BIN" "$DEST_DIR_BIN"

# 检查并复制 include 文件
check_source_dir "$SOURCE_DIR_INCLUDE"
check_dest_dir "$DEST_DIR_INCLUDE"
copy_files "$SOURCE_DIR_INCLUDE" "$DEST_DIR_INCLUDE"

# 检查并复制 lib 文件
check_source_dir "$SOURCE_DIR_LIB"
check_dest_dir "$DEST_DIR_LIB"
copy_files "$SOURCE_DIR_LIB" "$DEST_DIR_LIB"

echo "所有文件复制成功。"