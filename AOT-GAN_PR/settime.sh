#!/bin/bash

# 切换到特定的虚拟环境
source /root/miniconda3/bin/activate PRM

# 切换到特定的文件目录
cd /root/autodl-tmp/AOT-GAN-for-Inpainting-master/src

# 执行Python脚本
python train.py
