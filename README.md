# 项目名称

智能网联汽车辅助驾驶安全信息检测系统

# 学科

- 一级：计算机科学与技术
- 二级：计算机应用技术

# 时间

2021年1月到2021年4月

# 第一章 前言

道路交通安全一直是一个全球范围内的重要问题，频发的交通事故造成了巨大的经济损失和人员伤亡，给社会带来的危害显而易见。2020年我国道路交通事故万车死亡人数为1.66人，2019年我国交通发生数量为24.8万起，其中机动车事故更是高达21.5万起。因此，尽可能的减少交通事故的发生、降低其带来的危害迫在眉睫。

# 第二章 创意介绍

本系统在硬件层面通过采用Nvidia Jetson Nano+树莓派的组合，在保证性能的前提下，极大的减小了硬件体积和成本。软件层面通过CenterNet神经网络算法实现了对目标物体的识别；通过透视变换方程组的像素-距离比原理实现了对危险距离的标定；通过移植QT框架实现了车载中控的人机交互；通过CAN总线接口保留了系统功能扩展的可能性。

# 第三章 功能简介

本系统主要由预碰撞检测系统、车尾盲区检测系统和人机交互系统组成。

预碰撞检测系统可以完成对车前方的车辆、行人进行分辨和检测，并在即将发生碰撞时发出报警提醒。同时也可以实现检测路边交通警示标牌和车道偏离警告的功能。

车尾盲区检测系统可以完成对车尾盲区的障碍物检测，并实现在即将到达危险距离时对用户发出警告的功能。

人机交互系统主要实现前端显示和交互功能。同时我们也扩展了系统在手机端的应用，用户可以实现在手机App查看汽车盲区的检测视频。

# 第四章 特色综述

## 硬件平台设计：

在嵌入式平台部署神经网络完成实时推理。

在嵌入式小体积、低功耗场景下使用GPU完成实时运算。

软件程序可以无缝切换NVIDIA高算力平台，支持Jetson全系列嵌入式级GPU以及RTX、Tesla系列服务器级GPU，在预算充足的情况下可以获得更好的效果。

## 软件算法设计：

支持多视频流实时处理，程序可以支持8路、16路甚至更多视频输入，唯一受限的只是硬件显存与GPU算力，目前两路输入能达到25fps输出。

采用处于学术前沿的CenterNet网络进行目标识别，消耗较少的硬件资源的同时可以极大地提高神经网络传递效率。

利用单目摄像头数据，通过单目测距算法计算目标距离，并提供实时平面建模数据。

预留车载电子数据总线接口，可以实现辅助驾驶系统与车辆进行准确实时的数据交互。

## 用户交互设计：

移植开源项目Qt-Frameless-Window-DarkStyle，在车载中控实现方便的可视化用户交互。

基于Flutter+Web开发手机APP，可以实现通过车内局域网络查看辅助驾驶系统实时状况。

# 第五章 开发工具与技术

## 软件开发工具：

CMake、Jetpack、Deepstream、OpenCV、Pytorch、TensorRT、Qt、Axure、Git

## 相关技术：

**前端技术：**HTML、CSS、JavaScript、Nodejs、JQuery。

**嵌入式技术：**传感器数据收集及处理。

**视觉处理技术：**CUDA并行化运算、神经网络输入图像规范化、单目定位校准算法、单目测距算法。

**神经网络：**CenterNet、面向嵌入式平台的神经网络网络优化、面向推理框架的模型转换。

**网络技术：**RTSP视频传输

# 第六章 应用对象

本系统经过简单的适配改造，可以搭载在市面上所有常见的机动车上。

# 第七章 应用环境

本系统可以应用于常见的道路场景以及高速道路场景。

# 第八章 结语

本系统采用CenterNet神经网络算法进行盲区检测，包括预碰撞检测系统和车尾盲区检测系统，完成了车辆、行人检测，危险距离识别，交通警示标牌识别和车道偏离警告在内等功能，基本实现了汽车辅助驾驶的安全需求。

通过采用Nvidia Jetson Nano作为运算检测单元，树莓派作为人机交互单元，本系统实现了产品体积的极大缩小和产品成本的大幅减少。同时Can总线的设计，使得本系统的扩展性大大增加，保留了增加新功能、新外设的可能性。

智能网联汽车辅助驾驶安全信息检测系统完成了对汽车行驶过程的周围信息的检测和分析，对于未来无人驾驶的实现具有十分积极的意义和帮助。
