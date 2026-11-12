---
layout: post
title: "寒武纪 MLU 完整生态详解:BANG C 与 Neuware 软件栈全景"
date: 2026-11-12 11:30:00 +0800
excerpt: "逐模块梳理寒武纪 MLU 思元系列的软件生态:CNRT/CNDrv 运行时、BANG C 算子语言、CNNL 算子库、CNCL 通信、MagicMind 推理引擎,以及 torch_mlu(Catch)的框架适配,并给出与 CUDA 的对应关系。"
---

> 本文是[国产 AI 芯片生态全景:SDK、框架适配与开发者社区指南]({% post_url 2026-11-12-ai-chip-ecosystem-guide %})的寒武纪 MLU 部分展开。总文给出各家芯片的横向速查,本文专注把寒武纪软件生态逐模块讲透。

寒武纪(Cambricon)的 MLU(Machine Learning Unit,思元系列)是国内最早一批专用 AI 加速器之一。它走自建栈路线:以 **Neuware** 为软件栈总称,**BANG C** 为自定义算子语言,**torch_mlu(Catch)** 作为 PyTorch 接入层。架构上寒武纪强调为深度学习负载定制的智能处理单元。

> 说明:下文版本号为写作时的参考基线,实际以寒武纪官方 release notes 为准。

## 一、底层基础层

### 1. Neuware SDK
寒武纪软件栈的总称,相当于 CUDA Toolkit,涵盖驱动、运行时、编译器、算子库、通信库、推理引擎和工具。版本如 Neuware 3.5 / 3.8 / 3.10。

### 2. CNDrv(驱动层)
最底层的 MLU 设备驱动,负责设备枚举、内存管理、命令队列下发,类似 CUDA Driver API 层。

### 3. CNRT(Cambricon Runtime)
运行时库,封装设备/内存/队列/kernel 管理,是上层框架和应用的主要接口,角色类似 CUDA Runtime API。

## 二、编程模型核心概念

### BANG C(自定义算子语言)
寒武纪的异构编程语言,对标 CUDA C。
- 面向 MLU 的计算核(MLU Core)与向量/张量单元编程
- 提供对片上存储(NRAM/WRAM/SRAM)的显式管理
- 用于开发高性能自定义算子

### 计算与内存层次
- **MLU Core / IPU**:计算核心,内含张量与向量运算单元
- **NRAM(Neuron RAM)**:核内神经元数据片上存储
- **WRAM(Weight RAM)**:权重片上存储
- **SRAM**:簇内共享存储
- **DDR/HBM**:设备全局内存

寒武纪的内存层次比 GPU 更显式地区分"神经元"与"权重"存储,BANG C 编程时需要手动编排这几级存储间的搬运。

## 三、编译工具链

### 1. CNCC / BANG 编译器
编译 BANG C 源码为 MLU 可执行代码的编译器(角色对应 nvcc)。

### 2. CNGDB
GPU/MLU 调试器,基于 GDB,支持 device 侧断点调试。对应 cuda-gdb。

### 3. CNPerf
性能分析工具,采集算子耗时、硬件利用率。对应 Nsight Compute。

### 4. CNMon
设备监控工具,查看 MLU 利用率、显存、温度、功耗,角色类似 `nvidia-smi`。

## 四、算子库

### CNNL(Cambricon Neuware Neural network Library)
深度学习算子库,提供卷积、矩阵乘、归一化、激活、池化、Attention 等算子,对标 cuDNN + 部分 cuBLAS。是框架后端算子的主要来源。

此外还有面向通用计算的扩展算子库(如 CNNL Extra、矩阵/向量运算库),覆盖非神经网络类算子。

## 五、通信库

### CNCL(Cambricon Communications Library)
多 MLU / 多节点集合通信库,对标 NCCL。
- AllReduce、AllGather、ReduceScatter、Broadcast 等
- 多卡训练的通信底层
- 机内通过卡间高速互联,跨节点走 RDMA

## 六、AI 推理

### MagicMind
寒武纪的推理优化引擎,对标 TensorRT。
- 接收 ONNX / 框架模型,做图优化、算子融合、量化(INT8/INT16 等)
- 生成优化后的推理引擎并部署

### CNLight
轻量推理运行时,面向资源受限或轻量部署场景。

## 七、框架适配

### torch_mlu(Catch)
寒武纪维护的 PyTorch 适配层,又称 Catch(CAmbricon exTension for pytorCH)。
- 把 PyTorch 的张量运算映射到 CNNL/CNRT
- 版本跟随上游 PyTorch
- 关键版本绑定(参考):

```
Neuware 3.8 → torch_mlu 2.1 → PyTorch 2.1 → Python 3.8 / 3.10
Neuware 3.5 → torch_mlu 1.17 → PyTorch 1.13
```

### 主流框架适配状态

| 框架 | 适配状态 | 说明 |
|------|---------|------|
| **vLLM** | 有 MLU 适配分支 | 寒武纪官方维护;依赖较新 Neuware |
| **DeepSpeed** | 社区适配 | 通过 torch_mlu 路径 |
| **Megatron-LM** | 有适配工作 | 官方有测试用例 |

寒武纪也提供 TensorFlow 的适配(通过框架插件),但 PyTorch(torch_mlu)是当前主力路径。

## 八、多卡 / 多节点并行

- **数据并行**:torch_mlu + CNCL 走 DDP
- **张量/流水并行**:通过 Megatron 适配
- 机内卡间互联,跨节点 RDMA,CNCL 做集合通信

## 九、部署

寒武纪提供预装 Neuware + torch_mlu 的 Docker 镜像,推理侧可用 MagicMind 做服务化部署。生产推荐从官方镜像起手锁定版本。

## 十、与 CUDA 的对应关系速查

| CUDA 组件 | 寒武纪 MLU 对应 |
|-----------|---------------|
| CUDA Toolkit | Neuware SDK |
| CUDA Driver API | CNDrv |
| CUDA Runtime API | CNRT |
| CUDA C | BANG C |
| cuDNN / cuBLAS | CNNL(+ 扩展算子库) |
| NCCL | CNCL |
| TensorRT | MagicMind |
| cuda-gdb | CNGDB |
| Nsight Compute | CNPerf |
| nvidia-smi | CNMon |
| torch(CUDA) | torch_mlu(Catch) |
| Shared Memory | NRAM/WRAM/SRAM |

## 十一、开发者社区

- 开发者社区:[developer.cambricon.com](https://developer.cambricon.com)
- SDK 下载:[sdk.cambricon.com](https://sdk.cambricon.com)
- GitHub:[github.com/Cambricon](https://github.com/Cambricon)

## 总结导图

```
寒武纪 MLU 生态(自建栈)
├── 底层基础     CNDrv(驱动)→ CNRT(运行时)
├── 算子开发     BANG C(+ CNCC 编译器)
├── 工具链       CNGDB / CNPerf / CNMon
├── 算子库       CNNL(+ 扩展算子库)
├── 通信库       CNCL
├── AI 推理      MagicMind / CNLight
├── 框架适配     torch_mlu(Catch)/ vLLM-MLU / DeepSpeed / Megatron
└── 部署         官方 Docker + MagicMind 服务化
```

寒武纪是国产 AI 芯片里"自研栈"路线的代表之一:从 BANG C 到 CNNL 再到 MagicMind,整条链路自建,命名体系自成一套。好处是软硬件协同设计空间大、对专用负载优化到位;代价是迁移需要通过 torch_mlu 做适配,自定义算子要用 BANG C 重写。对深度学习推理这类核心场景,其成熟度在国产阵营中较高。
