---
layout: post
title: "海光 DCU 完整生态详解:ROCm 兼容路线下的 DTK 软件栈全景"
date: 2026-11-12 11:10:00 +0800
excerpt: "逐模块梳理海光 DCU 的软件生态:底层 ROCm 兼容运行时、HIP 编程模型、DTK 工具链、hipBLAS/hipDNN 数学库、HCCL 通信,以及 torch-dcu 与 vLLM/DeepSpeed 的框架适配,并给出与 CUDA 的对应关系。"
---

> 本文是[国产 AI 芯片生态全景:SDK、框架适配与开发者社区指南]({% post_url 2026-11-12-ai-chip-ecosystem-guide %})的海光 DCU 部分展开。总文给出各家芯片的横向速查,本文专注把海光 DCU 的软件生态逐模块讲透。

海光 DCU(Deep Computing Unit,深算系列)是海光信息面向 AI 训练与科学计算推出的 GPGPU 加速器。与多数国产芯片自建一套 API 不同,海光选择了 **ROCm 兼容路线**:在 AMD 开源的 ROCm 栈之上做二次适配与扩展,推出 DTK(Deep Computing Toolkit)。对开发者最大的意义是——大量基于 HIP 编写的代码无需修改即可在 DCU 上编译运行,迁移成本显著低于从零对接私有 API。

> 说明:下文版本号为写作时的参考基线,海光按季度迭代,实际版本以官方 release notes 为准。

## 一、底层基础层

### 1. ROCm 兼容架构
海光 DCU 的软件栈本质是 ROCm 的一个下游发行版。ROCm(Radeon Open Compute)是 AMD 的开源 GPU 计算平台,定位对标 CUDA。海光 DTK 对其做了针对自家 DCU 硬件的适配、性能调优与扩展。

### 2. DTK(Deep Computing Toolkit)
海光软件栈的总称,相当于 NVIDIA 的 CUDA Toolkit + AMD 的 ROCm 的角色。
- DTK 23.04 / 24.04 / 24.10 等版本,大致对应上游 ROCm 5.x / 6.x
- 包含编译器、运行时、数学库、通信库、性能工具的完整集合
- 向下屏蔽 DCU 硬件差异,向上提供 HIP 编程接口

### 3. HIP(Heterogeneous-Compute Interface for Portability)
核心编程接口,对齐 AMD HIP API。
- HIP 是 AMD 设计的可移植 GPU 编程模型,语法与 CUDA C++ 高度相似(`hipMalloc` ↔ `cudaMalloc`、`hipMemcpy` ↔ `cudaMemcpy`)
- 提供 `hipify` 工具,把 CUDA 源码自动转换为 HIP 代码
- 一份 HIP 代码理论上可同时跑在 AMD GPU 和海光 DCU 上

### 4. ROCr Runtime / 驱动层
底层运行时(对应 ROCm 的 ROCr、HSA runtime)负责 kernel 分发、内存管理、队列调度,通过内核态驱动与 DCU 硬件交互。

## 二、编程模型核心概念

### 线程层次
HIP 沿用与 CUDA 同构的层次模型:

```
Grid → Block(Work-group)→ Wavefront → Thread(Work-item)
```

- **Work-item**:最小执行单元,对应 CUDA 的 thread
- **Wavefront**:SIMD 调度单位,AMD 架构传统为 64 lane(CUDA warp 为 32),DCU 的具体宽度需以硬件手册为准
- **Work-group**:对应 CUDA 的 block,共享 LDS(Local Data Share)和 barrier 同步
- **Grid**:一次 kernel launch 的全部 work-group

### 内存层次
- **寄存器(VGPR/SGPR)**:每个 work-item 私有
- **LDS(Local Data Share)**:对应 CUDA 的 Shared Memory,work-group 内共享
- **L1/L2 Cache**
- **Global Memory(HBM)**:DCU 普遍采用 HBM,带宽是其面向训练/HPC 的关键
- **Constant Memory**

### 编程写法
用 `__global__` 标注 kernel,`hipLaunchKernelGGL` 或 `<<<>>>` 语法启动,与 CUDA C++ 几乎逐行对应。

## 三、编译工具链

### 1. hipcc
HIP 的编译器驱动,底层基于 LLVM/Clang,把 HIP 源码编译为 DCU 可执行的 GPU 代码。角色对应 `nvcc`。

### 2. hipify 工具
- `hipify-perl` / `hipify-clang`:把 `.cu` 的 CUDA 代码批量转换为 HIP
- 是 CUDA 项目迁移到 DCU 的第一步工具

### 3. rocgdb
GPU 调试器,基于 GDB,支持在 device kernel 内设断点。对应 cuda-gdb。

### 4. rocprof / rocprofiler
性能分析器,采集 kernel 执行时间、硬件 counter。对应 Nsight Compute/Systems 的部分能力。

### 5. roc-obj / llvm-objdump
反汇编与目标文件工具,查看生成的 GPU ISA。

## 四、核心数学库

海光 DTK 提供一套与 ROCm 同名、对标 cuXXX 的数学库:

| DTK/ROCm 库 | 功能 | 对标 CUDA |
|------|------|----------|
| hipBLAS / rocBLAS | 稠密线性代数 BLAS | cuBLAS |
| hipDNN / MIOpen | 深度学习算子(Conv/Pooling/Norm 等) | cuDNN |
| hipFFT / rocFFT | 快速傅里叶变换 | cuFFT |
| hipSPARSE / rocSPARSE | 稀疏矩阵运算 | cuSPARSE |
| hipSOLVER / rocSOLVER | 线性代数求解器 | cuSolver |
| hipRAND / rocRAND | 随机数生成 | cuRAND |

> 注:上游 ROCm 的深度学习算子库是 MIOpen,海光侧常以 hipDNN 形式对外对齐 cuDNN 语义,具体命名以 DTK 文档为准。

## 五、通信库

### HCCL(Hygon Collective Communications Library)
多 DCU / 多节点集合通信库,对标 NCCL(上游为 RCCL)。
- 提供 AllReduce、AllGather、ReduceScatter、Broadcast 等集合操作
- 多卡训练(DDP、张量并行)的通信底层
- 支持机内高速互联与跨节点 RDMA

## 六、AI 推理与框架适配

### torch-dcu(PyTorch 适配)
海光官方维护的 PyTorch 分支/插件包,把 PyTorch 的 CUDA 后端调用映射到 HIP 路径。
- 版本跟随上游 PyTorch 2.0 / 2.1 分支
- 用户代码里 `.cuda()`、`device='cuda'` 通常可直接沿用(HIP 在 PyTorch 中仍以 `cuda` 设备名暴露),迁移体验接近无感

### 框架适配状态

| 框架 | 适配状态 | 说明 |
|------|---------|------|
| **vLLM** | 有 DCU 分支 | 见 [github.com/Deep-Spark](https://github.com/Deep-Spark);依赖较新 DTK |
| **DeepSpeed** | 社区适配 | HIP 路径可走,部分算子需 patch |
| **Megatron-LM** | 有适配分支 | 通过 torch-dcu + HCCL 运行 TP/PP |
| **SGLang** | 实验性 | FlashInfer 对 DCU 的适配尚不完整 |

## 七、多卡 / 多节点并行

- **数据并行**:torch-dcu + HCCL 走 DDP
- **张量/流水并行**:通过 Megatron-LM 的 DCU 分支
- 机内多卡通过高速互联,跨节点依赖 InfiniBand/RoCE + RDMA

## 八、部署

海光官方提供 Docker 容器镜像,预装对应版本的 DTK + torch-dcu,是版本依赖最稳定的起手方式。生产部署推荐从官方镜像出发,避免手工拼装 DTK / 驱动 / 框架版本带来的兼容性问题。

## 九、与 CUDA 的对应关系速查

| CUDA 组件 | 海光 DCU 对应 |
|-----------|-------------|
| CUDA Toolkit | DTK(ROCm 兼容) |
| CUDA C++ | HIP C++ |
| nvcc | hipcc |
| cuBLAS / cuDNN / cuFFT | hipBLAS / hipDNN(MIOpen) / hipFFT |
| cuSPARSE / cuSolver / cuRAND | hipSPARSE / hipSOLVER / hipRAND |
| NCCL | HCCL(上游 RCCL) |
| cuda-gdb | rocgdb |
| Nsight | rocprof / rocprofiler |
| torch(CUDA) | torch-dcu |
| Warp(32) | Wavefront(传统 64) |
| Shared Memory | LDS |

## 十、开发者社区

- 开发者门户:[developer.hpccube.com](https://developer.hpccube.com)
- 开源平台:[github.com/Deep-Spark](https://github.com/Deep-Spark)
- 产品官网:[hygon.cn](https://www.hygon.cn)

## 总结导图

```
海光 DCU 生态(ROCm 兼容)
├── 底层基础     ROCr Runtime / 驱动 → HIP → DTK
├── 编译工具     hipcc / hipify / rocgdb / rocprof
├── 数学库       hipBLAS / hipDNN(MIOpen) / hipFFT / hipSPARSE / hipSOLVER
├── 通信库       HCCL(上游 RCCL)
├── 框架适配     torch-dcu / vLLM-dcu / DeepSpeed / Megatron
└── 部署         官方 Docker 容器
```

海光路线的核心取舍很清晰:**不自建一套 API,而是复用 ROCm/HIP 的成熟生态**。好处是迁移成本低、工具链完整;代价是性能与功能受上游 ROCm 迭代节奏影响,部分最新的推理加速特性(如 FlashInfer)落地会稍慢。对已有 HIP/ROCm 经验的团队,DCU 是上手最平滑的国产选项之一。
