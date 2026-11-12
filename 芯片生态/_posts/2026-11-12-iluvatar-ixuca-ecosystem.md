---
layout: post
title: "天数智芯 IXUCA 完整生态详解:天垓 GPU 的 Corex 软件栈全景"
date: 2026-11-12 12:20:00 +0800
excerpt: "逐模块梳理天数智芯 IXUCA/Corex 的软件生态:底层运行时与 CUDA 兼容机制、编译工具链、ixBLAS/ixDNN 数学库、ICCL 通信,以及 torch_corex 的框架适配与 Deep-Spark 开源平台,并给出与 CUDA 的对应关系。"
---

> 本文是[国产 AI 芯片生态全景:SDK、框架适配与开发者社区指南]({% post_url 2026-11-12-ai-chip-ecosystem-guide %})的天数智芯部分展开。总文给出各家芯片的横向速查,本文专注把天数智芯软件生态逐模块讲透。

天数智芯(Iluvatar CoreX)是国内较早做通用 GPU(GPGPU)的厂商,面向 **HPC 与 AI 双向市场**。代表产品是天垓(BI)系列,如天垓 100(BI-V100)、BI-V150,以及智铠推理系列。其软件栈为 **IXUCA(Iluvatar Unified Computing Architecture)**,也常以 **Corex** 之名出现;开源生态入口是与海光共用的 **Deep-Spark** 平台。天数走高 CUDA 兼容路线。

> 说明:下文版本号与部分组件命名为写作时的参考基线,天数部分底层细节公开资料有限,实际以官方 release notes 与文档为准。

## 一、底层基础层

### 1. IXUCA SDK(Corex 软件栈)
天数智芯软件栈总称,对标 CUDA Toolkit。版本如 IXUCA/Corex 3.1 / 4.0,包含运行时、编译器工具链、数学库(ix 系列)、通信库 ICCL、框架适配。

### 2. 运行时层
负责设备管理、内存分配、流/事件、kernel 调度,对标 CUDA Runtime/Driver。天数 GPU 定位通用计算,API 设计贴合 CUDA 语义以利迁移。

### 3. 天垓 GPU 架构
BI 系列定位 HPC + AI 通用计算,内含矩阵/向量计算单元,采用 HBM 满足高带宽需求,面向训练与科学计算。

## 二、编程模型核心概念

### CUDA 兼容机制
天数 GPU 以高 CUDA 兼容为设计目标:
- 大量 `cuda*` API 有对应实现,CUDA C++ 代码可较低成本编译运行
- 线程层次(Grid → Block → Warp → Thread)与内存层次(寄存器 → Shared Memory → L2 → Global Memory/HBM)贴合 CUDA 语义

> 底层 kernel 编程的公开细节有限,多数用户通过 torch_corex 高层算子即可,无需直接写 kernel。

## 三、编译工具链

IXUCA 提供编译器工具链(基于 LLVM),把设备代码编译为天垓 GPU 可执行码,角色对应 nvcc。配套调试与性能分析工具(对应 cuda-gdb / Nsight),以及设备监控工具查看 GPU 利用率、显存、功耗(角色类似 nvidia-smi)。

## 四、核心数学库

IXUCA 提供前缀 "ix" 的数学库,对标 cuXXX:

| IXUCA 库 | 功能 | 对标 CUDA |
|----------|------|----------|
| ixBLAS | 稠密线性代数 | cuBLAS |
| ixDNN | 深度学习算子 | cuDNN |
| ixFFT | 快速傅里叶变换 | cuFFT |
| ixSPARSE | 稀疏矩阵运算 | cuSPARSE |

> 具体库命名与覆盖范围以 IXUCA SDK 文档为准。

## 五、通信库

### ICCL(Iluvatar Collective Communications Library)
多 GPU / 多节点集合通信库,对标 NCCL。
- AllReduce、AllGather、ReduceScatter、Broadcast 等
- 多卡训练的通信底层
- API 贴合 NCCL 语义

## 六、AI 推理

天数结合 ixDNN 算子与图优化支持 GPU 推理,面向智铠推理系列做针对性优化;也支持通过 torch_corex 跑 vLLM 等推理框架。HPC 场景下其通用计算能力(双精度等)是区别于纯 AI 芯片的特点。

## 七、框架适配

### torch_corex / torch-iluvatar(PyTorch 适配)
天数维护的 PyTorch 适配包,把 PyTorch 运算映射到天垓 GPU。
- 版本如 2.0 / 2.1,跟随上游 PyTorch
- 得益于 CUDA 兼容,迁移改动较小

### 主流框架适配状态

| 框架 | 适配状态 | 说明 |
|------|---------|------|
| **vLLM** | 有适配工作 | 通过 torch_corex 路径 |
| **DeepSpeed** | 社区适配 | — |

框架适配与模型样例多托管在 Deep-Spark 开源平台。

## 八、多卡 / 多节点并行

- **数据并行**:torch_corex + ICCL 走 DDP
- **张量/流水并行**:通过框架并行策略适配
- 机内多卡互联,跨节点 RDMA,ICCL 做集合通信

## 九、部署

天数智芯依托 **Deep-Spark** 开源平台(与海光共用的 github.com/Deep-Spark)提供框架适配、模型样例与容器镜像。生产推荐从官方/Deep-Spark 镜像起手锁定版本依赖。

## 十、与 CUDA 的对应关系速查

| CUDA 组件 | 天数智芯 IXUCA 对应 |
|-----------|-------------------|
| CUDA Toolkit | IXUCA SDK(Corex) |
| CUDA Runtime/Driver | IXUCA 运行时 |
| CUDA C + nvcc | IXUCA 编译工具链 |
| cuBLAS / cuDNN / cuFFT | ixBLAS / ixDNN / ixFFT |
| cuSPARSE | ixSPARSE |
| NCCL | ICCL |
| nvidia-smi | 天数设备监控工具 |
| torch(CUDA) | torch_corex / torch-iluvatar |

## 十一、开发者社区

- 开发者门户:[developer.iluvatar.com](https://developer.iluvatar.com)
- 文档中心:[docs.iluvatar.com](https://docs.iluvatar.com)
- 开源平台:[github.com/Deep-Spark](https://github.com/Deep-Spark)(与海光共用)

## 总结导图

```
天数智芯 IXUCA 生态(高 CUDA 兼容,HPC+AI)
├── 底层基础     IXUCA 运行时 → 天垓 GPU 架构(HBM)
├── 软件栈       IXUCA SDK(Corex)
├── 编译工具     IXUCA 编译工具链(对标 nvcc)
├── 数学库       ixBLAS / ixDNN / ixFFT / ixSPARSE
├── 通信库       ICCL
├── 框架适配     torch_corex / vLLM / DeepSpeed
└── 部署         Deep-Spark 开源平台 + 官方容器
```

天数智芯的定位是**通用 GPU、面向 HPC 与 AI 双场景**,走高 CUDA 兼容路线(ix 系列库对标 cuXXX、ICCL 对标 NCCL)。与海光共用 Deep-Spark 开源平台是其生态上的一个特点。对既有 HPC 科学计算、又有 AI 需求的团队,天数提供了一个兼顾双精度通用计算与深度学习的国产选项。底层编程公开资料相对有限,实际落地以 torch_corex 适配与 Deep-Spark 镜像为主。
