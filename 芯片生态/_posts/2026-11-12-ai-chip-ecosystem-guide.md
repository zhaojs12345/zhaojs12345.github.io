---
layout: post
title: "国产 AI 芯片生态全景:SDK、框架适配与开发者社区指南"
date: 2026-11-12 10:00:00 +0800
excerpt: "覆盖英伟达 CUDA 生态与海光、华为昇腾、寒武纪、昆仑芯、燧原、摩尔线程、沐曦、天数智芯等国产 AI 芯片的 SDK 版本体系、主流框架适配状态与开发者社区入口。"
---

本文梳理英伟达及主要国产 AI 芯片的完整软件生态——包括底层运行时/编译器、主流训练推理框架的适配分支、关键依赖版本，以及各家官方开发者社区入口。版本号随厂商节奏持续迭代，以官方 release notes 为准，本文仅作参考基线。

## 分支深度解读导航

本文为总览，每款芯片的逐模块详解见下列分支长文：

- [CUDA 完整生态详解:从底层驱动到上层框架的全景梳理]({% post_url 2026-11-12-cuda-ecosystem-deep-dive %})
- [海光 DCU 完整生态详解:ROCm 兼容路线下的 DTK 软件栈全景]({% post_url 2026-11-12-hygon-dcu-ecosystem %})
- [华为昇腾 CANN 完整生态详解:从达芬奇架构到 MindSpore/MindIE 全栈]({% post_url 2026-11-12-ascend-cann-ecosystem %})
- [寒武纪 MLU 完整生态详解:BANG C 与 Neuware 软件栈全景]({% post_url 2026-11-12-cambricon-mlu-ecosystem %})
- [昆仑芯 XPU 完整生态详解:XRE/XDK 软件栈与 PaddlePaddle 协同全景]({% post_url 2026-11-12-kunlunxin-xpu-ecosystem %})
- [燧原 DCU(邃思/云燧)完整生态详解:TopsRider 软件栈全景]({% post_url 2026-11-12-enflame-gcu-ecosystem %})
- [摩尔线程 MUSA 完整生态详解:高度对标 CUDA 的国产 GPU 软件栈]({% post_url 2026-11-12-mthreads-musa-ecosystem %})
- [沐曦 MACA 完整生态详解:高 CUDA 兼容的曦云 GPU 软件栈]({% post_url 2026-11-12-metax-maca-ecosystem %})
- [天数智芯 IXUCA 完整生态详解:天垓 GPU 的 Corex 软件栈全景]({% post_url 2026-11-12-iluvatar-ixuca-ecosystem %})

---

## 一、NVIDIA CUDA 生态(基准参照)

NVIDIA 是当前 AI 训练推理的基准生态，其他国产芯片普遍以"兼容 CUDA 语义"为目标对齐。

> 本节只列核心 SDK 与框架适配的速查表。CUDA 生态从底层 Driver/Runtime API、PTX/SASS 编译链，到数学库、CUTLASS 模板库、TensorRT/NCCL 以及 RAPIDS 数据科学栈的逐模块详解，见分支长文:[CUDA 完整生态详解:从底层驱动到上层框架的全景梳理]({% post_url 2026-11-12-cuda-ecosystem-deep-dive %})。

### 核心 SDK

| 组件 | 当前主流版本 | 说明 |
|------|-------------|------|
| CUDA Toolkit | 12.1 / 12.4 / 12.6 | 编译器 + 运行时；vLLM/SGLang 以 12.1 为最低要求 |
| cuDNN | 8.9.x / 9.x | 深度学习算子库；cuDNN 9 对 Transformer 有专项优化 |
| NCCL | 2.18 / 2.21 | 多卡通信；版本需与 CUDA 对应 |
| TensorRT | 9.x / 10.x | 推理优化；LLM 场景常用 TRT-LLM 封装层 |
| cuBLAS / cuSPARSE | 随 CUDA Toolkit 发布 | 矩阵/稀疏算子 |

### 框架适配

| 框架 | 推荐 CUDA 版本 | 推荐 PyTorch 版本 | 备注 |
|------|--------------|-----------------|------|
| **vLLM** | ≥ 12.1 | ≥ 2.1 | 官方 pip 包，H100/A100 均支持 |
| **SGLang** | ≥ 12.1 | ≥ 2.1 | 依赖 FlashInfer；H100 FP8 需 CUDA 12.4+ |
| **DeepSpeed** | ≥ 11.8 | ≥ 2.0 | ZeRO-3 / Infinity；编译时需要 CUDA dev headers |
| **Megatron-LM** | ≥ 11.8 | ≥ 2.0 | TP/PP/SP 并行；依赖 Apex、TransformerEngine |
| **FlashAttention-2** | ≥ 11.6（sm80+） | ≥ 2.0 | Ampere+ 架构原生；H100 用 FA-3 |

### 开发者社区

- 官方文档：[docs.nvidia.com](https://docs.nvidia.com)
- CUDA 下载：[developer.nvidia.com/cuda-downloads](https://developer.nvidia.com/cuda-downloads)
- GitHub：[github.com/NVIDIA](https://github.com/NVIDIA)
- NGC 容器：[catalog.ngc.nvidia.com](https://catalog.ngc.nvidia.com)

---

## 二、海光 DCU（深算一号/深算二号）

海光走 ROCm 兼容路线，推出 DTK（Deep Computing Toolkit），是对 ROCm 的二次适配与扩展，使大量 HIP 代码无需修改即可运行。

> 📖 分支深度解读:[海光 DCU 完整生态详解]({% post_url 2026-11-12-hygon-dcu-ecosystem %})

### 核心 SDK

| 组件 | 版本 | 对应关系 |
|------|------|---------|
| DTK（Deep Computing Toolkit） | 23.04 / 24.04 / 24.10 | 对应 ROCm 5.x / 6.x |
| HIP | 随 DTK 发布 | 对齐 AMD HIP API |
| hipBLAS / hipDNN / hipFFT | 随 DTK 发布 | 对应 cuBLAS/cuDNN/cuFFT |
| HCCL | 随 DTK 发布 | 对应 NCCL，多 DCU 通信 |
| DTK PyTorch | torch 2.0 / 2.1 分支 | 官方维护 torch-dcu 包 |

### 框架适配

| 框架 | 适配状态 | 说明 |
|------|---------|------|
| **vLLM** | 有 DCU 分支 | 见 [github.com/Deep-Spark/vllm-dcu](https://github.com/Deep-Spark)；依赖 DTK 23.10+ |
| **DeepSpeed** | 社区适配 | HIP 路径可走，部分算子需 patch |
| **Megatron-LM** | 有适配分支 | 通过 torch-dcu + HCCL 运行 |
| **SGLang** | 实验性 | FlashInfer 对 DCU 的适配尚不完整 |

### 开发者社区

- 开发者门户：[developer.hpccube.com](https://developer.hpccube.com)
- DTK 文档：[github.com/Deep-Spark](https://github.com/Deep-Spark)
- 深算系列产品页：[hygon.cn](https://www.hygon.cn)

---

## 三、华为昇腾（Ascend NPU）

华为昇腾生态最为完整，拥有从编译器（ATC/CANN）到训练框架（MindSpore）再到推理引擎（MindIE）的完整链路，同时维护 torch_npu 以接入 PyTorch 生态。

> 📖 分支深度解读:[华为昇腾 CANN 完整生态详解]({% post_url 2026-11-12-ascend-cann-ecosystem %})

### 核心 SDK

| 组件 | 版本 | 说明 |
|------|------|------|
| **CANN**（Compute Architecture for Neural Networks） | 7.0 / 8.0 / 8.1 | 核心运行时+编译器；相当于 CUDA Toolkit |
| torch_npu | 2.1.0 / 2.3.1 | PyTorch 昇腾适配；版本号与 PyTorch 对齐 |
| MindSpore | 2.3 / 2.4 | 华为自研框架；对昇腾优化更深 |
| **MindIE** | 1.0 / 2.0 | 推理引擎；对标 TensorRT |
| HCCL | 随 CANN 发布 | 昇腾多卡通信库；对标 NCCL |
| Ascend C | CANN 7.0+ | 自定义算子语言；对标 CUDA C |

**关键版本绑定关系（参考）：**

```
CANN 8.0 → torch_npu 2.3.1 → Python 3.9 / 3.10
CANN 7.0 → torch_npu 2.1.0 → Python 3.8 / 3.9
```

### 框架适配

| 框架 | 适配状态 | 说明 |
|------|---------|------|
| **vLLM** | 官方支持，有专属分支 | [github.com/vllm-project/vllm-ascend](https://github.com/vllm-project/vllm-ascend)；依赖 CANN 8.0+ |
| **DeepSpeed** | 华为官方维护补丁 | 通过 torch_npu 拦截 CUDA 调用 |
| **Megatron-LM** | 有 Ascend 分支 | 华为内部维护，部分开源 |
| **SGLang** | 适配进行中 | 社区贡献；部分功能受限 |
| **LLaMA-Factory** | 支持 | 通过 torch_npu 透传 |

### 开发者社区

- 官方开发者社区：[hiascend.com/developer](https://www.hiascend.com/developer)
- CANN 文档：[hiascend.com/document](https://www.hiascend.com/document)
- torch_npu GitHub：[github.com/Ascend/pytorch](https://github.com/Ascend/pytorch)
- 模型库 MindSpore Hub：[gitee.com/mindspore](https://gitee.com/mindspore)
- 昇腾 ModelZoo：[gitee.com/ascend/ModelZoo-PyTorch](https://gitee.com/ascend/ModelZoo-PyTorch)

---

## 四、寒武纪 MLU（思元系列）

寒武纪推出 Neuware SDK，以 BANG C 为自定义算子语言，torch_mlu（也称 Catch）作为 PyTorch 接入层。

> 📖 分支深度解读:[寒武纪 MLU 完整生态详解]({% post_url 2026-11-12-cambricon-mlu-ecosystem %})

### 核心 SDK

| 组件 | 版本 | 说明 |
|------|------|------|
| **Neuware SDK** | 3.5 / 3.8 / 3.10 | 运行时 + BANG C 编译器 |
| torch_mlu（Catch） | 1.17 / 2.0 / 2.1 | PyTorch 适配层；版本跟随 PyTorch |
| **CNCL**（Cambricon NCCL） | 随 Neuware 发布 | 多 MLU 通信库 |
| MagicMind | 随 Neuware 发布 | 推理优化引擎；对标 TensorRT |
| CNLight | 单独发布 | 轻量推理运行时 |

**版本绑定（参考）：**

```
Neuware 3.8 → torch_mlu 2.1 → PyTorch 2.1 → Python 3.8 / 3.10
Neuware 3.5 → torch_mlu 1.17 → PyTorch 1.13
```

### 框架适配

| 框架 | 适配状态 | 说明 |
|------|---------|------|
| **vLLM** | 有 MLU 适配分支 | 寒武纪官方维护；依赖 Neuware 3.8+ |
| **DeepSpeed** | 社区适配 | 通过 torch_mlu 路径 |
| **Megatron-LM** | 有适配工作 | 官方有测试用例 |

### 开发者社区

- 开发者社区：[developer.cambricon.com](https://developer.cambricon.com)
- SDK 下载：[sdk.cambricon.com](https://sdk.cambricon.com)
- GitHub：[github.com/Cambricon](https://github.com/Cambricon)

---

## 五、昆仑芯 XPU（R200/R300 系列）

昆仑芯脱胎于百度 AI 加速器，专为推理和训练优化，XRE/XDK 是其核心运行时和开发套件，PaddlePaddle 是最成熟的上层框架。

> 📖 分支深度解读:[昆仑芯 XPU 完整生态详解]({% post_url 2026-11-12-kunlunxin-xpu-ecosystem %})

### 核心 SDK

| 组件 | 版本 | 说明 |
|------|------|------|
| **XRE**（Kunlun Runtime Environment） | 4.x / 5.x | 底层运行时 |
| **XDK**（Kunlun Developer Kit） | 4.x / 5.x | 编译器 + 工具链，与 XRE 版本对应 |
| XCCL | 随 XDK 发布 | 多 XPU 通信库 |
| torch_xpu（XPU 版） | 2.0 / 2.1 | PyTorch 昆仑适配；与 PaddlePaddle 并行维护 |
| **PaddlePaddle-XPU** | 2.5 / 2.6 | 支持最完整的上层框架 |

### 框架适配

| 框架 | 适配状态 | 说明 |
|------|---------|------|
| **vLLM** | 实验性适配 | 通过 torch_xpu 路径 |
| **DeepSpeed** | PaddlePaddle 路径较成熟 | PyTorch 路径社区维护 |
| **Megatron-LM** | 适配进行中 | — |

### 开发者社区

- 官方开发者门户：[developer.kunlunxin.com](https://developer.kunlunxin.com)
- XDK 文档：[kunlunxin.com/developer](https://www.kunlunxin.com/developer)
- GitHub：[github.com/kunlunxin](https://github.com/kunlunxin)

---

## 六、燧原科技 DCU（邃思 / 云燧系列）

燧原科技（Enflame）的芯片基于**邃思（SuiSi）架构的 DTU（Deep Thinking Unit，深度思考单元）**，加速卡品牌为**云燧（CloudBlazer）**（训练 T 系列、推理 i 系列），定位训练推理一体，TopsRider 是完整的软件栈名称。

> 📖 分支深度解读:[燧原 DCU（邃思/云燧）完整生态详解]({% post_url 2026-11-12-enflame-gcu-ecosystem %})

### 核心 SDK

| 组件 | 版本 | 说明 |
|------|------|------|
| **TopsRider SDK** | 2.3 / 3.0 | 完整软件栈；包含编译器+运行时+通信库 |
| TopsCC | 随 TopsRider 发布 | 自定义算子编译器，对标 NVCC |
| TCCL | 随 TopsRider 发布 | 多 GCU 通信库 |
| **torch_gcu** | 2.0 / 2.1 | PyTorch GCU 适配 |
| TopsInference | 随 TopsRider 发布 | 推理引擎 |

**版本绑定（参考）：**

```
TopsRider 3.0 → torch_gcu 2.1 → PyTorch 2.1 → Python 3.8 / 3.10
```

### 框架适配

| 框架 | 适配状态 | 说明 |
|------|---------|------|
| **vLLM** | 有适配分支 | 官方提供；依赖 TopsRider 3.0+ |
| **DeepSpeed** | 社区适配 | 通过 torch_gcu |
| **Megatron-LM** | 适配进行中 | — |

### 开发者社区

- 开发者社区：[developer.enflame.cn](https://developer.enflame.cn)
- 文档中心：[docs.enflame.cn](https://docs.enflame.cn)
- GitHub：[github.com/EnflameTechnology](https://github.com/EnflameTechnology)

---

## 七、摩尔线程 MUSA（MTT S80/S4000 系列）

摩尔线程的 MUSA（Moore Threads Unified System Architecture）接口设计高度对标 CUDA，API 语义几乎一一对应，迁移门槛低是其核心卖点。

> 📖 分支深度解读:[摩尔线程 MUSA 完整生态详解]({% post_url 2026-11-12-mthreads-musa-ecosystem %})

### 核心 SDK

| 组件 | 版本 | 对标 |
|------|------|------|
| **MUSA SDK** | 2.0 / 2.1 / 3.0 | CUDA Toolkit |
| muBLAS / muDNN / muFFT | 随 MUSA SDK | cuBLAS / cuDNN / cuFFT |
| MCCL | 随 MUSA SDK | NCCL |
| **torch_musa** | 2.0 / 2.2 | PyTorch 适配；版本跟随 PyTorch |
| MUSArt | 随 SDK 发布 | 推理运行时 |

**版本绑定（参考）：**

```
MUSA SDK 3.0 → torch_musa 2.2 → PyTorch 2.2 → Python 3.9 / 3.10
MUSA SDK 2.1 → torch_musa 2.0 → PyTorch 2.0
```

### 框架适配

| 框架 | 适配状态 | 说明 |
|------|---------|------|
| **vLLM** | 官方维护 MUSA 分支 | [github.com/MooreThreads/vllm-musa](https://github.com/MooreThreads)；依赖 MUSA 3.0+ |
| **SGLang** | 实验性 | — |
| **DeepSpeed** | torch_musa 路径可走 | — |
| **Megatron-LM** | 适配进行中 | — |

### 开发者社区

- 开发者门户：[developer.mthreads.com](https://developer.mthreads.com)
- MUSA 文档：[docs.mthreads.com](https://docs.mthreads.com)
- GitHub：[github.com/MooreThreads](https://github.com/MooreThreads)

---

## 八、沐曦 GPU（曦云 C500/曦计算系列）

沐曦采用 MACA（MetaX Accelerated Computing Architecture）软件栈，同样以高 CUDA 兼容性为卖点，支持直接编译大量 CUDA 代码。

> 📖 分支深度解读:[沐曦 MACA 完整生态详解]({% post_url 2026-11-12-metax-maca-ecosystem %})

### 核心 SDK

| 组件 | 版本 | 说明 |
|------|------|------|
| **MACA SDK** | 2.0 / 2.1 | 运行时 + 编译器；CUDA 语义兼容 |
| hipMACA | 随 SDK | HIP 层适配 |
| MCCL（沐曦版） | 随 SDK | 多卡通信库 |
| **torch_maca** | 2.0 / 2.1 | PyTorch 适配 |

### 框架适配

| 框架 | 适配状态 | 说明 |
|------|---------|------|
| **vLLM** | 有官方 MACA 适配 | 依赖 MACA 2.0+ |
| **DeepSpeed** | 社区适配 | — |

### 开发者社区

- 开发者社区：[developer.metax-tech.com](https://developer.metax-tech.com)
- 技术文档：[docs.metax-tech.com](https://docs.metax-tech.com)
- GitHub：[github.com/MetaX-ICS](https://github.com/MetaX-ICS)

---

## 九、天数智芯 GPU（天垓 100 系列）

天数智芯面向 HPC 与 AI 双向市场，IXUCA SDK 是其软件栈名称，iluvatar-corex 是开源生态入口。

> 📖 分支深度解读:[天数智芯 IXUCA 完整生态详解]({% post_url 2026-11-12-iluvatar-ixuca-ecosystem %})

### 核心 SDK

| 组件 | 版本 | 说明 |
|------|------|------|
| **IXUCA SDK**（iluvatar Unified Computing Architecture） | 3.1 / 4.0 | 运行时 + 编译器工具链 |
| ixBLAS / ixDNN | 随 SDK | 对应 cuBLAS/cuDNN |
| ICCL | 随 SDK | 多卡通信库 |
| **torch_corex** / torch-iluvatar | 2.0 / 2.1 | PyTorch 适配 |

### 框架适配

| 框架 | 适配状态 | 说明 |
|------|---------|------|
| **vLLM** | 有适配工作 | 通过 torch_corex 路径 |
| **DeepSpeed** | 社区适配 | — |

### 开发者社区

- 开发者门户：[developer.iluvatar.com](https://developer.iluvatar.com)
- GitHub：[github.com/Deep-Spark](https://github.com/Deep-Spark)（天数与海光共用 Deep-Spark 开源平台）
- 文档中心：[docs.iluvatar.com](https://docs.iluvatar.com)

---

## 十、各厂商芯片代际与英伟达对标性能速查

> ⚠️ **重要说明（务必先读）**：下表的"大致性能"是**相对所对标英伟达型号的粗略百分比**，仅供量级参考，**不构成选型依据**。原因有三：
> 1. **口径差异大**：同一张卡 FP16 / BF16 / FP8 / INT8 的相对表现可能差数倍，表中若未特别注明，默认按 **FP16/BF16 稠密算力或公开实测推理吞吐**的量级给出。
> 2. **纸面 ≠ 实测**：厂商标称峰值常接近甚至超过对标卡，但端到端（叠加编译器/算子库/通信库成熟度）实测通常**明显低于**纸面值，国产卡尤甚。表中尽量给**实测区间**，拿不到实测的用纸面并标注。
> 3. **数据来源混杂**：部分为厂商宣传、部分为第三方实测、部分为公开估算，均随版本迭代变化。**以各厂商官方 release 与实机实测为准。**

### 对标与性能概览

| 厂商 | 芯片代际 / 型号 | 大致年代 | 对标 NV 型号 | 大致性能（占对标卡的百分比） | 口径 / 备注 |
|------|---------------|---------|-------------|--------------------------|------------|
| **海光** | BW100（深算一号系） | 2021 | V100 | ~60–70% | ROCm/DTK 兼容路线早期 |
| | BW200（深算二号系） | 2023 | A100 | ~60–80% | FP16，软件生态拉低端到端；本项目实测卡 |
| | BW1000（新一代） | 2024+ | A100 / A800 | ~80–90% | 面向大模型训推，DTK 26.04 |
| **华为昇腾** | 910（2019） | 2019 | V100 → A100 | 纸面 FP16 ~320T（≈A100 相当），实测 ~70–90% | 达芬奇架构首代 |
| | 910B | 2023 | A100 | 纸面 FP16 ~400T（超 A100），实测 ~80% | 国内最成熟的 A100 替代之一 |
| | 910C | 2025 | H100 | 纸面 FP16 ~800T，推理实测 ~H100 的 60–80% | 双 910B chiplet 设计 |
| | 910D | 在研 | H100 / H200 | 待官方数据 | — |
| **寒武纪** | 思元 290 | 2021 | V100 | ~V100 级 | 训练卡首代 |
| | 思元 370 | 2022 | A10 / A30 | 对标中端推理卡 | 7nm，训推一体 |
| | 思元 590（MLU590） | 2023–24 | A100 | ~80% | 当前主力训练卡 |
| | 思元 690 | 新一代 | H100 方向 | 待公开 | — |
| **昆仑芯** | 昆仑 1 代（K200） | 2020 | T4 / V100（推理） | 对标推理卡 | 脱胎百度自研 |
| | 昆仑 2 代（R200 / R300） | 2021–22 | A10 / A30 | 对标中端 | — |
| | P800（3 代） | 2024 | A100 | ~A100 级 | 当前主力，PaddlePaddle 协同最佳 |
| **燧原科技** | 云燧 T20 / i20（邃思 2.0） | 2021 | A100（训练）/ T4（推理） | 对标 A100 级 / 中端推理 | T 训练、i 推理 |
| | 邃思 3.0 新一代 | 2023+ | A100+ | 待公开实测 | — |
| **摩尔线程** | S2000 / S3000 | 2022 | T4 / A10 | 早期数据中心卡 | — |
| | MTT S4000 | 2023 | A100 | ~30–50%（FP16 ~100T vs A100 312T；INT8 ~200 TOPS） | 迁移门槛低是卖点 |
| | MTT S5000 | 2025 | A100 | 单卡 Prefill 实测 ~国际高端卡的 46–54% | 当前主力 |
| **沐曦** | 曦云 C500（MXC500） | 2023 | A100 | ~80%（官方优化目标与现状） | 7nm，高 CUDA 兼容 |
| | 曦云 C550 | 2024 | A100 / A800 | 接近 A100 | 旗舰 OAM 模组 |
| **天数智芯** | 天垓 100（BI-V100） | 2021 | A100（部分） | 约 A100 的 50%± | HPC/AI 双向 |
| | 天垓 150（BI-V150） | 2023–24 | A100 | 较上代提升，~A100 级 | 当前主力 OAM |

**几个要点：**

- **"对标"≠"等同"**：厂商宣称对标 A100/H100，多指**内存容量、互联带宽、纸面峰值算力**对齐；端到端大模型训练/推理还要看算子库（如 FlagGems 等）、通信库、编译器成熟度，国产卡这块通常是真正的差距所在。
- **精度维度很关键**：INT8/FP8 推理场景国产卡相对差距往往比 FP16 训练场景小，部分推理卡在 INT8 上账面接近对标卡，但训练 FP16/BF16 差距更明显。
- **代际在快速迭代**：昇腾 910C、寒武纪 690、摩尔 S5000、沐曦 C550 等新一代都在显著收窄与 A100/H100 的差距，上表百分比会随软件栈优化持续变化。

---

## 十一、横向对比速查

| 芯片 | 软件栈名称 | PyTorch 适配包 | 通信库 | vLLM 适配 | 自定义算子语言 |
|------|-----------|--------------|--------|----------|-------------|
| NVIDIA | CUDA / cuDNN | torch（官方） | NCCL | 官方支持 | CUDA C |
| 海光 DCU | DTK（ROCm 兼容） | torch-dcu | HCCL | DCU 分支 | HIP C |
| 华为昇腾 | CANN | torch_npu | HCCL | vllm-ascend | Ascend C |
| 寒武纪 | Neuware | torch_mlu | CNCL | MLU 分支 | BANG C |
| 昆仑芯 | XRE / XDK | torch_xpu | XCCL | 实验性 | — |
| 燧原（邃思/云燧） | TopsRider | torch_gcu | TCCL | DTU 分支 | TopsCC |
| 摩尔线程 | MUSA SDK | torch_musa | MCCL | MUSA 分支 | MUSA C |
| 沐曦 | MACA SDK | torch_maca | MCCL | MACA 适配 | HIP/MACA |
| 天数智芯 | IXUCA SDK | torch_corex | ICCL | 适配进行中 | — |

---

## 附：常见问题

**框架版本不兼容怎么办？** 国产芯片的 torch 适配包版本号通常与上游 PyTorch 对齐，但有滞后。建议先用各家官方提供的容器镜像（Docker）起手，这是版本依赖最稳定的方式。

**有没有统一的多芯片框架？** FlagOpen 的 FlagGems 和 OneFlow 在尝试抽象底层异构后端；另外 PaddlePaddle 对多国产芯片的适配广度最宽，可以作为跨芯片参考。

**这些 SDK 能公开下载吗？** 华为昇腾、摩尔线程、海光、寒武纪都有公开的开发者门户可注册下载；部分厂商（如昆仑芯、天数）需要商务渠道或合作协议，以各家官网最新公告为准。
