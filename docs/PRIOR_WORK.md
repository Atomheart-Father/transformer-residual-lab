# 可借鉴开源成果核对 · 禁止无谓移植整套框架

本阶段只需要一个 MLX/Qwen3 基础、教师—并行残差诊断和数据评估；不是新造一个完整训练平台。以下参考链接截至 2026-10-19 本轮核实。仓库的许可证、代码接口和 checkpoint 对应版本仍需本地开发者使用前复核。

| 项目 | 上游与用途 | 本阶段决定 | 不能直接照搬的原因 |
|---|---|---|---|
| [mlx-lm](https://github.com/ml-explore/mlx-lm) / [qwen3.py](https://github.com/ml-explore/mlx-lm/blob/main/mlx_lm/models/qwen3.py) | 苹果 GPU 上 Qwen3 BF16 reference block、mask、norm、RoPE、LM head | **直接用作唯一参考实现**；冻结模型，封装现有 blocks | 任何手工重写 Attention 都要先经过等价验收 |
| [Qwen3-0.6B-Base-bf16](https://huggingface.co/mlx-community/Qwen3-0.6B-Base-bf16) | 已转换的 Base 模型，BF16，28 blocks | **直接下载使用**；锁 revision | 与 `Qwen3-0.6B` post-trained 检查点不同 |
| [CQIL](https://github.com/Photooon/CQIL) / [ACL 2024](https://aclanthology.org/2024.acl-long.394/) | 连续近独立层并行与 bypassing | **只借鉴** shared-anchor block 分组及质量对照 | 上游脚本主要针对 LLaMA/Qwen1.5，依赖 PyTorch/CUDA；不可认定可以直接用于 MLX |
| [Leveraging the true depth of LLMs](https://arxiv.org/abs/2502.02790) | 层并行导致的精度损失、实际吞吐量和有效深度 | **只借鉴** 对照方法与定义 | 作者特定框架/多 GPU，不等价于 M4 Pro 的端到端加速 |
| [Transformer Layers as Painters](https://arxiv.org/abs/2407.09298) | 中间层交换、并行、循环保真分析 | **研究参考** 层顺序干预方式 | 第一阶段不实现层重排与迭代修正 |
| [EvoPress](https://github.com/IST-DASLab/EvoPress) | 结构压缩进化搜索 | **暂不安装、不移植**；Plan A 自动 DAG 搜索时再借算法 | 本阶段不搜索，且 PyTorch/CUDA 依赖与 Mac 环境不匹配 |
| [AutoCircuit](https://github.com/UFO-101/auto-circuit) | 特定模型行为的 activation/edge patching | **暂不安装**；后续只有需要边级依赖时参考 | 改装解释性工具会膨胀首轮研究，不直接提供部署 DAG |
| [ACDC](https://github.com/ArthurConmy/Automatic-Circuit-Discovery) | 自动发现重要电路边 | **暂不安装** | 是解释性路径发现，不是可用的并行模型执行器 |
| [DenseFormer](https://github.com/epfml/DenseFormer) | 层表示的加权融合 | **Plan B 参考** 小量参数的系数合并 | 原结构深层仍串行，不应误称它实现了 Plan B |

## 直接复制与许可

“能抄”在此指采用经过许可的代码、引用论文、复现算法。最简单的实际方案是调用 MLX-LM 的 Python 对象 API，而不是复制粘贴一个新的 Qwen3 实现。CQIL/EvoPress 的仓库不能为了少写几行算法而把完整依赖和训练框架拉进本地。

真正要自行写的有限内容：

- 共享输入 `Block(anchor)-anchor` 的抽象计算与组内残差汇合；
- 原始串行教师层的状态采集/局部 teacher-conditioned 干预；
- 完整 vocab KL、teacher forcing NLL、逐层更新误差与统计；
- 实验 manifest、固定样本 token IDs、结果摘要。

不应自行重造：模型权重加载、tokenizer、Qwen3 block、RoPE、标准 causal attention、MLX GPU kernels。第二阶段若需要真正并发调度，再重新选技术栈；第一阶段不把 ONNX 作为硬依赖。
