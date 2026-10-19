# Transformer Residual Lab · Stage 1

**状态：研究设计及测量原型；未在 Apple Silicon 运行，尚无实测数据。**

目标：冻结 Qwen3-0.6B-Base BF16，测量原始串行 Transformer 的逐层残差与共享输入并行近似之间的差异，评估局部校正量、自由推进时的误差累积、最终 NLL/KL，并决定 Plan B 的轻量级补偿网络是否值得训练。

**本阶段不训练、不自动搜索 DAG、不声称获得 GPU 加速。** 当前 Python 计算图只是数学上可以并行，尚未实现 GPU 的真正并发调度。

- [完整阶段一设计](docs/STAGE1_EXPERIMENT.md)
- [本地运行与验收](docs/RUNBOOK.md)
- [前人工作、哪些直接复用](docs/PRIOR_WORK.md)
- [数据输出定义](docs/OUTPUT_SCHEMA.md)
- [报告模板](results/REPORT_TEMPLATE.md)
- [给本地研究助手的范围说明](docs/DEVELOPER_BRIEF.md)

主模型：`mlx-community/Qwen3-0.6B-Base-bf16`（28 层），Apple M4 Pro 24GB 统一内存，可用预算 <10GB，单份权重，MLX-LM。预训练 BF16 权重不做 Q4 量化。模型、数据、版本、采样种子需留存 manifest。

**实现原则**

1. 首先对比参考 `model(tokens)` 与手工 28 层串行，确保 mask/RoPE/norm/LM head 和输出等价，不能用参数调整掩盖差异。
2. 两类误差分别报告：教师状态条件下的局部误差（oracle 诊断）与候选模型从 embedding 自由推进产生的累积误差（可部署质量）。
3. 多层共享输入时只能对完整 block 的 **残差更新量** `B_i(anchor)-anchor` 求和，不要累加完整输出而重复计入 anchor。
4. 统计的 Gram 谱或低秩性不能证明误差可由小网络预测；该问题须留给下一轮学习实验。
5. MLX 异步/延迟执行需要同步；现阶段的 Python 循环不应标成真实 GPU 并发性能。

本仓库不包含模型权重、采样语料、第三方代码副本或伪造的 benchmark 曲线。外部项目遵守各自许可。
