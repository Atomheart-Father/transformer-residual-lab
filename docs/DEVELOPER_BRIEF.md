# 给本地开发者的研究任务说明

研究主体是冻结预训练 Transformer 的**串行与共享输入并行差值**，为下一轮共享轻量残差预测 Plan B 提供证据。用户已讨论 Plan A 的静态 DAG 与 Plan B 的小网络，但这次明确缩小执行范围：**先测差异，再判断值不值得做、怎么做**。

不要求替换用户现有 Harness、不修改用户其他项目、不安装整套 PyTorch/CUDA 研究工具、不训练补偿网络、不开发生产级 ONNX/DAG 引擎。允许工程师调整本原型不兼容的 MLX 调用，必须把 API 修正及对应 commit 记录下来。

验收顺序：`模型参考加载 → 手工串行等价 → shared-anchor 层差值 → deployable group rollout → 局部窗口与删层对照 → 真实评测报告`。

学术上必须回答：哪些层的误差最大？误差有没有互相抵消？何种并行度使 quality gap 不可接受？即便同一个输入产生的误差高度低秩，能否**仅凭这项证据**认为轻量网络能预测？答案是不能，需要下一轮学习实验才能验证。

如果实验在 Mac GPU 上出错，优先停止并报告：Mac RAM pressure、`mlx`/`mlx-lm` API mismatch、model revision、mask cache 语义差异。千万不要把静态语法成功冒充实际运行完成。
