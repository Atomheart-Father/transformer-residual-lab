"""配置：用户开关、pin、阶段表、CFG、派生路径。

这个模块只放"声明"，不放逻辑。RUN_S2 / RUN_S3 / RUN_S4 存在 CFG 里，
由 run(run_s4=True) 之类的入口写入，再由 _apply_run_switches() 展开。
"""

from pathlib import Path

NOTEBOOK_VERSION = "1.2.1"



# 固定 commit 的源码。sha256 是 **tarball 的**，不是解压后目录的。
# 换版本必须同时换 commit / url / sha256 / bytes / dir 五项，否则 S2 会拒绝解压。
S2_PINNED = {
    "commit": "d454a204ef2b63af357744809c05af897611a72e",
    "url": ("https://github.com/ggml-org/llama.cpp/archive/"
            "d454a204ef2b63af357744809c05af897611a72e.tar.gz"),
    "sha256": "f565ee976503e0b66997a05554d4881a252ced7687811f2a41df771203b54d7f",
    "bytes": 38419835,
    "dir": "llama.cpp-d454a204ef2b63af357744809c05af897611a72e",
}



# 量化目标格式。放在配置 cell（而不是 S4 cell）里，因为 _apply_run_switches()
# 在导入时就要引用它；放在 S4 里会变成"用到时才存在"的隐藏顺序依赖。
FORMATS = ["IQ2_XS", "IQ4_XS"]



# S3 用的 imatrix base：unsloth 的 Q8_0。**不是**量化目标 base（那是 S4 的事）。# 两个 sha256 来源互相独立、都已在 2026-10-22 核对一致：
#   · /api/models/.../tree/<rev>?expand=true  → lfs.oid
#   · resolve URL 响应头                      → x-linked-etag
# 值：0cfbf745…fe86e；字节 8,709,519,168（8.11 GiB）。
# revision 一并钉死：换 revision 必须重核 bytes + sha256，否则下载阶段会拒收。
S3_PINNED = {
    "repo": "unsloth/Qwen3-8B-GGUF",
    "revision": "a6adef130ffb23ddaf1a62fec9dced968c9bc482",
    "filename": "Qwen3-8B-Q8_0.gguf",
    "url": ("https://huggingface.co/unsloth/Qwen3-8B-GGUF/resolve/"
            "a6adef130ffb23ddaf1a62fec9dced968c9bc482/Qwen3-8B-Q8_0.gguf"),
    "sha256": "0cfbf745760f07a76ddeb358dd025a27f2e11d1ca9c9a4169a373d52990fe86e",
    "bytes": 8709519168,
    "kind": "q8_0",
    # HF 在 resolve URL 上回的真实 digest 头；下载时与上面预登记值交叉核对
    "etag_header": "x-linked-etag",
}



# S4 的量化 base：**同一个仓库、同一个 revision** 的原始 BF16。
# 这一点是"imatrix base 与量化 base 同源"的核验骨干 —— Q8_0 与 BF16 是
# unsloth 用同一次转换流程从同一份权重产出的两个文件，repo+revision 相同
# 就保证来自同一 checkpoint；反过来只靠"张量形状一致"证明不了（三个仓库的
# Q8_0 字节数都不同，形状却必然一致）。
# sha256 双源互证（2026-10-22 重新核对）：
#   · tree API lfs.oid = 5e416a2020fe63e76ea13c8979be35fc6070aaf3578f7876400c55c2f5c3eb30
#   · resolve URL 302 响应头 X-Linked-ETag（带引号）= 同一个值
#   · X-Linked-Size / content-length = 16388044384
# 注意：跟随 302 之后 CDN 响应里的裸 `etag`（c47a6551…）是 xet 分块 id，
# **不是**内容 sha256，不能用于校验；只有 X-Linked-ETag 可以。
S4_PINNED = {
    "repo": "unsloth/Qwen3-8B-GGUF",
    "revision": "a6adef130ffb23ddaf1a62fec9dced968c9bc482",
    "filename": "Qwen3-8B-BF16.gguf",
    "url": ("https://huggingface.co/unsloth/Qwen3-8B-GGUF/resolve/"
            "a6adef130ffb23ddaf1a62fec9dced968c9bc482/Qwen3-8B-BF16.gguf"),
    "sha256": "5e416a2020fe63e76ea13c8979be35fc6070aaf3578f7876400c55c2f5c3eb30",
    "bytes": 16388044384,
    "kind": "bf16",
    "etag_header": "x-linked-etag",
}



# 已通过真实 Kaggle T4 运行产出的 imatrix（S3，v0.4.0，ctx=512 / chunks=32，
# base = 上面的 Q8_0）。钉死它，S4 就能在新 Session 里直接复用这份 10.7 MB 的
# 文件，而不必重下 8.11 GiB 的 Q8_0、也不必重算 S3。
# sha256 与字节数来自用户回传的产物本身（2026-10-22 本地复核一致）。
IMATRIX_PINNED = {
    "sha256": "34062ab6d5dfaaef56ee659461c19f4e94254ec14481b07d5531459fbb16ac37",
    "bytes": 10701952,
    "n_tensors": 1008,
    "note": ("v0.4.0 真实 Kaggle T4 运行产物：S3，ctx=512 × chunks=32，"
             "base = S3_PINNED 的 Q8_0，校准语料 = wikitext-2-raw train"),
}



# 校准语料：wikitext-2-raw **train**（与 S6 的 test split 严格分离）。
# sha256 同样来自 tree API 的 lfs.oid，并与 resolve URL 的 x-linked-etag 一致。
CALIB_PINNED = {
    "repo": "Salesforce/wikitext",
    "revision": "b08601e04326c79dfdd32d625aee71d232d685c3",
    "path": "wikitext-2-raw-v1/train-00000-of-00001.parquet",
    "url": ("https://huggingface.co/datasets/Salesforce/wikitext/resolve/"
            "b08601e04326c79dfdd32d625aee71d232d685c3/"
            "wikitext-2-raw-v1/train-00000-of-00001.parquet"),
    "sha256": "e83889baabc497075506f91975be5fac0d45c5290b6b20582c8cd1e853d0c9f7",
    "bytes": 6357543,
    "split": "train",
    "config": "wikitext-2-raw-v1",
    "etag_header": "x-linked-etag",
}



# 评测语料：wikitext-2-raw **test**。与上面的 train split 严格分离（S6 会做
# 内容级重叠断言，重叠非零直接拒绝）。sha256 取自 HF tree API 的 lfs.oid
# （2026-10-22 查询），与 train 的取法一致。
EVAL_PINNED = {
    "repo": "Salesforce/wikitext",
    "revision": "b08601e04326c79dfdd32d625aee71d232d685c3",
    "path": "wikitext-2-raw-v1/test-00000-of-00001.parquet",
    "url": ("https://huggingface.co/datasets/Salesforce/wikitext/resolve/"
            "b08601e04326c79dfdd32d625aee71d232d685c3/"
            "wikitext-2-raw-v1/test-00000-of-00001.parquet"),
    "sha256": "5f1bea067869d04849c0f975a2b29c4ff47d867f484f5010ea5e861eab246d91",
    "bytes": 732610,
    "split": "test",
    "config": "wikitext-2-raw-v1",
    "etag_header": "x-linked-etag",
}



STAGES = {
    "S0_ENV":         True,    # 环境 / GPU / 磁盘检查（秒级）
    "S1_SMOKE":       True,    # 短时 smoke：合成 GGUF 往返 + 显存/磁盘微基准
    "S4_PREFLIGHT":   False,   # S4 前的只读预检（RUN_S4 自动开；下载/编译前 fail-closed）
    "S2_BUILD":       False,   # 由 RUN_S2 / RUN_S3 控制；手改这里也行，但开关更省事
    "S3_IMATRIX":     False,   # 生成 imatrix.gguf（RUN_S3=True 时连 S2 一起自动开）
    "S4_QUANT":       False,   # 产出 IQ2_XS / IQ4_XS（共用同一份 imatrix）
    "S5_BYTES":       False,   # 读文件算真实字节与 bpw
    "S6_NLL":         False,   # 配对 NLL + bootstrap CI
    "S7_QUALITY":     False,   # 有限配对质量检查
    "S8_BF16_ANCHOR": False,   # 高精度锚点（T4 上会主动拒绝）
    "S9_REPORT":      True,    # 汇总（每次重新生成）
    "SX_EXPLORE":     False,   # 第三方现成量化，仅探索
}



# 这些阶段即使 status=="ok" 也重跑
# S4_PREFLIGHT 必须在册：磁盘/暂存盘/imatrix 的状态每次运行都可能变
# （换 Session、清过 working、挂了新 Input），缓存一次 "ok" 会变成假阴性。
ALWAYS_RERUN = {"S9_REPORT", "S4_PREFLIGHT"}



# 前置阶段失败就不再启动（v0.4 / D-6）：S2 编译挂了，S3 不该再报一句
# "找不到 llama-imatrix"，而是直接说"因为 S2 失败了所以没跑"。
# 注意：只在前置**真的失败**（status=="error"）时拦；前置根本没开不拦。
# v1.0.1：S4_PREFLIGHT 是 S2–S5 的共同前置 —— 预检不过就不该花 28 分钟编译、
# 8 GiB 下载，最后才在下载 BF16 时发现磁盘不够。
STAGE_REQUIRES_OK = {
    "S2_BUILD":     ["S4_PREFLIGHT"],
    "S3_IMATRIX":   ["S2_BUILD", "S4_PREFLIGHT"],
    "S4_QUANT":     ["S2_BUILD", "S3_IMATRIX", "S4_PREFLIGHT"],
    "S5_BYTES":     ["S4_QUANT", "S4_PREFLIGHT"],
    # S6/S7 不再要求 S2 成功：v1.1 起它们从 HF 上已存的 GGUF 直接评测，
    # 二进制缺失由阶段自己的预检说清（而不是被 S2 的旧错误拦住）。
    "S6_NLL":       [],
    "S7_QUALITY":   [],
}



CFG = {
    # ---------- 用户开关（notebook 只改这三个）----------
    # run(run_s4=True) 或直接改这里都行。三个都 False 时 _apply_run_switches()
    # 一个字段都不改 —— 误点 Run All 不会联网、不会下载（有测试钉住）。
    "RUN_S2": False,
    "RUN_S3": False,
    "RUN_S4": False,
    # v1.1：只评测，不量化。打开后只跑 S6/S7/S9 —— 从 HF 上已存的 GGUF 直接
    # 取输入，**不编译、不下载 15 GiB BF16、不重做 S3、不重新量化**。
    # 默认 False：一个字节都不联网。
    # 打开后**自带小结果回传**（只 5 个白名单 JSON/MD，不传模型、不传日志）：
    # Kaggle 是可丢弃的算力机，结果不落 HF 就随 Session 一起没了。
    "RUN_EVAL": False,
    # ---------- 模型 ----------
    "MODEL_ID":    "Qwen/Qwen3-8B",
    "PARAM_COUNT": 8.19e9,
    "N_LAYERS":    36,
    "HIDDEN":      4096,
    "N_KV_HEADS":  8,
    "HEAD_DIM":    128,

    # ---------- 文件路径（S3 起需要，全部手动挂载到 /kaggle/input） ----------
    # BASE_GGUF = 量化 base。**正式路径必须是 f16/bf16**：
    #   从已量化的 base 再量化，llama-quantize 需要 --allow-requantize，
    #   且官方帮助明确警告 "can severely reduce quality"。
    "BASE_GGUF":    "",   # 例 "/kaggle/input/qwen3-8b-gguf/qwen3-8b-f16.gguf"
    "BASE_KIND":    "",   # 必须与文件一致："f16" / "bf16" / "q8_0" / "q4_k_m" ...
    # IMATRIX_BASE = 算 imatrix 用的 base，留空则与 BASE_GGUF 相同。
    #   在 T4 上可以让它指向 Q8_0（能进显存，快），而 BASE_GGUF 仍是 f16。
    #   两者必须来自同一 checkpoint（S4 会核对 architecture + 张量形状）。
    "IMATRIX_BASE": "",
    "IMATRIX_BASE_KIND": "",  # 留空则用 BASE_KIND
    "IMATRIX_GGUF": "",   # 留空则 S3 生成到 working。**不要指向 /kaggle/input（只读）**
    "CALIB_TXT":    "",   # imatrix 校准语料
    "EVAL_TXT":     "",   # NLL 评测语料（必须与校准集不同源）
    "EVAL_SOURCE":  "",   # 例 "wikitext-2-raw/test"
    "CALIB_SOURCE": "",   # 例 "wikitext-2-raw/train"
    "EXPLORE_GGUF": "",   # 第三方现成量化，仅探索

    # ---------- llama.cpp ----------
    "LLAMA_SRC":    "",      # 手动源码目录；留空则自动获取（RUN_S2=True）或找 /kaggle/input
    "LLAMA_BIN":    "",      # 已有构建的 bin 目录；留空则按 WORK/build-llama → /kaggle/input 扫描
    "LLAMA_COMMIT": "",      # RUN_S2=True 时自动填成 S2_PINNED["commit"]
    "S2_AUTO_FETCH_SOURCE": False,  # 由 RUN_S2 打开。True 才允许联网取源码
    "BUILD_CUDA":   True,    # False 则只构建 CPU 版（imatrix / quantize 都是 CPU 路径）
    "CUDA_ARCH":    "75",    # T4 = sm_75。留空则用上游默认的多架构列表（4 核上慢数倍）
    # v0.3.1：Kaggle T4 上 find_package(CUDAToolkit) 解析不出 CUDA::cuda_driver
    # IMPORTED target，ggml-cuda 的 target_link_libraries 直接 CMake Error。
    # GGML_CUDA_NO_VMM=ON 只跳过 libcuda 直链 + VMM 显存池（回退 cudaMalloc 池），
    # 算力内核不变。见 ggml/src/ggml-cuda/CMakeLists.txt:178-182 与 common.cuh:265。
    "CUDA_NO_VMM":  True,
    "BUILD_JOBS":   4,

    # ---------- S3 一键取数（RUN_S3=True 时自动打开） ----------
    # imatrix base 与校准语料都由 S3 自己下载；两个都 False 时 S3 退回
    # "读 CFG['IMATRIX_BASE'] / CFG['CALIB_TXT']" 的手动模式。
    "S3_AUTO_FETCH_MODEL": False,
    "S3_AUTO_FETCH_CALIB": False,
    # -ngl 之外的显存余量（GiB）。v0.4 起默认 2.0：Q8_0 8.11 GiB + 2.0 < 12.56 可用。
    "NGL_MARGIN_GB":       2.0,
    # imatrix 验证通过 + 结果记录写盘之后，是否删掉下载来的 Q8_0（磁盘账要求删）。
    # 置 False 可留着当 S8 的近似锚点，代价是 working 只剩约 10.9 GiB。
    "S3_DELETE_Q8_AFTER":  True,
    # v0.5：已有**已验证**的 imatrix 时，S3 直接跳过（不重下 Q8_0、不重算）。
    # 这是"新 Session 也能恢复"的关键：imatrix 只有 10.7 MB，Q8_0 是 8.11 GiB。
    "S3_SKIP_IF_IMATRIX_OK": False,
    # v0.5：imatrix 的 sha256 与 IMATRIX_PINNED 不符时，是否仍然放行。
    # 默认 False —— 换 imatrix 必须显式承认，否则 S4 会拿旧统计量冒充新结果。
    "IMATRIX_ALLOW_UNPINNED": False,

    # ---------- S4 一键量化（RUN_S4=True 时自动打开） ----------
    # 量化 base（BF16）从哪来。**默认 False**：RUN_S4 没开时一个字节都不下。
    # RUN_S4=True 时置 True，然后按优先级自动挑最省事的那条路：
    #   /kaggle/input 已挂 > working 之外的暂存盘 > working（装得下才下）
    "S4_AUTO_FETCH_BASE": False,
    # 下载前先回收"已验证过、但已经用完了"的 Q8_0 缓存（腾出 8.11 GiB）。
    # **默认 False**：只有 S4 真的要为 BF16 腾地方时才需要它（RUN_S4 分支打开）。
    "S4_RECLAIM_Q8_CACHE": False,
    # 量化期间的 RAM 余量建议值（GiB）。llama-quantize 是 CPU 路径 + mmap，
    # 但 --max-buffer-size 默认 8192 MiB，RAM 不够会 exit 137。
    "S4_RAM_MARGIN_GB": 4.0,
    # 每个格式的量化时间上限（秒）。v1.0.1：S4_QUANT 总预算从 1800 s 提到
    # 5400 s（原来 30 分钟要连做两次 8B CPU 量化，误杀风险实在太高），
    # 再给单个格式加一道帽，避免第一个把第二个的时间吃光。
    # **这是保护上限，不是耗时承诺** —— 真实耗时未实测，超了会留日志并
    # 在重跑时按 provenance 复用已完成的格式。
    "S4_PER_FORMAT_BUDGET_SEC": 3600,
    # S4_PREFLIGHT 磁盘账里 S2 build 目录的**估算**占用（GiB）。
    # 真实大小未观测（llama.cpp + CUDA，4 个二进制），只用于预检算术，
    # 刻意估高不估低；S0_ENV 会给实测值。
    "S2_BUILD_DIR_EST_GB": 3.0,

    # ---------- Hugging Face（v1.1：产物存 HF，代码/小记录存 GitHub）----------
    # 已存好的真实量化产物（S4/S5 在 Kaggle 上跑出来的，commit 097e692）。
    # **默认 Private**：所以下面所有下载都要 token；token 从环境/平台 secret/
    # HF 官方缓存里找，绝不打印，也绝不要求写进 notebook / git / issue。
    "HF_MODEL_REPO":     "Atomheart-Father/trlab-qwen3-8b-quant-pilot",
    # 想钉死可填 commit sha（HF 的 commit 不可变）；"main" 更方便但会漂。
    "HF_MODEL_REVISION": "main",
    # 允许 S6/S7 从 HF 取 GGUF（RUN_EVAL=True 会自动打开）。
    # 这不是"泄露数据"，是**读取输入**；默认 False 保证不联网。
    "HF_FETCH_ARTIFACTS": False,
    # 结果上传是单独 opt-in：默认 False = 一个字节都不外发。
    "HF_UPLOAD_AFTER_RUN": False,
    "HF_UPLOAD_REPO":     "",        # 留空则用 HF_MODEL_REPO
    "HF_TIMEOUT_SEC":     60,

    # ---------- S6/S7 评测（v1.1：从 HF 已存 GGUF 直接评测，不重量化）----------
    # **默认用 GPU（有就有效使用）**：Kaggle 上你开的就是 GPU Session，
    # 让 8B 在 4 核 CPU 上慢慢跑是浪费。`EVAL_ALLOW_GPU=True` 且检测到显存时，
    # 按显存算 `-ngl`（装不下才退 0 并**明确说为什么**）；没有 GPU 时才纯 CPU。
    # 要强制纯 CPU（例如想省额度、或只想验证链路）：`--set EVAL_ALLOW_GPU=false`。
    "EVAL_ALLOW_GPU":     True,
    # 纯 CPU 下的墙钟上限（秒）。估算超过它就 fail-closed 并给三个选项，
    # 而不是跑到一半被阶段预算杀掉。GPU 路径不看这个上限（速率假设差一个量级）。
    "EVAL_CPU_MAX_WALL_SEC": 5400,
    # 评测前是否强制核对 GGUF 与 HF tree 的 lfs.oid（4.5 GiB 约 2–4 s/个）。
    "EVAL_VERIFY_HASH":   True,
    # ---- 校准/评测隔离（v1.2.0）----
    # 旧版"见 1 行重叠就整轮拒绝"证明不了污染，也证明不了没污染。现在改成
    # 取证 + 确定性 clean holdout：剔除与校准集共享前 200 字符的**评测段**，
    # 两个量化用同一份 holdout。下面两个阈值是 fail-closed 的护栏：
    #   · 剔除比例超过 10% → 更像"把 train 当成 eval"，必须停下来看；
    #   · 剔除后不足 20 段 → 评测集太小，NLL 没意义。
    "ISOLATION_MAX_DROP_FRAC": 0.10,
    "ISOLATION_MIN_PARAS":     20,

    # ---------- 预算（硬上限，超了主动抛异常而不是被 OOM kill） ----------
    "MAX_WALL_SEC":      8 * 3600,
    "MAX_VRAM_GB":       14.0,
    "MIN_FREE_DISK_GB":  3.0,
    "STAGE_BUDGET_SEC": {
        "S0_ENV": 120, "S1_SMOKE": 240, "S2_BUILD": 2400,
        "S4_PREFLIGHT": 120,
        "S3_IMATRIX": 3600, "S4_QUANT": 5400, "S5_BYTES": 300,
        "S6_NLL": 5400, "S7_QUALITY": 1800, "S8_BF16_ANCHOR": 3600,
        "S9_REPORT": 120, "SX_EXPLORE": 1800,
    },

    # ---------- 实验规模（pilot；v1.1 起按"4 核纯 CPU 跑得完"定默认） ----------
    # 配对比较只要求**两个模型同设置**，不需要把 ctx 拉到 2048。默认
    # 8 组 × 1 chunk × 512 ctx = 4k token/模型、共 8k token，4 核 CPU 上
    # 估算 ≈0.9 h（假设 2.5 tok/s，**不是实测**；llama-perplexity 是批处理，
    # 实际通常更快）。改大之前先看 trlab/artifacts.py 的 eval_cost_estimate：
    # 旧的 8×4×2048 在纯 CPU 上估算 14.6 h，S6 会被阶段预算杀掉。
    "IMATRIX_CTX":           512,
    "IMATRIX_CHUNKS":        32,
    "NLL_CTX":               512,
    "NLL_CHUNKS_PER_GROUP":  1,
    "NLL_GROUPS":            8,      # pilot 级；正式 CI 需按独立文档重新设计
    "QUALITY_PROMPTS":       8,
    "QUALITY_N_PREDICT":     32,

    # ---------- 锚点 ----------
    "ANCHOR_EXTRA_GB":       2.0,    # 权重之外的 KV/context/workspace 估算
    "ANCHOR_ALLOW_APPROX":   False,  # True 才允许用 Q8_0 之类做"近似参照"

    # ---------- 开关 ----------
    "FORCE_RERUN":      False,   # 最后手段：忽略指纹强制全跑
    "ALLOW_REQUANTIZE": False,   # BASE_KIND 是量化类型时必须显式打开（二次量化，仅 pilot）
    "HASH_GGUF":        True,    # 对产物算 sha256
    "BOOTSTRAP_N":      10000,
    "SEED":             1234,
}



def _s4_common():
    """RUN_S4 与 RUN_S3 共用的那部分设置（S2 编译参数 + S3 取数参数）。"""
    STAGES["S2_BUILD"] = True
    STAGES["S3_IMATRIX"] = True
    CFG["S2_AUTO_FETCH_SOURCE"] = True
    CFG["S3_AUTO_FETCH_MODEL"] = True
    CFG["S3_AUTO_FETCH_CALIB"] = True
    CFG["S3_SKIP_IF_IMATRIX_OK"] = True
    CFG["LLAMA_COMMIT"] = S2_PINNED["commit"]
    CFG["CUDA_ARCH"] = "75"    # T4 = sm_75；不指定时上游会编九套架构
    CFG["CUDA_NO_VMM"] = True  # Kaggle T4 上 CUDA::cuda_driver target 找不到（v0.3.1）
    CFG["BUILD_CUDA"] = True   # 三个工具都有 CPU 路径，nvcc 缺失时预检会直接提示
    CFG["BUILD_JOBS"] = 4      # Kaggle 只有 4 核
    CFG["IMATRIX_BASE"] = ""            # 由 S3 下载后填入（v0.5 起不据此判来源）
    CFG["IMATRIX_BASE_KIND"] = S3_PINNED["kind"]
    CFG["IMATRIX_GGUF"] = ""            # 留空 → working/models/imatrix.gguf
    CFG["CALIB_TXT"] = ""               # 由 S3 下载 parquet 后转出来
    CFG["CALIB_SOURCE"] = "wikitext-2-raw/train"
    CFG["NGL_MARGIN_GB"] = 2.0
    CFG["S3_DELETE_Q8_AFTER"] = True



def _apply_run_switches():
    """把 RUN_S2 / RUN_S3 / RUN_S4 三个开关展开成各阶段需要的全部设置。

    三个都 False 时**什么都不改** —— 这一点是刻意的：在 Kaggle 上点 "Run All"
    不应该因为导入 notebook 就悄悄联网下载。想下载必须显式改开关。

    RUN_S3=True 时**隐含打开 S2_BUILD**：imatrix 需要 llama-imatrix 二进制，
    而 Kaggle 交互会话空闲 20 分钟就断，二进制可能已经丢了。所以 S3 不能假设
    "用户已经跑过 S2" —— S2 自己会先找已有二进制（WORK/build-llama /
    CFG['LLAMA_BIN'] / /kaggle/input 自动扫描），找到就零编译复用，找不到才重编。

    RUN_S4=True 时再往上叠 S4_QUANT + S5_BYTES，并把量化 base 设成 BF16。
    **不填 CFG['BASE_GGUF']**：base 从哪来由 S4 自己按优先级决定
    （/kaggle/input 已挂 > working 外暂存盘 > working 装得下才下），
    这样既不会因为用户忘了配路径而失败，也不会盲目往 20 GB 的 working 里塞 15 GiB。
    """
    if CFG["RUN_EVAL"]:
        # v1.1：只评测。输入从 HF 上已存的量化 GGUF 取，**不碰** S2/S3/S4：
        # 不编译（二进制缺失时 S6 自己的预检会说清）、不下载 BF16、不重量化。
        STAGES["S6_NLL"] = True
        STAGES["S7_QUALITY"] = True
        CFG["HF_FETCH_ARTIFACTS"] = True
        # Kaggle 是可丢弃的算力机：小结果不回传就随 Session 一起没了。
        # 只传白名单里的 5 个小文件（不传模型、不传日志、不传 secrets）。
        # 想彻底不外发：--set HF_UPLOAD_AFTER_RUN=false。
        CFG["HF_UPLOAD_AFTER_RUN"] = True
        if not (CFG["RUN_S2"] or CFG["RUN_S3"] or CFG["RUN_S4"]):
            return {"run_eval": True, "run_s2": False, "run_s3": False,
                    "run_s4": False,
                    "stages": [k for k, v in STAGES.items() if v],
                    "eval_models_from": f"hf:{CFG['HF_MODEL_REPO']}"
                                        f"@{CFG['HF_MODEL_REVISION']}",
                    "eval_corpus": f"{EVAL_PINNED['config']}/{EVAL_PINNED['split']}",
                    "eval_device": ("gpu_if_present" if CFG["EVAL_ALLOW_GPU"]
                                    else "cpu_forced"),
                    "upload_results": bool(CFG["HF_UPLOAD_AFTER_RUN"])}

    if not (CFG["RUN_S2"] or CFG["RUN_S3"] or CFG["RUN_S4"]):
        return {"run_s2": False, "run_s3": False, "run_s4": False,
                "run_eval": False}

    if CFG["RUN_S4"]:
        _s4_common()
        STAGES["S4_QUANT"] = True
        STAGES["S5_BYTES"] = True
        # v1.0.1：先跑只读预检再动工。预检不过，后面 28 分钟编译 + 8 GiB
        # 下载 + 15 GiB BF16 都省了 —— 磁盘不够要在最开始就说。
        STAGES["S4_PREFLIGHT"] = True
        # 量化 base = 同仓库同 revision 的原始 BF16。ALLOW_REQUANTIZE 保持 False：
        # 本路线禁止从 Q8_0 二次量化（上游自己就警告质量会严重下降）。
        CFG["BASE_KIND"] = S4_PINNED["kind"]
        CFG["S4_AUTO_FETCH_BASE"] = True
        CFG["S4_RECLAIM_Q8_CACHE"] = True
        CFG["S3_DELETE_Q8_AFTER"] = True
        return {"run_s2": True, "run_s3": True, "run_s4": True, "run_eval": bool(CFG["RUN_EVAL"]),
                "stages": [k for k, v in STAGES.items() if v],
                "commit": CFG["LLAMA_COMMIT"],
                "cuda_arch": CFG["CUDA_ARCH"],
                "imatrix_base": S3_PINNED["filename"],
                "quant_base": S4_PINNED["filename"],
                "calib_source": CFG["CALIB_SOURCE"],
                "formats": FORMATS}

    if CFG["RUN_S3"]:
        _s4_common()
        return {"run_s2": True, "run_s3": True, "run_s4": False, "run_eval": bool(CFG["RUN_EVAL"]),
                "stages": [k for k, v in STAGES.items() if v],
                "commit": CFG["LLAMA_COMMIT"],
                "cuda_arch": CFG["CUDA_ARCH"],
                "imatrix_base": S3_PINNED["filename"],
                "calib_source": CFG["CALIB_SOURCE"]}

    _s2_only()
    return {"run_s2": True, "run_s3": False, "run_s4": False, "run_eval": bool(CFG["RUN_EVAL"]),
            "stages": [k for k, v in STAGES.items() if v],
            "commit": CFG["LLAMA_COMMIT"],
            "cuda_arch": CFG["CUDA_ARCH"]}



def _s2_only():
    """只编 S2：开 S2_BUILD + 填好编译参数，**不碰** S3/S4/S5。

    v1.1.2：`RUN_EVAL=True` 在新 Kaggle Session 上找不到 `llama-perplexity` /
    `llama-cli` 时，由 `run()` 调这个函数自动补编 —— 用户仍然只有一个开关。
    这里只下载 36.6 MiB 的 pinned 源码并编译，**不下载任何模型、不重量化**。
    """
    STAGES["S2_BUILD"] = True
    CFG["S2_AUTO_FETCH_SOURCE"] = True
    CFG["LLAMA_COMMIT"] = S2_PINNED["commit"]
    CFG["CUDA_ARCH"] = "75"    # T4 = sm_75；不指定时上游会编九套架构
    CFG["CUDA_NO_VMM"] = True  # Kaggle T4 上 CUDA::cuda_driver target 找不到（v0.3.1）
    CFG["BUILD_CUDA"] = True   # 三个工具都有 CPU 路径，nvcc 缺失时预检会直接提示
    CFG["BUILD_JOBS"] = 4      # Kaggle 只有 4 核
    return STAGES["S2_BUILD"]



# 这条永远不允许改成 True。S9 会断言。
THIRD_PARTY_IN_COMPARISON = False



# ---- 派生路径 ----
ON_KAGGLE = Path("/kaggle/working").is_dir()



ROOT = Path("/kaggle/working") if ON_KAGGLE else Path.cwd() / "_trlab_run"



WORK = ROOT / "trlab"



OUT  = WORK / "out"



LOGS = WORK / "logs"



CKPT = WORK / "ckpt"



MDLS = WORK / "models"



BUILD = WORK / "build-llama"      # P0-A：绝不在 /kaggle/input 下建 build



SRC_CACHE = WORK / "src"          # 自动下载的源码解压到这里



DL_DIR    = WORK / "dl"           # tarball / 模型 / 语料的缓存（重跑不重下）



MODEL_DL  = DL_DIR / "models"     # 自动下载的 Q8_0 与校准 parquet（与产物分开，便于整目录回收）



IN   = Path("/kaggle/input")


def print_banner():
    """把当前开关与计划打到 stdout —— Kaggle 上这就是"这次会干什么"。"""
    print(f"trlab         : v{NOTEBOOK_VERSION}")

    print(f"运行位置      : {'Kaggle' if ON_KAGGLE else '本地/其他'}")

    print(f"工作目录      : {WORK}")

    print(f"llama.cpp build: {BUILD}")

    print(f"RUN_S2        : {CFG['RUN_S2']}"
          + ("（会下载 36.6 MiB 源码并编译；需要 Internet=On）" if CFG['RUN_S2']
             else "（不联网、不下载）"))

    print(f"RUN_S3        : {CFG['RUN_S3']}"
          + ("（一个开关跑完 S2+S3：确认二进制 → 下 Q8_0 + wikitext → imatrix → 回收）"
             if CFG['RUN_S3'] else "（S3 不启动，不下载任何模型）"))

    print(f"RUN_S4        : {CFG['RUN_S4']}"
          + ("（在 S3 之上再跑 S4 量化 + S5 字节统计；BF16 由 S4 自己取）"
             if CFG['RUN_S4'] else "（S4/S5 不启动，不下载 BF16）"))

    print(f"RUN_EVAL      : {CFG['RUN_EVAL']}"
          + ("（只评测 S6/S7/S9：输入取自 HF 已存 GGUF，不编译、不重量化、不下 BF16；"
             f"{'有 GPU 就用 GPU' if CFG['EVAL_ALLOW_GPU'] else '强制纯 CPU（-ngl 0）'}）"
             if CFG['RUN_EVAL'] else "（不评测）"))

    print(f"开启的阶段    : {[k for k, v in STAGES.items() if v]}")

    if CFG['RUN_S2']:
        print(f"源码          : {S2_PINNED['url']}")
        print(f"                commit {S2_PINNED['commit']}  sha256 {S2_PINNED['sha256'][:16]}…")

    if CFG['RUN_S3']:
        print(f"imatrix base  : 自动下载 {S3_PINNED['repo']}@{S3_PINNED['revision'][:12]}"
              f"/{S3_PINNED['filename']}（{S3_PINNED['bytes']/2**30:.2f} GiB）")
        print(f"                sha256 {S3_PINNED['sha256'][:16]}…（已与 HF lfs.oid / x-linked-etag 双源核对）")
        print(f"校准语料      : 自动下载 {CALIB_PINNED['repo']}@{CALIB_PINNED['revision'][:12]}"
              f"/{CALIB_PINNED['path'].split('/')[-1]}（{CALIB_PINNED['bytes']/2**20:.1f} MiB）")
        print(f"imatrix 参数  : ctx={CFG['IMATRIX_CTX']} chunks={CFG['IMATRIX_CHUNKS']}"
              f"（pilot：约 {CFG['IMATRIX_CTX']*CFG['IMATRIX_CHUNKS']/1000:.1f}k token）")
        print(f"-ngl 余量     : {CFG['NGL_MARGIN_GB']} GiB；OOM 时逐级降低卸载层数")
        print(f"Q8_0 回收     : {'imatrix 验证通过并记录后自动删' if CFG['S3_DELETE_Q8_AFTER'] else '保留'}")

    if CFG['RUN_S4']:
        print(f"imatrix 复用  : 已有且 sha256 与 pin 一致 → 跳过 S3，不重下 Q8_0、不重算")
        print(f"                pin sha256 {IMATRIX_PINNED['sha256'][:16]}…"
              f"（{IMATRIX_PINNED['bytes']/2**20:.1f} MiB，真实 Kaggle 运行产物）")
        print(f"量化 base     : 自动取 {S4_PINNED['repo']}@{S4_PINNED['revision'][:12]}"
              f"/{S4_PINNED['filename']}（{S4_PINNED['bytes']/2**30:.2f} GiB，kind={S4_PINNED['kind']}）")
        print(f"                sha256 {S4_PINNED['sha256'][:16]}…（已与 HF lfs.oid / x-linked-etag 双源核对）")
        print(f"                来源优先级：/kaggle/input 已挂 > working 外暂存盘 > working（装得下才下）")
        print(f"量化格式      : {'/'.join(FORMATS)}（共用同一份 imatrix）")
        print(f"二次量化      : 禁止（ALLOW_REQUANTIZE 保持 False；base 必须是 f16/bf16）")

    if CFG['RUN_EVAL']:
        print(f"评测输入      : HF {CFG['HF_MODEL_REPO']}@{CFG['HF_MODEL_REVISION']}"
              f"（{'已允许拉取' if CFG['HF_FETCH_ARTIFACTS'] else '拉取未允许'}）")
        print(f"评测语料      : {EVAL_PINNED['config']}/{EVAL_PINNED['split']}"
              f"（{EVAL_PINNED['bytes']/2**20:.2f} MiB，与校准集的 train split 严格分离）")
        print(f"隔离口径      : 取证 + clean holdout（剔除与 train 共享前 200 字符的评测段；"
              f"两个量化用同一份）")
        print(f"评测规模      : ctx={CFG['NLL_CTX']} × {CFG['NLL_GROUPS']} 组"
              f" × {CFG['NLL_CHUNKS_PER_GROUP']} chunk"
              f"（约 {CFG['NLL_CTX']*CFG['NLL_GROUPS']*CFG['NLL_CHUNKS_PER_GROUP']/1000:.1f}k token/模型）")
        print(f"执行设备      : {'有 GPU 就整模型卸载（-ngl 99）；装不下才退 CPU 并说明原因' if CFG['EVAL_ALLOW_GPU'] else '强制纯 CPU'}")
        print(f"结果回传      : {'开（只 5 个白名单 JSON/MD，不传模型/日志）' if CFG['HF_UPLOAD_AFTER_RUN'] else '关（结果只落本地，Session 结束即丢）'}")

    print(f"S4–S8         : "
          + ("S4/S5 开，S6–S8 与 SX 全关" if CFG['RUN_S4'] else
             ("S6/S7 开（只评测），S4/S5/S8/SX 关" if CFG['RUN_EVAL'] else
              ("全关（本版不量化、不下 BF16、不跑 8B）" if CFG['RUN_S3'] else "全关"))))

    print(f"量化 base     : {CFG['BASE_GGUF'] or '(未设置)'}  kind={CFG['BASE_KIND'] or '?'}")

    print(f"imatrix base  : {CFG['IMATRIX_BASE'] or '(S3 自动下载)'}"
          f"  kind={CFG['IMATRIX_BASE_KIND'] or CFG['BASE_KIND'] or '?'}")

    print(f"二次量化许可  : {'允许（仅 pilot，结果不可当高保真结论）' if CFG['ALLOW_REQUANTIZE'] else '禁止'}")

    print(f"网络下载      : "
          + ("S2 源码 + S3 的 Q8_0/wikitext + S4 的 BF16（均已 pin + sha256）"
             if (CFG["RUN_S2"] or CFG["RUN_S3"] or CFG["RUN_S4"]) else
             ("HF 上已存的 3 个 GGUF + wikitext test（size/lfs.oid 双源核对）"
              if CFG["RUN_EVAL"] else "禁止")))

    print(f"第三方量化入表: {'允许（不应发生）' if THIRD_PARTY_IN_COMPARISON else '禁止'}")
