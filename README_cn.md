# DwarfStar 4（中文版）

> 本文件是 [README.md](README.md) 的中文翻译与扩写版本，并额外补充：
>
> - **与 Qwen3.5-122B-A10B INT4 的对比**（DS4 + DeepSeek V4 Flash 的优势 / 劣势）
> - **Docker 容器化部署**（ARM64 + CUDA，针对 DGX Spark / GB10）
>
> 原始英文 README 是项目权威文档，本文档若出现版本滞后，请以英文版为准。

---

## 项目简介

**DwarfStar 4（DS4）** 是一个**专为 DeepSeek V4 Flash 模型**打造的小型本地推理引擎。它有意保持窄目标：

- **不是**通用 GGUF 加载器
- **不是**对其它推理库的包装
- 完全自包含，只链接 libc / pthread / Metal / CUDA

除了"以正确且快速的方式运行模型"外，项目目标还包括：DS4 专用的加载、Prompt 渲染、工具调用、KV 状态管理（RAM + 磁盘）、Server API —— 开箱即用，可直接对接 coding agent 或自带的 CLI。仓库还附带 GGUF / imatrix 生成工具、质量评测和速度评测。

支持的后端：

- **Metal**：主要目标，从 96GB RAM 的 MacBook 起
- **NVIDIA CUDA**：特别照顾 DGX Spark
- **AMD ROCm**：仅在 [rocm](https://github.com/antirez/ds4/tree/rocm) 分支维护

项目离不开 **llama.cpp 和 GGML**，详见英文 README 的致谢章节。

---

## 为什么选 DeepSeek V4 Flash？

作者列出的几条理由（README 第 27–34 行）：

1. **激活参数少 → 更快**（13B active / 284B total）
2. **思考长度自适应**：避开"max thinking"时，thinking section 比同类模型短得多（往往只有 1/5），且**长度正比于问题复杂度**。这让 V4 Flash 在打开思考模式时仍然实用
3. **1M tokens 上下文**
4. **总参数大 = 知识广**：284B 在长尾知识、小语种、冷门领域上明显强于 27B / 35B 级模型
5. **英语和意大利语写作质量高**（"feels a quasi-frontier model"）
6. **KV 缓存极度压缩** → 长上下文本地推理 + **KV 落盘持久化**可行
7. **2-bit 量化质量可用**：经过特殊量化（见下文），128GB Mac 能跑，部分用户报告 96GB Mac 也能跑 250k 上下文
8. **预期 DeepSeek 会持续发布 V4 Flash 更新版**

需要诚实说明的几点：

- 本地推理生态项目很多，新模型频出，注意力总在被"下一个模型"切走。**本项目刻意只押一个模型**：单模型 + 官方 logits 验证 + 长上下文测试 + 充分的 agent 集成
- 项目主要由 **GPT 5.5 强力辅助**编写，人主导思路、测试和调试。如果你不接受 AI 辅助开发的代码，本项目不适合你
- 项目的核心论点：**现代 MacBook 的 SSD 速度 + 极度压缩的 KV 缓存 = KV 应该被当成一等磁盘公民**
- 优化的图路径只针对 **macOS Metal** 和 **Linux CUDA**。CPU 路径只用于正确性检查和模型 / 分词器诊断。`make cpu` 可以构建 CPU-only Linux 版本。**注意：当前 macOS 版本在虚拟内存实现上有 bug，运行 CPU 路径会让内核崩溃**
- 这是 alpha 质量代码

---

## 与 Qwen3.5-122B-A10B INT4 的对比

本节是中文版独有内容，作为选型参考。

### 模型规格对比

| 维度 | **DeepSeek V4 Flash**（本仓库目标） | **Qwen3.5-122B-A10B INT4** |
| --- | --- | --- |
| 总参数 | **284B** | 122B |
| 激活参数 | **13B** | 10B |
| 架构 | MoE + MLA 压缩 KV + routed/shared experts | MoE（A10B = 激活 10B） |
| 上下文 | **1M tokens** | 通常 256K |
| 量化文件大小 | q2 ≈ 81 GB / q4 ≈ 153 GB | INT4 ≈ 65–75 GB |
| 本仓库支持 | ✅ 唯一目标 | ❌ 不支持 |

> **重要前提**：DS4 是 V4 Flash 专用引擎，**无法用来跑 Qwen**。因此下面对比的是"DS4 + V4 Flash"这一整套方案，与"llama.cpp / vLLM / SGLang + Qwen3.5"的整体差异，不是纯模型 benchmark。

### DS4 + V4 Flash 相对 Qwen3.5-122B-A10B INT4 的**优势**

1. **总知识容量约 2.3 倍**
   - MoE 总参决定知识广度，激活参数决定速度。V4 Flash 总参 284B vs Qwen 122B
   - 长尾知识、小语种、冷门领域上理论优势明显

2. **KV 缓存压缩 —— V4 Flash 最大杀器**
   - V4 Flash 用 **MLA（Multi-head Latent Attention）**，KV cache 被极度压缩
   - 1M context 在本地机器上**真能跑**（满 ctx 约 26GB 内存，indexer 占 22GB）
   - DS4 把 KV 当**一等磁盘公民**，命中前缀秒级 resume
   - Qwen3.5 用标准 GQA，1M context 的内存 / 磁盘资源需求要大得多。对 agent / 长对话场景这是**数量级差距**

3. **思考长度自适应**
   - V4 Flash thinking section 长度正比于问题复杂度，比同类模型短约 5 倍
   - Qwen3.5 思考模式 token 消耗一直是公认痛点
   - 直接转化为端到端延迟更低、成本更低

4. **2-bit 量化的"可用性"**
   - V4 Flash 使用**非对称量化**：只对 routed MoE experts 做 2-bit（up/gate IQ2_XXS，down Q2_K），shared experts / 投影层 / 路由 / 输出层全部高精度
   - 结果：81 GB 文件就能在 96–128GB MacBook 上跑，**质量可靠到可以 tool call**
   - Qwen3.5-122B INT4 是均匀 4-bit，体积接近但**精度策略不同**：V4 Flash 在更低位宽（2-bit）下专门做了质量保护，并对照官方 logits 做了回归测试

5. **端到端"成品"工程化**
   - DS4 不只是引擎：A) 引擎 + B) 专门 craft 的 GGUF + C) coding agent 验证 —— 三件套绑死配对
   - 官方 logits 在不同 context 长度下的回归测试（`tests/test-vectors/`）
   - 精确 tool-call DSML replay（解决 agent 多轮 tool 调用丢精度的问题）
   - 磁盘 KV 触发策略覆盖 cold / continued / evict / shutdown 四种时机
   - 跑 Qwen3.5 这一切都要自己组合，没有项目把这条链整体打磨过

6. **Apple Silicon Metal 一等公民**
   - DS4 主要目标平台明确是 Metal on macOS，CUDA 是次要目标
   - Qwen 在 Apple Silicon 上完全靠 llama.cpp 社区，没有专属优化

### DS4 + V4 Flash 的**劣势 / 代价**

1. **内存门槛更高**：q2 81GB / q4 153GB，比 Qwen3.5-122B INT4（65–75GB）大。低于 96GB RAM 没法跑
2. **生态绑死单一模型**：DS4 只跑 V4 Flash，换模型即作废。Qwen 可随时切 7B / 32B / 72B / 122B
3. **alpha 质量**：作者明说"alpha quality code, exists only for a few days"。Qwen + llama.cpp / vLLM 在稳定性上明显更成熟
4. **激活 13B vs 10B**：单 token 计算量略大；理论 t/s 上 Qwen 略占便宜，但实际取决于 KV bandwidth，V4 Flash 因 MLA 反而经常更快
5. **打榜成绩**：Qwen3.5 系列 benchmark 更新更激进，纯打榜分数 V4 Flash 不一定占优
6. **中文能力**：V4 Flash 重点宣传的是英语 / 意大利语；**Qwen 系列对中文是母语级**。中文场景这是关键差异

### 一句话结论

- 场景偏向 **超长上下文 agent + 高端 Mac / DGX Spark + 知识广度优先 + 英文 / 多语种**：选 DS4 + V4 Flash
- 场景偏向 **中文为主 / 需要灵活换模型 / 生产稳定 / 256K 上下文够用**：用 Qwen3.5 + vLLM 或 llama.cpp 更省心

---

## 模型权重

本实现**只支持本项目发布的 DeepSeek V4 Flash GGUF**。它不是通用 GGUF loader，任意 DeepSeek/GGUF 文件因 tensor 布局、量化组合、metadata、MTP 状态不匹配而无法工作。

仓库提供的 2-bit 量化不是噱头：行为良好、能在 coding agent 下可靠调用工具。2-bit 量化采用**极不对称策略**：

- 只对 routed MoE experts 量化：up/gate 用 `IQ2_XXS`，down 用 `Q2_K`
- 其他组件（shared experts、各种投影、路由）保留高精度

下载主模型（**推荐 imatrix 版本**）：

```sh
./download_model.sh q2-imatrix   # 96/128GB 内存机器，imatrix 调过的 q2
./download_model.sh q4-imatrix   # >=256GB 内存机器，imatrix 调过的 q4
```

旧版（不带 imatrix）仍可下载：

```sh
./download_model.sh q2
./download_model.sh q4
```

脚本从 `https://huggingface.co/antirez/deepseek-v4-gguf` 下载，文件落到 `./gguf/`，断点续传，并把 `./ds4flash.gguf` 链接到选中的模型。公开下载不需要鉴权；`--token TOKEN`、`HF_TOKEN`、本地 HF token 缓存都会被自动使用。

`./download_model.sh mtp` 拉可选的**推测解码 GGUF**。可配 q2/q4 使用，必须用 `--mtp` 显式开启。当前 MTP 路径仍是实验性，正确性兜底，加速很轻微。

要自己重新生成 GGUF 或采集 imatrix，看 [gguf-tools/README.md](gguf-tools/README.md)。

---

## 本机构建

```sh
make                  # macOS Metal
make cuda-spark       # Linux CUDA, DGX Spark / GB10
make cuda-generic     # Linux CUDA, 其它本地 CUDA GPU
make cpu              # 仅 CPU 诊断构建
```

`./ds4flash.gguf` 是两个二进制的默认模型路径。`-m` 选另外的 GGUF。`./ds4 --help` 和 `./ds4-server --help` 列出全部参数。

---

## Docker 部署（ARM64 + CUDA）

中文版独有章节。本仓库已经准备好一套针对 **DGX Spark / GB10** 等 ARM64 + NVIDIA 平台的容器化方案，从构建到访问 API 一条龙。仓库根目录的相关文件：

| 文件 | 作用 |
| --- | --- |
| `Dockerfile` | 多阶段构建：`nvidia/cuda:12.8.1-devel`（编译）→ `nvidia/cuda:12.8.1-runtime`（运行） |
| `build.sh` | 封装 `docker buildx`，强制 `linux/arm64`，构建完自动 `docker save` 成 tar |
| `docker-compose.yaml` | 端口 30001、GPU 申请、Volume 映射一站式配置 |
| `.dockerignore` | 排除权重、tar、KV 缓存、git 等 |

### 0. 前提

- Host 是 ARM64 Linux + NVIDIA GPU（如 DGX Spark / GB10、Grace Hopper、Orin）
- Host 装好 NVIDIA driver（≥ 570，对应 CUDA 12.8）+ **nvidia-container-toolkit**
- 已安装 Docker（含 `docker buildx` 和 `docker compose` v2）
- Apple Silicon Mac 可以**构建**镜像（buildx + QEMU），但**无法运行**（Docker Desktop for Mac 不透传 NVIDIA GPU）

检查 driver / CUDA 版本：

```sh
nvidia-smi | head -5
# 应看到:  Driver Version: 570.xx 或更高,  CUDA Version: 12.8 或更高
```

### 1. 克隆仓库

```sh
git clone https://github.com/<your-fork>/ds4.git
cd ds4
```

> 本仓库要求**模型权重位于宿主机的 `./gguf/` 目录**。下面的所有相对路径都从仓库根目录开始算。

### 2. 准备模型权重

权重 81 GB 起，**绝不会**打进镜像。下载到宿主机的 `./gguf/`：

```sh
# 96 / 128 GB 内存机器（推荐）
./download_model.sh q2-imatrix

# 256 GB 以上才考虑
# ./download_model.sh q4-imatrix
```

下载完成后，仓库根目录会出现一个软链：

```
./ds4flash.gguf  ->  ./gguf/<MODEL_FILE>.gguf
```

后续所有命令里出现的 `<MODEL_FILE>` 都指**你下载到 `./gguf/` 里的具体 GGUF 文件名**（例如 `DeepSeek-V4-Flash-IQ2XXS-w2Q2K-AProjQ8-SExpQ8-OutQ8-chat-v2.gguf`）。Compose 默认走软链，**不需要**你手动改文件名；但 `docker run` 直接启动时建议用绝对文件名。

### 3. 构建镜像

```sh
./build.sh                  # 默认 sm_120 (DGX Spark / GB10 Blackwell)
./build.sh sm_90            # Grace Hopper
./build.sh sm_87            # Orin / AGX
./build.sh generic          # 在目标机上跑 nvcc -arch=native
```

构建完成会做两件事：

1. 本地 Docker 里生成 image `ds4:cuda-arm64`
2. 在仓库根目录生成 `ds4-cuda-arm64.tar`（约 2~3 GB），方便离线拷贝到其它机器

环境变量覆盖：

| 变量 | 默认 | 用途 |
| --- | --- | --- |
| `IMAGE_NAME` | `ds4:cuda-arm64` | 镜像 tag |
| `CUDA_IMAGE_TAG` | `12.8.1-devel-ubuntu22.04` | 编译用 base 镜像 |
| `CUDA_RUNTIME_TAG` | `12.8.1-runtime-ubuntu22.04` | 运行用 base 镜像 |
| `CPU_FLAG` | 空（镜像默认 `-mcpu=neoverse-v2`，对应 GB10 Grace 核） | -mcpu 值 |
| `SAVE_TAR` | `1` | 设为 `0` 跳过 `docker save` |
| `TAR_PATH` | `./<image>.tar` | tar 输出路径 |

跨机部署：在另一台 ARM64 + NVIDIA 主机上 `docker load -i ds4-cuda-arm64.tar` 即可，无需重新构建。

### 4. 准备运行时目录

```sh
mkdir -p kv-cache traces
```

| 宿主机目录 | 容器内 | 用途 |
| --- | --- | --- |
| `./gguf` | `/models`（只读） | 上一步下载的 GGUF 权重 |
| `./ds4flash.gguf` | `/app/ds4flash.gguf`（只读） | 软链，指向你选中的权重 |
| `./kv-cache` | `/kv` | 磁盘 KV checkpoint，跨重启持久化 |
| `./traces` | `/traces` | 会话 trace，便于排错 |

### 5. 启动服务（推荐用 Compose）

```sh
docker compose up -d
docker compose logs -f
```

日志看到类似下面就说明加载成功（首次加载 81 GB 权重大约 30 秒 ~ 2 分钟）：

```
ds4-server: loading /app/ds4flash.gguf
ds4-server: listening on 0.0.0.0:30001
```

`docker-compose.yaml` 里默认执行的命令等价于：

```sh
./ds4-server \
  --model /app/ds4flash.gguf \
  --host 0.0.0.0 --port 30001 \
  --ctx 100000 \
  --kv-disk-dir /kv --kv-disk-space-mb 16384 \
  --trace /traces/session.trace
```

### 6. 验证服务

服务监听 **30001 端口**，提供 OpenAI / Anthropic 兼容 API：

```sh
# 列出模型
curl http://localhost:30001/v1/models

# 跑一次对话
curl http://localhost:30001/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{
    "model": "deepseek-v4-flash",
    "messages": [{"role":"user","content":"你好"}],
    "stream": false
  }'
```

> 想从其它机器访问：把 `localhost` 换成宿主机 IP，并确认防火墙放开了 30001。

### 7. 不用 Compose？直接 docker run

```sh
docker run --rm --gpus all \
  --ipc=host \
  -p 30001:30001 \
  -v "$PWD/gguf:/models:ro" \
  -v "$PWD/kv-cache:/kv" \
  -v "$PWD/traces:/traces" \
  ds4:cuda-arm64 \
  ./ds4-server \
    --model /models/<MODEL_FILE>.gguf \
    --host 0.0.0.0 --port 30001 \
    --ctx 100000 \
    --kv-disk-dir /kv --kv-disk-space-mb 16384
```

> 把 `<MODEL_FILE>.gguf` 替换为你 `./gguf/` 下的实际文件名。

### 8. 换模型 / 调参数

**换不同量化**（推荐用宿主软链法，不动 compose）：

```sh
./download_model.sh q4-imatrix    # 自动重链 ./ds4flash.gguf
docker compose restart
```

或者在 `docker-compose.yaml` 的 `command` 段直接写死具体文件名：

```yaml
- --model
- /models/<MODEL_FILE>.gguf
```

**MTP 推测解码 / CORS / 其它参数**：`docker-compose.yaml` 里相关行已经预置注释，按需取消注释或追加参数，**重启容器**生效（不用重新构建镜像）。

```sh
docker compose restart
```

### 常见坑

1. **`--host 0.0.0.0` 必填**，否则容器内只监听 127.0.0.1，宿主访问不到（compose 已经配好）
2. **driver 版本**：CUDA 12.8 runtime 要求 host driver ≥ 570。`nvidia-smi` 看一眼
3. **`--ctx` 决定 KV 启动预算**，启动日志会打印估算值。q2 + 128 GB 机器，**100k~300k 上下文**比较稳妥；1M 满 ctx 单 indexer 就要 ~22 GB
4. **不要把权重打进镜像**（81~153 GB），永远走 volume
5. **`./kv-cache` 落在容量充裕的盘**，长期使用会逼近 `--kv-disk-space-mb` 上限；删整目录即清空缓存
6. **首启很慢**：81 GB 权重需要从磁盘加载到 GPU，加 `--warm-weights` 会让首启更慢但首次推理更稳
7. **buildx 警告 `FromPlatformFlagConstDisallowed`** 已经修掉；若仍报错，确认 Docker 是较新版本（≥ 24）

---

## Speed 速度

以下是 Metal CLI 单跑数据，`--ctx 32768`、`--nothink`、贪心解码、`-n 256`。短 prompt 是个普通的意大利语小故事；长 prompt 测的是分块 prefill + 长上下文解码。Q4 需要大内存机器，M3 Max 上的 Q4 标 N/A。

| 机器 | 量化 | Prompt | Prefill | Generation |
| --- | ---: | ---: | ---: | ---: |
| MacBook Pro M3 Max, 128 GB | q2 | short | 58.52 t/s | 26.68 t/s |
| MacBook Pro M3 Max, 128 GB | q2 | 11709 tokens | 250.11 t/s | 21.47 t/s |
| MacBook Pro M3 Max, 128 GB | q4 | short | N/A | N/A |
| MacBook Pro M3 Max, 128 GB | q4 | long | N/A | N/A |
| Mac Studio M3 Ultra, 512 GB | q2 | short | 84.43 t/s | 36.86 t/s |
| Mac Studio M3 Ultra, 512 GB | q2 | 11709 tokens | 468.03 t/s | 27.39 t/s |
| Mac Studio M3 Ultra, 512 GB | q4 | short | 78.95 t/s | 35.50 t/s |
| Mac Studio M3 Ultra, 512 GB | q4 | 12018 tokens | 448.82 t/s | 26.62 t/s |
| DGX Spark GB10, 128 GB | q2 | 7047 tokens | 343.81 t/s | 13.75 t/s |

![M3 Max t/s](speed-bench/m3_max_ts.svg)

---

## ds4-bench：基准测试

`ds4-bench` 测量在 context 各前沿位置的**瞬时** prefill 和 generation 吞吐，而不是给一个全程平均值。它加载一次模型，遍历到 2048、4096、6144 等前沿，用增量 prefill 让每行只测新增的 token 区间。每到一个前沿，保存 KV 到内存，做固定贪心非 EOS 探针，恢复快照，继续 prefill。

```sh
./ds4-bench \
  -m ds4flash.gguf \
  --prompt-file speed-bench/promessi_sposi.txt \
  --ctx-start 2048 \
  --ctx-max 65536 \
  --step-incr 2048 \
  --gen-tokens 128
```

示例文件是 Project Gutenberg 的 Manzoni《I Promessi Sposi》清洗版本（去掉了 PG 页眉页脚）。`--step-incr` 是线性步长，`--step-mul` 是指数步长。输出 CSV，每行一个前沿：最新 prefill 区间的 tokens/sec、该前沿 generation tokens/sec、`kvcache_bytes`。

---

## ds4-eval：能力评测

`ds4-eval` 是个小型真实模型集成基准，**不是榜单跑分工具**，不应作为官方 GPQA / SuperGPQA / AIME / 安全基准的成绩上报。它内嵌一个 92 题子集，目的是让本地回归测试可用、可肉眼审计。

```sh
./ds4-eval -m ds4flash.gguf --trace /tmp/ds4-eval.txt
```

默认 `--tokens 16000`，思考模式开，并设有软 / 硬 `</think>` 预算截断保证模型还有空间产生可见答案。`ds4-eval` 内部从最大 prompt + 生成预算自动定 context，超过 1M tokens 会拒跑。TUI 操作：`p` 暂停，`q` 退出并打印报告，上下选择题目，回车跑下一题，`--plain` 关 TUI。

前 75 题穿插：25 GPQA Diamond + 25 SuperGPQA（人审过）+ 25 AIME 2025；后 17 题是 COMPSEC 子集，单函数 C/C++ 漏洞定位。对 V4 Flash 这类模型，应**当作硬性能力回归套件**而非 pass/fail：

- **GPQA Diamond**：研究生级理科多选题。即便模型能力强，小改动也容易回归
- **SuperGPQA**：跨专业知识、领域迁移题。模型卡分数本身就低于 GPQA Diamond
- **AIME 2025**：精确答案数学竞赛题，本评测里最不留情
- **COMPSEC**：从公开 CVE 简化得来，问题是定位漏洞引入的最佳源代码行，安全函数返回 `0`。**不是利用 prompt**

不要指望 92/92 满分。这个工具回答的工程问题是：改了 kernel / 量化 / prompt 渲染 / KV 缓存 / tool 流之后，模型在硬科学 + 广博知识 + 精确数学 + 安全代码这一组代表性混合任务上还能不能扛住。

---

## CLI 用法

单 prompt：

```sh
./ds4 -p "Explain Redis streams in one paragraph."
```

不带 `-p` 进入交互模式：

```sh
./ds4
ds4>
```

交互 CLI 是真正的多轮 DS4 chat，保存渲染过的 transcript 和活动 KV checkpoint，所以每一轮都接在前一轮上。常用命令：`/help`、`/think`、`/think-max`、`/nothink`、`/ctx N`、`/read FILE`、`/quit`。Ctrl+C 中断当前生成，回到 `ds4>`。

CLI 默认开思考模式，`--nothink` 或 `/nothink` 切直接回答。`--mtp MTP.gguf --mtp-draft 2` 开 MTP 推测解码（贪心解码下才有用，靠 `--mtp-margin` 置信度门限避免慢速部分接受，视作实验性轻微加速）。

---

## Server

本地启动 OpenAI / Anthropic 兼容服务：

```sh
./ds4-server --ctx 100000 --kv-disk-dir /tmp/ds4-kv --kv-disk-space-mb 8192
```

从其它目录启动时加 `--chdir /path/to/ds4`，让 `metal/*.metal` 这类相对路径资产正确解析。

服务器在内存中只保留**一个**可变的后端 / KV checkpoint，所以无状态客户端反复发送同一 prompt 的更长版本时，可以复用共享前缀，不用从零重新 prefill。请求解析和 socket 用客户端线程，**但推理本身串行**通过单个图 worker —— 当前服务器不会把多个独立请求做 batching，并发请求要排队。

支持的端点：

- `GET /v1/models`
- `GET /v1/models/deepseek-v4-flash`
- `POST /v1/chat/completions`
- `POST /v1/responses`
- `POST /v1/completions`
- `POST /v1/messages`

`/v1/chat/completions` 接受 OpenAI 风格 `messages`、`max_tokens`/`max_completion_tokens`、`temperature`、`top_p`、`top_k`、`min_p`、`seed`、`stream`、`stream_options.include_usage`、`tools`、`tool_choice`。工具 schema 被渲染成 DeepSeek 的 DSML 格式，生成的 DSML tool call 再映回 OpenAI tool call。

`/v1/responses` 接受 OpenAI Responses 风格 `input`、`instructions`、`tools`、`tool_choice`、`max_output_tokens`、`temperature`、`top_p`、`stream`、`reasoning`。**Codex CLI 优先用这个端点**。

`/v1/messages` 是 Anthropic 兼容端点，**Claude Code 风格客户端走这条**。接受 `system`、`messages`、`tools`、`tool_choice`、`max_tokens`、`temperature`、`top_p`、`top_k`、`stream`、`stop_sequences` 和 thinking 控制。工具调用以 `tool_use` 块返回。

默认采样：`temperature=1`、`top_p=1`、`min_p=0.05` —— 默认过滤是相对概率而非 nucleus mass。**思考模式下用固定采样默认值，忽略客户端采样参数**，对齐 DeepSeek 官方 API 行为。

Chat / Responses / Anthropic 三个端点都支持 SSE 流式：

- 思考模式下，reasoning 以原生 API 形态流出，而不是混进最终文本
- OpenAI chat 流在 DSML 调用一被识别就开始流 tool call：先发 tool header，再把参数字节作为 `tool_calls[].function.arguments` delta 持续推送
- Anthropic 端点先流 thinking 和文本，工具块生成完成时再发结构化 `tool_use` 块
- Responses 端点流出 Codex 期望的 Responses 事件生命周期：`response.output_text.delta`、function-call 参数事件、终态 `response.completed` / `response.incomplete` / `response.failed`

浏览器 JS 客户端跨源访问加 `--cors`。这只是改 HTTP 头，**不会**把服务暴露到 LAN —— 要远程连入显式加 `--host 0.0.0.0`。

### 工具调用：精确 replay 与归一化

V4 Flash 把工具调用以 DSML 文本输出。Agent 客户端下一次请求不会原样回传这段文本，而是发归一化的 OpenAI/Anthropic JSON tool call。**如果服务器重新渲染时哪怕字节级差一点，渲染前缀就匹配不上 KV checkpoint，下一轮就要重建**。

第一道防线是**精确 replay**：每个工具调用拿一个不可猜的 API tool ID，服务器在内存里维护 `tool id -> 模型采样的精确 DSML 块` 映射（radix tree 后端）。客户端回传 tool ID 时，prompt 渲染器直接用模型采样的精确 DSML 字节，而非"重新格式化的近似版"。这个 map 也能存进 KV 缓存文件，重启后仍能精确 replay。

**归一化只是兜底**：如果精确 DSML 块丢了，或用 `--disable-exact-dsml-tool-replay` 关掉了精确 replay，服务器会用确定性规则从 JSON 渲染出 DSML。工具调用轮结束后，服务器会把活动采样流和"客户端下一个请求会渲染出的 prompt"做对比，必要时重写活动 checkpoint，或回退到旧的磁盘 KV 快照只 replay 后缀，保持模型续写与无状态 API 对账一致。

生成中 DSML 语法和 payload 区别对待：模型在输出 DSML 标签、参数头、JSON 标点、闭合标记这类**协议结构**时，采样会被强制为 `temperature=0`，保证 tool call 可解析；但**参数 payload 内部**（`string=true` 参数体、JSON 字符串值，包括文件内容和编辑文本）用请求的正常采样设置 —— 否则长代码 / 长文件块会重复。

### 最小 OpenAI 例子

```sh
curl http://127.0.0.1:8000/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{
    "model":"deepseek-v4-flash",
    "messages":[{"role":"user","content":"列出三条 Redis 设计原则"}],
    "stream":true
  }'
```

### Agent 客户端接入

`ds4-server` 可以被任何会说 OpenAI 兼容 chat completions 的本地 coding agent 使用。客户端的 context limit 不应高于启动时的 `--ctx`：

```sh
./ds4-server --ctx 100000 --kv-disk-dir /tmp/ds4-kv --kv-disk-space-mb 8192
```

1M 完整上下文大约要 26GB 内存（光是 compressed indexer 就 ~22GB），按系统配置定 ctx。128GB RAM 跑 2-bit（已占 81GB），上 26GB 偏紧，**100k~300k 比较合理**。社区也有人在 96GB Mac 上跑 250k ctx，但需要先杀掉吃内存的进程。

`384000` 输出上限是为了避免模型被截断（模型最多能生成 384k tokens）。

**opencode**：在 `~/.config/opencode/opencode.json` 里加 provider 和 agent：

```json
{
  "$schema": "https://opencode.ai/config.json",
  "provider": {
    "ds4": {
      "name": "ds4.c (local)",
      "npm": "@ai-sdk/openai-compatible",
      "options": {
        "baseURL": "http://127.0.0.1:8000/v1",
        "apiKey": "dsv4-local"
      },
      "models": {
        "deepseek-v4-flash": {
          "name": "DeepSeek V4 Flash (ds4.c local)",
          "limit": {
            "context": 100000,
            "output": 384000
          }
        }
      }
    }
  },
  "agent": {
    "ds4": {
      "description": "DeepSeek V4 Flash served by local ds4-server",
      "model": "ds4/deepseek-v4-flash",
      "temperature": 0
    }
  }
}
```

**Pi**：在 `~/.pi/agent/models.json` 加 provider（字段同英文 README，完整内容见原文）。可以把 `~/.pi/agent/settings.json` 改成默认走 ds4。

**Codex CLI**：用 Responses 协议：

```toml
[model_providers.ds4]
name = "DS4"
base_url = "http://127.0.0.1:8000/v1"
wire_api = "responses"
stream_idle_timeout_ms = 1000000
```

```sh
codex --model deepseek-v4-flash -c model_provider=ds4
```

**Claude Code**：走 Anthropic 兼容端点，可以写一个 wrapper 脚本：

```sh
#!/bin/sh
unset ANTHROPIC_API_KEY

export ANTHROPIC_BASE_URL="${DS4_ANTHROPIC_BASE_URL:-http://127.0.0.1:8000}"
export ANTHROPIC_AUTH_TOKEN="${DS4_API_KEY:-dsv4-local}"
export ANTHROPIC_MODEL="deepseek-v4-flash"

export ANTHROPIC_CUSTOM_MODEL_OPTION="deepseek-v4-flash"
export ANTHROPIC_CUSTOM_MODEL_OPTION_NAME="DeepSeek V4 Flash local ds4"
export ANTHROPIC_CUSTOM_MODEL_OPTION_DESCRIPTION="ds4.c local GGUF"

export ANTHROPIC_DEFAULT_SONNET_MODEL="deepseek-v4-flash"
export ANTHROPIC_DEFAULT_HAIKU_MODEL="deepseek-v4-flash"
export ANTHROPIC_DEFAULT_OPUS_MODEL="deepseek-v4-flash"
export CLAUDE_CODE_SUBAGENT_MODEL="deepseek-v4-flash"

export CLAUDE_CODE_DISABLE_NONESSENTIAL_TRAFFIC=1
export CLAUDE_CODE_DISABLE_NONSTREAMING_FALLBACK=1
export CLAUDE_STREAM_IDLE_TIMEOUT_MS=600000

exec "$HOME/.local/bin/claude" "$@"
```

Claude Code 启动时常发一个 25k tokens 左右的大 prompt 才开始做事，**务必开 `--kv-disk-dir`**：第一次贵的 prefill 之后，磁盘 KV 让续写或重启会话能复用前缀，不用整段重跑。

---

## 思考模式

V4 Flash 有非思考、思考、Think Max 三档。服务器默认思考模式。`reasoning_effort=max` 请求 Think Max，但**只在上下文够大时才生效**（按模型卡推荐），上下文小自动降回普通思考。OpenAI `reasoning_effort=xhigh` 仍然映到普通思考，不是 Think Max。

要直接回答用 `thinking: {"type":"disabled"}`、`think:false`，或非思考别名 `deepseek-chat`。

---

## 磁盘 KV 缓存

Chat/completion API 是无状态的：agent 客户端通常每次都重发整段对话。`ds4-server` 先做最便宜的 token-prefix 精确匹配，匹配失败再用"渲染后字节"对比 checkpoint 解码字节。**内存里 live checkpoint 覆盖当前会话；磁盘 KV 是会话切换 / 服务器重启后让前缀仍可用的机制**。

出于内存原因当前只保留**一个** live KV cache。新无关会话顶替之后，旧 checkpoint 只有写过磁盘才能恢复。也就是说：**内存缓存 = 当前会话；磁盘缓存 = 跨会话 / 重启的 resume**。

启用：

```sh
./ds4-server --kv-disk-dir /tmp/ds4-kv --kv-disk-space-mb 8192
```

缓存键是渲染字节前缀的 SHA1，文件名 `<sha1>.kv`。文件用普通 `read`/`write` IO 写，**不用 mmap**，避免在已经 mmap 模型的进程里再加 VM mapping。

工具调用映射用不可猜 tool ID 维护精确 DSML replay，RAM map 默认 100000 条，`--tool-memory-max-ids` 调；`--disable-exact-dsml-tool-replay` 关掉走 JSON 兜底。

四个保存时机：

- `cold`：长 prompt 首次到稳定前缀、生成开始前
- `continued`：prefill 或生成到下一个绝对对齐前沿
- `evict`：无关请求要顶替活动会话前
- `shutdown`：服务器正常退出

`cold` 保存会刻意去掉一小段尾部 token、按 prefill chunk 边界向下对齐，避免后续请求在同 prompt 上追加文本时遇到 BPE 边界 retokenize 失配。默认值偏保守：

- 最少 512 token 才保存
- cold 最多保存 30000 token 的 prompt
- 尾部裁掉 32 token
- 向下对齐到 2048-token chunk

`continued` 用同一套对齐，只在活动图自然到达绝对前沿时写出（默认约每 10k token 一次），长生成会留下重启点而不持久化脆弱的最后几个 token。

可调参数：

- `--kv-cache-min-tokens`
- `--kv-cache-cold-max-tokens`
- `--kv-cache-continued-interval-tokens`
- `--kv-cache-boundary-trim-tokens`
- `--kv-cache-boundary-align-tokens`
- `--tool-memory-max-ids`
- `--disable-exact-dsml-tool-replay`

默认情况下，渲染前缀匹配时 checkpoint 可以**跨 2-bit / 4-bit 复用**。要严格同量化才复用，加 `--kv-cache-reject-different-quant`。

缓存目录是一次性资源：行为可疑就停服务删掉即可。文件里直接含明文 prompt 字节，可以用 hexdump 检查。

文件格式细节见英文 README"Disk KV Cache"小节，本文不重复。

---

## 后端

macOS 默认 Metal，CUDA 构建默认 CUDA：

```sh
./ds4 -p "Hello" --metal
./ds4 -p "Hello" --cuda
```

Linux 上 `make` 不会隐式选 CUDA target，而是打印可选目标。`make cuda-spark` 是 DGX Spark / GB10 当前最快路径（故意省 `nvcc -arch`）；`make cuda-generic` 是普通本地 CUDA 构建；交叉构建或需要明确目标用 `CUDA_ARCH`：

```sh
make cuda CUDA_ARCH=sm_120
make cuda CUDA_ARCH=native
```

CPU 参考 / 调试路径：

```sh
./ds4 -p "Hello" --cpu
make cpu
./ds4
./ds4 -p "Hello"
```

**不要**把 CPU 路径当作生产目标。CLI 和 `ds4-server` 都支持 CPU backend，KV session 和快照格式跟 Metal/CUDA 共享，但正常推理应该走 Metal / CUDA。

---

## Steering（方向激活引导）

仓库支持单向量激活引导，详见 `dir-steering` 目录。思路来自论文 [Refusal in Language Models Is Mediated by a Single Direction](https://arxiv.org/abs/2406.11717)。可以用它把模型调得更啰嗦或更简洁、让你的租车站聊天机器人少回答编程问题、给安全研究人员降低模型给出 dual-use 攻击建议的意愿 —— 比微调快得多。

---

## 测试向量

`tests/test-vectors` 保存了从官方 DeepSeek V4 Flash API 抓的短 / 长上下文 continuation 向量。请求用 `deepseek-v4-flash`、贪心解码、关思考、最大 `top_logprobs` 切片。本地用 `./ds4 --dump-logprobs` 生成同样格式的向量并按 token 字节对比，分词 / 模板 / 注意力回归就会在变成长生成失败前被抓到。

```sh
make test                  # ./ds4_test --all
./ds4_test --logprob-vectors
./ds4_test --server
```

---

## 调试

生成结果可疑时，三个工具通常就够找到第一手线索：

```sh
./ds4 --dump-tokens -p "..."
./ds4 --dump-logprobs /tmp/out.json --logprobs-top-k 20 --temp 0 -p "..."
./ds4-server --trace /tmp/ds4-trace.txt ...
```

- `--dump-tokens` 按 `-p` / `--prompt-file` 的字符串做分词，识别 DS4 协议 special token，然后直接退出。例如 DSML 工具闭标记拆成两个 token：`</` 和 `｜DSML｜`
- `--dump-logprobs` 存一段贪心 continuation + 每步 top 本地候选，帮你区分"采样选择问题"和"模型 logit 问题"
- `ds4-server --trace` 写整个 agent session 的渲染 prompt、缓存决策、生成文本、tool 解析事件

---

## 更多文档

- [CONTRIBUTING.md](CONTRIBUTING.md)：贡献者必读，正确性 / 速度回归
- [gguf-tools/README.md](gguf-tools/README.md)：离线 GGUF 生成、imatrix 采集、量化、质量检查
- [gguf-tools/imatrix/README.md](gguf-tools/imatrix/README.md)：routed-MoE imatrix 采集与使用
- [gguf-tools/imatrix/dataset/README.md](gguf-tools/imatrix/dataset/README.md)：校准 prompt 语料如何生成
- [gguf-tools/quality-testing/README.md](gguf-tools/quality-testing/README.md)：本地 GGUF 对照官方 V4 Flash 续写的打分
- [dir-steering/README.md](dir-steering/README.md)：方向激活数据、向量生成与使用
- [speed-bench/README.md](speed-bench/README.md)：基准 CSV 与图表生成
- [tests/test-vectors/README.md](tests/test-vectors/README.md)：官方 continuation 向量
