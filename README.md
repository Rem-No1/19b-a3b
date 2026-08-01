# Qwen3.6-19B-A3B DeepSpeed SFT 交付说明

本文档面向通过 GitHub 获取代码、通过 Docker Hub 获取镜像的使用者。按第 3、
4、6 节操作，即可启动训练并在每个已保存 checkpoint 上异步计算验证集 pass
rate，无需在宿主机单独安装 PyTorch、Transformers、DeepSpeed、flash-attn
或 vLLM。

交付地址：

- 代码：<https://github.com/Rem-No1/19b-a3b>
- 镜像：<https://hub.docker.com/r/iceswallow/qwen36-delivery/tags>
- 推荐训练标签：`iceswallow/qwen36-delivery:sft-1.1-fla`
- 异步验证标签：`iceswallow/qwen36-delivery:vllm-eval-1.0`

首次使用的最短路线是：准备宿主机 Docker/NVIDIA 环境 → 按第 3 节准备模型和
数据 → 按第 4.1、4.2 节获取代码和镜像 → 按第 4.3、4.4 节完成自检 → 复制
第 6.1 节完整命令并只修改顶部路径、GPU 编号、ZeRO/offload 选项。旧版离线
tar 和本地构建只作为无法访问 Docker Hub 时的备选流程。

> **v1.1-fla 训练镜像：** 推荐使用 `qwen36-sft:1.1-fla`。该镜像在原有
> FlashAttention 2 之外加入 Qwen3.5 linear-attention 训练所需的
> `flash-linear-attention 0.5.2`、`causal-conv1d 1.6.2.post1` 和
> `TileLang 0.1.12`。TileLang 用于避开 H100 + Triton 3.6.0 下 FLA
> gated-delta backward 的已知正确性保护报错；PyTorch 2.11.0 和 Triton
> 3.6.0 保持不变。旧的 `qwen36-sft:1.0` 不包含这条完整快速路径。

> **当前训练代码：** 已包含 v1.0.3 的热修复，包括百万级数据预处理 manifest
> 扫描优化，以及非主 rank 长时间等待预处理可能触发 DDP/NCCL timeout 的问题；
> 全参训练默认
> 学习率改为从 `5e-5` 余弦衰减到 `5e-6`，并支持通过 `--zero-stage 2|3`
> 选择 ZeRO 阶段、通过 `--offload` 选择是否使用 CPU offload，并修复连续验证
> checkpoint 时 vLLM 端口尚未释放的问题。`qwen36-sft:1.1-fla` 会嵌入构建
> 时的当前训练代码；继续只读挂载同一版本的 `train/`/`eval/`，便于后续只更新
> 代码而不重建镜像。旧 `qwen36-sft:1.0` 必须挂载新版代码。

> 安全提示：下文的 `hf-xxxxxxx` 是交付方提供的 Hugging Face 访问令牌。
> 不要把真实令牌写入代码、README、Shell 历史或 Git 仓库。若令牌已经公开，
> 应立即在 Hugging Face 中撤销并重新创建只读令牌。

## 1. 目录结构和代码用途

本项目用于对 Qwen3.6/Qwen3.5 架构的 19B-A3B、128-expert MoE 模型进行
单机或多机多卡监督微调（SFT）。训练基于 Hugging Face Transformers 和
DeepSpeed，支持 ZeRO-2 和 ZeRO-3，默认采用 BF16、FlashAttention 2、梯度检查点
以及 ZeRO-3 CPU offload。单机是默认模式；多机通过四个环境变量启用，无需修改
Python 训练代码。

数据支持 JSONL 文件或顶层为数组的 JSON 文件。每条训练记录至少需要包含
OpenAI/Qwen 风格的 `messages` 字段，也支持 `tools`、逐行
`enable_thinking` 和逐行 `last_assistant_only`。

```text
qwen36_sft/
├── README.md
├── Dockerfile
├── requirements.lock
├── requirements-fla.lock
├── launch_async_delivery_test.sh
├── train/
│   ├── run_qwen36_19b_a3b_sft_deepspeed.sh
│   ├── train_qwen36_19b_a3b_sft_deepspeed.py
│   ├── toolchat_data.py
│   ├── pass_rate_eval.py
│   ├── async_eval_markers.py
│   ├── verify_qwen35_fast_path.py
│   └── ds_config/
│       ├── qwen36_19b_a3b_zero2.json
│       └── qwen36_19b_a3b_zero3.json
├── eval/
│   ├── async_vllm_eval.py
│   ├── run_async_vllm_eval.sh
│   └── vllm_environment_manifest.json
└── tests/
    ├── test_async_eval.py
    ├── test_multinode_launcher.py
    └── test_training_preprocessing.py
```

各文件用途如下：

| 文件 | 用途 |
| --- | --- |
| `train/run_qwen36_19b_a3b_sft_deepspeed.sh` | 容器入口和单机/多机多卡启动器；校验节点与 GPU 参数、执行预检并调用 `torchrun`。 |
| `train/train_qwen36_19b_a3b_sft_deepspeed.py` | 主训练程序；加载模型和数据、构造 assistant-only loss mask、启动 Transformers Trainer。 |
| `train/toolchat_data.py` | JSON/JSONL 读取、逐文件抽样、chat template 编码和 assistant token mask。 |
| `train/ds_config/qwen36_19b_a3b_zero2.json` | DeepSpeed ZeRO-2 配置；`--offload` 只控制 optimizer state 是否放到 CPU。 |
| `train/ds_config/qwen36_19b_a3b_zero3.json` | 默认 DeepSpeed ZeRO-3 配置；`--offload` 控制参数和 optimizer state 是否放到 CPU。 |
| `train/pass_rate_eval.py` | 训练进程内生成式 pass rate 评测实现，适合短输出调试。 |
| `train/async_eval_markers.py` | 在 checkpoint 完整保存后生成异步评测 ready 标记。 |
| `eval/` | 独立 vLLM 评测工具；推荐的 HARP pass rate 验证需要启动一个 worker。 |
| `launch_async_delivery_test.sh` | 交付方内部异步联调脚本；接收方进行标准训练时不需要使用。 |
| `Dockerfile`、`requirements.lock`、`requirements-fla.lock` | 重建包含 Qwen3.5 linear-attention 快速路径的训练镜像时使用。 |
| `train/verify_qwen35_fast_path.py` | 在真实 GPU 上验证 causal-conv1d、FLA/TileLang 和 Qwen3.5 linear-attention 层的 BF16 forward/backward。 |

模型、数据、checkpoint、日志和缓存均位于宿主机，通过 Docker volume 挂载，
不会复制进镜像。

## 2. 已验证环境

以下是已经完成训练测试的一套环境，不代表最低硬件要求：

| 项目 | 已验证配置 |
| --- | --- |
| 宿主机 | Linux x86_64，Ubuntu 22.04.2 LTS |
| Docker | 27.2.1 |
| GPU | NVIDIA H100 80GB；训练使用 5 张卡 |
| NVIDIA Driver | 580.95.05 |
| 主机内存 | 2 TiB；DeepSpeed 配置会大量使用 CPU 内存 |
| 容器基础环境 | Ubuntu 24.04，CUDA Toolkit 13.0.2 |
| Python | 3.12 |
| PyTorch | 2.11.0，CUDA 13.0 |
| Transformers | 5.14.1 |
| DeepSpeed | 0.19.2 |
| Datasets | 5.0.0 |
| Accelerate | 1.14.0 |
| flash-attn | 2.8.3.post1 |
| flash-linear-attention / fla-core | 0.5.2 / 0.5.2 |
| causal-conv1d | 1.6.2.post1 |
| TileLang | 0.1.12 |
| PEFT | 0.19.1 |

已完成真实单机 5 卡训练测试。多机启动器已通过自动测试验证单机/多机
`torchrun` 参数构造和错误输入拦截，但交付方没有在真实多服务器集群上进行
端到端训练；接收方首次多机运行时应先使用少量数据和较小 `--max-steps` 冒烟。

宿主机必须提前安装：

- 可正常工作的 NVIDIA 驱动；
- Docker；
- NVIDIA Container Toolkit，使 Docker 能使用 `--gpus`。

导入镜像后，可执行以下命令检查容器能否访问 GPU：

```bash
docker run --rm --gpus all \
  --entrypoint nvidia-smi \
  qwen36-sft:1.1-fla
```

默认配置使用 DeepSpeed ZeRO-3，将参数、梯度和优化器状态分片，并把模型参数
与优化器卸载到 CPU。也可以选择 ZeRO-2；此时参数在每张 GPU 上完整保留，只有
梯度和优化器状态分片，因此需要明显更多的单卡显存。使用其他 GPU、较少的 CPU
内存或不同驱动时，需要重新评估兼容性、显存、主机内存和训练速度。24k/32k
上下文的资源消耗较高。

多机训练还要求节点之间能够互相通信，并且所有节点能够访问同一个共享输出
目录。推荐使用 NFS、Lustre、CephFS 等共享文件系统保存 checkpoint。

## 3. 下载数据和模型到本地

数据仓库：
<https://huggingface.co/datasets/Ice195/math100w>

模型仓库：
<https://huggingface.co/Ice195/19b-a3b>

下面以 `/data/qwen36-delivery` 为宿主机交付目录。接收方可以修改
`DELIVERY_ROOT`，但后续命令要保持一致。

### 3.1 安装 Hugging Face CLI

```bash
DELIVERY_ROOT=/data/qwen36-delivery

mkdir -p \
  "${DELIVERY_ROOT}/model" \
  "${DELIVERY_ROOT}/data" \
  "${DELIVERY_ROOT}/output" \
  "${DELIVERY_ROOT}/logs" \
  "${DELIVERY_ROOT}/cache"

python3 -m venv "${DELIVERY_ROOT}/hf-cli-env"
source "${DELIVERY_ROOT}/hf-cli-env/bin/activate"
python -m pip install --upgrade pip huggingface_hub
```

### 3.2 下载模型

模型仓库需要访问令牌。通过环境变量临时传入令牌，避免直接把令牌放进命令参数：

```bash
export HF_TOKEN='hf-xxxxxxx'

hf download Ice195/19b-a3b \
  --local-dir "${DELIVERY_ROOT}/model/19b-a3b"
```

如果仓库页面要求确认访问协议，需要先在浏览器登录对应 Hugging Face 账号并完成
授权。令牌所属账号必须拥有该模型仓库的读取权限。

### 3.3 下载数据

```bash
hf download Ice195/math100w \
  --repo-type dataset \
  --local-dir "${DELIVERY_ROOT}/data/math100w"
```

下载完成后立即从当前 Shell 清除令牌：

```bash
unset HF_TOKEN
deactivate
```

不要把 `hf-cli-env`、Hugging Face 缓存或任何包含令牌的文件提交到 Git。

### 3.4 检查下载结果

```bash
test -f "${DELIVERY_ROOT}/model/19b-a3b/config.json"
test -f "${DELIVERY_ROOT}/model/19b-a3b/model.safetensors.index.json"
test -f "${DELIVERY_ROOT}/model/19b-a3b/tokenizer_config.json"

find "${DELIVERY_ROOT}/data/math100w" -type f \
  \( -name '*.json' -o -name '*.jsonl' \) | sort
```

第 6 节的启动命令假定数据仓库下载后保留以下相对路径：

```text
math100w/
├── Nemotron-SFT-Math-v4/train25w.jsonl
├── OpenMathReasoning/train20w.jsonl
├── OpenR1-Math-220k/train.jsonl
├── OpenCodeReasoning-2/train20w.jsonl
├── general/qwen3_235b_thinking_2507_110k_sft.jsonl
├── Nemotron-SFT-Instruction-Following-Chat-v3-chat/train6w.jsonl
└── val/HARP/HARP_difficulty_2_sample_50.jsonl
```

如果仓库中的目录发生变化，只需同步修改启动命令中
`/datasets/...` 后面的相对路径。

## 4. 获取代码和 Docker 镜像

推荐直接从 GitHub 获取当前代码，并从 Docker Hub 拉取已经验证的两个镜像。
以下流程不会把模型或数据复制到镜像中；模型、数据、输出、日志和缓存仍位于
宿主机，通过只读或读写 volume mount 提供给容器。

### 4.1 获取当前训练代码

首次获取源码：

```bash
git clone https://github.com/Rem-No1/19b-a3b.git
cd 19b-a3b
git switch main
```

已有仓库时：

```bash
git fetch origin --tags
git switch main
git pull --ff-only origin main
```

`git pull --ff-only` 要求本地没有与远端冲突的提交或未处理的 rebase。交付环境
建议使用全新 clone，或者固定在交付方指定的 Git commit/tag。多机训练时所有
节点必须 checkout 同一个 commit/tag。

记下仓库的绝对路径，后续把它填入 `CODE_DIR`：

```bash
CODE_DIR="$(pwd)"
test -f "${CODE_DIR}/train/run_qwen36_19b_a3b_sft_deepspeed.sh"
test -f "${CODE_DIR}/eval/run_async_vllm_eval.sh"
git rev-parse HEAD
```

### 4.2 从 Docker Hub 拉取镜像（推荐）

镜像仓库是公开仓库，一般不需要 `docker login`。使用版本化标签，不要在正式
交付任务中依赖可能变化的 `sft-latest`：

```bash
docker pull iceswallow/qwen36-delivery:sft-1.1-fla
docker pull iceswallow/qwen36-delivery:vllm-eval-1.0

docker tag \
  iceswallow/qwen36-delivery:sft-1.1-fla \
  qwen36-sft:1.1-fla

docker tag \
  iceswallow/qwen36-delivery:vllm-eval-1.0 \
  qwen36-vllm-eval:1.0
```

后续命令统一使用较短的本地标签 `qwen36-sft:1.1-fla` 和
`qwen36-vllm-eval:1.0`。确认两个标签均存在：

```bash
docker image inspect qwen36-sft:1.1-fla >/dev/null
docker image inspect qwen36-vllm-eval:1.0 >/dev/null
docker images | grep -E 'qwen36-(sft|vllm-eval)'
```

如果拉取出现 `no matching manifest for linux/arm64`，说明宿主机不是交付镜像
验证过的 Linux x86_64/amd64 平台；不要用 `--platform` 强行开始正式训练。

### 4.3 检查 GPU、入口和代码挂载

先确认 NVIDIA Container Toolkit 能把 GPU 暴露给训练容器：

```bash
docker run --rm --gpus all \
  --entrypoint nvidia-smi \
  qwen36-sft:1.1-fla
```

再确认当前 checkout 的训练和验证代码能够被容器读取：

```bash
CODE_DIR=/absolute/path/to/19b-a3b

docker run --rm \
  -v "${CODE_DIR}/train:/app/train:ro" \
  qwen36-sft:1.1-fla --help

docker run --rm \
  -v "${CODE_DIR}/eval:/app/eval:ro" \
  qwen36-vllm-eval:1.0 \
  --help
```

训练镜像已嵌入构建时的代码，但正式命令仍只读挂载当前仓库的 `train/` 和
`eval/`，使容器实际运行的代码与当前 checkout 完全一致。代码更新后无需重新
下载镜像；volume mount 会覆盖镜像内置目录。不要把整个项目挂到 `/app`，否则
可能遮蔽镜像内其他运行环境文件。

### 4.4 验证 FLA/FlashAttention 训练快速路径

开始正式训练前，必须在真实 GPU 上运行一次训练级自检。这里的 `MODEL_DIR` 是
宿主机模型目录，例如第 3 节中的
`${DELIVERY_ROOT}/model/19b-a3b`，不是 Git 仓库目录：

```bash
MODEL_DIR=/absolute/path/to/model/19b-a3b
CACHE_DIR=/absolute/path/to/cache

mkdir -p "${CACHE_DIR}"

docker run --rm \
  --gpus all \
  -v "${MODEL_DIR}:/model:ro" \
  -v "${CACHE_DIR}:/cache" \
  --entrypoint python \
  qwen36-sft:1.1-fla \
  /app/train/verify_qwen35_fast_path.py \
  --model-path /model \
  --sequence-length 128
```

预期最后输出包含：

```text
'chunk_rule_impl': 'fla.ops.gated_delta_rule.chunk'
'causal_conv_impl': 'causal_conv1d.causal_conv1d_interface'
'forward_backward': 'ok'
```

若出现 `forward_backward: ok`，说明 linear-attention 的 FLA/TileLang 路径、
causal-conv1d 以及 BF16 forward/backward 均可用。仅仅能导入 `flash_attn` 不等于
Qwen3.5 linear-attention 快速路径可用。首次运行某个 shape 时 TileLang 会编译
内核；训练命令把宿主机 cache 挂到 `/cache`，后续容器可以复用。

### 4.5 可选：运行代码回归测试

使用交付镜像运行热修复回归测试：

```bash
CODE_DIR=/absolute/path/to/19b-a3b

docker run --rm \
  -v "${CODE_DIR}:/workspace:ro" \
  --entrypoint python \
  qwen36-sft:1.1-fla \
  -m unittest discover \
  -s /workspace/tests \
  -p test_training_preprocessing.py \
  -v
```

预期最后输出 `Ran 13 tests` 和 `OK`（缺少可选依赖时个别测试可能显示 skipped）。

### 4.6 离线 tar 备选流程

只有无法访问 Docker Hub，并且交付方另行提供
`qwen36-sft-with-vllm-multinode-1.0.tar` 时，才使用本节。先校验：

```bash
sha256sum qwen36-sft-with-vllm-multinode-1.0.tar
```

该旧归档的正确 SHA-256 为：

```text
64383790cdf82f5b64f92ac21a73ae3a7b9e277980158df105995194cf8ee51f
```

一致后导入；不要用普通 `tar -xf` 解压：

```bash
docker load -i qwen36-sft-with-vllm-multinode-1.0.tar
```

该归档里的 `qwen36-vllm-eval:1.0` 可以继续用于异步验证，但
`qwen36-sft:1.0` 是旧训练镜像，不包含 Qwen3.5 linear-attention 所需的完整
FLA 快速路径，不能代替推荐的 `qwen36-sft:1.1-fla`。离线环境要训练当前版本，
应由交付方同时提供新镜像归档，或按第 4.7 节在本机重建。

### 4.7 本地重建训练镜像（可选）

正常交付直接使用第 4.2 节的预构建镜像，无需执行本节。只有需要审计构建过程、
修改依赖或无法获得新镜像归档时，才在仓库根目录执行：

```bash
docker build --progress=plain \
  -t qwen36-sft:1.1-fla \
  .
```

`causal-conv1d` 对当前 CUDA 13/PyTorch 2.11/Python 3.12 组合没有官方预编译
wheel，首次构建会从源码编译 CUDA forward/backward 内核。官方构建脚本会生成
多个 GPU 架构，数分钟内没有新日志是正常现象。不要因为这一阶段较慢而中断。

构建完成后，必须重新执行第 4.3 和 4.4 节的 GPU/快速路径自检，再开始训练。

## 5. 训练脚本完整参数及其含义

查看当前训练代码的实时帮助：

```bash
CODE_DIR=/absolute/path/to/19b-a3b

docker run --rm \
  -v "${CODE_DIR}/train:/app/train:ro" \
  qwen36-sft:1.1-fla --help
```

所有布尔参数均接受 `true/false`、`1/0`、`yes/no`、`on/off`。下表的默认值
指通过训练镜像入口脚本启动时的有效默认值。

### 5.1 启动器和路径参数

| 参数 | 默认值 | 含义 |
| --- | --- | --- |
| `--gpus` | `CUDA_VISIBLE_DEVICES`，否则 `0` | 宿主机 GPU 编号，逗号分隔。启动器按数量创建相同数量的 `torchrun` 进程。 |
| `--zero-stage` | `ZERO_STAGE`，否则 `3` | 选择内置的 ZeRO-2 或 ZeRO-3 配置，只接受 `2` 或 `3`。也可使用同名环境变量。 |
| `--model-path` | `/model` | 模型目录。必须包含 `config.json`、`model.safetensors.index.json`、tokenizer 文件及全部权重分片。 |
| `--data-files` | 无，必填 | 一个或多个训练 JSON/JSONL 文件。 |
| `--eval-data-files` | 不启用 | 可选验证文件；提供后按 `--eval-steps` 在训练进程内验证。 |
| `--output-dir` | `/output/${RUN_NAME}` | checkpoint、Trainer 状态和数据清单输出目录。 |
| `--deepspeed` | 由 `--zero-stage` 选择 | DeepSpeed 配置文件路径。通常使用 `--zero-stage` 即可；直接运行 Python 时可显式传入。 |
| `--offload` | `true` | ZeRO-2 下控制 optimizer state CPU offload；ZeRO-3 下同时控制参数和 optimizer state CPU offload。 |
| `--expected-num-experts` | `128` | 预检时要求模型具有的 routed expert 数量。 |
| `--resume-from-checkpoint` | 不启用 | 从指定 `checkpoint-N` 恢复模型、优化器、scheduler 和训练进度。 |
| `--run-name` | 启动器自动生成 | 本次任务名称；同时用于默认输出子目录。 |

#### 选择 ZeRO-2 或 ZeRO-3

ZeRO 阶段和 CPU offload 是两个独立选项。启动器默认使用 ZeRO-3，完整组合如下：

| 目标配置 | `ZERO_STAGE` / `--zero-stage` | `--offload` | GPU 中保留的主要模型状态 |
| --- | --- | --- | --- |
| ZeRO-3 + CPU offload（默认） | `3` | `true` | 参数、梯度和 optimizer state 均分片；参数及 optimizer state 可卸载到 CPU |
| ZeRO-3，不 offload | `3` | `false` | 参数、梯度和 optimizer state 均分片并保留在 GPU |
| ZeRO-2 + optimizer offload | `2` | `true` | 参数在每张 GPU 完整保留；梯度和 optimizer state 分片，optimizer state/计算卸载到 CPU |
| ZeRO-2，完全不 offload | `2` | `false` | 参数在每张 GPU 完整保留；梯度和 optimizer state 分片并保留在 GPU |

例如，选择 ZeRO-2 且完全关闭 CPU offload：

```bash
ZERO_STAGE=2
OFFLOAD=false
```

第 6 节的完整命令已经把这两个值放在命令开头，并分别传给
`-e ZERO_STAGE="${ZERO_STAGE}"` 和 `--offload "${OFFLOAD}"`。只需修改这两行，
不需要修改 Python、Shell 或 DeepSpeed JSON 文件。

也可以在 `docker run` 的镜像名称之前设置 `-e ZERO_STAGE=2` 或
`-e ZERO_STAGE=3`。若设置了 `DEEPSPEED_CONFIG`，自定义配置文件优先于
`ZERO_STAGE/--zero-stage` 选择的内置文件；显式传入 `--deepspeed` 时又以该
命令行值为准。训练日志会打印最终实际使用的 ZeRO 阶段和配置路径。

ZeRO-2 不分片模型参数，通常只适合单卡显存充足或 GPU 数量很多的环境；ZeRO-3
显存占用更低。切换阶段后先以目标序列长度运行 `--max-steps 2` 做显存测试。
不要使用 ZeRO-2 直接恢复 ZeRO-3 的 optimizer checkpoint，反向切换也一样；
更换阶段时应创建新任务并从 Hugging Face 模型权重启动。

### 5.2 数据和序列参数

| 参数 | 默认值 | 含义 |
| --- | --- | --- |
| `--max-seq-length` | `24000` | 单条样本最大 token 数。超过上限的样本会被丢弃，不是截断。 |
| `--max-samples` | `-1` | 合并逐文件抽样结果后，训练集最多保留多少条；`<=0` 表示不限。 |
| `--max-eval-samples` | `-1` | 验证集合并后最多保留多少条；`<=0` 表示不限，此处建议使用全部数据。 |
| `--max-samples-per-file` | 不限制 | 每个训练文件分别随机抽取的最大样本数，数值数量必须与 `--data-files` 一致；`<=0` 表示该文件不限。 |
| `--max-eval-samples-per-file` | 不限制 | 每个验证文件的抽样上限，数值数量必须与 `--eval-data-files` 一致。 |
| `--dataset-num-proc` | `8` | 数据预处理使用的 CPU 进程数。 |
| `--seed` | `3407` | 数据抽样、打乱和训练随机种子。 |
| `--attn-implementation` | `flash_attention_2` | Transformers attention 后端。 |
| `--enable-thinking` | `true` | 所有训练文件的 thinking chat template 全局默认值。 |
| `--enable-thinking-per-file` | 不设置 | 为每个训练文件分别指定 thinking 开关；数量和顺序必须与 `--data-files` 一致。 |
| `--eval-enable-thinking-per-file` | 不设置 | 为每个验证文件分别指定 thinking 开关。 |
| `--last-assistant-only` | `false` | `false` 训练每个 assistant turn；`true` 只训练最后一个 assistant turn。 |
| `--last-assistant-only-per-file` | 不设置 | 为每个训练文件覆盖上一项；数量和顺序必须与 `--data-files` 一致。 |
| `--eval-last-assistant-only-per-file` | 不设置 | 为每个验证文件分别设置是否只保留最后一个 assistant turn。 |
| `--mask-empty-think` | `true` | 不监督 chat template 自动插入的空 `<think></think>` 前缀。 |

若数据行自身包含 `enable_thinking` 或 `last_assistant_only`，行级设置的优先级
高于逐文件参数；逐文件参数又高于全局参数。

v1.0.3 在生成 sampling manifest 时只批量读取
`source_file_index`、`last_assistant_only`、`n_tokens` 和 `has_loss`
四个元数据字段，不再为统计操作反序列化大体积的 `input_ids` 和 `labels`。
这不会删除或修改训练 Dataset 中的 token 和 label；Trainer 仍使用完整数据。
Filtering 同样只读取 `has_loss` 和 `n_tokens` 来计算保留索引。

### 5.3 batch、优化器和训练步数

| 参数 | 默认值 | 含义 |
| --- | --- | --- |
| `--per-device-train-batch-size` | `1` | 每张 GPU 每个 micro step 的训练样本数。 |
| `--per-device-eval-batch-size` | `1` | 每张 GPU 的验证 batch size。 |
| `--gradient-accumulation-steps` | `8` | 累积多少个 micro step 后进行一次 optimizer/global step。 |
| `--num-train-epochs` | `1.0` | 训练 epoch 数；当 `--max-steps > 0` 时由后者控制停止。 |
| `--max-steps` | `-1` | 最大 optimizer/global steps；`-1` 表示按 epoch 训练。 |
| `--learning-rate` | 全参 `5e-5`；LoRA `2e-4` | 无 warmup 时为初始/最大学习率；显式传入时覆盖自动默认值。 |
| `--min-learning-rate` | 全参 `5e-6`；LoRA 不设置 | `cosine_with_min_lr` 调度器的最终最低学习率，不能超过 `--learning-rate`。 |
| `--warmup-ratio` | `0.0` | 总训练步数中用于 warmup 的比例。 |
| `--weight-decay` | `0.0` | AdamW weight decay。 |
| `--lr-scheduler-type` | 全参 `cosine_with_min_lr`；LoRA `constant_with_warmup` | Transformers 学习率 scheduler。全参默认从 `5e-5` 余弦衰减到 `5e-6`。 |
| `--optim` | `adamw_torch` | Transformers optimizer 名称。DeepSpeed JSON 中的 AdamW 参数使用 `auto` 与 Trainer 同步。 |
| `--logging-steps` | `1` | 每多少个 global steps 输出一次训练日志。 |
| `--ddp-timeout` | `86400` | 分布式操作最长等待秒数；百万级数据预处理期间非主 rank 会等待主 rank，默认允许等待 24 小时。 |
| `--gradient-checkpointing` | `true` | 用额外计算换取更低的 activation 显存。 |
| `--freeze-vision-tower` | `true` | 冻结文本 SFT 不使用的视觉塔参数。 |

内置 DeepSpeed JSON 有意不定义 `scheduler`：学习率调度完全交给
Transformers Trainer，避免 DeepSpeed 的 `WarmupLR` 覆盖
`cosine_with_min_lr`。如果自行替换 DeepSpeed 配置，也应保持不设置
`scheduler`，否则上述余弦衰减参数不会生效。

`--offload true` 是默认且更节省显存的配置，但 CPU-GPU 数据交换可能降低训练
速度。ZeRO-3 会 offload 参数与 optimizer state；ZeRO-2 不支持参数 offload，
因此只 offload optimizer state。传入 `--offload false` 后，两种阶段都不会使用
CPU offload。

有效全局 batch size 的计算公式是：

```text
节点数 × 每节点进程数 × per-device-train-batch-size
       × gradient-accumulation-steps
```

例如单机 5 卡、batch size 1、梯度累积 8 时，有效全局 batch size 为
`1 × 5 × 1 × 8 = 40`；两台机器各 8 卡且其他参数不变时为
`2 × 8 × 1 × 8 = 128`。增加节点数会改变优化语义，应重新评估学习率、
训练步数和梯度累积。

全参训练默认 `warmup_ratio=0`，因此第一个 optimizer step 使用接近 `5e-5`
的学习率，随后按半个余弦周期平滑衰减，并在训练最后一步达到 `5e-6`。如果显式
设置 `--warmup-ratio > 0`，则学习率会先从接近 0 升至 `5e-5`，再开始余弦
衰减；此时 `5e-5` 是 warmup 后的峰值，而不是第一个 step 的学习率。

### 5.4 保存、训练进程内验证和异步验证参数

| 参数 | 默认值 | 含义 |
| --- | --- | --- |
| `--save-steps` | `70` | 每多少个 optimizer/global steps 保存一次 checkpoint。 |
| `--save-total-limit` | `10` | 最多保留多少个 checkpoint；超出后由 Trainer 轮转删除旧项。 |
| `--save-only-model` | `false` | `false` 同时保存 optimizer、scheduler 等断点续训状态；`true` 只保存模型。 |
| `--eval-steps` | `70` | 提供 `--eval-data-files` 后，每多少个 global steps 验证一次。 |
| `--eval-metric` | `loss` | `loss` 使用 teacher forcing；`pass_rate` 生成答案后比较最终答案。 |
| `--eval-max-new-tokens` | `256` | `pass_rate` 模式下每题最大生成 token 数，必须小于 `--max-seq-length`。 |
| `--eval-generation-enable-thinking` | `false` | `pass_rate` 生成时是否启用 Qwen thinking。 |
| `--async-eval-markers` | `false` | checkpoint 完整保存后是否写 vLLM worker 使用的 ready 标记；推荐验证方式需要设为 `true`，且不能和 `--eval-data-files` 同时使用。 |

`pass_rate` 会在训练进程内用 Transformers 自回归生成，长 thinking 输出可能
非常慢，并会暂停训练。因此第 6 节使用独立 vLLM worker 异步计算 pass rate，
训练进程不传 `--eval-data-files`，而是传 `--async-eval-markers true`。

异步验证只评测已经完整保存的 checkpoint，因此验证间隔由 `--save-steps`
决定，不使用 `--eval-steps`。例如 `--save-steps 10` 表示每 10 个 global
steps 保存并验证一次；如果必须每 5 步验证，需要同时改成 `--save-steps 5`。

全参数 DeepSpeed checkpoint 同时包含模型和优化器状态，单个 checkpoint 可能占用
约 250 GB。`--save-total-limit 10` 的最坏磁盘需求可能接近 2.5 TB，启动前必须
检查输出盘空间。异步验证速度可能低于 checkpoint 保存速度；如果待验证队列中的
旧 checkpoint 超出 `--save-total-limit`，Trainer 可能在 worker 开始验证前将其
删除。正式训练时应按磁盘容量和最大验证积压量提高该参数，确保所有 ready
checkpoint 在验证完成前一直保留。

### 5.5 LoRA 参数

| 参数 | 默认值 | 含义 |
| --- | --- | --- |
| `--use-lora` | `false` | 是否使用 PEFT LoRA；`false` 为全参数训练。 |
| `--lora-r` | `16` | LoRA rank。 |
| `--lora-alpha` | `32` | LoRA scaling alpha。 |
| `--lora-dropout` | `0.0` | LoRA dropout。 |
| `--target-modules` | `q_proj k_proj v_proj o_proj gate_proj up_proj down_proj` | 注入 LoRA adapter 的模块名列表。 |
| `--use-rslora` | `false` | 是否启用 rank-stabilized LoRA。 |

后五项仅在 `--use-lora true` 时生效。

### 5.6 日志和诊断参数

| 参数 | 默认值 | 含义 |
| --- | --- | --- |
| `--report-to` | `none` | Transformers 日志后端，例如 `none`、`tensorboard` 或 `wandb`。 |
| `--preflight-only` | `false` | 只检查模型结构、文件、依赖和 GPU，不进入训练。 |
| `--skip-gpu-check` | `false` | 跳过 CUDA 可用性检查，仅用于静态预检。 |

启动器还支持以下环境变量：

| 环境变量 | 默认值 | 含义 |
| --- | --- | --- |
| `MODEL_PATH` | `/model` | 模型挂载路径。 |
| `OUTPUT_ROOT` | `/output` | 默认输出根目录。 |
| `OUTPUT_DIR` | `${OUTPUT_ROOT}/${RUN_NAME}` | 显式指定本次输出目录。 |
| `RUN_NAME` | 自动生成时间戳名称 | 本次训练名称。 |
| `NNODES` | `1` | 参与同一个训练任务的服务器总数；大于 1 时进入多机模式。 |
| `NODE_RANK` | `0` | 当前服务器编号，范围是 `0` 到 `NNODES-1`，每台必须唯一。 |
| `MASTER_ADDR` | 单机不需要 | 多机必填，填写 node 0 可被其他节点访问的 IP 或主机名。 |
| `MASTER_PORT` | `29508` | 所有节点一致的 rendezvous 端口，必须能在节点间访问。 |
| `NPROC_PER_NODE` | GPU 列表长度 | 本机训练进程数，通常不要手动设置。 |
| `PRECHECK_ONLY` | `0` | 设置为 `1` 时完成入口预检后退出。 |
| `SKIP_GPU_CHECK` | `0` | 设置为 `1` 时跳过 GPU 预检。 |
| `RUN_BACKGROUND` | `0` | 设置为 `1` 时由入口脚本自行 `nohup`；Docker 启动示例不需要它。 |
| `LOG_DIR` | `OUTPUT_ROOT`，未设置时为代码目录下的 `logs` | `RUN_BACKGROUND=1` 时的日志目录。 |
| `RUN_LOG_FILE` | 自动生成 | `RUN_BACKGROUND=1` 时显式指定日志文件。 |
| `ZERO_STAGE` | `3` | 选择内置 ZeRO 配置，只接受 `2` 或 `3`；等价于 `--zero-stage`。 |
| `DEEPSPEED_CONFIG` | 不设置 | 使用自定义 DeepSpeed JSON；设置后覆盖 `ZERO_STAGE/--zero-stage` 的内置文件选择。 |
| `EXPECTED_NUM_EXPERTS` | `128` | 模型 expert 数预期值。 |

### 5.7 异步 vLLM 验证参数

验证容器使用 `qwen36-vllm-eval:1.0`。它持续观察训练输出目录，发现 ready
checkpoint 后启动 vLLM、并发生成 HARP 答案、计算 pass rate、保存结果，然后
关闭该次 vLLM server 并继续等待下一个 checkpoint。

| 参数 | 示例值 | 含义 |
| --- | --- | --- |
| `--run-dir` | `/output/${RUN_NAME}` | 与训练进程完全相同的 run 输出目录。 |
| `--dataset-file` | `/datasets/val/HARP/HARP_difficulty_2_sample_50.jsonl` | 验证集文件；当前交付示例使用 HARP difficulty 2 的 50 题。 |
| `--metadata-source` | `/model` | 原始模型目录，用于为 checkpoint 补齐 processor 等元数据。 |
| `--gpus` | `0,7` | 验证使用的 GPU，不能和同机训练 GPU 重叠。 |
| `--tensor-parallel-size` | `2` | vLLM tensor parallel 数量，必须等于 `--gpus` 中 GPU 的数量。 |
| `--max-model-len` | `24000` | vLLM 最大上下文长度。 |
| `--max-tokens` | `18000` | 每题最多生成 token 数，必须小于 `--max-model-len`。 |
| `--thinking` | `true` | 是否启用 Qwen thinking 模式。 |
| `--temperature` | `0` | 生成温度；`0` 用于确定性 pass rate 验证。 |
| `--concurrency` | `16` | 同时提交给 vLLM 的题目数量。 |
| `--gpu-memory-utilization` | `0.90` | vLLM 可使用的 GPU 显存比例。 |
| `--poll-seconds` | `15` | 没有新 checkpoint 时的轮询间隔。 |
| `--request-timeout` | `3600` | 单个生成请求的超时时间，单位为秒。 |
| `--request-retries` | `3` | 单个请求失败后的最大尝试次数。 |
| `--server-start-timeout` | `1800` | 等待 vLLM server 完成模型加载的最长秒数。 |
| `--max-task-attempts` | `3` | 每个 checkpoint 验证任务的最大尝试次数。 |
| `--limit` | 不限制 | 只验证前 N 题；建议首次冒烟时设为 `1`。 |
| `--retry-failed` | `false` | 启动时是否清除失败标记并重试失败 checkpoint。 |
| `--exit-when-training-complete` | `true` | 训练完成且所有 checkpoint 均处理后自动退出 worker。 |

同一个 `RUN_NAME` 只能启动一个验证 worker。验证 GPU 可以在训练服务器上预留，
也可以位于一台能访问相同模型、验证集和共享输出目录的独立服务器。

## 6. 训练脚本启动代码

### 6.0 先选择 ZeRO 阶段和 offload

单机和多机命令开头都有下面两个变量：

```bash
# 默认配置：ZeRO-3 + CPU offload
ZERO_STAGE=3
OFFLOAD=true
```

按目标硬件把这两行改成以下任意一组：

- ZeRO-3，不使用 CPU offload：`ZERO_STAGE=3`、`OFFLOAD=false`；
- ZeRO-2，只将 optimizer state/计算卸载到 CPU：`ZERO_STAGE=2`、`OFFLOAD=true`；
- ZeRO-2，完全关闭 CPU offload：`ZERO_STAGE=2`、`OFFLOAD=false`。

每次只保留其中一组有效赋值。ZeRO-2 的每张 GPU 都保存完整模型参数，正式训练前
应保留目标上下文长度并临时增加 `--max-steps 2` 做显存冒烟测试。多机训练时，
所有节点的 `ZERO_STAGE` 和 `OFFLOAD` 必须完全相同。

单机和多机示例均使用六个完整训练文件和 32k 上下文，均为每卡 batch size 1、
梯度累积 8、全参数训练、每 10 个 global steps 保存 checkpoint、最多保留
10 个 checkpoint，
并训练 1 个 epoch。每个 checkpoint 会使用
`/datasets/val/HARP/HARP_difficulty_2_sample_50.jsonl` 异步计算一次 pass
rate；验证不会阻塞训练。

Docker 的 `--gpus all` 只负责把宿主机 GPU 暴露给容器，真正参与训练的卡由
镜像后的脚本参数 `--gpus` 决定。

以下训练示例都假定已按第 4.1、4.2 节获取当前源码和镜像。训练容器通过
`-v "${CODE_DIR}/train:/app/train:ro"` 覆盖镜像内置训练代码，验证容器通过
`-v "${CODE_DIR}/eval:/app/eval:ro"` 覆盖镜像内置验证代码。

### 6.1 单机多卡

下面示例在一台 8 卡服务器上使用 GPU `1,2,3,4,6` 训练，GPU `0,7` 进行
vLLM 验证，GPU `5` 保留。训练有效全局 batch size 是
`1 × 5 × 1 × 8 = 40`。单机模式不需要设置任何多机环境变量。

先启动验证 worker，再启动训练。worker 可以先启动并等待训练生成第一个
checkpoint。

```bash
DELIVERY_ROOT=/data/qwen36-delivery
CODE_DIR=/absolute/path/to/19b-a3b
MODEL_DIR="${DELIVERY_ROOT}/model/19b-a3b"
DATA_DIR="${DELIVERY_ROOT}/data/math100w"
OUTPUT_ROOT="${DELIVERY_ROOT}/output"
LOG_DIR="${DELIVERY_ROOT}/logs"
CACHE_DIR="${DELIVERY_ROOT}/cache"

# 训练配置：当前为默认 ZeRO-3 + offload；ZeRO-2 无 offload 改为 2/false。
ZERO_STAGE=3
OFFLOAD=true

mkdir -p \
  "${OUTPUT_ROOT}" \
  "${LOG_DIR}" \
  "${CACHE_DIR}/torch" \
  "${CACHE_DIR}/vllm"

test -f "${CODE_DIR}/train/train_qwen36_19b_a3b_sft_deepspeed.py"
test -f "${CODE_DIR}/eval/async_vllm_eval.py"

RUN_NAME="qwen36-19b-a3b-sft-$(date +%Y%m%d_%H%M%S)"
TRAIN_CONTAINER="${RUN_NAME}-train"
EVAL_CONTAINER="${RUN_NAME}-eval"
TRAIN_LOG_FILE="${LOG_DIR}/${RUN_NAME}.train.log"
EVAL_LOG_FILE="${LOG_DIR}/${RUN_NAME}.async-eval.log"

nohup docker run --rm \
  --name "${EVAL_CONTAINER}" \
  --gpus all \
  --ipc=host \
  --shm-size=64g \
  --ulimit memlock=-1 \
  -e PYTHONUNBUFFERED=1 \
  -v "${CODE_DIR}/eval:/app/eval:ro" \
  -v "${MODEL_DIR}:/model:ro" \
  -v "${DATA_DIR}:/datasets:ro" \
  -v "${OUTPUT_ROOT}:/output" \
  -v "${CACHE_DIR}/vllm:/cache" \
  qwen36-vllm-eval:1.0 \
  --run-dir "/output/${RUN_NAME}" \
  --dataset-file /datasets/val/HARP/HARP_difficulty_2_sample_50.jsonl \
  --metadata-source /model \
  --gpus 0,7 \
  --tensor-parallel-size 2 \
  --max-model-len 24000 \
  --max-tokens 18000 \
  --thinking true \
  --temperature 0 \
  --concurrency 16 \
  --poll-seconds 15 \
  >"${EVAL_LOG_FILE}" 2>&1 &

EVAL_PID=$!

nohup docker run --rm \
  --name "${TRAIN_CONTAINER}" \
  --gpus all \
  --ipc=host \
  --shm-size=64g \
  --ulimit memlock=-1 \
  -v "${CODE_DIR}/train:/app/train:ro" \
  -v "${MODEL_DIR}:/model:ro" \
  -v "${DATA_DIR}:/datasets:ro" \
  -v "${OUTPUT_ROOT}:/output" \
  -v "${CACHE_DIR}/torch:/cache" \
  -e MODEL_PATH=/model \
  -e OUTPUT_ROOT=/output \
  -e RUN_NAME="${RUN_NAME}" \
  -e ZERO_STAGE="${ZERO_STAGE}" \
  -e PYTHONUNBUFFERED=1 \
  qwen36-sft:1.1-fla \
  --gpus 1,2,3,4,6 \
  --data-files \
    /datasets/Nemotron-SFT-Math-v4/train25w.jsonl \
    /datasets/OpenMathReasoning/train20w.jsonl \
    /datasets/OpenR1-Math-220k/train.jsonl \
    /datasets/OpenCodeReasoning-2/train20w.jsonl \
    /datasets/general/qwen3_235b_thinking_2507_110k_sft.jsonl \
    /datasets/Nemotron-SFT-Instruction-Following-Chat-v3-chat/train6w.jsonl \
  --max-samples-per-file \
    -1 -1 -1 -1 -1 -1 \
  --last-assistant-only-per-file \
    false false false false false true \
  --enable-thinking-per-file \
    true true true true true true \
  --max-seq-length 32000 \
  --per-device-train-batch-size 1 \
  --gradient-accumulation-steps 8 \
  --num-train-epochs 1 \
  --learning-rate 5e-5 \
  --min-learning-rate 5e-6 \
  --lr-scheduler-type cosine_with_min_lr \
  --offload "${OFFLOAD}" \
  --use-lora false \
  --freeze-vision-tower true \
  --save-steps 10 \
  --save-total-limit 10 \
  --save-only-model false \
  --async-eval-markers true \
  --report-to none \
  --dataset-num-proc 64 \
  --ddp-timeout 86400 \
  >"${TRAIN_LOG_FILE}" 2>&1 &

TRAIN_PID=$!

echo "训练 PID=${TRAIN_PID}"
echo "验证 PID=${EVAL_PID}"
echo "训练容器=${TRAIN_CONTAINER}"
echo "验证容器=${EVAL_CONTAINER}"
echo "训练日志=${TRAIN_LOG_FILE}"
echo "验证日志=${EVAL_LOG_FILE}"
echo "输出=${OUTPUT_ROOT}/${RUN_NAME}"
echo "验证汇总=${OUTPUT_ROOT}/${RUN_NAME}/async_eval/results.jsonl"
```

### 6.2 多机多卡

下面以两台服务器、每台 8 张 GPU 为例。两台服务器必须满足：

- 已导入完全相同的 `qwen36-sft:1.1-fla` 镜像；
- 已 checkout 完全相同的 Git commit/tag，并挂载相同的 `train/` 代码；
- 模型和数据内容完全相同，可以是每台本地副本，也可以是共享只读目录；
- node 0 的 `MASTER_ADDR:MASTER_PORT` 能被其他节点访问；
- 节点之间允许 PyTorch/NCCL 通信，Docker 使用 `--network host`；
- 宿主机 `SHARED_OUTPUT_ROOT` 是所有节点可见的同一个 NFS、Lustre 或 CephFS
  目录，并在容器中统一挂载为 `/output`；
- `NNODES`、`MASTER_ADDR`、`MASTER_PORT`、`RUN_NAME` 和所有训练参数在全部
  节点上完全一致；
- `NODE_RANK` 每台唯一，必须依次为 `0,1,...,NNODES-1`。

所有训练节点还必须传入 `--async-eval-markers true`。整个多机训练任务只启动
一个 `qwen36-vllm-eval:1.0` worker；不要在每个训练节点重复启动。

不要让每台服务器分别使用自己的本地输出盘冒充同一个 `/output`。DeepSpeed
checkpoint 由多个 global rank 共同保存；输出不共享会得到分散或不完整的
checkpoint。

首先确定固定配置。例如：

```text
node 0 IP: 10.20.30.40，NODE_RANK=0
node 1 IP: 10.20.30.41，NODE_RANK=1
NNODES=2
MASTER_ADDR=10.20.30.40
MASTER_PORT=29508
RUN_NAME=qwen36-19b-a3b-sft-multinode-001
```

在 node 0 上先执行下面的完整命令，把 `NODE_RANK` 设为 `0`；随后在 node 1
执行同一命令，只把 `NODE_RANK` 改为 `1`。更多节点以此类推。

```bash
DELIVERY_ROOT=/data/qwen36-delivery
CODE_DIR=/absolute/path/to/19b-a3b
MODEL_DIR="${DELIVERY_ROOT}/model/19b-a3b"
DATA_DIR="${DELIVERY_ROOT}/data/math100w"
LOCAL_LOG_DIR="${DELIVERY_ROOT}/logs"
LOCAL_CACHE_DIR="${DELIVERY_ROOT}/cache"

# 训练配置：当前为默认 ZeRO-3 + offload；ZeRO-2 无 offload 改为 2/false。
# 所有节点必须使用相同的值。
ZERO_STAGE=3
OFFLOAD=true

# 必须是已在所有节点挂载好的同一个共享文件系统目录。
SHARED_OUTPUT_ROOT=/shared/qwen36-output

# 以下五项除 NODE_RANK 外，所有节点必须完全相同；上面的 ZERO_STAGE 和
# OFFLOAD 也必须在所有节点保持一致。
NNODES=2
NODE_RANK=0
MASTER_ADDR=10.20.30.40
MASTER_PORT=29508
RUN_NAME=qwen36-19b-a3b-sft-multinode-001

mkdir -p "${LOCAL_LOG_DIR}" "${LOCAL_CACHE_DIR}"
test -d "${SHARED_OUTPUT_ROOT}"
test -f "${CODE_DIR}/train/train_qwen36_19b_a3b_sft_deepspeed.py"

CONTAINER_NAME="${RUN_NAME}-node${NODE_RANK}"
LOG_FILE="${LOCAL_LOG_DIR}/${CONTAINER_NAME}.log"

nohup docker run --rm \
  --name "${CONTAINER_NAME}" \
  --network host \
  --gpus all \
  --ipc=host \
  --shm-size=64g \
  --ulimit memlock=-1 \
  -v "${CODE_DIR}/train:/app/train:ro" \
  -v "${MODEL_DIR}:/model:ro" \
  -v "${DATA_DIR}:/datasets:ro" \
  -v "${SHARED_OUTPUT_ROOT}:/output" \
  -v "${LOCAL_CACHE_DIR}:/cache" \
  -e MODEL_PATH=/model \
  -e OUTPUT_ROOT=/output \
  -e RUN_NAME="${RUN_NAME}" \
  -e ZERO_STAGE="${ZERO_STAGE}" \
  -e NNODES="${NNODES}" \
  -e NODE_RANK="${NODE_RANK}" \
  -e MASTER_ADDR="${MASTER_ADDR}" \
  -e MASTER_PORT="${MASTER_PORT}" \
  -e PYTHONUNBUFFERED=1 \
  -e NCCL_DEBUG=INFO \
  qwen36-sft:1.1-fla \
  --gpus 0,1,2,3,4,5,6,7 \
  --data-files \
    /datasets/Nemotron-SFT-Math-v4/train25w.jsonl \
    /datasets/OpenMathReasoning/train20w.jsonl \
    /datasets/OpenR1-Math-220k/train.jsonl \
    /datasets/OpenCodeReasoning-2/train20w.jsonl \
    /datasets/general/qwen3_235b_thinking_2507_110k_sft.jsonl \
    /datasets/Nemotron-SFT-Instruction-Following-Chat-v3-chat/train6w.jsonl \
  --max-samples-per-file \
    -1 -1 -1 -1 -1 -1 \
  --last-assistant-only-per-file \
    false false false false false true \
  --enable-thinking-per-file \
    true true true true true true \
  --max-seq-length 32000 \
  --per-device-train-batch-size 1 \
  --gradient-accumulation-steps 8 \
  --num-train-epochs 1 \
  --learning-rate 5e-5 \
  --min-learning-rate 5e-6 \
  --lr-scheduler-type cosine_with_min_lr \
  --offload "${OFFLOAD}" \
  --use-lora false \
  --freeze-vision-tower true \
  --save-steps 10 \
  --save-total-limit 10 \
  --save-only-model false \
  --async-eval-markers true \
  --report-to none \
  --dataset-num-proc 64 \
  --ddp-timeout 86400 \
  >"${LOG_FILE}" 2>&1 &

TRAIN_PID=$!

echo "docker run PID=${TRAIN_PID}"
echo "NODE_RANK=${NODE_RANK}/${NNODES}"
echo "容器名称=${CONTAINER_NAME}"
echo "日志=${LOG_FILE}"
echo "共享输出=${SHARED_OUTPUT_ROOT}/${RUN_NAME}"
```

在一台具有两张空闲 GPU、且能访问同一共享输出目录的服务器上启动唯一的验证
worker。下面示例假设该服务器上的 GPU `0,1` 专用于验证；它可以是独立验证
服务器，也可以是预留了两张卡的训练服务器。

```bash
DELIVERY_ROOT=/data/qwen36-delivery
CODE_DIR=/absolute/path/to/19b-a3b
MODEL_DIR="${DELIVERY_ROOT}/model/19b-a3b"
DATA_DIR="${DELIVERY_ROOT}/data/math100w"
EVAL_LOG_DIR="${DELIVERY_ROOT}/logs"
EVAL_CACHE_DIR="${DELIVERY_ROOT}/cache/vllm"

# 与训练节点相同的共享目录和 RUN_NAME。
SHARED_OUTPUT_ROOT=/shared/qwen36-output
RUN_NAME=qwen36-19b-a3b-sft-multinode-001

EVAL_CONTAINER="${RUN_NAME}-eval"
EVAL_LOG_FILE="${EVAL_LOG_DIR}/${RUN_NAME}.async-eval.log"

mkdir -p "${EVAL_LOG_DIR}" "${EVAL_CACHE_DIR}"
test -d "${SHARED_OUTPUT_ROOT}/${RUN_NAME}"
test -f "${CODE_DIR}/eval/async_vllm_eval.py"

nohup docker run --rm \
  --name "${EVAL_CONTAINER}" \
  --gpus all \
  --ipc=host \
  --shm-size=64g \
  --ulimit memlock=-1 \
  -e PYTHONUNBUFFERED=1 \
  -v "${CODE_DIR}/eval:/app/eval:ro" \
  -v "${MODEL_DIR}:/model:ro" \
  -v "${DATA_DIR}:/datasets:ro" \
  -v "${SHARED_OUTPUT_ROOT}:/output" \
  -v "${EVAL_CACHE_DIR}:/cache" \
  qwen36-vllm-eval:1.0 \
  --run-dir "/output/${RUN_NAME}" \
  --dataset-file /datasets/val/HARP/HARP_difficulty_2_sample_50.jsonl \
  --metadata-source /model \
  --gpus 0,1 \
  --tensor-parallel-size 2 \
  --max-model-len 24000 \
  --max-tokens 18000 \
  --thinking true \
  --temperature 0 \
  --concurrency 16 \
  --poll-seconds 15 \
  >"${EVAL_LOG_FILE}" 2>&1 &

EVAL_PID=$!

echo "验证 PID=${EVAL_PID}"
echo "验证容器=${EVAL_CONTAINER}"
echo "验证日志=${EVAL_LOG_FILE}"
echo "验证汇总=${SHARED_OUTPUT_ROOT}/${RUN_NAME}/async_eval/results.jsonl"
```

两台 8 卡、每卡 batch size 1、梯度累积 8 时，有效全局 batch size 为：

```text
2 × 8 × 1 × 8 = 128
```

如果要保持与单机 8 卡相同的有效全局 batch size，应相应降低
`--gradient-accumulation-steps`。增加总 GPU 数后也应重新评估学习率和训练步数。

首次多机运行建议临时增加：

```bash
--max-steps 2
```

先确认所有节点完成初始化、执行训练 step 并生成完整 checkpoint，再开始正式任务。
如果 NCCL 选择了错误网卡，可在所有节点的 `docker run` 中增加相同的环境变量：

```bash
-e NCCL_SOCKET_IFNAME=实际训练网卡名
```

InfiniBand/RoCE 所需设备映射、RDMA 驱动和 NCCL 环境变量取决于接收方集群，
应由集群管理员配置。

### 6.3 监控和停止

单机查看状态：

```bash
tail -f "${TRAIN_LOG_FILE}"
tail -f "${EVAL_LOG_FILE}"
docker ps --filter "name=${TRAIN_CONTAINER}"
docker ps --filter "name=${EVAL_CONTAINER}"
nvidia-smi
```

多机需要分别查看每台服务器的 node 日志：

```bash
tail -f "${LOG_FILE}"
docker ps --filter "name=${CONTAINER_NAME}"
nvidia-smi
```

在验证 worker 所在服务器查看：

```bash
tail -f "${EVAL_LOG_FILE}"
docker ps --filter "name=${EVAL_CONTAINER}"
tail -n 20 "${SHARED_OUTPUT_ROOT}/${RUN_NAME}/async_eval/results.jsonl"
```

每个 checkpoint 的文件位于：

```text
${OUTPUT_ROOT}/${RUN_NAME}/async_eval/checkpoint-XXXXXXXX/metrics.json
${OUTPUT_ROOT}/${RUN_NAME}/async_eval/checkpoint-XXXXXXXX/predictions.jsonl
${OUTPUT_ROOT}/${RUN_NAME}/async_eval/checkpoint-XXXXXXXX/vllm_server.log
${OUTPUT_ROOT}/${RUN_NAME}/async_eval/results.jsonl
```

其中 `XXXXXXXX` 是八位补零的 global step，例如 step 10 对应
`checkpoint-00000010`。

`results.jsonl` 是所有 checkpoint 的 pass rate 汇总。验证日志中的进度条格式为
`vLLM验证(step=N): 50/50`。训练完成且所有 ready checkpoint 处理完后，worker
默认自动退出。

后台日志不是交互式终端，tqdm 使用回车符刷新同一行，因此 `tail -f` 中的进度条
可能不连续换行，甚至在一段时间后集中显示；这不表示训练卡住。上述命令已设置
`PYTHONUNBUFFERED=1` 以便普通日志及时落盘。判断训练是否前进时，同时观察日志中
的 `loss`/`learning_rate`/`epoch`、checkpoint 目录更新时间和 `nvidia-smi`。
训练结束后，Trainer 汇总中的 `train_runtime` 是完整训练耗时。

多机任务中任一节点失败，整个分布式任务通常都会失败。停止训练时，应在每台
服务器上执行：

```bash
docker stop "${CONTAINER_NAME}"
```

如需提前停止验证 worker，在它所在的服务器执行：

```bash
docker stop "${EVAL_CONTAINER}"
```

### 6.4 从 checkpoint 恢复

保持原来的 `RUN_NAME`、节点数量、节点 rank、共享输出挂载和训练参数，并在
所有节点的训练参数末尾增加同一个容器内 checkpoint 路径，例如：

```bash
--resume-from-checkpoint "/output/${RUN_NAME}/checkpoint-70"
```

恢复时必须重新启动所有节点。只有 `--save-only-model false` 保存出的完整
checkpoint 才能恢复 optimizer、scheduler 和精确训练步数。验证 worker 可以
使用相同 `RUN_NAME` 和 `--run-dir` 重新启动；已经成功生成 `.done` 标记的
checkpoint 不会重复评测。
