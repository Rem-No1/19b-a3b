# Qwen3.6-19B-A3B DeepSpeed SFT 交付说明

本文档面向收到代码目录和 Docker 镜像文件的使用者。按第 3、4、6 节操作，
即可启动训练并在每个已保存 checkpoint 上异步计算验证集 pass rate，无需在
宿主机单独安装 PyTorch、Transformers、DeepSpeed、flash-attn 或 vLLM。

> **v1.0.1 代码热修复：** 修复百万级数据预处理时 manifest 扫描过慢，以及
> 非主 rank 长时间等待预处理可能触发 DDP/NCCL timeout 的问题；全参训练默认
> 学习率改为从 `5e-5` 余弦衰减到 `5e-6`。v1.0.1 源码继续使用原
> `qwen36-sft:1.0` 镜像，但训练容器必须按第 4.3 节只读挂载新版 `train/`
> 目录；不挂载时仍会执行镜像内置的 v1.0.0 代码。

> 安全提示：下文的 `hf-xxxxxxx` 是交付方提供的 Hugging Face 访问令牌。
> 不要把真实令牌写入代码、README、Shell 历史或 Git 仓库。若令牌已经公开，
> 应立即在 Hugging Face 中撤销并重新创建只读令牌。

## 1. 目录结构和代码用途

本项目用于对 Qwen3.6/Qwen3.5 架构的 19B-A3B、128-expert MoE 模型进行
单机或多机多卡监督微调（SFT）。训练基于 Hugging Face Transformers 和
DeepSpeed，默认采用 BF16、FlashAttention 2、梯度检查点以及 ZeRO-3 CPU
offload。单机是默认模式；多机通过四个环境变量启用，无需修改 Python 训练代码。

数据支持 JSONL 文件或顶层为数组的 JSON 文件。每条训练记录至少需要包含
OpenAI/Qwen 风格的 `messages` 字段，也支持 `tools`、逐行
`enable_thinking` 和逐行 `last_assistant_only`。

```text
qwen36_sft/
├── README.md
├── Dockerfile
├── requirements.lock
├── launch_async_delivery_test.sh
├── train/
│   ├── run_qwen36_19b_a3b_sft_deepspeed.sh
│   ├── train_qwen36_19b_a3b_sft_deepspeed.py
│   ├── toolchat_data.py
│   ├── pass_rate_eval.py
│   ├── async_eval_markers.py
│   └── ds_config/
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
| `train/ds_config/qwen36_19b_a3b_zero3.json` | DeepSpeed ZeRO-3、CPU 参数 offload 和 CPU optimizer offload 配置。 |
| `train/pass_rate_eval.py` | 训练进程内生成式 pass rate 评测实现，适合短输出调试。 |
| `train/async_eval_markers.py` | 在 checkpoint 完整保存后生成异步评测 ready 标记。 |
| `eval/` | 独立 vLLM 评测工具；推荐的 HARP pass rate 验证需要启动一个 worker。 |
| `launch_async_delivery_test.sh` | 交付方内部异步联调脚本；接收方进行标准训练时不需要使用。 |
| `Dockerfile`、`requirements.lock` | 重建训练镜像时使用；收到镜像 tar 后不需要重建。 |

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
  qwen36-sft:1.0
```

当前配置使用 DeepSpeed ZeRO-3，将参数、梯度和优化器状态分片，并把模型参数
与优化器卸载到 CPU。使用其他 GPU、较少的 CPU 内存或不同驱动时，需要重新评估
兼容性、显存、主机内存和训练速度。24k 上下文的资源消耗较高。

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
└── split/
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

## 4. 使用交付的 Docker 镜像文件

交付的文件名为：

```text
qwen36-sft-with-vllm-multinode-1.0.tar
```

它是 `docker save` 生成的未压缩 Docker 镜像归档，包含：

- `qwen36-sft:1.0`：训练镜像；
- `qwen36-vllm-eval:1.0`：推荐验证流程使用的独立 vLLM 评测镜像。

### 4.1 校验文件

将终端切换到 tar 所在目录并执行：

```bash
sha256sum qwen36-sft-with-vllm-multinode-1.0.tar
```

正确的 SHA-256 为：

```text
64383790cdf82f5b64f92ac21a73ae3a7b9e277980158df105995194cf8ee51f
```

若校验值不同，说明文件在传输过程中损坏，不要继续导入。

### 4.2 导入镜像

```bash
docker load -i qwen36-sft-with-vllm-multinode-1.0.tar
```

不要使用普通 `tar -xf` 解压该文件。导入后确认两个镜像存在：

```bash
docker image inspect qwen36-sft:1.0 >/dev/null
docker image inspect qwen36-vllm-eval:1.0 >/dev/null
docker images | grep -E 'qwen36-(sft|vllm-eval)'
```

标准训练只使用 `qwen36-sft:1.0`。模型和数据不需要重新打进镜像，也不需要
重新执行 `docker build`。

### 4.3 获取 v1.0.1 热修复代码

首次获取源码：

```bash
git clone https://github.com/Rem-No1/19b-a3b.git
cd 19b-a3b
git checkout v1.0.1
```

已有仓库时：

```bash
git fetch origin --tags
git checkout v1.0.1
```

原镜像不包含热修复代码。运行训练容器时必须把仓库中的 `train/` 只读挂载到
`/app/train`：

```bash
CODE_DIR=/absolute/path/to/19b-a3b

docker run --rm \
  -v "${CODE_DIR}/train:/app/train:ro" \
  qwen36-sft:1.0 --help
```

无需重新下载镜像；volume mount 会在运行时覆盖镜像内置的旧 `train/` 目录。
模型、数据、输出和 cache 的挂载方式不变。多机训练时所有节点必须 checkout
同一个 Git tag，并挂载完全相同的代码。

使用交付镜像运行热修复回归测试：

```bash
CODE_DIR=/absolute/path/to/19b-a3b

docker run --rm \
  -v "${CODE_DIR}:/workspace:ro" \
  --entrypoint python \
  qwen36-sft:1.0 \
  -m unittest discover \
  -s /workspace/tests \
  -p test_training_preprocessing.py \
  -v
```

预期最后输出 `Ran 9 tests` 和 `OK`。

## 5. 训练脚本完整参数及其含义

查看 v1.0.1 训练代码的实时帮助：

```bash
CODE_DIR=/absolute/path/to/19b-a3b

docker run --rm \
  -v "${CODE_DIR}/train:/app/train:ro" \
  qwen36-sft:1.0 --help
```

所有布尔参数均接受 `true/false`、`1/0`、`yes/no`、`on/off`。下表的默认值
指通过训练镜像入口脚本启动时的有效默认值。

### 5.1 启动器和路径参数

| 参数 | 默认值 | 含义 |
| --- | --- | --- |
| `--gpus` | `CUDA_VISIBLE_DEVICES`，否则 `0` | 宿主机 GPU 编号，逗号分隔。启动器按数量创建相同数量的 `torchrun` 进程。 |
| `--model-path` | `/model` | 模型目录。必须包含 `config.json`、`model.safetensors.index.json`、tokenizer 文件及全部权重分片。 |
| `--data-files` | 无，必填 | 一个或多个训练 JSON/JSONL 文件。 |
| `--eval-data-files` | 不启用 | 可选验证文件；提供后按 `--eval-steps` 在训练进程内验证。 |
| `--output-dir` | `/output/${RUN_NAME}` | checkpoint、Trainer 状态和数据清单输出目录。 |
| `--deepspeed` | 内置 ZeRO-3 JSON | DeepSpeed 配置文件路径。 |
| `--expected-num-experts` | `128` | 预检时要求模型具有的 routed expert 数量。 |
| `--resume-from-checkpoint` | 不启用 | 从指定 `checkpoint-N` 恢复模型、优化器、scheduler 和训练进度。 |
| `--run-name` | 启动器自动生成 | 本次任务名称；同时用于默认输出子目录。 |

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

v1.0.1 在生成 sampling manifest 时只批量读取
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

全参数 ZeRO-3 checkpoint 同时包含模型和优化器分片，单个 checkpoint 可能占用
约 250 GB。`--save-total-limit 10` 的最坏磁盘需求可能接近 2.5 TB，启动前必须
检查输出盘空间。

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
| `DEEPSPEED_CONFIG` | 镜像内置 JSON | 替换 DeepSpeed 配置文件。 |
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

单机示例使用六个完整训练文件和 32k 上下文；多机首次联调示例从六个文件合计
抽取 10,000 条并使用 24k 上下文。两者均为每卡 batch size 1、梯度累积 8、
全参数训练、每 10 个 global steps 保存 checkpoint、最多保留 10 个 checkpoint，
并训练 1 个 epoch。每个 checkpoint 会使用
`/datasets/val/HARP/HARP_difficulty_2_sample_50.jsonl` 异步计算一次 pass
rate；验证不会阻塞训练。

Docker 的 `--gpus all` 只负责把宿主机 GPU 暴露给容器，真正参与训练的卡由
镜像后的脚本参数 `--gpus` 决定。

以下训练示例都假定已按第 4.3 节获取 v1.0.1 源码，并通过
`-v "${CODE_DIR}/train:/app/train:ro"` 覆盖镜像内置训练代码。

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

mkdir -p \
  "${OUTPUT_ROOT}" \
  "${LOG_DIR}" \
  "${CACHE_DIR}/torch" \
  "${CACHE_DIR}/vllm"

test -f "${CODE_DIR}/train/train_qwen36_19b_a3b_sft_deepspeed.py"

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
  qwen36-sft:1.0 \
  --gpus 1,2,3,4,6 \
  --data-files \
    /datasets/split/Nemotron-SFT-Math-v4/train25w.jsonl \
    /datasets/split/OpenMathReasoning/train20w.jsonl \
    /datasets/split/OpenR1-Math-220k/train.jsonl \
    /datasets/split/OpenCodeReasoning-2/train20w.jsonl \
    /datasets/split/general/qwen3_235b_thinking_2507_110k_sft.jsonl \
    /datasets/split/Nemotron-SFT-Instruction-Following-Chat-v3-chat/train6w.jsonl \
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

- 已导入完全相同的 `qwen36-sft:1.0` 镜像；
- 已 checkout 完全相同的 v1.0.1 Git tag，并挂载相同的 `train/` 代码；
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

不要让每台服务器分别使用自己的本地输出盘冒充同一个 `/output`。ZeRO-3
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

# 必须是已在所有节点挂载好的同一个共享文件系统目录。
SHARED_OUTPUT_ROOT=/shared/qwen36-output

# 以下五项除 NODE_RANK 外，所有节点必须完全相同。
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
  -e NNODES="${NNODES}" \
  -e NODE_RANK="${NODE_RANK}" \
  -e MASTER_ADDR="${MASTER_ADDR}" \
  -e MASTER_PORT="${MASTER_PORT}" \
  -e NCCL_DEBUG=INFO \
  qwen36-sft:1.0 \
  --gpus 0,1,2,3,4,5,6,7 \
  --data-files \
    /datasets/split/Nemotron-SFT-Math-v4/train25w.jsonl \
    /datasets/split/OpenMathReasoning/train20w.jsonl \
    /datasets/split/OpenR1-Math-220k/train.jsonl \
    /datasets/split/OpenCodeReasoning-2/train20w.jsonl \
    /datasets/split/general/qwen3_235b_thinking_2507_110k_sft.jsonl \
    /datasets/split/Nemotron-SFT-Instruction-Following-Chat-v3-chat/train6w.jsonl \
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

nohup docker run --rm \
  --name "${EVAL_CONTAINER}" \
  --gpus all \
  --ipc=host \
  --shm-size=64g \
  --ulimit memlock=-1 \
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
${OUTPUT_ROOT}/${RUN_NAME}/async_eval/checkpoint-N/metrics.json
${OUTPUT_ROOT}/${RUN_NAME}/async_eval/checkpoint-N/predictions.jsonl
${OUTPUT_ROOT}/${RUN_NAME}/async_eval/checkpoint-N/vllm_server.log
${OUTPUT_ROOT}/${RUN_NAME}/async_eval/results.jsonl
```

`results.jsonl` 是所有 checkpoint 的 pass rate 汇总。验证日志中的进度条格式为
`vLLM验证(step=N): 50/50`。训练完成且所有 ready checkpoint 处理完后，worker
默认自动退出。

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
