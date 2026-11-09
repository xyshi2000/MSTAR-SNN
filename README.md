# SparseNeuron

SEW-ResNet 神经元剪枝，支持 CIFAR、ImageNet 和 MSTAR。

## 环境

使用 Python 3.10+，先安装与 GPU/CUDA 匹配的 PyTorch 与 torchvision，再安装
`requirements.txt` 中的其余依赖。默认依赖文件选择 CUDA 12 的 CuPy 包；CUDA 13
环境应将 `cupy-cuda12x` 替换为 `cupy-cuda13x`，不要同时安装两者。训练和评估均使用
现有 CUDA/CuPy 路径，需要 NVIDIA GPU；不支持通过 `--device cpu` 运行。

```bash
pip install -r requirements.txt
```

## MSTAR 数据与模型

`--data-path` 指向 SOC 根目录，目录结构如下：

```text
soc/
  train/{class}/*.jpg
  test/{class}/*.jpg
```

训练与测试必须具有相同的十类类别及标签映射。标准 SOC 数据为 17° 训练集 2747 张、
15° 测试集 2425 张。代码允许同样十类结构的子集用于短程检查，不按样本总数限制读取。

预处理沿用本地 `mstar-ann-snn-baseline`：灰度单通道、128×128 中心裁剪，只用训练集
计算均值和标准差；训练时增加反射填充 4 像素后的随机裁剪，不做翻转或颜色增强。
`--no-aug` 可关闭平移增强。预处理参数和类别映射保存在 `config.json` 与 checkpoint
的 `mstar_metadata` 字段中，恢复及独立评估使用这些参数并核对类别。

默认模型为单通道 `sew_resnet18_imagenet`，使用标准 7×7 stem、LIF 脉冲神经元
和十类分类头（含原有 boost 读出）。所有 SEW 模型统一使用 LIF，默认 tau=2、阈值=1、
复位电位=0、ATan 代理梯度，采用多步 CuPy 后端。静态图像重复 T=4 步，对 logits 取时间平均，
每个 batch 后重置膜电位。它与参考目录中的 post-add LIF ResNet-18 结构不同，
不能加载该目录附带的 ANN/SNN 权重，其报告精度也不代表本项目结果。

## 训练与剪枝

从项目根目录运行，命令行参数会覆盖 YAML 配置：

```bash
# 普通训练：100 轮，无剪枝、无微调。
python main.py --config configs/image_classification/sew_resnet18_mstar_train.yaml --data-path /path/to/soc

# 从头剪枝：100 轮搜索连续掩码，再固定二值掩码微调 40 轮。
python main.py --config configs/image_classification/sew_resnet18_mstar_prune.yaml --data-path /path/to/soc
```

两套配方使用 seed=0、batch=64、AdamW、lr=0.001、weight decay=0.0001、Cosine 和 AMP。
AdamW 沿用当前项目对 bias/norm 免衰减的参数分组；Cosine 最低学习率为 0。
SEW 神经元定义、参数分组与调度细节并不完全等同于原基线。

剪枝配方默认掩码初始化 `[0, 0]`、温度从 5 升至 1000、惩罚系数 `1e-5`、微调
学习率 0.0001。这些值是起始配置，需要根据精度与剩余神经元比例调整，不保证目标
稀疏率。`--penalty-lmbda` 可调整惩罚强度；此流程从头联合训练权重与掩码。

MSTAR 配置开启 `eval_final_only`：训练和微调期间不读取测试图像进行推理、不使用
测试准确率选择权重。训练结束后只评估最终阶段权重。关闭该开关则沿用项目每轮
评估、按微调测试准确率保存最佳权重的行为。

输出包含 `config.json`、`log.log`、TensorBoard 记录和以下权重：

- `checkpoint_latest.pth`：最近一轮，包含 `phase`、优化器、调度器及 AMP scaler 状态。
- `checkpoint_sparsified.pth`：搜索或普通训练结束时的权重。
- `checkpoint_final.pth`：最终微调权重；无微调时为最终搜索或普通训练权重。

最终评估在 `test/metrics.json` 保存准确率（百分数）、剩余神经元数和每张图像 SOPs，
剪枝模型还会输出掩码可视化。TensorBoard 在仅最终评估模式下只记录训练和稀疏指标。
运行实验时使用独立输出目录，避免混入旧实验文件。

评估还会单独遍历测试集测量平均推理时间，在日志和 `test/metrics.json` 的
`inference_time_ms_per_sample` 输出毫秒/样本。计时前移除 SOP 统计钩子，使用首批
输入预热 5 次，每批计时前后同步 CUDA。计时包含全部 T 步前向、logits 时间平均和
膜电位重置，不包含数据加载、CPU 到 GPU 传输、指标计算或可视化，使用 FP32。
平均值为所有计时批次总耗时除以实际样本总数，包含不足整批的尾批；同时保存
`inference_batch_size` 和 `inference_warmup_steps`。默认是批量推理摊到每个样本的
耗时；测量逐个样本推理延迟时，在独立评估命令中指定 `--batch-size 1`。

## 恢复与独立评估

恢复时使用原实验 YAML 和对应阶段。`checkpoint_latest.pth` 的 `phase` 是 `search`
或 `finetune`，必须与 `--resume-type` 匹配；普通训练也属于 `search` 阶段。
checkpoint 会恢复模型类型、时间步、剪枝开关、评估方式及预处理，其他训练参数使用
指定 YAML/命令行。恢复优化器、调度器与 scaler，但不保证随机序列逐步复现。

```bash
# 搜索中断后恢复；普通训练同理，更换为 train 配置及输出路径。
python main.py --config configs/image_classification/sew_resnet18_mstar_prune.yaml --data-path /path/to/soc --resume logs/sew_resnet18_mstar_prune/checkpoint_latest.pth --resume-type search

# 微调阶段中断后恢复。
python main.py --config configs/image_classification/sew_resnet18_mstar_prune.yaml --data-path /path/to/soc --resume logs/sew_resnet18_mstar_prune/checkpoint_latest.pth --resume-type finetune

# 独立评估最终权重。模型与预处理从 checkpoint 恢复，可只提供 soc/test 目录树。
python main.py --test-only --resume logs/sew_resnet18_mstar_prune/checkpoint_final.pth --resume-type test --data-path /path/to/soc --output-dir logs/mstar_eval --batch-size 64 --workers 4
```

短程检查可在训练命令后附加 `--epoch-search 1 --epoch-finetune 1`；普通训练使用
`--epoch-search 1 --epoch-finetune 0`。本次整合未添加测试代码。
