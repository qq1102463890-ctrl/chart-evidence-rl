# 多对象训练

`train.py` 是从原实验整理出的单卡训练入口。它读取已封存的题序和训练清单，先校验图像与起点权重，再传给 TRL。SFT 训练助手回答部分，GRPO 对同一道题的四次采样应用逐项奖励。推理前所有组都固定使用最终更新的权重。

| 文件 | 用途 |
|---|---|
| `sft.jsonl.gz`、`grpo.jsonl.gz` | 英文提示、原图相对路径和摘要、读数容差、参考顺序 |
| `reward.py` | 80/10/10 奖励，包括图轴默认单位、显式单位折算、重复标签和缺项处理 |
| `protocol.json` | 基座版本、LoRA 起点摘要、两个种子、训练配置与最终适配器摘要 |
| `requirements.txt` | 原实验所用核心 Python 依赖版本 |

### 运行条件

原实验采用 Qwen3-VL-2B-Instruct、BF16、SDPA 和 LoRA（rank 16、alpha 32、dropout 0.05）。GPU 训练需要提供公开 PlotQA 训练图，按照清单放在 `IMAGES/plotqa/train/png/`。SFT 还需要原项目的双对象 LoRA 起点；GRPO 需要对应的多对象 SFT 起点。适配器权重单独保存在原实验环境，仓库提供 SHA-256 核对。仓库首页的 CPU 复算已包含所需全部输入。

以下为从仓库根目录启动各阶段的命令示例。请替换大写路径，第二个 GRPO 种子为 20261126，使用独立输出目录。整理发布时验证了训练清单、目标生成和奖励函数，实际成绩来自此前完成的训练。

```bash
python -m training.train sft --images IMAGES --adapter INITIAL_ADAPTER --output outputs/sft
python -m training.train grpo --images IMAGES --adapter MULTI_SFT_ADAPTER --seed 20261125 --output outputs/seed1
```

奖励规则有 13 个无需 GPU 的固定检查案例，覆盖缺失读数、重复冲突、边界、单位和排序。可在仓库根目录执行：

```bash
python -c "from training.reward import engineered; print(engineered())"
```

### 从公开基座开始自己的实验

有训练图和合适 GPU 的读者，可以用新初始化的 LoRA 启动同一训练配方，再从自己的 SFT 结果继续 GRPO。这个选项改变了 SFT 起点，所得成绩应独立报告。

```bash
python -m training.train sft --images IMAGES --fresh-lora --output outputs/new-sft
python -m training.train grpo --images IMAGES --adapter outputs/new-sft/adapter --seed 20261125 --output outputs/new-grpo
```

## 实验设计

| 阶段 | 样本与设置 | 学习目标 |
|---|---|---|
| 多对象监督微调（SFT） | 单张 RTX 5090；8,000 张 PlotQA 训练图；3～6 个对象各 2,000 张；1,000 次更新 | 从已有双对象 LoRA 权重继续训练，生成对象、读数和完整排序 |
| 组相对策略优化（GRPO） | 同一 SFT 起点；256 道 PlotQA 训练题；每题采样 4 次；每个种子 64 次更新 | 80% 按对象核对读数，10% 按对象对核对先后，10% 奖励整题全对 |
| 评测 | PlotQA 测试图按对象数每档 128 张，ChartQA 每档 64 张；每张图一道题 | 使用固定最终权重、相同提示及贪心解码；分别评估读数与排序 |

例如 4 个对象有 6 对先后关系。如果读对 3 个数、6 对顺序都正确，训练奖励为 **70/100**（读数 60 分、顺序 10 分、整题 0 分）。两个训练种子是 20261125 和 20261126。
