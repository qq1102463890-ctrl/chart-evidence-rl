# 题单与证据

每行对应一张图和一道派生题。两个文件均为 UTF-8 JSON Lines，经 gzip 压缩。

| 文件 | 来源划分 | 图数 | 目标读数 |
|---|---|---:|---:|
| `plotqa.jsonl.gz` | PlotQA test | 512 | 2,304 |
| `chartqa.jsonl.gz` | ChartQA test | 256 | 1,152 |

`qid` 含来源、图号及对象数，`question` 是英文原题，`targets` 是题目所问对象。`gold.evidence` 保存每个对象的原表值、单位及允许区间两端，`gold.ordering` 是参考排序。`responses` 内的三个字符串为模型直接生成的文本。键 `sft`、`seed1`、`seed2` 分别对应监督微调及两次强化学习。`expected.json` 是从原实验归档导出的独立校验计数。

题单构建时，从源表匹配对象和数值，用图轴刻度确定容差，并核对柱形几何。每档抽取 3、4、5 或 6 个能区分顺序的对象。实验按图和图表家族排查与项目既有题目的重复。基座预训练对公开图片的覆盖情况未知。

PlotQA 原图及数据表取自版本 `0d448b31e8b1868192e1cca19fe1dc991ea72551`。示例图片的 SHA-256 为 `accdb348520c59da1427ff62c3e2a0a0c4df8e54d3249be600dd36fe7960b40e`。[PlotQA 作者仓库](https://github.com/NiteshMethani/PlotQA) · [锁定的数据快照](https://huggingface.co/datasets/Dodon/PlotQA_dataset/tree/0d448b31e8b1868192e1cca19fe1dc991ea72551)

ChartQA 使用官方测试图、表和图形标注，数据版本为 `af8b6f5c08c95085271561c2a3f9d15f2b5a9031`。图片和数据的使用请参考各上游的说明。[ChartQA 作者仓库](https://github.com/vis-nlp/ChartQA) · [锁定的数据快照](https://huggingface.co/datasets/ahmed-masry/ChartQA/tree/af8b6f5c08c95085271561c2a3f9d15f2b5a9031)
