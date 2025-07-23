# README

**CoLA任务RL训练代码库**

[[English](README_en.md)|中文]

## 目录

- [项目简介](#项目简介)
- [更新日志](#更新日志)
- [主要结果](#主要结果)
- [如何使用](#如何使用)
- [待办事项](#待办事项)

## 项目简介

本仓库主要聚焦于利用`Qwen3`系列模型，通过强化学习（`Reinforcement Learning`, `RL`）技术，完成`GLUE`基准中的`CoLA`（`Corpus of Linguistic Acceptability`）子任务的句子可接受性分类。代码实现了数据预处理、模型训练和评估全流程，方便快速上手与复现相关研究。

## 更新日志

[25/07/23]修复数据准备错误，新增`SFT`数据准备代码和训练脚本（基于`LLaMA-Factory`框架），新增`REMAX`、`DAPO`训练脚本，新增`Text Classification`代码，新增`DeepSeek R1 0528`蒸馏`COLA`数据集。

[25/06/22]完成数据处理流程、模型`GRPO`训练脚本（基于`verl`框架）和[文档编写](docs)

## 主要结果

- 对比指标[Matthews相关系数](https://en.wikipedia.org/wiki/Phi_coefficient)（MCC）
- 提示词：

```python
prompt = """
Decide whether the following sentence is grammatically acceptable or not. If it is grammatically correct, answer "acceptable". If not, answer "unacceptable". Only output "acceptable" or "unacceptable", and do not output any other information.

Sentence: {sentence}

Your answer:
"""
```

|      Model       | Shot Setting | 验证集 | 测试集（kaggle） |
| :--------------: | :----------: | :----: | :--------------: |
|    Qwen3-0.6B    |  zero-shot   | 0.223  |      待测试      |
| DeepSeek V3 0324 |  zero-shot   | 0.726  |      待测试      |
| DeepSeek R1 0120 |  zero-shot   | 0.636  |      待测试      |
| DeepSeek R1 0528 |  zero-shot   | 0.658  |      待测试      |

## 如何使用

### 环境搭建

> [!TIP]
> 参阅[文档](docs/verl框架训练与Debug.md)。

### 模型下载

- 从[魔搭社区](https://modelscope.cn/home)或[Huggingface](https://huggingface.co/models)下载`Qwen3`系列模型到`model`文件夹下。

### GRPO训练

- 修改脚本`run_grpo_qwen3_0.6b.sh`，修改`wandb api key`、工作目录和训练`GPU`编号。
- 启动训练：

```shell
bash run_grpo_qwen3_0.6b.sh
```

## 待办事项

- 对比不同RL算法对CoLA分类的效果。
- 对比不同参数量模型对CoLA分类的效果。
- 上传`wandb`报告。

## 致谢

- [CoLA Dataset](https://nyu-mll.github.io/CoLA/)

- [Verl Framework](https://github.com/volcengine/verl)
