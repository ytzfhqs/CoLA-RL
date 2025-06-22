# README

**CoLA Task RL Training Codebase**

[English | [中文](README.md)]

## Table of Contents

- [Project Introduction](#project-introduction)
- [Changelog](#changelog)
- [Main Results](#main-results)
- [How to Use](#how-to-use)
- [TODO](#TODO)

## Project Introduction

This repository focuses on leveraging the `Qwen3` series models to perform sentence acceptability classification for the `CoLA` (Corpus of Linguistic Acceptability) subtask within the `GLUE` benchmark, using Reinforcement Learning (RL) methods. The codebase implements the complete pipeline of data preprocessing, model training, and evaluation, making it easy to get started and reproduce related research.

## Changelog

[25/06/22] Completed the data processing pipeline, `GRPO` training script (based on the `verl` framework), and [documentation](docs).

## Main Results

- Evaluation Metric: [Matthews Correlation Coefficient (MCC)](https://en.wikipedia.org/wiki/Phi_coefficient)
- Prompt Template:

```python
prompt = """
Decide whether the following sentence is grammatically acceptable or not. If it is grammatically correct, answer "acceptable". If not, answer "unacceptable". Only output "acceptable" or "unacceptable", and do not output any other information.

Sentence: {sentence}

Your answer:
"""
```

|      Model       | Shot Setting | Validation | Test (Kaggle) |
| :--------------: | :----------: | :--------: | :-----------: |
|    Qwen3-0.6B    |  zero-shot   |   0.223    |   TBA         |
| DeepSeek V3 0324 |  zero-shot   |   0.726    |   TBA         |
| DeepSeek R1 0120 |  zero-shot   |   0.636    |   TBA         |

## How to Use

### Environment Setup

> [!TIP]
> See the [documentation](docs/verl框架训练与Debug.md) for details.

### Model Download

- Download `Qwen3` series models from [ModelScope](https://modelscope.cn/home) or [HuggingFace](https://huggingface.co/models) and place them under the `model` directory.

### GRPO Training

- Edit the script `run_grpo_qwen3_0.6b.sh` to set up your `wandb API key`, working directory, and training GPU ID.
- Start training:

```shell
bash run_grpo_qwen3_0.6b.sh
```

## TODO

- Compare the effect of different RL algorithms on CoLA classification.
- Compare the effect of models with different parameter sizes on CoLA classification.
- Upload `wandb` reports.

## Acknowledgments

- [CoLA Dataset](https://nyu-mll.github.io/CoLA/)
- [Verl Framework](https://github.com/volcengine/verl)