<p>
  <h1>
    <img src="./assets/logo.svg" height=150px align="right"/>
   <var>&micro<sup>2</sup></var>Tokenizer: Differentiable Multi-Scale Multi-Modal Tokenizer for Radiology Report Generation
  </h1>
</p>

[![PWC](https://img.shields.io/badge/%F0%9F%93%8E%20arXiv-Paper-red)](https://u2tokenizer.github.io/static/pdfs/%CE%BC2_Tokenizer.pdf)
[![PWC](https://img.shields.io/badge/%F0%9F%8C%8E%20Website-Official%20Page-blue)](https://u2tokenizer.github.io/)
[![PWC](https://img.shields.io/badge/🤗%20Dataset-CT--RATE--Thinking-yellow)](https://huggingface.co/datasets/AlpachinoNLP/CT-RATE-Thinking)
<br>
[![PWC](https://img.shields.io/badge/🤗%20Model-u2Qwen3--1.7B--Instruct-green)](https://huggingface.co/AlpachinoNLP/u2Qwen3-1.7B-Instruct)
[![PWC](https://img.shields.io/badge/🤗%20Model-u2Qwen3--4B--Instruct-green)](https://huggingface.co/AlpachinoNLP/u2Qwen3-4B-Instruct)
[![PWC](https://img.shields.io/badge/🤗%20Model-u2Qwen3--4B--Thinking-green)](https://huggingface.co/AlpachinoNLP/u2Qwen3-4B-Thinking)
---
> 🎉🎉🎉 Our Paper accepted by the 28th conference of The Medical Image Computing and Computer Assisted Intervention Society (MICCAI). See you in Daejeon, Korea from September 23-27, 2025.

<p align="center">
  <img src="./assets/cover.svg">
</p>


This repository contains the official paper for μ² Tokenizer, a novel approach for automated radiology report generation (RRG) introduced in the paper "μ² Tokenizer: Differentiable Multi-Scale Multi-Modal Tokenizer for Radiology Report Generation".

Our proposed model, μ²LLM, leverages a multi-scale, multi-modal architecture to generate accurate and clinically salient radiology reports from CT scans.

## 👋 Introduction

<img src="./assets/ullm.svg">

we introduce μ²LLM, a multi-scale multimodal large language model. At its core is the novel μ² Tokenizer, an intermediate layer that intelligently fuses visual features from CT scans with textual information. The model is further refined using Direct Preference Optimization (DPO), guided by the specialized medical report evaluation metric, GREEN, to ensure the generated reports align with expert standards.

<img src="./assets/dpo.svg">

Our experimental results on four large-scale CT datasets show that μ²LLM outperforms existing methods, highlighting its potential for generating high-quality radiology reports even with limited training data.

## 🚀 Quickstart
Here, we can easily use our model based on Hugging Face.

```python
coming soon...
```

## 🤖 Model
| Model | Download Link |
|-------|---------------|
| μ²Qwen3-4B-Thinking | [HuggingFace](https://huggingface.co/AlpachinoNLP/u2Qwen3-4B-Thinking) |
| μ²Qwen3-4B-Instruct | [HuggingFace](https://huggingface.co/AlpachinoNLP/u2Qwen3-4B-Instruct) |
| μ²Qwen3-1.7B-Instruct | [HuggingFace](https://huggingface.co/AlpachinoNLP/u2Qwen3-1.7B-Instruct) |

## ⚙️ Installation
```bash
git clone https://github.com/Siyou-Li/u2Tokenizer.git
cd u2Tokenizer
pip install -r requirements.txt
```
Ensure that the NVIDIA CUDA version 11.8 or above to be compatible with PyTorch 2.2.2.

## 💿 Data

| Dataset | Description | Download |
|---------|-------------|----------|
| CT-RATE-Thinking | Reasoning-augmented VQA pairs and report-level thinking narratives derived from CT-RATE, used for SFT training of μ²LLM. Contains ~666K English VQA pairs, ~666K Chinese VQA pairs, ~42K English report thinking, and ~42K Chinese report thinking across train/val splits. | [HuggingFace](https://huggingface.co/datasets/AlpachinoNLP/CT-RATE-Thinking) |
| CT-RATE | Source CT volumes and original radiology reports (50,188 scans, 21,340 patients). Required to pair with CT-RATE-Thinking. | [HuggingFace](https://huggingface.co/datasets/ibrahimhamamci/CT-RATE) |

The CT-RATE-Thinking dataset was generated from CT-RATE reports using the five-stage CT Report Reasoning Synthesis pipeline described in §2.4 of our paper. See the [dataset README](https://huggingface.co/datasets/AlpachinoNLP/CT-RATE-Thinking) for the full schema and usage instructions.

## 🚄 Training
Coming soon...


## 🧰 System Hardware requirements

For training, stage 1 and 2 use a 4 * 80GB A100 GPU. For inference, a single 40GB A40 GPU is used. For loading model checkpoint, approximately 39GB of CPU memory is required.

## 🫡 Acknowledgements


## ✨ Cite our work

If you find this repo useful, please consider citing: 

```bibtex
@misc{li2025mu2tokenizerdifferentiablemultiscalemultimodal,
      title={${\mu}^2$Tokenizer: Differentiable Multi-Scale Multi-Modal Tokenizer for Radiology Report Generation}, 
      author={Siyou Li and Pengyao Qin and Huanan Wu and Dong Nie and Arun J. Thirunavukarasu and Juntao Yu and Le Zhang},
      year={2025},
      eprint={2507.00316},
      archivePrefix={arXiv},
      primaryClass={cs.LG},
      url={https://arxiv.org/abs/2507.00316}, 
}
```